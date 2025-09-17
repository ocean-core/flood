"""
集成测试模块
测试完整的模型运行流程
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from cama_flood_py.model.controller import CaMaFloodModel
from cama_flood_py.model.config import ConfigManager, create_default_config
from cama_flood_py.core.physics import PhysicsEngine
from cama_flood_py.core.hydraulics import HydraulicsCalculator
from cama_flood_py.core.storage import StorageCalculator


class TestModelIntegration:
    """模型集成测试类"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 创建测试配置文件
        self.config_path = self.temp_path / "test_config.yaml"
        create_default_config(str(self.config_path))
        
        # 修改配置为测试设置
        config_manager = ConfigManager(str(self.config_path))
        config = config_manager.config
        
        # 设置短时间模拟
        config.time.start_year = 2000
        config.time.start_month = 1
        config.time.start_day = 1
        config.time.end_year = 2000
        config.time.end_month = 1
        config.time.end_day = 2  # 只模拟1天
        config.time.timestep = 3600.0  # 1小时步长
        
        # 设置输出目录
        config.output.directory = str(self.temp_path)
        config.output.frequency = 1  # 每小时输出
        
        # 设置输入目录（使用临时目录，文件不存在但不会报错）
        config.input.map_directory = str(self.temp_path / "map")
        config.input.forcing_directory = str(self.temp_path / "forcing")
        
        config_manager.save_config(str(self.config_path))
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = CaMaFloodModel(str(self.config_path))
        
        # 验证初始状态
        assert not model.is_initialized
        assert model.current_step == 0
        
        # 初始化模型（会有警告但不会失败）
        model.initialize()
        
        # 验证初始化后状态
        assert model.is_initialized
        assert model.physics_engine is not None
        assert model.hydraulics_calculator is not None
        assert model.storage_calculator is not None
        assert model.time_manager is not None
    
    def test_single_step_advance(self):
        """测试单步推进"""
        model = CaMaFloodModel(str(self.config_path))
        model.initialize()
        
        # 记录初始状态
        initial_time = model.time_manager.current_time
        initial_step = model.current_step
        
        # 执行一步
        model.advance_one_step()
        
        # 验证状态更新
        assert model.time_manager.current_time > initial_time
        assert model.current_step == initial_step + 1
    
    def test_short_simulation(self):
        """测试短时间模拟"""
        model = CaMaFloodModel(str(self.config_path))
        model.initialize()
        
        # 记录初始状态
        initial_results = model.get_results()
        
        # 运行几步
        for _ in range(5):
            if not model.time_manager.is_simulation_finished():
                model.advance_one_step()
        
        # 验证模拟进展
        final_results = model.get_results()
        assert final_results['simulation_info']['current_step'] > 0
        assert final_results['simulation_info']['current_time'] > initial_results['simulation_info']['current_time']
    
    def test_water_balance_conservation(self):
        """测试水量平衡守恒"""
        model = CaMaFloodModel(str(self.config_path))
        model.initialize()
        
        # 设置初始存储量
        initial_river_storage = np.sum(model.physics_engine.river_storage)
        initial_flood_storage = np.sum(model.physics_engine.flood_storage)
        initial_total = initial_river_storage + initial_flood_storage
        
        # 运行几步
        for _ in range(3):
            if not model.time_manager.is_simulation_finished():
                model.advance_one_step()
        
        # 检查水量平衡
        final_river_storage = np.sum(model.physics_engine.river_storage)
        final_flood_storage = np.sum(model.physics_engine.flood_storage)
        final_total = final_river_storage + final_flood_storage
        
        # 水量应该守恒（考虑数值误差和径流输入）
        # 由于有径流输入，最终水量应该增加
        assert final_total >= initial_total
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with CaMaFloodModel(str(self.config_path)) as model:
            model.initialize()
            
            # 验证模型可用
            assert model.is_initialized
            
            # 执行一些操作
            model.advance_one_step()
            
        # 上下文退出后，模型应该正常终止（无异常）


class TestComponentIntegration:
    """组件集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.nseq = 100
        
        # 创建各组件
        self.physics = PhysicsEngine(self.nseq)
        self.hydraulics = HydraulicsCalculator(self.nseq, 0.03, 0.10)
        self.storage = StorageCalculator(self.nseq)
        
        # 设置测试地形
        self._setup_test_topography()
    
    def _setup_test_topography(self):
        """设置测试地形"""
        # 创建简单的河网：线性连接
        self.physics.river_elevation = np.linspace(100, 50, self.nseq)  # 下游坡度
        self.physics.river_width.fill(50.0)
        self.physics.river_height.fill(5.0)
        self.physics.river_length.fill(1000.0)
        self.physics.catchment_area.fill(1e6)
        self.physics.next_distance.fill(1000.0)
        
        # 设置河网连接
        self.physics.next_index = np.arange(1, self.nseq + 1)
        self.physics.next_index[-1] = 0  # 最后一个为出口
        
        # 设置初始存储量
        self.physics.river_storage.fill(10000.0)  # 10,000 m³
        self.physics.flood_storage.fill(0.0)
    
    def test_physics_hydraulics_coupling(self):
        """测试物理过程与水力学的耦合"""
        # 计算洪水阶段
        self.physics.calculate_flood_stage()
        
        # 验证水深计算
        assert np.all(self.physics.river_depth > 0)
        assert np.all(self.physics.surface_elevation > self.physics.river_elevation)
        
        # 计算流量
        river_outflow, flood_outflow = self.hydraulics.calculate_outflow(
            self.physics.river_depth,
            self.physics.flood_depth,
            self.physics.surface_elevation,
            self.physics.river_width,
            self.physics.river_length,
            self.physics.next_distance,
            self.physics.next_index
        )
        
        # 验证流量计算
        assert np.all(river_outflow >= 0)
        assert np.all(flood_outflow >= 0)
        assert river_outflow[-1] == 0  # 出口流量为0
        assert np.any(river_outflow[:-1] > 0)  # 有流量产生
    
    def test_hydraulics_storage_coupling(self):
        """测试水力学与存储量的耦合"""
        # 先计算流量
        self.physics.calculate_flood_stage()
        
        river_outflow, flood_outflow = self.hydraulics.calculate_outflow(
            self.physics.river_depth,
            self.physics.flood_depth,
            self.physics.surface_elevation,
            self.physics.river_width,
            self.physics.river_length,
            self.physics.next_distance,
            self.physics.next_index
        )
        
        # 计算入流
        runoff = np.full(self.nseq, 1.0)  # 1 m³/s径流
        river_inflow = self.hydraulics.calculate_inflow(
            river_outflow, self.physics.next_index, runoff
        )
        
        # 更新存储量
        dt = 3600.0  # 1小时
        new_river_storage, new_flood_storage = self.storage.update_storage(
            self.physics.river_storage,
            self.physics.flood_storage,
            river_inflow,
            river_outflow,
            np.zeros(self.nseq),  # 无漫滩入流
            flood_outflow,
            dt
        )
        
        # 验证存储量更新
        assert np.all(new_river_storage >= 0)
        assert np.all(new_flood_storage >= 0)
        
        # 验证质量守恒（粗略检查）
        total_initial = np.sum(self.physics.river_storage + self.physics.flood_storage)
        total_final = np.sum(new_river_storage + new_flood_storage)
        total_input = np.sum(runoff) * dt
        
        # 最终存储应该接近初始存储加输入（考虑出流）
        assert total_final > total_initial  # 有净输入
    
    def test_full_cycle_integration(self):
        """测试完整循环集成"""
        dt = 3600.0
        runoff = np.full(self.nseq, 0.5)  # 0.5 m³/s径流
        
        # 执行多个时间步
        for step in range(5):
            # 1. 洪水阶段计算
            self.physics.calculate_flood_stage()
            
            # 2. 流量计算
            river_outflow, flood_outflow = self.hydraulics.calculate_outflow(
                self.physics.river_depth,
                self.physics.flood_depth,
                self.physics.surface_elevation,
                self.physics.river_width,
                self.physics.river_length,
                self.physics.next_distance,
                self.physics.next_index
            )
            
            # 3. 入流计算
            river_inflow = self.hydraulics.calculate_inflow(
                river_outflow, self.physics.next_index, runoff
            )
            
            # 4. 存储量更新
            new_river_storage, new_flood_storage = self.storage.update_storage(
                self.physics.river_storage,
                self.physics.flood_storage,
                river_inflow,
                river_outflow,
                np.zeros(self.nseq),
                flood_outflow,
                dt
            )
            
            # 5. 更新状态
            self.physics.river_storage[:] = new_river_storage
            self.physics.flood_storage[:] = new_flood_storage
            
            # 验证每步结果
            assert np.all(self.physics.river_storage >= 0)
            assert np.all(self.physics.flood_storage >= 0)
            assert np.all(np.isfinite(self.physics.river_storage))
            assert np.all(np.isfinite(self.physics.flood_storage))


class TestPerformance:
    """性能测试"""
    
    def test_large_scale_performance(self):
        """测试大规模性能"""
        # 创建大规模网格
        nseq = 50000  # 5万网格
        
        physics = PhysicsEngine(nseq)
        hydraulics = HydraulicsCalculator(nseq, 0.03, 0.10)
        
        # 设置随机地形
        physics.river_elevation = np.random.rand(nseq) * 100
        physics.river_width = np.random.rand(nseq) * 100 + 10
        physics.river_height = np.random.rand(nseq) * 10 + 1
        physics.river_length.fill(1000.0)
        physics.catchment_area = np.random.rand(nseq) * 1e6
        physics.next_distance.fill(1000.0)
        physics.next_index = np.random.randint(0, nseq, nseq)
        
        # 设置随机存储量
        physics.river_storage = np.random.rand(nseq) * 1e5
        physics.flood_storage = np.random.rand(nseq) * 1e4
        
        # 性能测试
        import time
        
        start_time = time.time()
        physics.calculate_flood_stage()
        flood_stage_time = time.time() - start_time
        
        start_time = time.time()
        river_outflow, flood_outflow = hydraulics.calculate_outflow(
            physics.river_depth,
            physics.flood_depth,
            physics.surface_elevation,
            physics.river_width,
            physics.river_length,
            physics.next_distance,
            physics.next_index
        )
        outflow_time = time.time() - start_time
        
        # 验证性能（应在合理时间内完成）
        assert flood_stage_time < 2.0  # 洪水阶段计算应在2秒内
        assert outflow_time < 2.0      # 流量计算应在2秒内
        
        # 验证结果正确性
        assert np.all(np.isfinite(physics.river_depth))
        assert np.all(np.isfinite(river_outflow))
        assert np.all(river_outflow >= 0)


if __name__ == "__main__":
    pytest.main([__file__])
