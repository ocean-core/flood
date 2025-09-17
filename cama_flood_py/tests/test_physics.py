"""
物理过程模块单元测试
"""

import pytest
import numpy as np
from cama_flood_py.core.physics import PhysicsEngine, calculate_flood_stage_numba


class TestPhysicsEngine:
    """物理引擎测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.nseq = 100
        self.physics = PhysicsEngine(self.nseq)
        
        # 设置测试地形数据
        self.physics.river_elevation.fill(10.0)  # 河床高程10m
        self.physics.river_width.fill(50.0)      # 河道宽度50m
        self.physics.river_height.fill(5.0)      # 河道深度5m
        self.physics.river_length.fill(1000.0)   # 河道长度1000m
        self.physics.catchment_area.fill(1e6)    # 集水面积1km²
    
    def test_flood_stage_calculation_empty_storage(self):
        """测试空存储量的洪水阶段计算"""
        # 设置零存储量
        self.physics.river_storage.fill(0.0)
        self.physics.flood_storage.fill(0.0)
        
        # 计算洪水阶段
        self.physics.calculate_flood_stage()
        
        # 验证结果
        assert np.all(self.physics.river_depth == 0.0)
        assert np.all(self.physics.flood_depth == 0.0)
        assert np.all(self.physics.surface_elevation == 10.0)  # 等于河床高程
    
    def test_flood_stage_calculation_river_only(self):
        """测试仅河道存储的洪水阶段计算"""
        # 设置河道存储量（未满）
        river_capacity = 50.0 * 5.0 * 1000.0  # width * height * length = 250,000 m³
        self.physics.river_storage.fill(river_capacity * 0.5)  # 50%容量
        self.physics.flood_storage.fill(0.0)
        
        # 计算洪水阶段
        self.physics.calculate_flood_stage()
        
        # 验证结果
        expected_depth = (river_capacity * 0.5) / (50.0 * 1000.0)  # storage / (width * length)
        assert np.allclose(self.physics.river_depth, expected_depth)
        assert np.all(self.physics.flood_depth == 0.0)
        assert np.allclose(self.physics.surface_elevation, 10.0 + expected_depth)
    
    def test_flood_stage_calculation_with_overflow(self):
        """测试河道溢流的洪水阶段计算"""
        # 设置超过河道容量的存储量
        river_capacity = 50.0 * 5.0 * 1000.0  # 250,000 m³
        total_storage = river_capacity * 1.5  # 150%容量
        self.physics.river_storage.fill(river_capacity)
        self.physics.flood_storage.fill(total_storage - river_capacity)
        
        # 计算洪水阶段
        self.physics.calculate_flood_stage()
        
        # 验证结果
        assert np.allclose(self.physics.river_depth, 5.0)  # 等于河道深度
        assert np.all(self.physics.flood_depth > 0.0)      # 有漫滩水深
        assert np.all(self.physics.surface_elevation > 15.0)  # 高于河道顶部
    
    def test_numba_function_consistency(self):
        """测试Numba函数与类方法的一致性"""
        # 设置测试数据
        self.physics.river_storage.fill(100000.0)
        self.physics.flood_storage.fill(50000.0)
        
        # 使用类方法计算
        self.physics.calculate_flood_stage()
        class_result = (
            self.physics.river_depth.copy(),
            self.physics.flood_depth.copy(),
            self.physics.surface_elevation.copy()
        )
        
        # 使用Numba函数计算
        numba_result = calculate_flood_stage_numba(
            self.physics.river_storage,
            self.physics.flood_storage,
            self.physics.river_elevation,
            self.physics.river_width,
            self.physics.river_height,
            self.physics.river_length,
            self.physics.catchment_area
        )
        
        # 验证一致性
        np.testing.assert_allclose(class_result[0], numba_result[0], rtol=1e-10)
        np.testing.assert_allclose(class_result[1], numba_result[1], rtol=1e-10)
        np.testing.assert_allclose(class_result[2], numba_result[2], rtol=1e-10)
    
    def test_state_dict_operations(self):
        """测试状态字典操作"""
        # 设置初始状态
        self.physics.river_storage.fill(1000.0)
        self.physics.flood_storage.fill(500.0)
        
        # 获取状态字典
        state_dict = self.physics.get_state_dict()
        
        # 验证状态字典内容
        assert 'river_storage' in state_dict
        assert 'flood_storage' in state_dict
        assert np.all(state_dict['river_storage'] == 1000.0)
        assert np.all(state_dict['flood_storage'] == 500.0)
        
        # 修改状态字典
        new_state = {
            'river_storage': np.full(self.nseq, 2000.0),
            'flood_storage': np.full(self.nseq, 1000.0)
        }
        
        # 设置新状态
        self.physics.set_state_dict(new_state)
        
        # 验证状态更新
        assert np.all(self.physics.river_storage == 2000.0)
        assert np.all(self.physics.flood_storage == 1000.0)


class TestFloodStageNumba:
    """Numba函数专项测试"""
    
    def test_edge_cases(self):
        """测试边界情况"""
        nseq = 5
        
        # 创建测试数据
        river_storage = np.array([0.0, 1000.0, 100000.0, 500000.0, 1000000.0])
        flood_storage = np.array([0.0, 0.0, 0.0, 100000.0, 200000.0])
        river_elevation = np.full(nseq, 10.0)
        river_width = np.array([0.0, 50.0, 50.0, 50.0, 50.0])  # 第一个为0测试除零
        river_height = np.full(nseq, 5.0)
        river_length = np.full(nseq, 1000.0)
        catchment_area = np.full(nseq, 1e6)
        
        # 计算结果
        river_depth, flood_depth, surface_elevation = calculate_flood_stage_numba(
            river_storage, flood_storage, river_elevation,
            river_width, river_height, river_length, catchment_area
        )
        
        # 验证边界情况
        assert river_depth[0] == 0.0  # 零宽度情况
        assert flood_depth[0] == 0.0
        assert surface_elevation[0] == 10.0
        
        # 验证正常情况
        assert river_depth[1] > 0.0
        assert flood_depth[1] == 0.0  # 未溢流
        
        # 验证溢流情况
        assert river_depth[3] > 0.0
        assert flood_depth[3] > 0.0  # 有溢流
    
    def test_performance(self):
        """测试性能（大规模数据）"""
        nseq = 100000
        
        # 创建大规模测试数据
        river_storage = np.random.rand(nseq) * 1e6
        flood_storage = np.random.rand(nseq) * 1e5
        river_elevation = np.full(nseq, 10.0)
        river_width = np.full(nseq, 50.0)
        river_height = np.full(nseq, 5.0)
        river_length = np.full(nseq, 1000.0)
        catchment_area = np.full(nseq, 1e6)
        
        # 计算时间
        import time
        start_time = time.time()
        
        river_depth, flood_depth, surface_elevation = calculate_flood_stage_numba(
            river_storage, flood_storage, river_elevation,
            river_width, river_height, river_length, catchment_area
        )
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # 验证性能（应该在合理时间内完成）
        assert computation_time < 1.0  # 应在1秒内完成
        
        # 验证结果合理性
        assert np.all(river_depth >= 0.0)
        assert np.all(flood_depth >= 0.0)
        assert np.all(surface_elevation >= 10.0)


if __name__ == "__main__":
    pytest.main([__file__])
