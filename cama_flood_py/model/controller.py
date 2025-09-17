"""
主控制器模块
对应Fortran源码: MAIN_cmf.F90 和 cmf_drv_*_mod.F90
"""

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import time

from .config import ConfigManager
from .time_manager import TimeManager, AdaptiveTimeManager
from ..core.physics import PhysicsEngine
from ..core.hydraulics import HydraulicsCalculator
from ..core.storage import StorageCalculator, WaterBalanceMonitor
from ..data.io_manager import IOManager
from ..utils.constants import ModelConstants


class CaMaFloodModel:
    """
    CaMa-Flood主控制器
    对应Fortran: MAIN_cmf (MAIN_cmf.F90) 和控制模块
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化CaMa-Flood模型
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # 初始化组件
        self.time_manager = None
        self.physics_engine = None
        self.hydraulics_calculator = None
        self.storage_calculator = None
        self.io_manager = None
        self.water_balance_monitor = None
        
        # 模型状态
        self.is_initialized = False
        self.current_step = 0
        self.total_steps = 0
        
        # 性能统计
        self.performance_stats = {
            'initialization_time': 0.0,
            'simulation_time': 0.0,
            'io_time': 0.0,
            'physics_time': 0.0
        }
        
        self.logger.info("CaMa-Flood模型实例创建完成")
    
    def initialize(self) -> None:
        """
        模型初始化
        对应Fortran: CMF_DRV_INIT (cmf_drv_control_mod.F90:89-178行)
        """
        start_time = time.time()
        
        self.logger.info("开始模型初始化...")
        
        # 验证配置
        if not self.config_manager.validate_config():
            raise ValueError("配置验证失败")
        
        # 初始化时间管理器
        self._initialize_time_manager()
        
        # 初始化I/O管理器
        self._initialize_io_manager()
        
        # 初始化物理引擎和计算器
        self._initialize_physics_components()
        
        # 加载地图数据
        self._load_map_data()
        
        # 初始化状态变量
        self._initialize_state_variables()
        
        # 初始化诊断变量
        self._initialize_diagnostic_variables()
        
        # 初始化输出
        self._initialize_output()
        
        self.is_initialized = True
        self.performance_stats['initialization_time'] = time.time() - start_time
        
        self.logger.info(f"模型初始化完成，耗时: {self.performance_stats['initialization_time']:.2f}秒")
    
    def run(self) -> None:
        """
        运行模型主循环
        对应Fortran: MAIN_cmf主循环 (MAIN_cmf.F90:58-98行)
        """
        if not self.is_initialized:
            raise RuntimeError("模型未初始化，请先调用initialize()")
        
        start_time = time.time()
        self.logger.info("开始模型模拟...")
        
        # 主时间循环
        while not self.time_manager.is_simulation_finished():
            step_start_time = time.time()
            
            # 时间步进
            self.advance_one_step()
            
            # 输出结果
            if self.time_manager.should_output(self.config.output.frequency):
                self._write_output()
            
            # 写入重启文件 (可选)
            if self.current_step % 1000 == 0:  # 每1000步写一次重启文件
                self._write_restart()
            
            # 更新性能统计
            step_time = time.time() - step_start_time
            self.performance_stats['physics_time'] += step_time
            
            # 进度报告
            if self.current_step % 100 == 0:
                progress = self.time_manager.get_progress()
                self.logger.info(f"模拟进度: {progress:.1%}, 当前时间: {self.time_manager.format_current_time()}")
        
        self.performance_stats['simulation_time'] = time.time() - start_time
        self.logger.info(f"模型模拟完成，总耗时: {self.performance_stats['simulation_time']:.2f}秒")
        
        # 输出性能统计
        self._log_performance_stats()
    
    def advance_one_step(self) -> None:
        """
        推进一个时间步
        对应Fortran: CMF_DRV_ADVANCE (cmf_drv_advance_mod.F90:46-159行)
        """
        # 1. 更新时间
        if isinstance(self.time_manager, AdaptiveTimeManager):
            # 自适应时间步长
            max_velocity = self.hydraulics_calculator.get_max_velocity()
            grid_spacing = 10000.0  # 假设网格间距，实际应从配置获取
            dt = self.time_manager.update_time_adaptive(max_velocity, grid_spacing)
        else:
            # 固定时间步长
            self.time_manager.update_time()
            dt = self.time_manager.timestep
        
        # 2. 读取强迫数据
        self._update_forcing_data()
        
        # 3. 物理过程推进
        self._advance_physics(dt)
        
        # 4. 更新计数器
        self.current_step += 1
    
    def _advance_physics(self, dt: float) -> None:
        """
        物理过程推进
        对应Fortran: CMF_PHYSICS_ADVANCE调用序列
        
        Args:
            dt: 时间步长
        """
        # 1. 洪水阶段计算
        self.physics_engine.calculate_flood_stage()
        
        # 2. 流量计算
        river_outflow, flood_outflow = self.hydraulics_calculator.calculate_outflow(
            self.physics_engine.river_depth,
            self.physics_engine.flood_depth,
            self.physics_engine.surface_elevation,
            self.physics_engine.river_width,
            self.physics_engine.river_length,
            self.physics_engine.next_distance,
            self.physics_engine.next_index
        )
        
        # 更新物理引擎中的流量
        self.physics_engine.river_outflow[:] = river_outflow
        self.physics_engine.flood_outflow[:] = flood_outflow
        
        # 3. 入流计算
        river_inflow = self.hydraulics_calculator.calculate_inflow(
            self.physics_engine.river_outflow,
            self.physics_engine.next_index,
            self.current_runoff  # 当前径流强迫
        )
        self.physics_engine.river_inflow[:] = river_inflow
        
        # 4. 存储量更新
        new_river_storage, new_flood_storage = self.storage_calculator.update_storage(
            self.physics_engine.river_storage,
            self.physics_engine.flood_storage,
            self.physics_engine.river_inflow,
            self.physics_engine.river_outflow,
            np.zeros_like(self.physics_engine.river_inflow),  # 漫滩入流（简化为0）
            self.physics_engine.flood_outflow,
            dt
        )
        
        # 更新存储量
        self.physics_engine.river_storage[:] = new_river_storage
        self.physics_engine.flood_storage[:] = new_flood_storage
        
        # 5. 水量平衡检查
        balance_stats = self.water_balance_monitor.check_water_balance(
            self.physics_engine.river_storage - (new_river_storage - self.physics_engine.river_storage),
            self.physics_engine.flood_storage - (new_flood_storage - self.physics_engine.flood_storage),
            new_river_storage,
            new_flood_storage,
            self.physics_engine.river_inflow,
            self.physics_engine.river_outflow,
            np.zeros_like(self.physics_engine.river_inflow),
            self.physics_engine.flood_outflow,
            dt
        )
        
        # 记录严重的水量平衡误差
        if balance_stats['max_absolute_error'] > 1e-3:
            self.logger.warning(f"水量平衡误差较大: {balance_stats['max_absolute_error']:.2e}")
    
    def _initialize_time_manager(self) -> None:
        """初始化时间管理器"""
        start_time = self.config_manager.get_start_datetime()
        end_time = self.config_manager.get_end_datetime()
        timestep = self.config.time.timestep
        
        if self.config.model.adaptive_timestep:
            self.time_manager = AdaptiveTimeManager(start_time, end_time, timestep)
        else:
            self.time_manager = TimeManager(start_time, end_time, timestep)
        
        self.total_steps = self.time_manager.total_steps
        self.logger.info(f"时间管理器初始化完成，总步数: {self.total_steps}")
    
    def _initialize_io_manager(self) -> None:
        """初始化I/O管理器"""
        self.io_manager = IOManager(self.config.output.directory)
        
        # 验证输入文件
        if not self.io_manager.validate_input_files(self.config):
            self.logger.warning("部分输入文件验证失败，将使用默认值")
    
    def _initialize_physics_components(self) -> None:
        """初始化物理计算组件"""
        # 估算网格数量（实际应从地图文件获取）
        nseq = 1440 * 720  # 15分分辨率全球网格
        
        self.physics_engine = PhysicsEngine(nseq)
        self.hydraulics_calculator = HydraulicsCalculator(
            nseq, 
            self.config.physics.manning_river,
            self.config.physics.manning_flood
        )
        self.storage_calculator = StorageCalculator(nseq)
        self.water_balance_monitor = WaterBalanceMonitor(nseq)
        
        self.logger.info(f"物理计算组件初始化完成，网格数: {nseq}")
    
    def _load_map_data(self) -> None:
        """加载地图数据"""
        try:
            map_data = self.io_manager.load_map_data(self.config)
            self.physics_engine.set_map_data(map_data)
            self.logger.info("地图数据加载完成")
        except Exception as e:
            self.logger.error(f"地图数据加载失败: {e}")
            raise
    
    def _initialize_state_variables(self) -> None:
        """初始化状态变量"""
        # 如果有重启文件，从重启文件读取
        restart_file = Path(self.config.output.directory) / "restart.nc"
        if restart_file.exists() and self.config.model.restart_mode:
            try:
                state_data = self.io_manager.read_restart_file("restart.nc")
                self.physics_engine.set_state_dict(state_data)
                self.logger.info("从重启文件加载状态变量")
            except Exception as e:
                self.logger.warning(f"重启文件读取失败，使用默认初值: {e}")
                self._set_default_initial_conditions()
        else:
            self._set_default_initial_conditions()
    
    def _set_default_initial_conditions(self) -> None:
        """设置默认初始条件"""
        # 简单的初始条件：河道有少量水，漫滩为空
        self.physics_engine.river_storage.fill(100.0)  # 100 m³初始存储
        self.physics_engine.flood_storage.fill(0.0)
        self.logger.info("设置默认初始条件")
    
    def _initialize_diagnostic_variables(self) -> None:
        """初始化诊断变量"""
        # 计算初始的诊断变量
        self.physics_engine.calculate_flood_stage()
        self.logger.info("诊断变量初始化完成")
    
    def _initialize_output(self) -> None:
        """初始化输出"""
        # 创建输出目录
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"输出目录创建: {output_dir}")
    
    def _update_forcing_data(self) -> None:
        """更新强迫数据"""
        # 简化实现：使用常数径流
        # 实际应从强迫数据文件读取并插值
        self.current_runoff = np.ones(self.physics_engine.nseq) * 0.1  # 0.1 m³/s
    
    def _write_output(self) -> None:
        """写入输出文件"""
        try:
            # 准备输出数据
            output_data = {}
            coordinates = {
                'time': [self.time_manager.current_time],
                'lat': np.linspace(-90, 90, 720),
                'lon': np.linspace(-180, 180, 1440)
            }
            
            # 选择输出变量
            state_dict = self.physics_engine.get_state_dict()
            for var_name in self.config.output.variables:
                if var_name in state_dict:
                    # 将一维数据重塑为二维地图
                    data_1d = state_dict[var_name]
                    data_2d = data_1d.reshape(720, 1440)  # ny, nx
                    output_data[var_name] = np.expand_dims(data_2d, axis=0)  # 添加时间维度
            
            # 生成输出文件名
            filename = self.time_manager.get_output_filename(
                f"output_{self.config.output.tag}", "nc"
            )
            
            # 写入文件
            if self.config.output.format == "netcdf":
                self.io_manager.write_output_netcdf(filename, output_data, coordinates)
            else:
                # 二进制格式输出
                for var_name, data in output_data.items():
                    binary_filename = f"{var_name}_{self.time_manager.get_output_filename('', 'bin')}"
                    self.io_manager.write_output_binary(binary_filename, data.squeeze())
            
            self.logger.debug(f"输出文件写入: {filename}")
            
        except Exception as e:
            self.logger.error(f"输出文件写入失败: {e}")
    
    def _write_restart(self) -> None:
        """写入重启文件"""
        try:
            state_data = self.physics_engine.get_state_dict()
            coordinates = {
                'lat': np.linspace(-90, 90, 720),
                'lon': np.linspace(-180, 180, 1440)
            }
            
            # 重塑数据为二维
            for var_name in state_data:
                data_1d = state_data[var_name]
                state_data[var_name] = data_1d.reshape(720, 1440)
            
            self.io_manager.create_restart_file("restart.nc", state_data, coordinates)
            self.logger.debug("重启文件写入完成")
            
        except Exception as e:
            self.logger.error(f"重启文件写入失败: {e}")
    
    def _log_performance_stats(self) -> None:
        """记录性能统计"""
        stats = self.performance_stats
        self.logger.info("=== 性能统计 ===")
        self.logger.info(f"初始化时间: {stats['initialization_time']:.2f}秒")
        self.logger.info(f"模拟时间: {stats['simulation_time']:.2f}秒")
        self.logger.info(f"物理计算时间: {stats['physics_time']:.2f}秒")
        self.logger.info(f"平均每步时间: {stats['physics_time']/max(self.current_step, 1):.4f}秒")
        self.logger.info(f"总步数: {self.current_step}")
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('cama_flood.log')
            ]
        )
    
    def get_results(self) -> Dict[str, Any]:
        """
        获取模拟结果
        
        Returns:
            dict: 模拟结果字典
        """
        if not self.is_initialized:
            return {}
        
        return {
            'state_variables': self.physics_engine.get_state_dict(),
            'performance_stats': self.performance_stats.copy(),
            'simulation_info': {
                'total_steps': self.total_steps,
                'current_step': self.current_step,
                'start_time': self.time_manager.start_time,
                'end_time': self.time_manager.end_time,
                'current_time': self.time_manager.current_time
            }
        }
    
    def finalize(self) -> None:
        """
        模型终止处理
        对应Fortran: CMF_DRV_END (cmf_drv_control_mod.F90:335行)
        """
        if self.is_initialized:
            # 写入最终输出
            self._write_output()
            self._write_restart()
            
            # 清理资源
            self.logger.info("模型终止处理完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.finalize()
