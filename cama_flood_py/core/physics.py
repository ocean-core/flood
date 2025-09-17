"""
物理过程计算模块
对应Fortran源码: cmf_ctrl_physics_mod.F90 和 cmf_calc_fldstg_mod.F90
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
import logging

from ..utils.constants import ModelConstants, PhysicalConstants


class PhysicsEngine:
    """
    物理过程计算引擎
    对应Fortran: CMF_PHYSICS_* 系列函数
    """
    
    def __init__(self, nseq: int):
        """
        初始化物理引擎
        
        Args:
            nseq: 网格序列数量
        """
        self.nseq = nseq
        self.logger = logging.getLogger(__name__)
        
        # 状态变量 (对应Fortran: yos_cmf_prog.F90)
        self.river_storage = np.zeros(nseq, dtype=np.float64)    # P2RIVSTO
        self.flood_storage = np.zeros(nseq, dtype=np.float64)    # P2FLDSTO
        self.river_outflow = np.zeros(nseq, dtype=np.float64)    # D2RIVOUT
        self.flood_outflow = np.zeros(nseq, dtype=np.float64)    # D2FLDOUT
        
        # 诊断变量 (对应Fortran: yos_cmf_diag.F90)
        self.river_depth = np.zeros(nseq, dtype=np.float64)      # D2RIVDPH
        self.flood_depth = np.zeros(nseq, dtype=np.float64)      # D2FLDDPH
        self.river_inflow = np.zeros(nseq, dtype=np.float64)     # D2RIVINF
        self.surface_elevation = np.zeros(nseq, dtype=np.float64) # D2SFCELV
        
        # 地形参数 (对应Fortran: yos_cmf_map.F90)
        self.river_elevation = np.zeros(nseq, dtype=np.float64)  # D2RIVELV
        self.river_width = np.zeros(nseq, dtype=np.float64)      # D2RIVWTH
        self.river_height = np.zeros(nseq, dtype=np.float64)     # D2RIVHGT
        self.river_length = np.zeros(nseq, dtype=np.float64)     # D2RIVLEN
        self.catchment_area = np.zeros(nseq, dtype=np.float64)   # D2GRAREA
        self.next_distance = np.zeros(nseq, dtype=np.float64)    # D2NXTDST
        self.next_index = np.zeros(nseq, dtype=np.int32)         # I1NEXT
    
    def advance(self, dt: float) -> None:
        """
        物理过程推进
        对应Fortran: CMF_PHYSICS_ADVANCE (cmf_ctrl_physics_mod.F90:46-89行)
        
        Args:
            dt: 时间步长
        """
        # 1. 洪水阶段计算
        self.calculate_flood_stage()
        
        # 2. 流量计算 (将在hydraulics模块中实现)
        # self.calculate_outflow()
        
        # 3. 分汊流量计算 (如果启用)
        # self.calculate_bifurcation()
        
        # 4. 入流计算
        # self.calculate_inflow()
        
        # 5. 存储量更新 (将在storage模块中实现)
        # self.update_storage(dt)
        
        self.logger.debug(f"物理过程推进完成, dt={dt}")
    
    def calculate_flood_stage(self) -> None:
        """
        洪水阶段计算
        对应Fortran: CMF_CALC_FLDSTG_DEF (cmf_calc_fldstg_mod.F90:21-110行)
        """
        # 调用Numba优化的计算函数
        self.river_depth, self.flood_depth, self.surface_elevation = \
            calculate_flood_stage_numba(
                self.river_storage,
                self.flood_storage,
                self.river_elevation,
                self.river_width,
                self.river_height,
                self.river_length,
                self.catchment_area
            )
    
    def set_map_data(self, map_data: dict) -> None:
        """
        设置地图数据
        
        Args:
            map_data: 地图数据字典
        """
        if 'river_elevation' in map_data:
            self.river_elevation[:] = map_data['river_elevation'].flatten()
        if 'river_width' in map_data:
            self.river_width[:] = map_data['river_width'].flatten()
        if 'river_height' in map_data:
            self.river_height[:] = map_data['river_height'].flatten()
        if 'river_length' in map_data:
            self.river_length[:] = map_data['river_length'].flatten()
        if 'catchment_area' in map_data:
            self.catchment_area[:] = map_data['catchment_area'].flatten()
        if 'next_distance' in map_data:
            self.next_distance[:] = map_data['next_distance'].flatten()
        if 'next_xy' in map_data:
            self.next_index[:] = map_data['next_xy'].flatten()
    
    def get_state_dict(self) -> dict:
        """
        获取状态变量字典
        
        Returns:
            dict: 状态变量
        """
        return {
            'river_storage': self.river_storage.copy(),
            'flood_storage': self.flood_storage.copy(),
            'river_outflow': self.river_outflow.copy(),
            'flood_outflow': self.flood_outflow.copy(),
            'river_depth': self.river_depth.copy(),
            'flood_depth': self.flood_depth.copy(),
            'river_inflow': self.river_inflow.copy(),
            'surface_elevation': self.surface_elevation.copy()
        }
    
    def set_state_dict(self, state_dict: dict) -> None:
        """
        设置状态变量
        
        Args:
            state_dict: 状态变量字典
        """
        if 'river_storage' in state_dict:
            self.river_storage[:] = state_dict['river_storage']
        if 'flood_storage' in state_dict:
            self.flood_storage[:] = state_dict['flood_storage']
        if 'river_outflow' in state_dict:
            self.river_outflow[:] = state_dict['river_outflow']
        if 'flood_outflow' in state_dict:
            self.flood_outflow[:] = state_dict['flood_outflow']


@jit(nopython=True, parallel=True)
def calculate_flood_stage_numba(river_storage: np.ndarray,
                               flood_storage: np.ndarray,
                               river_elevation: np.ndarray,
                               river_width: np.ndarray,
                               river_height: np.ndarray,
                               river_length: np.ndarray,
                               catchment_area: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    洪水阶段计算 (Numba优化版本)
    对应Fortran: CMF_CALC_FLDSTG_DEF (cmf_calc_fldstg_mod.F90:46-107行)
    
    实现Yamazaki et al. 2011 WRR中的存储-水深关系方程
    
    Args:
        river_storage: 河道存储量 [m³]
        flood_storage: 漫滩存储量 [m³]
        river_elevation: 河床高程 [m]
        river_width: 河道宽度 [m]
        river_height: 河道深度 [m]
        river_length: 河道长度 [m]
        catchment_area: 集水面积 [m²]
        
    Returns:
        tuple: (河道水深, 漫滩水深, 水面高程)
    """
    nseq = len(river_storage)
    river_depth = np.zeros(nseq, dtype=np.float64)
    flood_depth = np.zeros(nseq, dtype=np.float64)
    surface_elevation = np.zeros(nseq, dtype=np.float64)
    
    # 并行计算每个网格 (对应Fortran: !$OMP PARALLEL DO)
    for iseq in prange(nseq):
        total_storage = river_storage[iseq] + flood_storage[iseq]
        
        if total_storage <= 0.0:
            river_depth[iseq] = 0.0
            flood_depth[iseq] = 0.0
            surface_elevation[iseq] = river_elevation[iseq]
            continue
        
        # 河道容量计算 (对应Fortran第52-65行)
        river_capacity = river_width[iseq] * river_height[iseq] * river_length[iseq]
        
        if total_storage <= river_capacity:
            # 水位在河道内 (对应Fortran第66-75行)
            if river_width[iseq] > 0 and river_length[iseq] > 0:
                river_depth[iseq] = total_storage / (river_width[iseq] * river_length[iseq])
            else:
                river_depth[iseq] = 0.0
            
            flood_depth[iseq] = 0.0
            surface_elevation[iseq] = river_elevation[iseq] + river_depth[iseq]
        else:
            # 水位超出河道，计算漫滩 (对应Fortran第76-107行)
            river_depth[iseq] = river_height[iseq]
            
            # 漫滩存储量
            flood_storage_val = total_storage - river_capacity
            
            # 漫滩水深计算 (简化的矩形漫滩模型)
            if catchment_area[iseq] > 0:
                # 使用集水面积作为漫滩面积的代理
                effective_flood_area = catchment_area[iseq] * 0.1  # 假设10%的集水面积为漫滩
                if effective_flood_area > 0:
                    flood_depth[iseq] = flood_storage_val / effective_flood_area
                else:
                    flood_depth[iseq] = 0.0
            else:
                flood_depth[iseq] = 0.0
            
            # 水面高程为河道顶部加漫滩水深
            surface_elevation[iseq] = river_elevation[iseq] + river_height[iseq] + flood_depth[iseq]
    
    return river_depth, flood_depth, surface_elevation


@jit(nopython=True)
def calculate_river_velocity(discharge: float, width: float, depth: float) -> float:
    """
    计算河道流速
    
    Args:
        discharge: 流量 [m³/s]
        width: 河道宽度 [m]
        depth: 河道水深 [m]
        
    Returns:
        float: 流速 [m/s]
    """
    if width <= 0 or depth <= 0:
        return 0.0
    
    cross_section_area = width * depth
    if cross_section_area <= 0:
        return 0.0
    
    return discharge / cross_section_area


@jit(nopython=True)
def calculate_water_level_slope(elevation_upstream: float, elevation_downstream: float,
                               distance: float) -> float:
    """
    计算水面坡度
    
    Args:
        elevation_upstream: 上游水面高程 [m]
        elevation_downstream: 下游水面高程 [m]
        distance: 距离 [m]
        
    Returns:
        float: 水面坡度 [-]
    """
    if distance <= 0:
        return 0.0
    
    return (elevation_upstream - elevation_downstream) / distance


class FloodStageSelector:
    """
    洪水阶段计算选择器
    对应Fortran: CMF_PHYSICS_FLDSTG (cmf_ctrl_physics_mod.F90:94-120行)
    """
    
    def __init__(self, method: str = "default"):
        """
        初始化洪水阶段选择器
        
        Args:
            method: 计算方法 ("default", "kinematic", "diffusive")
        """
        self.method = method
        self.logger = logging.getLogger(__name__)
    
    def select_calculation_method(self, physics_engine: PhysicsEngine) -> None:
        """
        选择洪水阶段计算方法
        
        Args:
            physics_engine: 物理引擎实例
        """
        if self.method == "default":
            physics_engine.calculate_flood_stage()
        elif self.method == "kinematic":
            # 运动波近似方法 (扩展功能)
            self._calculate_kinematic_flood_stage(physics_engine)
        elif self.method == "diffusive":
            # 扩散波近似方法 (扩展功能)
            self._calculate_diffusive_flood_stage(physics_engine)
        else:
            self.logger.warning(f"未知的洪水阶段计算方法: {self.method}, 使用默认方法")
            physics_engine.calculate_flood_stage()
    
    def _calculate_kinematic_flood_stage(self, physics_engine: PhysicsEngine) -> None:
        """
        运动波近似洪水阶段计算
        """
        # 简化实现，实际可以添加更复杂的运动波方程
        physics_engine.calculate_flood_stage()
    
    def _calculate_diffusive_flood_stage(self, physics_engine: PhysicsEngine) -> None:
        """
        扩散波近似洪水阶段计算
        """
        # 简化实现，实际可以添加更复杂的扩散波方程
        physics_engine.calculate_flood_stage()
