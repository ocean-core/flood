"""
水力学计算模块
对应Fortran源码: cmf_calc_outflw_mod.F90
"""

import numpy as np
from numba import jit, prange
from typing import Tuple
import logging

from ..utils.constants import ModelConstants, PhysicalConstants


class HydraulicsCalculator:
    """
    水力学计算器
    对应Fortran: CMF_CALC_OUTFLW 系列函数
    """
    
    def __init__(self, nseq: int, manning_river: float = ModelConstants.DEFAULT_MANNING_RIVER,
                 manning_flood: float = ModelConstants.DEFAULT_MANNING_FLOOD):
        """
        初始化水力学计算器
        
        Args:
            nseq: 网格序列数量
            manning_river: 河道Manning系数
            manning_flood: 漫滩Manning系数
        """
        self.nseq = nseq
        self.manning_river = manning_river
        self.manning_flood = manning_flood
        self.logger = logging.getLogger(__name__)
        
        # 流量相关变量
        self.river_outflow = np.zeros(nseq, dtype=np.float64)
        self.flood_outflow = np.zeros(nseq, dtype=np.float64)
        self.flow_velocity = np.zeros(nseq, dtype=np.float64)
    
    def calculate_outflow(self, river_depth: np.ndarray, flood_depth: np.ndarray,
                         surface_elevation: np.ndarray, river_width: np.ndarray,
                         river_length: np.ndarray, next_distance: np.ndarray,
                         next_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算河道和漫滩出流
        对应Fortran: CMF_CALC_OUTFLW (cmf_calc_outflw_mod.F90:32-185行)
        
        Args:
            river_depth: 河道水深 [m]
            flood_depth: 漫滩水深 [m]
            surface_elevation: 水面高程 [m]
            river_width: 河道宽度 [m]
            river_length: 河道长度 [m]
            next_distance: 到下游距离 [m]
            next_index: 下游网格索引
            
        Returns:
            tuple: (河道出流, 漫滩出流)
        """
        # 调用Numba优化的计算函数
        self.river_outflow, self.flood_outflow, self.flow_velocity = \
            calculate_outflow_numba(
                river_depth, flood_depth, surface_elevation,
                river_width, river_length, next_distance, next_index,
                self.manning_river, self.manning_flood
            )
        
        return self.river_outflow.copy(), self.flood_outflow.copy()
    
    def calculate_inflow(self, river_outflow: np.ndarray, next_index: np.ndarray,
                        runoff_forcing: np.ndarray) -> np.ndarray:
        """
        计算河道入流
        对应Fortran: CMF_CALC_INFLOW (cmf_calc_outflw_mod.F90:191-380行)
        
        Args:
            river_outflow: 河道出流 [m³/s]
            next_index: 下游网格索引
            runoff_forcing: 径流强迫 [m³/s]
            
        Returns:
            np.ndarray: 河道入流 [m³/s]
        """
        return calculate_inflow_numba(river_outflow, next_index, runoff_forcing)
    
    def get_max_velocity(self) -> float:
        """
        获取最大流速（用于CFL条件）
        
        Returns:
            float: 最大流速 [m/s]
        """
        return np.max(self.flow_velocity)


@jit(nopython=True, parallel=True)
def calculate_outflow_numba(river_depth: np.ndarray, flood_depth: np.ndarray,
                           surface_elevation: np.ndarray, river_width: np.ndarray,
                           river_length: np.ndarray, next_distance: np.ndarray,
                           next_index: np.ndarray, manning_river: float,
                           manning_flood: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Manning公式流量计算 (Numba优化版本)
    对应Fortran: CMF_CALC_OUTFLW (cmf_calc_outflw_mod.F90:43-182行)
    
    Args:
        river_depth: 河道水深 [m]
        flood_depth: 漫滩水深 [m]
        surface_elevation: 水面高程 [m]
        river_width: 河道宽度 [m]
        river_length: 河道长度 [m]
        next_distance: 到下游距离 [m]
        next_index: 下游网格索引
        manning_river: 河道Manning系数
        manning_flood: 漫滩Manning系数
        
    Returns:
        tuple: (河道出流, 漫滩出流, 流速)
    """
    nseq = len(river_depth)
    river_outflow = np.zeros(nseq, dtype=np.float64)
    flood_outflow = np.zeros(nseq, dtype=np.float64)
    flow_velocity = np.zeros(nseq, dtype=np.float64)
    
    # 并行计算每个网格 (对应Fortran: !$OMP PARALLEL DO SIMD)
    for iseq in prange(nseq):
        # 获取下游网格信息
        next_idx = next_index[iseq]
        
        if next_idx <= 0 or next_idx >= nseq:
            # 出口网格或无效索引
            river_outflow[iseq] = 0.0
            flood_outflow[iseq] = 0.0
            flow_velocity[iseq] = 0.0
            continue
        
        # 计算水面坡度 (对应Fortran第55-70行)
        distance = next_distance[iseq]
        if distance <= 0:
            river_outflow[iseq] = 0.0
            flood_outflow[iseq] = 0.0
            flow_velocity[iseq] = 0.0
            continue
        
        water_slope = (surface_elevation[iseq] - surface_elevation[next_idx]) / distance
        
        # 河道流量计算 (对应Fortran第75-120行)
        river_depth_val = river_depth[iseq]
        if river_depth_val > 0:
            # Manning公式: Q = (1/n) * A * R^(2/3) * S^(1/2)
            river_area = river_width[iseq] * river_depth_val
            river_perimeter = river_width[iseq] + 2.0 * river_depth_val
            
            if river_perimeter > 0:
                hydraulic_radius = river_area / river_perimeter
                
                if water_slope > 0:
                    velocity = (hydraulic_radius**(2.0/3.0) * np.sqrt(water_slope)) / manning_river
                    river_outflow[iseq] = velocity * river_area
                    flow_velocity[iseq] = velocity
                else:
                    river_outflow[iseq] = 0.0
                    flow_velocity[iseq] = 0.0
            else:
                river_outflow[iseq] = 0.0
                flow_velocity[iseq] = 0.0
        else:
            river_outflow[iseq] = 0.0
            flow_velocity[iseq] = 0.0
        
        # 漫滩流量计算 (对应Fortran第140-170行)
        flood_depth_val = flood_depth[iseq]
        if flood_depth_val > 0:
            # 简化的漫滩流量计算
            # 假设漫滩宽度为河道宽度的5倍
            effective_flood_width = river_width[iseq] * 5.0
            flood_area = flood_depth_val * effective_flood_width
            flood_perimeter = effective_flood_width + 2.0 * flood_depth_val
            
            if flood_perimeter > 0:
                flood_hydraulic_radius = flood_area / flood_perimeter
                
                if water_slope > 0:
                    flood_velocity = (flood_hydraulic_radius**(2.0/3.0) * 
                                    np.sqrt(water_slope)) / manning_flood
                    flood_outflow[iseq] = flood_velocity * flood_area
                    
                    # 更新最大流速
                    if flood_velocity > flow_velocity[iseq]:
                        flow_velocity[iseq] = flood_velocity
                else:
                    flood_outflow[iseq] = 0.0
            else:
                flood_outflow[iseq] = 0.0
        else:
            flood_outflow[iseq] = 0.0
    
    return river_outflow, flood_outflow, flow_velocity


@jit(nopython=True, parallel=True)
def calculate_inflow_numba(river_outflow: np.ndarray, next_index: np.ndarray,
                          runoff_forcing: np.ndarray) -> np.ndarray:
    """
    河道入流计算 (Numba优化版本)
    对应Fortran: CMF_CALC_INFLOW (cmf_calc_outflw_mod.F90:191-380行)
    
    Args:
        river_outflow: 河道出流 [m³/s]
        next_index: 下游网格索引
        runoff_forcing: 径流强迫 [m³/s]
        
    Returns:
        np.ndarray: 河道入流 [m³/s]
    """
    nseq = len(river_outflow)
    river_inflow = np.zeros(nseq, dtype=np.float64)
    
    # 首先设置径流强迫作为基础入流
    for iseq in prange(nseq):
        river_inflow[iseq] = runoff_forcing[iseq]
    
    # 累加上游流量 (对应Fortran第210-250行)
    for iseq in range(nseq):
        next_idx = next_index[iseq]
        
        if next_idx > 0 and next_idx < nseq:
            # 将当前网格的出流加到下游网格的入流中
            river_inflow[next_idx] += river_outflow[iseq]
    
    return river_inflow


@jit(nopython=True)
def calculate_manning_flow_simple(depth: float, width: float, slope: float,
                                 manning_n: float) -> float:
    """
    简化的Manning公式流量计算
    
    Args:
        depth: 水深 [m]
        width: 宽度 [m]
        slope: 坡度 [-]
        manning_n: Manning系数
        
    Returns:
        float: 流量 [m³/s]
    """
    if depth <= 0 or width <= 0 or slope <= 0 or manning_n <= 0:
        return 0.0
    
    area = width * depth
    perimeter = width + 2 * depth
    hydraulic_radius = area / perimeter
    
    velocity = (hydraulic_radius**(2.0/3.0) * np.sqrt(slope)) / manning_n
    discharge = velocity * area
    
    return discharge


@jit(nopython=True)
def calculate_critical_depth(discharge: float, width: float, gravity: float = 9.8) -> float:
    """
    计算临界水深
    
    Args:
        discharge: 流量 [m³/s]
        width: 宽度 [m]
        gravity: 重力加速度 [m/s²]
        
    Returns:
        float: 临界水深 [m]
    """
    if discharge <= 0 or width <= 0:
        return 0.0
    
    # 对于矩形断面: yc = (q²/g)^(1/3), 其中q = Q/B
    unit_discharge = discharge / width
    critical_depth = (unit_discharge * unit_discharge / gravity)**(1.0/3.0)
    
    return critical_depth


@jit(nopython=True)
def calculate_froude_number(velocity: float, depth: float, gravity: float = 9.8) -> float:
    """
    计算Froude数
    
    Args:
        velocity: 流速 [m/s]
        depth: 水深 [m]
        gravity: 重力加速度 [m/s²]
        
    Returns:
        float: Froude数
    """
    if depth <= 0:
        return 0.0
    
    return velocity / np.sqrt(gravity * depth)


class LocalInertialSolver:
    """
    局部惯性方程求解器
    对应Fortran中的高级流量计算方法
    """
    
    def __init__(self, dt: float = 3600.0, theta: float = 0.8):
        """
        初始化局部惯性求解器
        
        Args:
            dt: 时间步长 [s]
            theta: 隐式权重因子 (0.5-1.0)
        """
        self.dt = dt
        self.theta = theta
        self.logger = logging.getLogger(__name__)
    
    def solve_local_inertial(self, discharge_prev: np.ndarray, depth: np.ndarray,
                            width: np.ndarray, slope: np.ndarray,
                            manning_n: float) -> np.ndarray:
        """
        求解局部惯性方程
        
        Args:
            discharge_prev: 前一时刻流量 [m³/s]
            depth: 水深 [m]
            width: 宽度 [m]
            slope: 坡度 [-]
            manning_n: Manning系数
            
        Returns:
            np.ndarray: 新的流量 [m³/s]
        """
        return solve_local_inertial_numba(
            discharge_prev, depth, width, slope, manning_n, self.dt, self.theta
        )


@jit(nopython=True, parallel=True)
def solve_local_inertial_numba(discharge_prev: np.ndarray, depth: np.ndarray,
                              width: np.ndarray, slope: np.ndarray,
                              manning_n: float, dt: float, theta: float) -> np.ndarray:
    """
    局部惯性方程求解 (Numba优化版本)
    
    局部惯性方程: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x + gAn²|Q|Q/(A²R^(4/3)) = 0
    简化为: Q^(n+1) = (Q^n + dt*g*A*S) / (1 + dt*g*n²|Q|/(A*R^(4/3)))
    
    Args:
        discharge_prev: 前一时刻流量
        depth: 水深
        width: 宽度
        slope: 坡度
        manning_n: Manning系数
        dt: 时间步长
        theta: 隐式权重
        
    Returns:
        np.ndarray: 新的流量
    """
    nseq = len(discharge_prev)
    discharge_new = np.zeros(nseq, dtype=np.float64)
    gravity = 9.8
    
    for iseq in prange(nseq):
        if depth[iseq] <= 0 or width[iseq] <= 0:
            discharge_new[iseq] = 0.0
            continue
        
        area = width[iseq] * depth[iseq]
        perimeter = width[iseq] + 2.0 * depth[iseq]
        hydraulic_radius = area / perimeter
        
        # 重力项
        gravity_term = gravity * area * slope[iseq]
        
        # 摩擦项
        q_abs = abs(discharge_prev[iseq])
        friction_coeff = gravity * manning_n * manning_n * q_abs / (area * hydraulic_radius**(4.0/3.0))
        
        # 局部惯性方程求解
        numerator = discharge_prev[iseq] + dt * gravity_term
        denominator = 1.0 + dt * friction_coeff
        
        discharge_new[iseq] = numerator / denominator
    
    return discharge_new
