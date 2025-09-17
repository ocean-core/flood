"""
存储量更新模块
对应Fortran源码: cmf_calc_stonxt_mod.F90
"""

import numpy as np
from numba import jit, prange
from typing import Tuple
import logging

from ..utils.constants import ModelConstants


class StorageCalculator:
    """
    存储量计算器
    对应Fortran: CMF_CALC_STONXT 系列函数
    """
    
    def __init__(self, nseq: int):
        """
        初始化存储量计算器
        
        Args:
            nseq: 网格序列数量
        """
        self.nseq = nseq
        self.logger = logging.getLogger(__name__)
        
        # 水量平衡检查
        self.water_balance_error = np.zeros(nseq, dtype=np.float64)
        self.max_error_tolerance = 1e-6
    
    def update_storage(self, river_storage: np.ndarray, flood_storage: np.ndarray,
                      river_inflow: np.ndarray, river_outflow: np.ndarray,
                      flood_inflow: np.ndarray, flood_outflow: np.ndarray,
                      dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新存储量
        对应Fortran: CMF_CALC_STONXT (cmf_calc_stonxt_mod.F90:20-135行)
        
        Args:
            river_storage: 河道存储量 [m³]
            flood_storage: 漫滩存储量 [m³]
            river_inflow: 河道入流 [m³/s]
            river_outflow: 河道出流 [m³/s]
            flood_inflow: 漫滩入流 [m³/s]
            flood_outflow: 漫滩出流 [m³/s]
            dt: 时间步长 [s]
            
        Returns:
            tuple: (新的河道存储量, 新的漫滩存储量)
        """
        new_river_storage, new_flood_storage, self.water_balance_error = \
            update_storage_numba(
                river_storage, flood_storage,
                river_inflow, river_outflow,
                flood_inflow, flood_outflow,
                dt, self.max_error_tolerance
            )
        
        # 检查水量平衡
        max_error = np.max(np.abs(self.water_balance_error))
        if max_error > self.max_error_tolerance:
            self.logger.warning(f"水量平衡误差超过阈值: {max_error:.2e}")
        
        return new_river_storage, new_flood_storage
    
    def get_water_balance_error(self) -> np.ndarray:
        """
        获取水量平衡误差
        
        Returns:
            np.ndarray: 水量平衡误差
        """
        return self.water_balance_error.copy()


@jit(nopython=True, parallel=True)
def update_storage_numba(river_storage: np.ndarray, flood_storage: np.ndarray,
                        river_inflow: np.ndarray, river_outflow: np.ndarray,
                        flood_inflow: np.ndarray, flood_outflow: np.ndarray,
                        dt: float, tolerance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    存储量更新计算 (Numba优化版本)
    对应Fortran: CMF_CALC_STONXT (cmf_calc_stonxt_mod.F90:23-135行)
    
    连续性方程: dS/dt = Inflow - Outflow
    
    Args:
        river_storage: 河道存储量
        flood_storage: 漫滩存储量
        river_inflow: 河道入流
        river_outflow: 河道出流
        flood_inflow: 漫滩入流
        flood_outflow: 漫滩出流
        dt: 时间步长
        tolerance: 容差
        
    Returns:
        tuple: (新河道存储量, 新漫滩存储量, 水量平衡误差)
    """
    nseq = len(river_storage)
    new_river_storage = np.zeros(nseq, dtype=np.float64)
    new_flood_storage = np.zeros(nseq, dtype=np.float64)
    water_balance_error = np.zeros(nseq, dtype=np.float64)
    
    # 并行计算每个网格 (对应Fortran第23-50行变量声明和初始化)
    for iseq in prange(nseq):
        # 河道存储更新 (对应Fortran第52-80行)
        # dS/dt = Inflow - Outflow
        river_storage_change = (river_inflow[iseq] - river_outflow[iseq]) * dt
        new_river_storage[iseq] = max(0.0, river_storage[iseq] + river_storage_change)
        
        # 漫滩存储更新 (对应Fortran第81-120行)
        flood_storage_change = (flood_inflow[iseq] - flood_outflow[iseq]) * dt
        new_flood_storage[iseq] = max(0.0, flood_storage[iseq] + flood_storage_change)
        
        # 水量平衡检查 (对应Fortran第121-135行)
        total_change = river_storage_change + flood_storage_change
        total_flow_change = (river_inflow[iseq] + flood_inflow[iseq] - 
                           river_outflow[iseq] - flood_outflow[iseq]) * dt
        
        water_balance_error[iseq] = abs(total_change - total_flow_change)
        
        # 如果存储量变为负值，记录误差
        if river_storage[iseq] + river_storage_change < 0:
            water_balance_error[iseq] += abs(river_storage[iseq] + river_storage_change)
        
        if flood_storage[iseq] + flood_storage_change < 0:
            water_balance_error[iseq] += abs(flood_storage[iseq] + flood_storage_change)
    
    return new_river_storage, new_flood_storage, water_balance_error


@jit(nopython=True)
def calculate_exchange_flow(river_storage: float, flood_storage: float,
                           river_capacity: float, exchange_rate: float = 0.1) -> Tuple[float, float]:
    """
    计算河道和漫滩之间的交换流量
    
    Args:
        river_storage: 河道存储量 [m³]
        flood_storage: 漫滩存储量 [m³]
        river_capacity: 河道容量 [m³]
        exchange_rate: 交换率 [1/s]
        
    Returns:
        tuple: (河道到漫滩流量, 漫滩到河道流量)
    """
    river_to_flood = 0.0
    flood_to_river = 0.0
    
    if river_storage > river_capacity:
        # 河道溢流到漫滩
        overflow = river_storage - river_capacity
        river_to_flood = overflow * exchange_rate
    elif flood_storage > 0 and river_storage < river_capacity:
        # 漫滩回流到河道
        available_capacity = river_capacity - river_storage
        potential_return = flood_storage * exchange_rate
        flood_to_river = min(potential_return, available_capacity)
    
    return river_to_flood, flood_to_river


@jit(nopython=True, parallel=True)
def apply_storage_constraints(river_storage: np.ndarray, flood_storage: np.ndarray,
                             min_storage: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用存储量约束条件
    
    Args:
        river_storage: 河道存储量
        flood_storage: 漫滩存储量
        min_storage: 最小存储量
        
    Returns:
        tuple: (约束后的河道存储量, 约束后的漫滩存储量)
    """
    nseq = len(river_storage)
    constrained_river = np.zeros(nseq, dtype=np.float64)
    constrained_flood = np.zeros(nseq, dtype=np.float64)
    
    for iseq in prange(nseq):
        constrained_river[iseq] = max(min_storage, river_storage[iseq])
        constrained_flood[iseq] = max(min_storage, flood_storage[iseq])
    
    return constrained_river, constrained_flood


class AdaptiveStorageUpdater:
    """
    自适应存储量更新器
    根据数值稳定性调整时间步长
    """
    
    def __init__(self, nseq: int, max_change_rate: float = 0.1):
        """
        初始化自适应更新器
        
        Args:
            nseq: 网格数量
            max_change_rate: 最大变化率 (相对于当前存储量)
        """
        self.nseq = nseq
        self.max_change_rate = max_change_rate
        self.logger = logging.getLogger(__name__)
    
    def update_with_subcycling(self, river_storage: np.ndarray, flood_storage: np.ndarray,
                              river_inflow: np.ndarray, river_outflow: np.ndarray,
                              flood_inflow: np.ndarray, flood_outflow: np.ndarray,
                              dt: float) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        使用子循环的自适应存储量更新
        
        Args:
            river_storage: 河道存储量
            flood_storage: 漫滩存储量
            river_inflow: 河道入流
            river_outflow: 河道出流
            flood_inflow: 漫滩入流
            flood_outflow: 漫滩出流
            dt: 原始时间步长
            
        Returns:
            tuple: (新河道存储量, 新漫滩存储量, 子步数)
        """
        # 估算需要的子步数
        max_substeps = self._estimate_substeps(
            river_storage, flood_storage,
            river_inflow, river_outflow,
            flood_inflow, flood_outflow, dt
        )
        
        if max_substeps <= 1:
            # 不需要子循环
            new_river, new_flood, _ = update_storage_numba(
                river_storage, flood_storage,
                river_inflow, river_outflow,
                flood_inflow, flood_outflow,
                dt, 1e-6
            )
            return new_river, new_flood, 1
        
        # 执行子循环
        sub_dt = dt / max_substeps
        current_river = river_storage.copy()
        current_flood = flood_storage.copy()
        
        for step in range(max_substeps):
            current_river, current_flood, _ = update_storage_numba(
                current_river, current_flood,
                river_inflow, river_outflow,
                flood_inflow, flood_outflow,
                sub_dt, 1e-6
            )
        
        return current_river, current_flood, max_substeps
    
    def _estimate_substeps(self, river_storage: np.ndarray, flood_storage: np.ndarray,
                          river_inflow: np.ndarray, river_outflow: np.ndarray,
                          flood_inflow: np.ndarray, flood_outflow: np.ndarray,
                          dt: float) -> int:
        """
        估算所需的子步数
        
        Returns:
            int: 子步数
        """
        max_substeps = 1
        
        for iseq in range(self.nseq):
            # 计算相对变化率
            river_change_rate = abs(river_inflow[iseq] - river_outflow[iseq]) * dt
            flood_change_rate = abs(flood_inflow[iseq] - flood_outflow[iseq]) * dt
            
            if river_storage[iseq] > 0:
                river_relative_change = river_change_rate / river_storage[iseq]
                if river_relative_change > self.max_change_rate:
                    required_substeps = int(river_relative_change / self.max_change_rate) + 1
                    max_substeps = max(max_substeps, required_substeps)
            
            if flood_storage[iseq] > 0:
                flood_relative_change = flood_change_rate / flood_storage[iseq]
                if flood_relative_change > self.max_change_rate:
                    required_substeps = int(flood_relative_change / self.max_change_rate) + 1
                    max_substeps = max(max_substeps, required_substeps)
        
        return min(max_substeps, 100)  # 限制最大子步数


class WaterBalanceMonitor:
    """
    水量平衡监控器
    """
    
    def __init__(self, nseq: int):
        """
        初始化水量平衡监控器
        
        Args:
            nseq: 网格数量
        """
        self.nseq = nseq
        self.cumulative_error = np.zeros(nseq, dtype=np.float64)
        self.error_history = []
        self.logger = logging.getLogger(__name__)
    
    def check_water_balance(self, river_storage_old: np.ndarray, flood_storage_old: np.ndarray,
                           river_storage_new: np.ndarray, flood_storage_new: np.ndarray,
                           river_inflow: np.ndarray, river_outflow: np.ndarray,
                           flood_inflow: np.ndarray, flood_outflow: np.ndarray,
                           dt: float) -> dict:
        """
        检查水量平衡
        
        Returns:
            dict: 水量平衡统计信息
        """
        # 计算存储量变化
        river_change = river_storage_new - river_storage_old
        flood_change = flood_storage_new - flood_storage_old
        total_storage_change = river_change + flood_change
        
        # 计算净流量
        net_flow = (river_inflow + flood_inflow - river_outflow - flood_outflow) * dt
        
        # 计算误差
        balance_error = total_storage_change - net_flow
        
        # 更新累积误差
        self.cumulative_error += balance_error
        
        # 统计信息
        stats = {
            'max_absolute_error': np.max(np.abs(balance_error)),
            'mean_absolute_error': np.mean(np.abs(balance_error)),
            'total_error': np.sum(balance_error),
            'max_cumulative_error': np.max(np.abs(self.cumulative_error)),
            'relative_error': np.max(np.abs(balance_error) / (np.abs(total_storage_change) + 1e-10))
        }
        
        self.error_history.append(stats['max_absolute_error'])
        
        return stats
    
    def reset_cumulative_error(self) -> None:
        """重置累积误差"""
        self.cumulative_error.fill(0.0)
        self.error_history.clear()
