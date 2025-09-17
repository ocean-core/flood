"""
诊断变量计算模块
对应Fortran源码中的诊断变量计算
"""

import numpy as np
from numba import jit, prange
from typing import Dict, Tuple
import logging

from ..utils.constants import ModelConstants, PhysicalConstants


class DiagnosticsCalculator:
    """
    诊断变量计算器
    计算各种诊断和统计变量
    """
    
    def __init__(self, nseq: int):
        """
        初始化诊断计算器
        
        Args:
            nseq: 网格序列数量
        """
        self.nseq = nseq
        self.logger = logging.getLogger(__name__)
        
        # 诊断变量
        self.flow_velocity = np.zeros(nseq, dtype=np.float64)
        self.froude_number = np.zeros(nseq, dtype=np.float64)
        self.water_level_slope = np.zeros(nseq, dtype=np.float64)
        self.residence_time = np.zeros(nseq, dtype=np.float64)
        
        # 统计变量
        self.max_discharge = np.zeros(nseq, dtype=np.float64)
        self.max_depth = np.zeros(nseq, dtype=np.float64)
        self.flood_duration = np.zeros(nseq, dtype=np.float64)
    
    def calculate_flow_diagnostics(self, river_outflow: np.ndarray, 
                                  river_depth: np.ndarray,
                                  river_width: np.ndarray,
                                  surface_elevation: np.ndarray,
                                  next_distance: np.ndarray,
                                  next_index: np.ndarray) -> None:
        """
        计算流动诊断变量
        
        Args:
            river_outflow: 河道出流 [m³/s]
            river_depth: 河道水深 [m]
            river_width: 河道宽度 [m]
            surface_elevation: 水面高程 [m]
            next_distance: 到下游距离 [m]
            next_index: 下游网格索引
        """
        self.flow_velocity, self.froude_number, self.water_level_slope = \
            calculate_flow_diagnostics_numba(
                river_outflow, river_depth, river_width,
                surface_elevation, next_distance, next_index
            )
    
    def calculate_residence_time(self, river_storage: np.ndarray,
                               river_outflow: np.ndarray) -> None:
        """
        计算水体停留时间
        
        Args:
            river_storage: 河道存储量 [m³]
            river_outflow: 河道出流 [m³/s]
        """
        self.residence_time = calculate_residence_time_numba(
            river_storage, river_outflow
        )
    
    def update_statistics(self, river_outflow: np.ndarray,
                         river_depth: np.ndarray,
                         flood_depth: np.ndarray,
                         dt: float) -> None:
        """
        更新统计变量
        
        Args:
            river_outflow: 河道出流 [m³/s]
            river_depth: 河道水深 [m]
            flood_depth: 漫滩水深 [m]
            dt: 时间步长 [s]
        """
        # 更新最大值
        self.max_discharge = np.maximum(self.max_discharge, river_outflow)
        self.max_depth = np.maximum(self.max_depth, river_depth)
        
        # 更新洪水持续时间
        flood_mask = flood_depth > 0.1  # 漫滩水深超过10cm认为是洪水
        self.flood_duration[flood_mask] += dt / 3600.0  # 转换为小时
    
    def get_diagnostics_dict(self) -> Dict[str, np.ndarray]:
        """
        获取诊断变量字典
        
        Returns:
            dict: 诊断变量
        """
        return {
            'flow_velocity': self.flow_velocity.copy(),
            'froude_number': self.froude_number.copy(),
            'water_level_slope': self.water_level_slope.copy(),
            'residence_time': self.residence_time.copy(),
            'max_discharge': self.max_discharge.copy(),
            'max_depth': self.max_depth.copy(),
            'flood_duration': self.flood_duration.copy()
        }
    
    def reset_statistics(self) -> None:
        """重置统计变量"""
        self.max_discharge.fill(0.0)
        self.max_depth.fill(0.0)
        self.flood_duration.fill(0.0)


@jit(nopython=True, parallel=True)
def calculate_flow_diagnostics_numba(river_outflow: np.ndarray,
                                    river_depth: np.ndarray,
                                    river_width: np.ndarray,
                                    surface_elevation: np.ndarray,
                                    next_distance: np.ndarray,
                                    next_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算流动诊断变量 (Numba优化版本)
    
    Args:
        river_outflow: 河道出流
        river_depth: 河道水深
        river_width: 河道宽度
        surface_elevation: 水面高程
        next_distance: 到下游距离
        next_index: 下游网格索引
        
    Returns:
        tuple: (流速, Froude数, 水面坡度)
    """
    nseq = len(river_outflow)
    flow_velocity = np.zeros(nseq, dtype=np.float64)
    froude_number = np.zeros(nseq, dtype=np.float64)
    water_level_slope = np.zeros(nseq, dtype=np.float64)
    gravity = 9.8
    
    for iseq in prange(nseq):
        # 计算流速
        if river_depth[iseq] > 0 and river_width[iseq] > 0:
            cross_area = river_depth[iseq] * river_width[iseq]
            flow_velocity[iseq] = river_outflow[iseq] / cross_area
            
            # 计算Froude数
            froude_number[iseq] = flow_velocity[iseq] / np.sqrt(gravity * river_depth[iseq])
        
        # 计算水面坡度
        next_idx = next_index[iseq]
        if next_idx > 0 and next_idx < nseq and next_distance[iseq] > 0:
            elevation_diff = surface_elevation[iseq] - surface_elevation[next_idx]
            water_level_slope[iseq] = elevation_diff / next_distance[iseq]
    
    return flow_velocity, froude_number, water_level_slope


@jit(nopython=True, parallel=True)
def calculate_residence_time_numba(river_storage: np.ndarray,
                                  river_outflow: np.ndarray) -> np.ndarray:
    """
    计算水体停留时间 (Numba优化版本)
    
    Args:
        river_storage: 河道存储量 [m³]
        river_outflow: 河道出流 [m³/s]
        
    Returns:
        np.ndarray: 停留时间 [s]
    """
    nseq = len(river_storage)
    residence_time = np.zeros(nseq, dtype=np.float64)
    
    for iseq in prange(nseq):
        if river_outflow[iseq] > 0:
            residence_time[iseq] = river_storage[iseq] / river_outflow[iseq]
        else:
            residence_time[iseq] = 0.0
    
    return residence_time


class FloodAnalyzer:
    """
    洪水分析器
    提供洪水事件分析功能
    """
    
    def __init__(self, nseq: int, flood_threshold: float = 0.1):
        """
        初始化洪水分析器
        
        Args:
            nseq: 网格数量
            flood_threshold: 洪水阈值 [m]
        """
        self.nseq = nseq
        self.flood_threshold = flood_threshold
        self.logger = logging.getLogger(__name__)
        
        # 洪水事件追踪
        self.flood_start_time = np.full(nseq, -1.0)  # 洪水开始时间
        self.flood_peak_depth = np.zeros(nseq)       # 洪水峰值深度
        self.flood_volume = np.zeros(nseq)           # 洪水总量
        self.is_flooding = np.zeros(nseq, dtype=bool) # 当前是否洪水
    
    def analyze_flood_event(self, flood_depth: np.ndarray,
                           current_time: float, dt: float) -> Dict[str, np.ndarray]:
        """
        分析洪水事件
        
        Args:
            flood_depth: 漫滩水深 [m]
            current_time: 当前时间 [s]
            dt: 时间步长 [s]
            
        Returns:
            dict: 洪水事件信息
        """
        return analyze_flood_event_numba(
            flood_depth, self.flood_threshold, current_time, dt,
            self.flood_start_time, self.flood_peak_depth,
            self.flood_volume, self.is_flooding
        )
    
    def get_flood_statistics(self) -> Dict[str, float]:
        """
        获取洪水统计信息
        
        Returns:
            dict: 洪水统计
        """
        flooded_cells = np.sum(self.is_flooding)
        total_flood_volume = np.sum(self.flood_volume)
        max_flood_depth = np.max(self.flood_peak_depth)
        
        return {
            'flooded_cells': int(flooded_cells),
            'flood_fraction': flooded_cells / self.nseq,
            'total_flood_volume': total_flood_volume,
            'max_flood_depth': max_flood_depth,
            'mean_flood_depth': np.mean(self.flood_peak_depth[self.is_flooding]) if flooded_cells > 0 else 0.0
        }


@jit(nopython=True, parallel=True)
def analyze_flood_event_numba(flood_depth: np.ndarray, flood_threshold: float,
                             current_time: float, dt: float,
                             flood_start_time: np.ndarray,
                             flood_peak_depth: np.ndarray,
                             flood_volume: np.ndarray,
                             is_flooding: np.ndarray) -> Dict:
    """
    洪水事件分析 (Numba优化版本)
    
    注意：由于Numba限制，返回简化的结果
    """
    nseq = len(flood_depth)
    
    for iseq in prange(nseq):
        current_depth = flood_depth[iseq]
        
        if current_depth > flood_threshold:
            # 洪水状态
            if not is_flooding[iseq]:
                # 洪水开始
                is_flooding[iseq] = True
                flood_start_time[iseq] = current_time
                flood_peak_depth[iseq] = current_depth
                flood_volume[iseq] = 0.0
            else:
                # 洪水持续
                flood_peak_depth[iseq] = max(flood_peak_depth[iseq], current_depth)
            
            # 累积洪水量
            flood_volume[iseq] += current_depth * dt
        else:
            # 非洪水状态
            if is_flooding[iseq]:
                # 洪水结束
                is_flooding[iseq] = False
    
    # 返回空字典（Numba限制）
    return {}


class PerformanceProfiler:
    """
    性能分析器
    监控模型各部分的计算性能
    """
    
    def __init__(self):
        """初始化性能分析器"""
        self.timers = {}
        self.call_counts = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, name: str) -> None:
        """开始计时"""
        import time
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        import time
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        
        # 更新统计
        if name not in self.call_counts:
            self.call_counts[name] = {'total_time': 0.0, 'count': 0}
        
        self.call_counts[name]['total_time'] += elapsed
        self.call_counts[name]['count'] += 1
        
        return elapsed
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """获取性能报告"""
        report = {}
        
        for name, stats in self.call_counts.items():
            report[name] = {
                'total_time': stats['total_time'],
                'count': stats['count'],
                'average_time': stats['total_time'] / stats['count'],
                'percentage': 0.0  # 将在后面计算
            }
        
        # 计算百分比
        total_time = sum(stats['total_time'] for stats in self.call_counts.values())
        if total_time > 0:
            for name in report:
                report[name]['percentage'] = (report[name]['total_time'] / total_time) * 100
        
        return report
    
    def log_performance_report(self) -> None:
        """记录性能报告"""
        report = self.get_performance_report()
        
        self.logger.info("=== 性能分析报告 ===")
        for name, stats in sorted(report.items(), key=lambda x: x[1]['total_time'], reverse=True):
            self.logger.info(
                f"{name}: {stats['total_time']:.3f}s "
                f"({stats['percentage']:.1f}%, "
                f"{stats['count']} calls, "
                f"{stats['average_time']*1000:.1f}ms avg)"
            )
    
    def reset(self) -> None:
        """重置统计"""
        self.timers.clear()
        self.call_counts.clear()
