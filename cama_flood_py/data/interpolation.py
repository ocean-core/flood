"""
数据插值模块
提供时空数据插值功能
"""

import numpy as np
from typing import Tuple, Optional
import logging


class DataInterpolator:
    """
    数据插值器
    提供各种插值方法
    """
    
    def __init__(self):
        """初始化插值器"""
        self.logger = logging.getLogger(__name__)
    
    def temporal_interpolation(self, time_points: np.ndarray, values: np.ndarray,
                             target_time: float, method: str = 'linear') -> float:
        """
        时间插值
        
        Args:
            time_points: 时间点数组
            values: 对应的数值数组
            target_time: 目标时间
            method: 插值方法
            
        Returns:
            float: 插值结果
        """
        if len(time_points) == 0 or len(values) == 0:
            return 0.0
        
        if method == 'linear':
            return self._linear_interpolation(time_points, values, target_time)
        elif method == 'nearest':
            return self._nearest_interpolation(time_points, values, target_time)
        else:
            raise ValueError(f"不支持的插值方法: {method}")
    
    def _linear_interpolation(self, x: np.ndarray, y: np.ndarray, target: float) -> float:
        """线性插值"""
        if target <= x[0]:
            return y[0]
        if target >= x[-1]:
            return y[-1]
        
        idx = np.searchsorted(x, target)
        x1, x2 = x[idx-1], x[idx]
        y1, y2 = y[idx-1], y[idx]
        
        weight = (target - x1) / (x2 - x1)
        return y1 + weight * (y2 - y1)
    
    def _nearest_interpolation(self, x: np.ndarray, y: np.ndarray, target: float) -> float:
        """最近邻插值"""
        idx = np.argmin(np.abs(x - target))
        return y[idx]
    
    def spatial_interpolation(self, source_coords: np.ndarray, source_values: np.ndarray,
                            target_coords: np.ndarray, method: str = 'nearest') -> np.ndarray:
        """
        空间插值
        
        Args:
            source_coords: 源坐标 (N, 2)
            source_values: 源数值 (N,)
            target_coords: 目标坐标 (M, 2)
            method: 插值方法
            
        Returns:
            np.ndarray: 插值结果 (M,)
        """
        if method == 'nearest':
            from scipy.spatial import cKDTree
            tree = cKDTree(source_coords)
            distances, indices = tree.query(target_coords)
            return source_values[indices]
        else:
            raise ValueError(f"不支持的空间插值方法: {method}")
