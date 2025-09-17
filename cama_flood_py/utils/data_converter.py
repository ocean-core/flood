"""
数据转换工具
对应Fortran源码: cmf_utils_mod.F90中的数据转换函数
"""

import numpy as np
import struct
from typing import Tuple, Union, Optional
import sys

from .constants import ModelConstants, GridConstants


class DataConverter:
    """
    数据转换器
    对应Fortran中的数据格式转换函数
    """
    
    @staticmethod
    def vector_to_map(vector_data: np.ndarray, nx: int, ny: int, 
                     missing_value: float = ModelConstants.MISSING_VALUE) -> np.ndarray:
        """
        向量数据转换为地图格式
        对应Fortran: vecD2mapR (cmf_utils_mod.F90:45-64行)
        
        Args:
            vector_data: 一维向量数据
            nx: 经度网格数
            ny: 纬度网格数
            missing_value: 缺失值
            
        Returns:
            np.ndarray: 二维地图数据 (ny, nx)
        """
        # 初始化地图数组
        map_data = np.full((ny, nx), missing_value, dtype=np.float64)
        
        # 将向量数据填入地图
        if len(vector_data) > 0:
            # 假设向量数据按行优先顺序排列
            valid_size = min(len(vector_data), nx * ny)
            map_data.flat[:valid_size] = vector_data[:valid_size]
        
        return map_data
    
    @staticmethod
    def map_to_vector(map_data: np.ndarray, 
                     missing_value: float = ModelConstants.MISSING_VALUE) -> np.ndarray:
        """
        地图数据转换为向量格式
        对应Fortran: mapR2vecD (cmf_utils_mod.F90:143-160行)
        
        Args:
            map_data: 二维地图数据
            missing_value: 缺失值
            
        Returns:
            np.ndarray: 一维向量数据（去除缺失值）
        """
        # 展平为一维数组
        vector_data = map_data.flatten()
        
        # 过滤缺失值
        valid_mask = vector_data != missing_value
        return vector_data[valid_mask]
    
    @staticmethod
    def convert_endian(data: bytes, from_endian: str = 'big', 
                      to_endian: str = 'little') -> bytes:
        """
        字节序转换
        对应Fortran: CONV_END (cmf_utils_mod.F90:431-445行)
        
        Args:
            data: 原始字节数据
            from_endian: 源字节序 ('big' 或 'little')
            to_endian: 目标字节序 ('big' 或 'little')
            
        Returns:
            bytes: 转换后的字节数据
        """
        if from_endian == to_endian:
            return data
        
        # 假设数据是4字节浮点数数组
        float_count = len(data) // 4
        
        if from_endian == 'big':
            format_from = f'>{float_count}f'
        else:
            format_from = f'<{float_count}f'
        
        if to_endian == 'big':
            format_to = f'>{float_count}f'
        else:
            format_to = f'<{float_count}f'
        
        # 解包并重新打包
        values = struct.unpack(format_from, data)
        return struct.pack(format_to, *values)
    
    @staticmethod
    def read_fortran_binary(file_path: str, dtype: np.dtype = np.float32,
                           shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """
        读取Fortran二进制文件
        处理Fortran记录长度标识
        
        Args:
            file_path: 文件路径
            dtype: 数据类型
            shape: 数据形状
            
        Returns:
            np.ndarray: 读取的数据
        """
        with open(file_path, 'rb') as f:
            # 读取第一个记录长度标识（4字节）
            record_len_bytes = f.read(4)
            if len(record_len_bytes) != 4:
                raise ValueError("文件格式错误：无法读取记录长度")
            
            record_len = struct.unpack('<I', record_len_bytes)[0]
            
            # 读取实际数据
            data_bytes = f.read(record_len)
            if len(data_bytes) != record_len:
                raise ValueError(f"文件格式错误：期望{record_len}字节，实际读取{len(data_bytes)}字节")
            
            # 读取结束记录长度标识
            end_record_len_bytes = f.read(4)
            if len(end_record_len_bytes) != 4:
                raise ValueError("文件格式错误：无法读取结束记录长度")
            
            end_record_len = struct.unpack('<I', end_record_len_bytes)[0]
            if record_len != end_record_len:
                raise ValueError("文件格式错误：记录长度不匹配")
            
            # 转换为numpy数组
            data = np.frombuffer(data_bytes, dtype=dtype)
            
            if shape:
                data = data.reshape(shape)
            
            return data
    
    @staticmethod
    def write_fortran_binary(file_path: str, data: np.ndarray) -> None:
        """
        写入Fortran二进制文件
        添加Fortran记录长度标识
        
        Args:
            file_path: 文件路径
            data: 要写入的数据
        """
        with open(file_path, 'wb') as f:
            # 获取数据字节
            data_bytes = data.tobytes()
            record_len = len(data_bytes)
            
            # 写入记录长度标识
            f.write(struct.pack('<I', record_len))
            
            # 写入数据
            f.write(data_bytes)
            
            # 写入结束记录长度标识
            f.write(struct.pack('<I', record_len))
    
    @staticmethod
    def check_data_consistency(data1: np.ndarray, data2: np.ndarray,
                             tolerance: float = 1e-6) -> Tuple[bool, float, float]:
        """
        检查两个数据数组的一致性
        用于验证Fortran和Python版本的输出
        
        Args:
            data1: 第一个数据数组
            data2: 第二个数据数组
            tolerance: 容差
            
        Returns:
            tuple: (是否一致, 最大相对误差, 平均相对误差)
        """
        if data1.shape != data2.shape:
            return False, float('inf'), float('inf')
        
        # 计算相对误差
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_error = np.abs(data2 - data1) / np.abs(data1)
            relative_error = np.where(np.isfinite(relative_error), relative_error, 0)
        
        max_error = np.max(relative_error)
        mean_error = np.mean(relative_error)
        
        is_consistent = max_error < tolerance
        
        return is_consistent, max_error, mean_error
    
    @staticmethod
    def interpolate_time_series(time_points: np.ndarray, values: np.ndarray,
                               target_time: float, method: str = 'linear') -> float:
        """
        时间序列插值
        对应Fortran中的时间插值功能
        
        Args:
            time_points: 时间点数组
            values: 对应的数值数组
            target_time: 目标时间点
            method: 插值方法 ('linear', 'nearest')
            
        Returns:
            float: 插值结果
        """
        if len(time_points) != len(values):
            raise ValueError("时间点和数值数组长度不匹配")
        
        if len(time_points) == 0:
            return 0.0
        
        if len(time_points) == 1:
            return values[0]
        
        # 确保时间点是排序的
        sort_indices = np.argsort(time_points)
        sorted_times = time_points[sort_indices]
        sorted_values = values[sort_indices]
        
        if method == 'nearest':
            # 最近邻插值
            idx = np.argmin(np.abs(sorted_times - target_time))
            return sorted_values[idx]
        
        elif method == 'linear':
            # 线性插值
            if target_time <= sorted_times[0]:
                return sorted_values[0]
            elif target_time >= sorted_times[-1]:
                return sorted_values[-1]
            else:
                # 找到插值区间
                idx = np.searchsorted(sorted_times, target_time) - 1
                t1, t2 = sorted_times[idx], sorted_times[idx + 1]
                v1, v2 = sorted_values[idx], sorted_values[idx + 1]
                
                # 线性插值
                weight = (target_time - t1) / (t2 - t1)
                return v1 + weight * (v2 - v1)
        
        else:
            raise ValueError(f"不支持的插值方法: {method}")
    
    @staticmethod
    def calculate_grid_area(lat: float, lon_resolution: float, 
                           lat_resolution: float) -> float:
        """
        计算网格面积
        
        Args:
            lat: 纬度（度）
            lon_resolution: 经度分辨率（度）
            lat_resolution: 纬度分辨率（度）
            
        Returns:
            float: 网格面积（平方米）
        """
        from ..utils.constants import PhysicalConstants
        
        # 转换为弧度
        lat_rad = lat * PhysicalConstants.DEG_TO_RAD
        lon_res_rad = lon_resolution * PhysicalConstants.DEG_TO_RAD
        lat_res_rad = lat_resolution * PhysicalConstants.DEG_TO_RAD
        
        # 地球半径
        R = PhysicalConstants.EARTH_RADIUS
        
        # 计算面积
        area = R * R * lon_res_rad * np.sin(lat_rad + lat_res_rad/2) * lat_res_rad
        
        return abs(area)
    
    @staticmethod
    def calculate_grid_area(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        计算网格面积
        
        Args:
            lon: 经度数组 [度]
            lat: 纬度数组 [度]
            
        Returns:
            np.ndarray: 网格面积 [m²]
        """
        # 地球半径
        earth_radius = 6371000.0  # m
        
        # 转换为弧度
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        
        # 计算网格大小（假设规则网格）
        if len(lon) > 1 and len(lat) > 1:
            dlon = abs(lon_rad[1] - lon_rad[0])
            dlat = abs(lat_rad[1] - lat_rad[0])
        else:
            dlon = np.radians(0.25)  # 默认0.25度
            dlat = np.radians(0.25)
        
        # 计算面积
        area = earth_radius**2 * dlon * dlat * np.cos(lat_rad)
        
        return area


def interpolate_time_series(time_points: np.ndarray, values: np.ndarray, 
                          target_time: float, method: str = 'linear') -> float:
    """
    时间序列插值
    
    Args:
        time_points: 时间点数组
        values: 对应的数值数组
        target_time: 目标时间
        method: 插值方法 ('linear', 'nearest')
        
    Returns:
        float: 插值结果
    """
    if len(time_points) == 0 or len(values) == 0:
        return 0.0
    
    if len(time_points) != len(values):
        raise ValueError("时间点和数值数组长度不匹配")
    
    # 边界处理
    if target_time <= time_points[0]:
        return values[0]
    if target_time >= time_points[-1]:
        return values[-1]
    
    # 查找插值区间
    idx = np.searchsorted(time_points, target_time)
    
    if method == 'nearest':
        # 最近邻插值
        if abs(target_time - time_points[idx-1]) < abs(target_time - time_points[idx]):
            return values[idx-1]
        else:
            return values[idx]
    
    elif method == 'linear':
        # 线性插值
        t1, t2 = time_points[idx-1], time_points[idx]
        v1, v2 = values[idx-1], values[idx]
        
        weight = (target_time - t1) / (t2 - t1)
        return v1 + weight * (v2 - v1)
    
    else:
        raise ValueError(f"不支持的插值方法: {method}")


def interpolate_spatial_data(source_data: np.ndarray, source_coords: np.ndarray,
                           target_coords: np.ndarray, method: str = 'nearest') -> np.ndarray:
    """
    空间数据插值
    
    Args:
        source_data: 源数据
        source_coords: 源坐标 (N, 2)
        target_coords: 目标坐标 (M, 2)
        method: 插值方法
        
    Returns:
        np.ndarray: 插值后的数据
    """
    if method == 'nearest':
        # 最近邻插值
        from scipy.spatial import cKDTree
        tree = cKDTree(source_coords)
        distances, indices = tree.query(target_coords)
        return source_data[indices]
    else:
        raise ValueError(f"不支持的空间插值方法: {method}")


def create_coordinate_arrays(nx: int, ny: int, 
                            west: float = -180.0, east: float = 180.0,
                            south: float = -90.0, north: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建坐标数组
    
    Args:
        nx: 经度网格数
        ny: 纬度网格数
        west: 西边界
        east: 东边界
        south: 南边界
        north: 北边界
        
    Returns:
        tuple: (经度数组, 纬度数组)
    """
    lon = np.linspace(west, east, nx, endpoint=False)
    lat = np.linspace(south, north, ny, endpoint=False)
    
    return lon, lat
