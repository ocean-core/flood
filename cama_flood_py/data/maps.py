"""
地图数据管理模块
对应Fortran源码中的地图数据处理
"""

import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import xarray as xr

from ..utils.constants import ModelConstants
from ..utils.data_converter import DataConverter


class MapDataManager:
    """
    地图数据管理器
    处理河网地形、参数等地图数据的读取和管理
    """
    
    def __init__(self, nseq: int, map_dir: str = "./map"):
        """
        初始化地图数据管理器
        
        Args:
            nseq: 网格序列数量
            map_dir: 地图数据目录
        """
        self.nseq = nseq
        self.map_dir = Path(map_dir)
        self.logger = logging.getLogger(__name__)
        
        # 地形数据
        self.river_elevation = np.zeros(nseq, dtype=np.float64)    # 河床高程 [m]
        self.river_width = np.zeros(nseq, dtype=np.float64)       # 河道宽度 [m]
        self.river_height = np.zeros(nseq, dtype=np.float64)      # 河岸高度 [m]
        self.river_length = np.zeros(nseq, dtype=np.float64)      # 河段长度 [m]
        self.catchment_area = np.zeros(nseq, dtype=np.float64)    # 集水面积 [m²]
        
        # 河网连接
        self.next_index = np.zeros(nseq, dtype=np.int32)          # 下游网格索引
        self.next_distance = np.zeros(nseq, dtype=np.float64)     # 到下游距离 [m]
        
        # 坐标信息
        self.longitude = np.zeros(nseq, dtype=np.float64)         # 经度 [度]
        self.latitude = np.zeros(nseq, dtype=np.float64)          # 纬度 [度]
        
        # 数据转换器
        self.converter = DataConverter()
        
    def load_map_data(self, file_pattern: str = "*.bin") -> bool:
        """
        加载地图数据文件
        
        Args:
            file_pattern: 文件模式
            
        Returns:
            bool: 是否成功加载
        """
        try:
            # 查找地图文件
            files = list(self.map_dir.glob(file_pattern))
            if not files:
                self.logger.warning(f"未找到地图数据文件: {file_pattern}")
                return False
            
            # 加载各种地图数据
            success = True
            success &= self._load_elevation_data()
            success &= self._load_river_parameters()
            success &= self._load_network_topology()
            success &= self._load_coordinates()
            
            if success:
                self.logger.info("成功加载地图数据")
            
            return success
            
        except Exception as e:
            self.logger.error(f"加载地图数据失败: {e}")
            return False
    
    def _load_elevation_data(self) -> bool:
        """加载高程数据"""
        try:
            # 查找河床高程文件
            elv_files = list(self.map_dir.glob("*elv*"))
            if elv_files:
                data = self._read_binary_file(elv_files[0])
                if data is not None:
                    self.river_elevation[:len(data)] = data[:self.nseq]
                    return True
            
            # 使用默认值
            self.river_elevation = np.linspace(100, 0, self.nseq)  # 默认坡度
            self.logger.warning("使用默认河床高程数据")
            return True
            
        except Exception as e:
            self.logger.error(f"加载高程数据失败: {e}")
            return False
    
    def _load_river_parameters(self) -> bool:
        """加载河道参数"""
        try:
            # 河道宽度
            width_files = list(self.map_dir.glob("*wth*"))
            if width_files:
                data = self._read_binary_file(width_files[0])
                if data is not None:
                    self.river_width[:len(data)] = data[:self.nseq]
            else:
                self.river_width.fill(50.0)  # 默认50m宽度
            
            # 河岸高度
            height_files = list(self.map_dir.glob("*hgt*"))
            if height_files:
                data = self._read_binary_file(height_files[0])
                if data is not None:
                    self.river_height[:len(data)] = data[:self.nseq]
            else:
                self.river_height.fill(5.0)  # 默认5m高度
            
            # 河段长度
            length_files = list(self.map_dir.glob("*len*"))
            if length_files:
                data = self._read_binary_file(length_files[0])
                if data is not None:
                    self.river_length[:len(data)] = data[:self.nseq]
            else:
                self.river_length.fill(1000.0)  # 默认1000m长度
            
            # 集水面积
            area_files = list(self.map_dir.glob("*area*"))
            if area_files:
                data = self._read_binary_file(area_files[0])
                if data is not None:
                    self.catchment_area[:len(data)] = data[:self.nseq]
            else:
                self.catchment_area.fill(1e6)  # 默认1 km²
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载河道参数失败: {e}")
            return False
    
    def _load_network_topology(self) -> bool:
        """加载河网拓扑"""
        try:
            # 下游索引
            next_files = list(self.map_dir.glob("*next*"))
            if next_files:
                data = self._read_binary_file(next_files[0], dtype=np.int32)
                if data is not None:
                    self.next_index[:len(data)] = data[:self.nseq]
            else:
                # 创建简单的线性连接
                self.next_index = np.arange(1, self.nseq + 1)
                self.next_index[-1] = 0  # 最后一个为出口
            
            # 到下游距离
            dist_files = list(self.map_dir.glob("*dist*"))
            if dist_files:
                data = self._read_binary_file(dist_files[0])
                if data is not None:
                    self.next_distance[:len(data)] = data[:self.nseq]
            else:
                self.next_distance.fill(1000.0)  # 默认1000m距离
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载河网拓扑失败: {e}")
            return False
    
    def _load_coordinates(self) -> bool:
        """加载坐标信息"""
        try:
            # 经度
            lon_files = list(self.map_dir.glob("*lon*"))
            if lon_files:
                data = self._read_binary_file(lon_files[0])
                if data is not None:
                    self.longitude[:len(data)] = data[:self.nseq]
            else:
                self.longitude = np.linspace(-180, 180, self.nseq)
            
            # 纬度
            lat_files = list(self.map_dir.glob("*lat*"))
            if lat_files:
                data = self._read_binary_file(lat_files[0])
                if data is not None:
                    self.latitude[:len(data)] = data[:self.nseq]
            else:
                self.latitude = np.linspace(-90, 90, self.nseq)
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载坐标信息失败: {e}")
            return False
    
    def _read_binary_file(self, file_path: Path, dtype=np.float32) -> Optional[np.ndarray]:
        """读取二进制文件"""
        try:
            data = np.fromfile(file_path, dtype=dtype)
            return data.astype(np.float64) if dtype != np.int32 else data.astype(np.int32)
        except Exception as e:
            self.logger.error(f"读取二进制文件失败 {file_path}: {e}")
            return None
    
    def get_map_dict(self) -> Dict[str, np.ndarray]:
        """
        获取地图数据字典
        
        Returns:
            dict: 地图数据
        """
        return {
            'river_elevation': self.river_elevation.copy(),
            'river_width': self.river_width.copy(),
            'river_height': self.river_height.copy(),
            'river_length': self.river_length.copy(),
            'catchment_area': self.catchment_area.copy(),
            'next_index': self.next_index.copy(),
            'next_distance': self.next_distance.copy(),
            'longitude': self.longitude.copy(),
            'latitude': self.latitude.copy()
        }
    
    def validate_map_data(self) -> Dict[str, bool]:
        """
        验证地图数据
        
        Returns:
            dict: 验证结果
        """
        validation = {}
        
        # 检查地形数据
        validation['elevation_valid'] = np.all(np.isfinite(self.river_elevation))
        validation['width_valid'] = np.all(self.river_width > 0)
        validation['height_valid'] = np.all(self.river_height > 0)
        validation['length_valid'] = np.all(self.river_length > 0)
        validation['area_valid'] = np.all(self.catchment_area > 0)
        
        # 检查河网连接
        validation['topology_valid'] = (
            np.all(self.next_index >= 0) and 
            np.all(self.next_index < self.nseq)
        )
        
        # 检查坐标
        validation['coordinates_valid'] = (
            np.all(np.abs(self.longitude) <= 180) and
            np.all(np.abs(self.latitude) <= 90)
        )
        
        return validation
    
    def create_sample_map(self) -> None:
        """创建示例地图数据"""
        # 创建简单的河网
        self.river_elevation = np.linspace(1000, 0, self.nseq)  # 从1000m到0m
        self.river_width = np.random.uniform(10, 100, self.nseq)  # 10-100m宽度
        self.river_height = np.random.uniform(2, 10, self.nseq)   # 2-10m高度
        self.river_length = np.random.uniform(500, 2000, self.nseq)  # 500-2000m长度
        self.catchment_area = np.random.uniform(1e5, 1e7, self.nseq)  # 0.1-10 km²
        
        # 线性河网连接
        self.next_index = np.arange(1, self.nseq + 1)
        self.next_index[-1] = 0  # 最后一个为出口
        self.next_distance = self.river_length.copy()
        
        # 随机坐标
        self.longitude = np.random.uniform(-180, 180, self.nseq)
        self.latitude = np.random.uniform(-60, 60, self.nseq)
        
        self.logger.info("创建了示例地图数据")


def create_sample_map_data(nseq: int) -> Dict[str, np.ndarray]:
    """
    创建示例地图数据
    
    Args:
        nseq: 网格数量
        
    Returns:
        dict: 示例地图数据
    """
    map_manager = MapDataManager(nseq)
    map_manager.create_sample_map()
    return map_manager.get_map_dict()
