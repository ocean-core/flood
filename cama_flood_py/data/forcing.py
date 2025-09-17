"""
强迫数据管理模块
对应Fortran源码中的强迫数据处理
"""

import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
import xarray as xr

from ..utils.constants import ModelConstants
from ..utils.data_converter import interpolate_time_series


class ForcingDataManager:
    """
    强迫数据管理器
    处理径流、降水等强迫数据的读取和时间插值
    """
    
    def __init__(self, nseq: int, forcing_dir: str = "./forcing"):
        """
        初始化强迫数据管理器
        
        Args:
            nseq: 网格序列数量
            forcing_dir: 强迫数据目录
        """
        self.nseq = nseq
        self.forcing_dir = Path(forcing_dir)
        self.logger = logging.getLogger(__name__)
        
        # 当前强迫数据
        self.runoff = np.zeros(nseq, dtype=np.float64)      # 径流 [m³/s]
        self.precipitation = np.zeros(nseq, dtype=np.float64)  # 降水 [mm/day]
        self.evaporation = np.zeros(nseq, dtype=np.float64)    # 蒸发 [mm/day]
        
        # 数据缓存
        self.forcing_cache = {}
        self.time_cache = {}
        
        # 插值参数
        self.interpolation_method = 'linear'
        
    def load_forcing_data(self, variable: str, file_pattern: str) -> bool:
        """
        加载强迫数据文件
        
        Args:
            variable: 变量名 ('runoff', 'precipitation', 'evaporation')
            file_pattern: 文件模式
            
        Returns:
            bool: 是否成功加载
        """
        try:
            files = list(self.forcing_dir.glob(file_pattern))
            if not files:
                self.logger.warning(f"未找到强迫数据文件: {file_pattern}")
                return False
            
            # 读取第一个文件作为示例
            file_path = files[0]
            
            if file_path.suffix == '.nc':
                # NetCDF格式
                data = self._load_netcdf_forcing(file_path, variable)
            else:
                # 二进制格式
                data = self._load_binary_forcing(file_path, variable)
            
            if data is not None:
                self.forcing_cache[variable] = data
                self.logger.info(f"成功加载强迫数据: {variable}")
                return True
            
        except Exception as e:
            self.logger.error(f"加载强迫数据失败 {variable}: {e}")
        
        return False
    
    def _load_netcdf_forcing(self, file_path: Path, variable: str) -> Optional[np.ndarray]:
        """加载NetCDF格式强迫数据"""
        try:
            with xr.open_dataset(file_path) as ds:
                if variable in ds.variables:
                    data = ds[variable].values
                    return data
                else:
                    self.logger.warning(f"变量 {variable} 不存在于文件 {file_path}")
                    return None
        except Exception as e:
            self.logger.error(f"读取NetCDF文件失败 {file_path}: {e}")
            return None
    
    def _load_binary_forcing(self, file_path: Path, variable: str) -> Optional[np.ndarray]:
        """加载二进制格式强迫数据"""
        try:
            # 假设为单精度浮点数
            data = np.fromfile(file_path, dtype=np.float32)
            
            # 重塑为合适的形状 (时间, 空间)
            if len(data) % self.nseq == 0:
                ntime = len(data) // self.nseq
                data = data.reshape(ntime, self.nseq)
                return data.astype(np.float64)
            else:
                self.logger.warning(f"数据大小不匹配: {len(data)} vs {self.nseq}")
                return None
                
        except Exception as e:
            self.logger.error(f"读取二进制文件失败 {file_path}: {e}")
            return None
    
    def update_forcing(self, current_time: float, time_step: float) -> None:
        """
        更新当前时刻的强迫数据
        
        Args:
            current_time: 当前时间 [s]
            time_step: 时间步长 [s]
        """
        # 径流数据更新
        if 'runoff' in self.forcing_cache:
            self.runoff = self._interpolate_forcing('runoff', current_time)
        else:
            # 使用默认值
            self.runoff.fill(0.1)  # 0.1 m³/s 默认径流
        
        # 降水数据更新
        if 'precipitation' in self.forcing_cache:
            self.precipitation = self._interpolate_forcing('precipitation', current_time)
        else:
            self.precipitation.fill(0.0)
        
        # 蒸发数据更新
        if 'evaporation' in self.forcing_cache:
            self.evaporation = self._interpolate_forcing('evaporation', current_time)
        else:
            self.evaporation.fill(0.0)
    
    def _interpolate_forcing(self, variable: str, current_time: float) -> np.ndarray:
        """
        插值强迫数据到当前时间
        
        Args:
            variable: 变量名
            current_time: 当前时间 [s]
            
        Returns:
            np.ndarray: 插值后的数据
        """
        if variable not in self.forcing_cache:
            return np.zeros(self.nseq)
        
        data = self.forcing_cache[variable]
        
        # 简化的时间插值：使用第一个时间步的数据
        if data.ndim == 2:
            return data[0, :]  # 返回第一个时间步
        else:
            return data[:self.nseq]  # 确保长度匹配
    
    def get_forcing_dict(self) -> Dict[str, np.ndarray]:
        """
        获取当前强迫数据字典
        
        Returns:
            dict: 强迫数据
        """
        return {
            'runoff': self.runoff.copy(),
            'precipitation': self.precipitation.copy(),
            'evaporation': self.evaporation.copy()
        }
    
    def validate_forcing_data(self) -> Dict[str, bool]:
        """
        验证强迫数据
        
        Returns:
            dict: 验证结果
        """
        validation = {}
        
        # 检查径流数据
        validation['runoff_valid'] = (
            np.all(np.isfinite(self.runoff)) and 
            np.all(self.runoff >= 0)
        )
        
        # 检查降水数据
        validation['precipitation_valid'] = (
            np.all(np.isfinite(self.precipitation)) and 
            np.all(self.precipitation >= 0)
        )
        
        # 检查蒸发数据
        validation['evaporation_valid'] = (
            np.all(np.isfinite(self.evaporation)) and 
            np.all(self.evaporation >= 0)
        )
        
        return validation


class ForcingPreprocessor:
    """
    强迫数据预处理器
    处理原始强迫数据的格式转换和质量控制
    """
    
    def __init__(self):
        """初始化预处理器"""
        self.logger = logging.getLogger(__name__)
    
    def convert_units(self, data: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
        """
        单位转换
        
        Args:
            data: 输入数据
            from_unit: 原始单位
            to_unit: 目标单位
            
        Returns:
            np.ndarray: 转换后的数据
        """
        conversion_factors = {
            ('mm/day', 'm/s'): 1.0 / (1000 * 86400),  # mm/day to m/s
            ('mm/hour', 'm/s'): 1.0 / (1000 * 3600),  # mm/hour to m/s
            ('kg/m2/s', 'm/s'): 1.0 / 1000,           # kg/m2/s to m/s
            ('m3/s', 'm3/s'): 1.0,                    # no conversion
        }
        
        factor = conversion_factors.get((from_unit, to_unit), 1.0)
        return data * factor
    
    def quality_control(self, data: np.ndarray, variable: str) -> np.ndarray:
        """
        数据质量控制
        
        Args:
            data: 输入数据
            variable: 变量名
            
        Returns:
            np.ndarray: 质量控制后的数据
        """
        # 处理缺失值
        data = np.where(np.isnan(data), 0.0, data)
        data = np.where(np.isinf(data), 0.0, data)
        
        # 变量特定的范围检查
        if variable == 'runoff':
            # 径流不能为负值，上限为合理值
            data = np.clip(data, 0.0, 1000.0)  # 最大1000 m³/s
        elif variable == 'precipitation':
            # 降水不能为负值
            data = np.clip(data, 0.0, 1000.0)  # 最大1000 mm/day
        elif variable == 'evaporation':
            # 蒸发不能为负值
            data = np.clip(data, 0.0, 50.0)    # 最大50 mm/day
        
        return data
    
    def spatial_interpolation(self, data: np.ndarray, 
                            source_coords: Tuple[np.ndarray, np.ndarray],
                            target_coords: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        空间插值
        
        Args:
            data: 源数据
            source_coords: 源坐标 (lon, lat)
            target_coords: 目标坐标 (lon, lat)
            
        Returns:
            np.ndarray: 插值后的数据
        """
        # 简化实现：最近邻插值
        from scipy.spatial import cKDTree
        
        source_points = np.column_stack(source_coords)
        target_points = np.column_stack(target_coords)
        
        tree = cKDTree(source_points)
        distances, indices = tree.query(target_points)
        
        return data[indices]


def create_sample_forcing_data(nseq: int, ntime: int = 24) -> Dict[str, np.ndarray]:
    """
    创建示例强迫数据
    
    Args:
        nseq: 空间网格数
        ntime: 时间步数
        
    Returns:
        dict: 示例强迫数据
    """
    # 创建时间变化的径流数据
    time_factor = np.sin(np.linspace(0, 2*np.pi, ntime))
    spatial_factor = np.random.rand(nseq) * 0.5 + 0.5
    
    runoff = np.outer(time_factor * 2 + 3, spatial_factor)  # 1-5 m³/s
    precipitation = np.random.rand(ntime, nseq) * 10       # 0-10 mm/day
    evaporation = np.random.rand(ntime, nseq) * 5          # 0-5 mm/day
    
    return {
        'runoff': runoff,
        'precipitation': precipitation,
        'evaporation': evaporation
    }
