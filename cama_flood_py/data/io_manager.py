"""
输入输出管理模块
对应Fortran源码中的文件I/O处理
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from ..utils.data_converter import DataConverter
from ..utils.constants import ModelConstants, FileConstants


class IOManager:
    """
    输入输出管理器
    对应Fortran中的文件I/O函数
    """
    
    def __init__(self, base_path: str = "./"):
        """
        初始化I/O管理器
        
        Args:
            base_path: 基础路径
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        
        # 数据缓存
        self._map_data_cache = {}
        self._forcing_data_cache = {}
    
    def read_binary_map_file(self, filename: str, nx: int, ny: int,
                            dtype: np.dtype = np.float32) -> np.ndarray:
        """
        读取二进制地图文件
        对应Fortran中的地图数据读取
        
        Args:
            filename: 文件名
            nx: 经度网格数
            ny: 纬度网格数
            dtype: 数据类型
            
        Returns:
            np.ndarray: 地图数据
        """
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"地图文件不存在: {file_path}")
        
        try:
            # 使用DataConverter读取Fortran二进制文件
            data = DataConverter.read_fortran_binary(
                str(file_path), dtype=dtype, shape=(ny, nx)
            )
            
            self.logger.info(f"成功读取地图文件: {filename}, 形状: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"读取地图文件失败 {filename}: {e}")
            raise
    
    def read_forcing_file(self, filename: str, nx: int, ny: int,
                         time_steps: int = 1) -> np.ndarray:
        """
        读取强迫数据文件
        对应Fortran中的强迫数据读取
        
        Args:
            filename: 文件名
            nx: 经度网格数
            ny: 纬度网格数
            time_steps: 时间步数
            
        Returns:
            np.ndarray: 强迫数据 (time, ny, nx)
        """
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"强迫文件不存在: {file_path}")
        
        try:
            if filename.endswith('.nc'):
                # NetCDF格式
                with xr.open_dataset(file_path) as ds:
                    # 假设变量名为'runoff'
                    data = ds['runoff'].values
            else:
                # 二进制格式
                total_size = nx * ny * time_steps
                data = DataConverter.read_fortran_binary(
                    str(file_path), dtype=np.float32, shape=(time_steps, ny, nx)
                )
            
            self.logger.info(f"成功读取强迫文件: {filename}, 形状: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"读取强迫文件失败 {filename}: {e}")
            raise
    
    def write_output_netcdf(self, filename: str, data_dict: Dict[str, np.ndarray],
                           coordinates: Dict[str, np.ndarray],
                           attributes: Dict[str, Any] = None) -> None:
        """
        写入NetCDF输出文件
        对应Fortran: CMF_OUTPUT_WRITE
        
        Args:
            filename: 输出文件名
            data_dict: 数据字典 {变量名: 数据数组}
            coordinates: 坐标字典 {坐标名: 坐标数组}
            attributes: 属性字典
        """
        file_path = self.base_path / filename
        
        try:
            # 创建xarray数据集
            data_vars = {}
            
            for var_name, data in data_dict.items():
                if data.ndim == 2:  # 2D数据 (lat, lon)
                    data_vars[var_name] = (['lat', 'lon'], data)
                elif data.ndim == 3:  # 3D数据 (time, lat, lon)
                    data_vars[var_name] = (['time', 'lat', 'lon'], data)
                else:
                    self.logger.warning(f"不支持的数据维度: {var_name}, 维度: {data.ndim}")
            
            # 创建数据集
            ds = xr.Dataset(data_vars, coords=coordinates)
            
            # 添加属性
            if attributes:
                ds.attrs.update(attributes)
            
            # 添加变量属性
            variable_attrs = {
                'rivout': {
                    'long_name': 'River discharge',
                    'units': 'm3/s',
                    'description': 'River channel discharge'
                },
                'rivsto': {
                    'long_name': 'River storage',
                    'units': 'm3',
                    'description': 'River channel storage'
                },
                'rivdph': {
                    'long_name': 'River depth',
                    'units': 'm',
                    'description': 'River channel depth'
                },
                'flddph': {
                    'long_name': 'Flood depth',
                    'units': 'm',
                    'description': 'Floodplain depth'
                },
                'sfcelv': {
                    'long_name': 'Surface elevation',
                    'units': 'm',
                    'description': 'Water surface elevation'
                }
            }
            
            for var_name in data_vars.keys():
                if var_name in variable_attrs:
                    ds[var_name].attrs.update(variable_attrs[var_name])
            
            # 写入文件
            encoding = {}
            if 'compression' in ds.attrs and ds.attrs['compression']:
                encoding = {var: {'zlib': True, 'complevel': 6} 
                          for var in data_vars.keys()}
            
            ds.to_netcdf(file_path, encoding=encoding)
            
            self.logger.info(f"成功写入NetCDF文件: {filename}")
            
        except Exception as e:
            self.logger.error(f"写入NetCDF文件失败 {filename}: {e}")
            raise
    
    def write_output_binary(self, filename: str, data: np.ndarray) -> None:
        """
        写入二进制输出文件
        对应Fortran二进制输出格式
        
        Args:
            filename: 输出文件名
            data: 输出数据
        """
        file_path = self.base_path / filename
        
        try:
            DataConverter.write_fortran_binary(str(file_path), data)
            self.logger.info(f"成功写入二进制文件: {filename}")
            
        except Exception as e:
            self.logger.error(f"写入二进制文件失败 {filename}: {e}")
            raise
    
    def handle_netcdf_error(self, error: Exception, operation: str) -> None:
        """
        处理NetCDF错误
        对应Fortran: NCERROR (cmf_utils_mod.F90:554-567行)
        
        Args:
            error: 异常对象
            operation: 操作描述
        """
        error_msg = f"NetCDF错误 - 操作: {operation}, 错误: {str(error)}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def load_map_data(self, config) -> Dict[str, np.ndarray]:
        """
        加载所有地图数据
        对应Fortran中的地图初始化
        
        Args:
            config: 配置对象
            
        Returns:
            dict: 地图数据字典
        """
        map_dir = Path(config.input.map_directory)
        map_data = {}
        
        # 地图文件列表
        map_files = {
            'next_xy': config.input.next_xy,
            'catchment_area': config.input.catchment_area,
            'river_elevation': config.input.river_elevation,
            'river_width': config.input.river_width,
            'river_height': config.input.river_height,
            'river_length': config.input.river_length,
            'next_distance': config.input.next_distance
        }
        
        # 假设网格尺寸（实际应从配置或文件中获取）
        nx, ny = 1440, 720  # 15分分辨率全球网格
        
        for key, filename in map_files.items():
            file_path = map_dir / filename
            
            if file_path.exists():
                try:
                    if key == 'next_xy':
                        # 整数类型的索引数据
                        data = self.read_binary_map_file(
                            str(file_path), nx, ny, dtype=np.int32
                        )
                    else:
                        # 浮点类型的物理量数据
                        data = self.read_binary_map_file(
                            str(file_path), nx, ny, dtype=np.float32
                        )
                    
                    map_data[key] = data
                    self.logger.info(f"加载地图数据: {key}")
                    
                except Exception as e:
                    self.logger.warning(f"无法加载地图文件 {filename}: {e}")
                    # 使用默认值
                    if key == 'next_xy':
                        map_data[key] = np.zeros((ny, nx), dtype=np.int32)
                    else:
                        map_data[key] = np.ones((ny, nx), dtype=np.float32)
            else:
                self.logger.warning(f"地图文件不存在: {file_path}")
                # 创建默认数据
                if key == 'next_xy':
                    map_data[key] = np.zeros((ny, nx), dtype=np.int32)
                else:
                    map_data[key] = np.ones((ny, nx), dtype=np.float32)
        
        return map_data
    
    def create_restart_file(self, filename: str, state_data: Dict[str, np.ndarray],
                           coordinates: Dict[str, np.ndarray]) -> None:
        """
        创建重启文件
        对应Fortran: CMF_RESTART_WRITE
        
        Args:
            filename: 重启文件名
            state_data: 状态数据
            coordinates: 坐标数据
        """
        try:
            # 添加重启文件特定属性
            restart_attrs = {
                'title': 'CaMa-Flood Restart File',
                'source': 'CaMa-Flood Python',
                'conventions': 'CF-1.6',
                'history': f'Created by CaMa-Flood Python at {np.datetime64("now")}'
            }
            
            self.write_output_netcdf(filename, state_data, coordinates, restart_attrs)
            self.logger.info(f"创建重启文件: {filename}")
            
        except Exception as e:
            self.logger.error(f"创建重启文件失败: {e}")
            raise
    
    def read_restart_file(self, filename: str) -> Dict[str, np.ndarray]:
        """
        读取重启文件
        
        Args:
            filename: 重启文件名
            
        Returns:
            dict: 状态数据
        """
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"重启文件不存在: {file_path}")
        
        try:
            with xr.open_dataset(file_path) as ds:
                state_data = {}
                for var_name in ds.data_vars:
                    state_data[var_name] = ds[var_name].values
                
            self.logger.info(f"读取重启文件: {filename}")
            return state_data
            
        except Exception as e:
            self.logger.error(f"读取重启文件失败: {e}")
            raise
    
    def validate_input_files(self, config) -> bool:
        """
        验证输入文件的存在性和有效性
        
        Args:
            config: 配置对象
            
        Returns:
            bool: 文件是否有效
        """
        valid = True
        
        # 检查地图文件
        map_dir = Path(config.input.map_directory)
        if not map_dir.exists():
            self.logger.error(f"地图数据目录不存在: {map_dir}")
            valid = False
        
        # 检查强迫数据目录
        forcing_dir = Path(config.input.forcing_directory)
        if not forcing_dir.exists():
            self.logger.error(f"强迫数据目录不存在: {forcing_dir}")
            valid = False
        
        return valid
