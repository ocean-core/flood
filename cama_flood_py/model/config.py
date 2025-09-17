"""
配置管理模块
对应Fortran中的namelist配置系统
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.constants import ModelConstants, PhysicalConstants


@dataclass
class ModelConfig:
    """
    模型运行配置
    对应Fortran: &NRUNVER namelist
    """
    adaptive_timestep: bool = True
    bifurcation_flow: bool = True
    dam_operation: bool = False
    restart_mode: bool = False
    sediment_transport: bool = False


@dataclass
class PhysicsConfig:
    """
    物理参数配置
    对应Fortran: &NPARAM namelist
    """
    manning_river: float = ModelConstants.DEFAULT_MANNING_RIVER
    manning_flood: float = ModelConstants.DEFAULT_MANNING_FLOOD
    downstream_distance: float = 10000.0
    cfl_coefficient: float = ModelConstants.DEFAULT_CFL
    gravity: float = PhysicalConstants.GRAVITY


@dataclass
class TimeConfig:
    """
    时间配置
    对应Fortran: &NSIMTIME namelist
    """
    start_year: int = 2000
    start_month: int = 1
    start_day: int = 1
    start_hour: int = 0
    end_year: int = 2001
    end_month: int = 1
    end_day: int = 1
    end_hour: int = 0
    timestep: float = ModelConstants.DEFAULT_DT


@dataclass
class InputConfig:
    """
    输入文件配置
    对应Fortran: &NMAP 和 &NFORCE namelist
    """
    # 地图数据目录和文件
    map_directory: str = "./map/glb_15min/"
    next_xy: str = "nextxy.bin"
    catchment_area: str = "ctmare.bin"
    river_elevation: str = "elevtn.bin"
    river_width: str = "rivwth_gwdlr.bin"
    river_height: str = "rivhgt.bin"
    river_length: str = "rivlen.bin"
    next_distance: str = "nxtdst.bin"
    
    # 强迫数据配置
    forcing_directory: str = "./input/ELSE_GPCC/"
    forcing_prefix: str = "Roff__ELSE_GPCC"
    forcing_suffix: str = ".one"
    time_interpolation: bool = True


@dataclass
class OutputConfig:
    """
    输出配置
    对应Fortran: &NOUTPUT namelist
    """
    directory: str = "./out"
    variables: list = field(default_factory=lambda: ["rivout", "rivsto", "rivdph"])
    tag: str = "2000"
    frequency: int = 24  # 小时
    format: str = "netcdf"  # netcdf 或 binary
    compression: bool = True


@dataclass
class CaMaFloodConfig:
    """
    完整的CaMa-Flood配置
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class ConfigManager:
    """
    配置管理器
    负责读取、验证和管理配置文件
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = CaMaFloodConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            # 更新配置对象
            if 'model' in config_dict:
                self.config.model = ModelConfig(**config_dict['model'])
            
            if 'physics' in config_dict:
                self.config.physics = PhysicsConfig(**config_dict['physics'])
            
            if 'time' in config_dict:
                self.config.time = TimeConfig(**config_dict['time'])
            
            if 'input' in config_dict:
                # 处理输入配置中的列表字段
                input_config = config_dict['input'].copy()
                if 'variables' in input_config and isinstance(input_config['variables'], list):
                    pass  # 保持列表格式
                self.config.input = InputConfig(**input_config)
            
            if 'output' in config_dict:
                # 处理输出配置中的列表字段
                output_config = config_dict['output'].copy()
                if 'variables' in output_config and isinstance(output_config['variables'], list):
                    pass  # 保持列表格式
                self.config.output = OutputConfig(**output_config)
                
            self.config_path = config_path
            
        except Exception as e:
            raise ValueError(f"配置文件加载失败: {e}")
    
    def save_config(self, config_path: str) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config_path: 配置文件路径
        """
        config_dict = {
            'model': {
                'adaptive_timestep': self.config.model.adaptive_timestep,
                'bifurcation_flow': self.config.model.bifurcation_flow,
                'dam_operation': self.config.model.dam_operation,
                'restart_mode': self.config.model.restart_mode,
                'sediment_transport': self.config.model.sediment_transport
            },
            'physics': {
                'manning_river': self.config.physics.manning_river,
                'manning_flood': self.config.physics.manning_flood,
                'downstream_distance': self.config.physics.downstream_distance,
                'cfl_coefficient': self.config.physics.cfl_coefficient,
                'gravity': self.config.physics.gravity
            },
            'time': {
                'start_year': self.config.time.start_year,
                'start_month': self.config.time.start_month,
                'start_day': self.config.time.start_day,
                'start_hour': self.config.time.start_hour,
                'end_year': self.config.time.end_year,
                'end_month': self.config.time.end_month,
                'end_day': self.config.time.end_day,
                'end_hour': self.config.time.end_hour,
                'timestep': self.config.time.timestep
            },
            'input': {
                'map_directory': self.config.input.map_directory,
                'next_xy': self.config.input.next_xy,
                'catchment_area': self.config.input.catchment_area,
                'river_elevation': self.config.input.river_elevation,
                'river_width': self.config.input.river_width,
                'river_height': self.config.input.river_height,
                'river_length': self.config.input.river_length,
                'next_distance': self.config.input.next_distance,
                'forcing_directory': self.config.input.forcing_directory,
                'forcing_prefix': self.config.input.forcing_prefix,
                'forcing_suffix': self.config.input.forcing_suffix,
                'time_interpolation': self.config.input.time_interpolation
            },
            'output': {
                'directory': self.config.output.directory,
                'variables': self.config.output.variables,
                'tag': self.config.output.tag,
                'frequency': self.config.output.frequency,
                'format': self.config.output.format,
                'compression': self.config.output.compression
            }
        }
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            raise ValueError(f"配置文件保存失败: {e}")
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            bool: 配置是否有效
        """
        try:
            # 验证时间配置
            start_date = datetime(
                self.config.time.start_year,
                self.config.time.start_month,
                self.config.time.start_day,
                self.config.time.start_hour
            )
            end_date = datetime(
                self.config.time.end_year,
                self.config.time.end_month,
                self.config.time.end_day,
                self.config.time.end_hour
            )
            
            if start_date >= end_date:
                raise ValueError("开始时间必须早于结束时间")
            
            # 验证物理参数
            if self.config.physics.manning_river <= 0:
                raise ValueError("河道Manning系数必须大于0")
            
            if self.config.physics.manning_flood <= 0:
                raise ValueError("漫滩Manning系数必须大于0")
            
            if not (0 < self.config.physics.cfl_coefficient <= 1):
                raise ValueError("CFL系数必须在(0,1]范围内")
            
            # 验证时间步长
            if self.config.time.timestep <= 0:
                raise ValueError("时间步长必须大于0")
            
            # 验证输入文件路径
            map_dir = Path(self.config.input.map_directory)
            if not map_dir.exists():
                print(f"警告: 地图数据目录不存在: {map_dir}")
            
            forcing_dir = Path(self.config.input.forcing_directory)
            if not forcing_dir.exists():
                print(f"警告: 强迫数据目录不存在: {forcing_dir}")
            
            return True
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False
    
    def get_start_datetime(self) -> datetime:
        """获取模拟开始时间"""
        return datetime(
            self.config.time.start_year,
            self.config.time.start_month,
            self.config.time.start_day,
            self.config.time.start_hour
        )
    
    def get_end_datetime(self) -> datetime:
        """获取模拟结束时间"""
        return datetime(
            self.config.time.end_year,
            self.config.time.end_month,
            self.config.time.end_day,
            self.config.time.end_hour
        )
    
    def get_simulation_duration(self) -> float:
        """
        获取模拟时长（秒）
        
        Returns:
            float: 模拟时长（秒）
        """
        start = self.get_start_datetime()
        end = self.get_end_datetime()
        return (end - start).total_seconds()


def create_default_config(output_path: str) -> None:
    """
    创建默认配置文件
    
    Args:
        output_path: 输出配置文件路径
    """
    config_manager = ConfigManager()
    config_manager.save_config(output_path)
    print(f"默认配置文件已创建: {output_path}")
