"""
物理常数和模型常数定义
对应Fortran源码中的参数定义
"""

import numpy as np


class PhysicalConstants:
    """
    物理常数
    对应Fortran: yos_cmf_param.F90中的物理参数
    """
    
    # 重力加速度 [m/s²]
    GRAVITY = 9.8
    
    # 水的密度 [kg/m³]
    WATER_DENSITY = 1000.0
    
    # 地球半径 [m]
    EARTH_RADIUS = 6.371e6
    
    # 度到弧度转换
    DEG_TO_RAD = np.pi / 180.0
    
    # 弧度到度转换
    RAD_TO_DEG = 180.0 / np.pi


class ModelConstants:
    """
    模型常数
    对应Fortran: yos_cmf_param.F90中的模型参数
    """
    
    # 数值精度相关
    EPSILON = 1.0e-10          # 数值计算精度
    MISSING_VALUE = -9999.0    # 缺失值标识
    
    # 默认Manning粗糙度系数
    DEFAULT_MANNING_RIVER = 0.03   # 河道Manning系数
    DEFAULT_MANNING_FLOOD = 0.10   # 漫滩Manning系数
    
    # 时间步长相关
    DEFAULT_DT = 3600.0           # 默认时间步长 [秒]
    MIN_DT = 60.0                 # 最小时间步长 [秒]
    MAX_DT = 86400.0              # 最大时间步长 [秒]
    
    # CFL条件相关
    DEFAULT_CFL = 0.7             # 默认CFL系数
    MAX_CFL = 0.9                 # 最大CFL系数
    
    # 收敛判据
    CONVERGENCE_TOLERANCE = 1.0e-6
    MAX_ITERATIONS = 100
    
    # 文件格式相关
    BINARY_RECORD_LENGTH = 4      # Fortran二进制记录长度标识
    
    # 单位转换
    SECONDS_PER_DAY = 86400.0
    MINUTES_PER_DAY = 1440.0
    HOURS_PER_DAY = 24.0


class GridConstants:
    """
    网格相关常数
    对应Fortran: yos_cmf_map.F90中的网格参数
    """
    
    # 默认网格分辨率
    DEFAULT_RESOLUTION = 0.25     # 度
    
    # 网格维度限制
    MAX_NX = 1440                 # 最大经度网格数
    MAX_NY = 720                  # 最大纬度网格数
    MAX_NSEQ = MAX_NX * MAX_NY    # 最大序列网格数
    
    # 地理范围
    MIN_LON = -180.0
    MAX_LON = 180.0
    MIN_LAT = -90.0
    MAX_LAT = 90.0


class FileConstants:
    """
    文件相关常数
    对应Fortran中的文件处理参数
    """
    
    # 文件扩展名
    BINARY_EXT = ".bin"
    NETCDF_EXT = ".nc"
    TEXT_EXT = ".txt"
    
    # 默认文件名
    DEFAULT_CONFIG = "config.yaml"
    DEFAULT_RESTART = "restart.nc"
    DEFAULT_OUTPUT = "output.nc"
    
    # 文件读写模式
    READ_MODE = "rb"
    WRITE_MODE = "wb"
    APPEND_MODE = "ab"
