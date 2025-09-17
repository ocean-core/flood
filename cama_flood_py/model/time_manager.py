"""
时间管理模块
对应Fortran源码: cmf_ctrl_time_mod.F90 和相关时间函数
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
import calendar

from ..utils.constants import ModelConstants


class TimeManager:
    """
    时间管理器
    对应Fortran: CMF_TIME_* 系列函数
    """
    
    def __init__(self, start_time: datetime, end_time: datetime, 
                 timestep: float = ModelConstants.DEFAULT_DT):
        """
        初始化时间管理器
        
        Args:
            start_time: 模拟开始时间
            end_time: 模拟结束时间  
            timestep: 时间步长（秒）
        """
        self.start_time = start_time
        self.end_time = end_time
        self.timestep = timestep
        self.current_time = start_time
        
        # 计算总时间步数
        total_seconds = (end_time - start_time).total_seconds()
        self.total_steps = int(total_seconds / timestep)
        self.current_step = 0
        
        # 时间格式化相关
        self.reference_time = datetime(1900, 1, 1)  # 参考时间
    
    def update_time(self, dt: Optional[float] = None) -> None:
        """
        更新当前时间
        对应Fortran: CMF_TIME_UPDATE (cmf_drv_advance_mod.F90:142行)
        
        Args:
            dt: 时间步长，如果为None则使用默认timestep
        """
        if dt is None:
            dt = self.timestep
            
        self.current_time += timedelta(seconds=dt)
        self.current_step += 1
    
    def get_current_minutes(self) -> float:
        """
        获取当前时间对应的分钟数（从参考时间开始）
        对应Fortran时间表示方式
        
        Returns:
            float: 分钟数
        """
        delta = self.current_time - self.reference_time
        return delta.total_seconds() / 60.0
    
    def minutes_to_date(self, minutes: float) -> Tuple[int, int, int, int]:
        """
        将分钟数转换为日期格式
        对应Fortran: MIN2DATE (cmf_utils_mod.F90:268-309行)
        
        Args:
            minutes: 分钟数
            
        Returns:
            tuple: (年, 月, 日, 小时)
        """
        target_time = self.reference_time + timedelta(minutes=minutes)
        return (target_time.year, target_time.month, 
                target_time.day, target_time.hour)
    
    def date_to_minutes(self, year: int, month: int, day: int, hour: int = 0) -> float:
        """
        将日期转换为分钟数
        
        Args:
            year: 年
            month: 月
            day: 日
            hour: 小时
            
        Returns:
            float: 分钟数
        """
        target_time = datetime(year, month, day, hour)
        delta = target_time - self.reference_time
        return delta.total_seconds() / 60.0
    
    def split_date(self, date_int: int) -> Tuple[int, int, int]:
        """
        分解YYYYMMDD格式的日期
        对应Fortran: SPLITDATE (cmf_utils_mod.F90:375-384行)
        
        Args:
            date_int: YYYYMMDD格式的整数日期
            
        Returns:
            tuple: (年, 月, 日)
        """
        year = date_int // 10000
        month = (date_int % 10000) // 100
        day = date_int % 100
        return (year, month, day)
    
    def combine_date(self, year: int, month: int, day: int) -> int:
        """
        组合日期为YYYYMMDD格式
        
        Args:
            year: 年
            month: 月
            day: 日
            
        Returns:
            int: YYYYMMDD格式的整数
        """
        return year * 10000 + month * 100 + day
    
    def is_leap_year(self, year: int) -> bool:
        """
        判断是否为闰年
        
        Args:
            year: 年份
            
        Returns:
            bool: 是否为闰年
        """
        return calendar.isleap(year)
    
    def days_in_month(self, year: int, month: int) -> int:
        """
        获取指定月份的天数
        
        Args:
            year: 年
            month: 月
            
        Returns:
            int: 天数
        """
        return calendar.monthrange(year, month)[1]
    
    def get_julian_day(self, year: int, month: int, day: int) -> int:
        """
        获取儒略日
        
        Args:
            year: 年
            month: 月
            day: 日
            
        Returns:
            int: 儒略日（一年中的第几天）
        """
        target_date = datetime(year, month, day)
        start_of_year = datetime(year, 1, 1)
        return (target_date - start_of_year).days + 1
    
    def is_simulation_finished(self) -> bool:
        """
        检查模拟是否已完成
        
        Returns:
            bool: 模拟是否完成
        """
        return self.current_time >= self.end_time
    
    def get_progress(self) -> float:
        """
        获取模拟进度（0-1）
        
        Returns:
            float: 进度百分比
        """
        if self.total_steps == 0:
            return 1.0
        return min(self.current_step / self.total_steps, 1.0)
    
    def get_remaining_time(self) -> timedelta:
        """
        获取剩余模拟时间
        
        Returns:
            timedelta: 剩余时间
        """
        return max(self.end_time - self.current_time, timedelta(0))
    
    def get_elapsed_time(self) -> timedelta:
        """
        获取已经过的模拟时间
        
        Returns:
            timedelta: 已过时间
        """
        return self.current_time - self.start_time
    
    def format_current_time(self, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        格式化当前时间
        
        Args:
            format_str: 时间格式字符串
            
        Returns:
            str: 格式化的时间字符串
        """
        return self.current_time.strftime(format_str)
    
    def get_output_filename(self, prefix: str, suffix: str = "") -> str:
        """
        生成输出文件名（包含时间戳）
        
        Args:
            prefix: 文件名前缀
            suffix: 文件名后缀
            
        Returns:
            str: 文件名
        """
        time_str = self.current_time.strftime("%Y%m%d_%H%M")
        if suffix:
            return f"{prefix}_{time_str}.{suffix}"
        else:
            return f"{prefix}_{time_str}"
    
    def should_output(self, output_frequency_hours: int) -> bool:
        """
        判断是否应该输出结果
        
        Args:
            output_frequency_hours: 输出频率（小时）
            
        Returns:
            bool: 是否应该输出
        """
        if output_frequency_hours <= 0:
            return False
            
        # 计算从开始时间到现在的小时数
        elapsed_hours = (self.current_time - self.start_time).total_seconds() / 3600.0
        
        # 检查是否到达输出时间点
        return elapsed_hours % output_frequency_hours < (self.timestep / 3600.0)
    
    def reset(self) -> None:
        """
        重置时间管理器到初始状态
        """
        self.current_time = self.start_time
        self.current_step = 0
    
    def __str__(self) -> str:
        """
        字符串表示
        """
        return (f"TimeManager(current={self.format_current_time()}, "
                f"step={self.current_step}/{self.total_steps}, "
                f"progress={self.get_progress():.1%})")
    
    def __repr__(self) -> str:
        """
        详细字符串表示
        """
        return (f"TimeManager(start={self.start_time}, "
                f"end={self.end_time}, "
                f"timestep={self.timestep}, "
                f"current={self.current_time}, "
                f"step={self.current_step})")


class AdaptiveTimeManager(TimeManager):
    """
    自适应时间步长管理器
    对应Fortran中的自适应时间步长功能
    """
    
    def __init__(self, start_time: datetime, end_time: datetime,
                 initial_timestep: float = ModelConstants.DEFAULT_DT,
                 min_timestep: float = ModelConstants.MIN_DT,
                 max_timestep: float = ModelConstants.MAX_DT,
                 cfl_target: float = ModelConstants.DEFAULT_CFL):
        """
        初始化自适应时间管理器
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            initial_timestep: 初始时间步长
            min_timestep: 最小时间步长
            max_timestep: 最大时间步长
            cfl_target: 目标CFL数
        """
        super().__init__(start_time, end_time, initial_timestep)
        
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.cfl_target = cfl_target
        self.adaptive_timestep = initial_timestep
    
    def calculate_adaptive_timestep(self, max_velocity: float, 
                                  grid_spacing: float) -> float:
        """
        计算自适应时间步长
        基于CFL条件: dt = CFL * dx / v_max
        
        Args:
            max_velocity: 最大流速
            grid_spacing: 网格间距
            
        Returns:
            float: 自适应时间步长
        """
        if max_velocity <= 0:
            return self.max_timestep
        
        # CFL条件计算
        cfl_timestep = self.cfl_target * grid_spacing / max_velocity
        
        # 限制在允许范围内
        adaptive_dt = np.clip(cfl_timestep, self.min_timestep, self.max_timestep)
        
        self.adaptive_timestep = adaptive_dt
        return adaptive_dt
    
    def update_time_adaptive(self, max_velocity: float, 
                           grid_spacing: float) -> float:
        """
        使用自适应时间步长更新时间
        
        Args:
            max_velocity: 最大流速
            grid_spacing: 网格间距
            
        Returns:
            float: 实际使用的时间步长
        """
        dt = self.calculate_adaptive_timestep(max_velocity, grid_spacing)
        self.update_time(dt)
        return dt
