"""
CaMa-Flood Python重构版本
全球洪水模拟模型的Python实现

对应Fortran原版: CaMa-Flood v4.2.0
"""

__version__ = "4.2.0"
__author__ = "CaMa-Flood Team"
__description__ = "CaMa-Flood洪水模拟模型Python重构版本"

from .model.controller import CaMaFloodModel
from .core.physics import PhysicsEngine
from .data.io_manager import IOManager
from .model.time_manager import TimeManager

__all__ = [
    "CaMaFloodModel",
    "PhysicsEngine", 
    "IOManager",
    "TimeManager"
]
