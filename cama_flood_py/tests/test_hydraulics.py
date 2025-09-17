"""
水力学计算模块单元测试
"""

import pytest
import numpy as np
from cama_flood_py.core.hydraulics import (
    HydraulicsCalculator, 
    calculate_outflow_numba,
    calculate_inflow_numba,
    calculate_manning_flow_simple
)


class TestHydraulicsCalculator:
    """水力学计算器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.nseq = 50
        self.hydraulics = HydraulicsCalculator(
            self.nseq, 
            manning_river=0.03,
            manning_flood=0.10
        )
        
        # 设置测试数据
        self.river_depth = np.full(self.nseq, 2.0)      # 2m水深
        self.flood_depth = np.zeros(self.nseq)          # 无漫滩
        self.surface_elevation = np.linspace(100, 90, self.nseq)  # 下游坡度
        self.river_width = np.full(self.nseq, 50.0)     # 50m宽度
        self.river_length = np.full(self.nseq, 1000.0)  # 1000m长度
        self.next_distance = np.full(self.nseq, 1000.0) # 1000m间距
        self.next_index = np.arange(1, self.nseq + 1)   # 下游索引
        self.next_index[-1] = 0  # 最后一个网格为出口
    
    def test_manning_flow_calculation(self):
        """测试Manning公式流量计算"""
        depth = 2.0
        width = 50.0
        slope = 0.001  # 0.1%坡度
        manning_n = 0.03
        
        discharge = calculate_manning_flow_simple(depth, width, slope, manning_n)
        
        # 验证结果合理性
        assert discharge > 0
        
        # 手工计算验证
        area = width * depth  # 100 m²
        perimeter = width + 2 * depth  # 54 m
        hydraulic_radius = area / perimeter  # 1.85 m
        expected_velocity = (hydraulic_radius**(2/3) * np.sqrt(slope)) / manning_n
        expected_discharge = expected_velocity * area
        
        np.testing.assert_allclose(discharge, expected_discharge, rtol=1e-10)
    
    def test_outflow_calculation(self):
        """测试出流计算"""
        river_outflow, flood_outflow = self.hydraulics.calculate_outflow(
            self.river_depth,
            self.flood_depth,
            self.surface_elevation,
            self.river_width,
            self.river_length,
            self.next_distance,
            self.next_index
        )
        
        # 验证结果
        assert len(river_outflow) == self.nseq
        assert len(flood_outflow) == self.nseq
        assert np.all(river_outflow >= 0)  # 流量非负
        assert np.all(flood_outflow >= 0)
        
        # 验证出口网格流量为0
        assert river_outflow[-1] == 0.0
        assert flood_outflow[-1] == 0.0
        
        # 验证有坡度的网格有流量
        assert np.any(river_outflow[:-1] > 0)
    
    def test_inflow_calculation(self):
        """测试入流计算"""
        # 先计算出流
        river_outflow, _ = self.hydraulics.calculate_outflow(
            self.river_depth,
            self.flood_depth,
            self.surface_elevation,
            self.river_width,
            self.river_length,
            self.next_distance,
            self.next_index
        )
        
        # 设置径流强迫
        runoff_forcing = np.full(self.nseq, 1.0)  # 1 m³/s径流
        
        # 计算入流
        river_inflow = self.hydraulics.calculate_inflow(
            river_outflow, self.next_index, runoff_forcing
        )
        
        # 验证结果
        assert len(river_inflow) == self.nseq
        assert np.all(river_inflow >= 0)
        
        # 验证第一个网格只有径流强迫
        assert river_inflow[0] == 1.0
        
        # 验证下游网格包含上游流量
        assert river_inflow[1] > 1.0  # 应该包含上游出流
    
    def test_zero_depth_handling(self):
        """测试零水深处理"""
        # 设置零水深
        zero_depth = np.zeros(self.nseq)
        
        river_outflow, flood_outflow = self.hydraulics.calculate_outflow(
            zero_depth,
            zero_depth,
            self.surface_elevation,
            self.river_width,
            self.river_length,
            self.next_distance,
            self.next_index
        )
        
        # 验证零水深时流量为零
        assert np.all(river_outflow == 0.0)
        assert np.all(flood_outflow == 0.0)
    
    def test_negative_slope_handling(self):
        """测试负坡度处理"""
        # 设置逆向坡度（上游低于下游）
        reverse_elevation = np.linspace(90, 100, self.nseq)
        
        river_outflow, flood_outflow = self.hydraulics.calculate_outflow(
            self.river_depth,
            self.flood_depth,
            reverse_elevation,
            self.river_width,
            self.river_length,
            self.next_distance,
            self.next_index
        )
        
        # 验证负坡度时流量为零（简化模型）
        assert np.all(river_outflow == 0.0)
        assert np.all(flood_outflow == 0.0)


class TestNumbaFunctions:
    """Numba函数专项测试"""
    
    def test_outflow_numba_consistency(self):
        """测试Numba函数一致性"""
        nseq = 10
        
        # 创建测试数据
        river_depth = np.full(nseq, 2.0)
        flood_depth = np.zeros(nseq)
        surface_elevation = np.linspace(100, 90, nseq)
        river_width = np.full(nseq, 50.0)
        river_length = np.full(nseq, 1000.0)
        next_distance = np.full(nseq, 1000.0)
        next_index = np.arange(1, nseq + 1)
        next_index[-1] = 0
        
        # 调用Numba函数
        river_out, flood_out, velocity = calculate_outflow_numba(
            river_depth, flood_depth, surface_elevation,
            river_width, river_length, next_distance, next_index,
            0.03, 0.10
        )
        
        # 验证结果形状和类型
        assert river_out.shape == (nseq,)
        assert flood_out.shape == (nseq,)
        assert velocity.shape == (nseq,)
        assert river_out.dtype == np.float64
        
        # 验证物理合理性
        assert np.all(river_out >= 0)
        assert np.all(flood_out >= 0)
        assert np.all(velocity >= 0)
    
    def test_inflow_numba_mass_conservation(self):
        """测试入流Numba函数的质量守恒"""
        nseq = 5
        
        # 创建简单的线性网络
        river_outflow = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        next_index = np.array([1, 2, 3, 4, 0])  # 线性连接
        runoff_forcing = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        river_inflow = calculate_inflow_numba(
            river_outflow, next_index, runoff_forcing
        )
        
        # 验证质量守恒
        # 第一个网格：只有径流
        assert river_inflow[0] == 1.0
        
        # 第二个网格：径流 + 上游出流
        assert river_inflow[1] == 1.0 + 10.0
        
        # 验证总入流 = 总径流 + 总出流（除出口）
        total_inflow = np.sum(river_inflow)
        total_runoff = np.sum(runoff_forcing)
        total_outflow_internal = np.sum(river_outflow[:-1])  # 除出口外
        
        expected_total = total_runoff + total_outflow_internal
        np.testing.assert_allclose(total_inflow, expected_total)


class TestEdgeCases:
    """边界情况测试"""
    
    def test_single_grid(self):
        """测试单网格情况"""
        hydraulics = HydraulicsCalculator(1, 0.03, 0.10)
        
        river_depth = np.array([2.0])
        flood_depth = np.array([0.0])
        surface_elevation = np.array([100.0])
        river_width = np.array([50.0])
        river_length = np.array([1000.0])
        next_distance = np.array([1000.0])
        next_index = np.array([0])  # 出口
        
        river_out, flood_out = hydraulics.calculate_outflow(
            river_depth, flood_depth, surface_elevation,
            river_width, river_length, next_distance, next_index
        )
        
        # 单网格且为出口，流量应为0
        assert river_out[0] == 0.0
        assert flood_out[0] == 0.0
    
    def test_invalid_indices(self):
        """测试无效索引处理"""
        nseq = 5
        
        river_depth = np.full(nseq, 2.0)
        flood_depth = np.zeros(nseq)
        surface_elevation = np.linspace(100, 90, nseq)
        river_width = np.full(nseq, 50.0)
        river_length = np.full(nseq, 1000.0)
        next_distance = np.full(nseq, 1000.0)
        next_index = np.array([-1, 10, 2, 3, 0])  # 包含无效索引
        
        river_out, flood_out, velocity = calculate_outflow_numba(
            river_depth, flood_depth, surface_elevation,
            river_width, river_length, next_distance, next_index,
            0.03, 0.10
        )
        
        # 无效索引的网格流量应为0
        assert river_out[0] == 0.0  # next_index = -1
        assert river_out[1] == 0.0  # next_index = 10 (超出范围)
    
    def test_extreme_values(self):
        """测试极值情况"""
        # 极大水深
        large_depth = 1000.0
        discharge = calculate_manning_flow_simple(
            large_depth, 50.0, 0.001, 0.03
        )
        assert np.isfinite(discharge)
        assert discharge > 0
        
        # 极小坡度
        small_slope = 1e-10
        discharge = calculate_manning_flow_simple(
            2.0, 50.0, small_slope, 0.03
        )
        assert discharge >= 0
        
        # 极大Manning系数
        large_manning = 10.0
        discharge = calculate_manning_flow_simple(
            2.0, 50.0, 0.001, large_manning
        )
        assert discharge >= 0
        assert discharge < calculate_manning_flow_simple(2.0, 50.0, 0.001, 0.03)


if __name__ == "__main__":
    pytest.main([__file__])
