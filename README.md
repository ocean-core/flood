# Flood Python重构版本

Flood洪水模拟模型的Python重构实现，对应Fortran原版v4.2.0。

## 项目结构

```
cama_flood_py/
├── core/                    # 核心计算模块
│   ├── physics.py          # 物理过程计算 (对应CMF_PHYSICS_*)
│   ├── hydraulics.py       # 水力学计算 (对应CMF_CALC_OUTFLW)
│   ├── storage.py          # 存储量计算 (对应CMF_CALC_STONXT)
│   └── diagnostics.py      # 诊断变量计算
├── data/                    # 数据处理模块
│   ├── io_manager.py       # 输入输出管理
│   ├── forcing.py          # 强迫数据处理
│   ├── maps.py             # 地图数据处理
│   └── interpolation.py    # 插值算法
├── model/                   # 模型控制模块
│   ├── controller.py       # 主控制器 (对应MAIN_cmf)
│   ├── time_manager.py     # 时间管理 (对应CMF_TIME_*)
│   └── config.py           # 配置管理
├── utils/                   # 工具模块
│   ├── data_converter.py   # 数据转换 (对应vecD2mapR等)
│   └── constants.py        # 物理常数
├── visualization/           # 可视化模块
│   ├── plotter.py          # 静态图表
│   └── animator.py         # 动画生成
└── tests/                   # 测试模块
    ├── test_physics.py     # 物理过程测试
    ├── test_hydraulics.py  # 水力学测试
    └── test_integration.py # 集成测试
```

## 安装

```bash
# 使用 uv, 自动安装依赖和创建虚拟环境. 替代 pip install xx
uv sync

# 开发安装
pip install -e .

# 包含可选依赖
pip install -e ".[dev,viz,hpc]"
```

## 快速开始

```python
from cama_flood_py import CaMaFloodModel

# 创建模型实例
model = CaMaFloodModel("config.yaml")

# 初始化模型
model.initialize()

# 运行模拟
model.run()

# 获取结果
results = model.get_results()
```

## 配置文件

使用YAML格式配置文件，替代Fortran的namelist：

```yaml
model:
  adaptive_timestep: true
  bifurcation_flow: true

physics:
  manning_river: 0.03
  manning_flood: 0.10

time:
  start_year: 2000
  start_month: 1
  start_day: 1
  end_year: 2001
  end_month: 1
  end_day: 1
```

## 与Fortran版本对应关系

| Fortran函数           | Python函数                                 | 功能描述     |
| --------------------- | ------------------------------------------ | ------------ |
| `CMF_DRV_INIT`        | `CaMaFloodModel.initialize()`              | 模型初始化   |
| `CMF_DRV_ADVANCE`     | `CaMaFloodModel.advance()`                 | 时间步进     |
| `CMF_PHYSICS_ADVANCE` | `PhysicsEngine.advance()`                  | 物理过程推进 |
| `CMF_CALC_FLDSTG_DEF` | `PhysicsEngine.calculate_flood_stage()`    | 洪水阶段计算 |
| `CMF_CALC_OUTFLW`     | `HydraulicsCalculator.calculate_outflow()` | 流量计算     |

## 性能优化

- 使用Numba JIT编译加速核心计算循环
- NumPy数组优化内存布局
- 支持多核并行计算
- 可选GPU加速（CUDA）

## 开发状态

- ✅ 项目结构设计
- ✅ 核心物理模块 (physics, hydraulics, storage)
- ✅ 配置管理系统 (YAML配置)
- ✅ 时间管理模块 (自适应时间步长)
- ✅ 数据I/O模块 (NetCDF/二进制支持)
- ✅ 强迫数据和地图数据管理
- ✅ 单元测试开发 (26个测试全部通过)
- ✅ 性能优化 (Numba JIT加速)
- ✅ 命令行接口和Python API
- ✅ 诊断和分析工具

## 安装和使用

### 安装依赖

```bash
# 使用 uv, 自动安装依赖和创建虚拟环境
uv sync

# 如果用了uv, 下面手动安装依赖可不执行
# 安装Python依赖
pip install numpy numba xarray netcdf4 pyyaml pytest

# 可选：安装可视化依赖
pip install matplotlib cartopy
```

### 快速开始

#### 1. 创建配置文件

```bash
# 使用命令行工具创建默认配置
python -m cama_flood_py.cli create-config my_config.yaml

# 或者直接使用Python
python -c "from cama_flood_py.model.config import create_default_config; create_default_config('my_config.yaml')"
```

#### 2. 运行模拟

**命令行方式:**
```bash
# 如果使用uv, python  => uv run
# 运行模拟
python -m cama_flood_py.cli run my_config.yaml

# 详细输出
python -m cama_flood_py.cli run my_config.yaml --verbose
```

**Python API方式:**
```python
from cama_flood_py import CaMaFloodModel

# 使用上下文管理器（推荐）
with CaMaFloodModel("my_config.yaml") as model:
    model.initialize()
    model.run()

# 或者手动管理
model = CaMaFloodModel("my_config.yaml")
try:
    model.initialize()
    model.run()
finally:
    model.finalize()
```

#### 3. 配置文件示例

```yaml
# 模型设置
model:
  adaptive_timestep: true
  bifurcation_flow: true
  
# 物理参数
physics:
  manning_river: 0.03
  manning_flood: 0.10
  
# 时间设置
time:
  start_year: 2000
  start_month: 1
  start_day: 1
  end_year: 2000
  end_month: 1
  end_day: 2
  timestep: 3600.0  # 秒
  
# 输入输出
input:
  map_directory: "./map"
  forcing_directory: "./forcing"
  
output:
  directory: "./output"
  frequency: 24  # 小时
  variables: ["discharge", "depth", "storage"]
```

### 高级用法

#### 1. 自定义物理过程

```python
from cama_flood_py.core.physics import PhysicsEngine

# 创建自定义物理引擎
physics = PhysicsEngine(nseq=1000)
physics.set_flood_stage_method('advanced')  # 使用高级方法
```

#### 2. 性能监控

```python
from cama_flood_py.core.diagnostics import PerformanceProfiler

profiler = PerformanceProfiler()
with CaMaFloodModel("config.yaml") as model:
    model.set_profiler(profiler)
    model.run()
    
# 查看性能报告
profiler.log_performance_report()
```

#### 3. 批量处理

```python
import glob
from cama_flood_py import CaMaFloodModel

# 批量处理多个配置文件
for config_file in glob.glob("configs/*.yaml"):
    print(f"运行 {config_file}")
    with CaMaFloodModel(config_file) as model:
        model.initialize()
        model.run()
```

### 测试

```bash
# 运行所有测试
python -m pytest cama_flood_py/tests/ -v

# 运行特定测试
python -m pytest cama_flood_py/tests/test_physics.py -v

# 运行性能测试
python -m pytest cama_flood_py/tests/test_integration.py::TestPerformance -v
```

### 故障排除

**常见问题:**

1. **内存不足**: 对于大规模模拟，确保有足够的RAM
2. **Numba编译慢**: 首次运行时Numba需要编译，后续运行会更快
3. **文件路径错误**: 确保输入数据文件路径正确
4. **依赖缺失**: 使用`pip install -r requirements.txt`安装所有依赖

**调试模式:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 然后运行模拟，会输出详细日志
