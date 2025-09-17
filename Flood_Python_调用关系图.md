# Flood Python 调用关系图

## 整体架构图

```mermaid
graph TB
    CLI[命令行接口<br/>cli.py] --> Controller[主控制器<br/>controller.py]
    Controller --> TimeManager[时间管理<br/>time_manager.py]
    Controller --> ConfigManager[配置管理<br/>config.py]
    Controller --> PhysicsEngine[物理引擎<br/>physics.py]
    Controller --> HydraulicsCalculator[水力学计算<br/>hydraulics.py]
    Controller --> StorageCalculator[存储计算<br/>storage.py]
    Controller --> IOManager[I/O管理<br/>io_manager.py]
    Controller --> DiagnosticsCalculator[诊断计算<br/>diagnostics.py]
    
    IOManager --> ForcingDataManager[强迫数据<br/>forcing.py]
    IOManager --> MapDataManager[地图数据<br/>maps.py]
    IOManager --> DataInterpolator[数据插值<br/>interpolation.py]
    
    PhysicsEngine --> DataConverter[数据转换<br/>data_converter.py]
    HydraulicsCalculator --> DataConverter
    StorageCalculator --> DataConverter
    
    ConfigManager --> Constants[常量定义<br/>constants.py]
    PhysicsEngine --> Constants
    HydraulicsCalculator --> Constants
    StorageCalculator --> Constants
```

## 核心模拟流程图

```mermaid
sequenceDiagram
    participant CLI as 命令行接口
    participant Controller as 主控制器
    participant Config as 配置管理
    participant Time as 时间管理
    participant IO as I/O管理
    participant Physics as 物理引擎
    participant Hydraulics as 水力学计算
    participant Storage as 存储计算
    participant Diagnostics as 诊断计算

    CLI->>Controller: 创建模型实例
    Controller->>Config: 加载配置文件
    Controller->>IO: 初始化I/O管理器
    Controller->>Time: 初始化时间管理器
    Controller->>Physics: 初始化物理引擎
    Controller->>Hydraulics: 初始化水力学计算器
    Controller->>Storage: 初始化存储计算器
    Controller->>Diagnostics: 初始化诊断计算器
    
    loop 时间循环
        Controller->>Time: 更新当前时间
        Controller->>IO: 读取强迫数据
        Controller->>Physics: 计算洪水阶段
        Controller->>Hydraulics: 计算出流量
        Controller->>Hydraulics: 计算入流量
        Controller->>Storage: 更新存储量
        Controller->>Diagnostics: 计算诊断变量
        Controller->>IO: 写入输出数据
        
        alt 需要重启文件
            Controller->>IO: 写入重启文件
        end
        
        alt 模拟结束
            break 退出循环
        end
    end
    
    Controller->>IO: 关闭文件
    Controller->>Diagnostics: 输出性能报告
```

## 物理过程详细调用图

```mermaid
graph TD
    PhysicsAdvance[物理过程推进<br/>physics.advance] --> FloodStage[洪水阶段计算<br/>calculate_flood_stage]
    FloodStage --> FloodStageNumba[Numba加速计算<br/>calculate_flood_stage_numba]
    
    PhysicsAdvance --> OutflowCalc[出流计算<br/>hydraulics.calculate_outflow]
    OutflowCalc --> OutflowNumba[Numba加速计算<br/>calculate_outflow_numba]
    OutflowNumba --> ManningFlow[Manning公式<br/>calculate_manning_flow_simple]
    
    PhysicsAdvance --> InflowCalc[入流计算<br/>hydraulics.calculate_inflow]
    InflowCalc --> InflowNumba[Numba加速计算<br/>calculate_inflow_numba]
    
    PhysicsAdvance --> StorageUpdate[存储更新<br/>storage.update_storage]
    StorageUpdate --> StorageNumba[Numba加速计算<br/>update_storage_numba]
    StorageUpdate --> AdaptiveSubcycle[自适应子循环<br/>adaptive_subcycle_update]
    
    PhysicsAdvance --> DiagnosticsCalc[诊断计算<br/>diagnostics.calculate_flow_diagnostics]
    DiagnosticsCalc --> DiagnosticsNumba[Numba加速计算<br/>calculate_flow_diagnostics_numba]
```

## 数据流图

```mermaid
graph LR
    subgraph "输入数据"
        ConfigFile[配置文件<br/>YAML]
        MapData[地图数据<br/>二进制/NetCDF]
        ForcingData[强迫数据<br/>径流/降水/蒸发]
    end
    
    subgraph "核心计算"
        Physics[物理引擎]
        Hydraulics[水力学计算]
        Storage[存储计算]
    end
    
    subgraph "输出数据"
        OutputFiles[输出文件<br/>NetCDF]
        RestartFiles[重启文件<br/>二进制]
        LogFiles[日志文件<br/>文本]
        DiagnosticData[诊断数据<br/>统计信息]
    end
    
    ConfigFile --> Physics
    MapData --> Physics
    ForcingData --> Physics
    
    Physics --> Hydraulics
    Hydraulics --> Storage
    Storage --> Physics
    
    Physics --> OutputFiles
    Hydraulics --> OutputFiles
    Storage --> OutputFiles
    Storage --> RestartFiles
    Physics --> LogFiles
    Hydraulics --> DiagnosticData
```

## 类继承和组合关系图

```mermaid
classDiagram
    class CaMaFloodModel {
        +config_path: str
        +is_initialized: bool
        +current_step: int
        +initialize()
        +run()
        +advance_one_step()
        +finalize()
    }
    
    class PhysicsEngine {
        +nseq: int
        +river_storage: ndarray
        +flood_storage: ndarray
        +calculate_flood_stage()
        +advance()
        +get_state_dict()
    }
    
    class HydraulicsCalculator {
        +nseq: int
        +manning_river: float
        +manning_flood: float
        +calculate_outflow()
        +calculate_inflow()
    }
    
    class StorageCalculator {
        +nseq: int
        +update_storage()
        +adaptive_subcycle_update()
        +check_water_balance()
    }
    
    class TimeManager {
        +current_time: float
        +timestep: float
        +update_time()
        +is_simulation_finished()
    }
    
    class ConfigManager {
        +config: Config
        +load_config()
        +save_config()
        +validate_config()
    }
    
    class IOManager {
        +read_map_data()
        +read_forcing_data()
        +write_output()
        +write_restart()
    }
    
    CaMaFloodModel --> PhysicsEngine
    CaMaFloodModel --> HydraulicsCalculator
    CaMaFloodModel --> StorageCalculator
    CaMaFloodModel --> TimeManager
    CaMaFloodModel --> ConfigManager
    CaMaFloodModel --> IOManager
    
    IOManager --> ForcingDataManager
    IOManager --> MapDataManager
    IOManager --> DataInterpolator
```

## Numba加速函数调用图

```mermaid
graph TB
    subgraph "Python主函数"
        PyPhysics[PhysicsEngine.calculate_flood_stage]
        PyHydraulics[HydraulicsCalculator.calculate_outflow]
        PyStorage[StorageCalculator.update_storage]
        PyDiagnostics[DiagnosticsCalculator.calculate_flow_diagnostics]
    end
    
    subgraph "Numba JIT函数"
        NumbaFloodStage[calculate_flood_stage_numba<br/>@jit parallel=True]
        NumbaOutflow[calculate_outflow_numba<br/>@jit parallel=True]
        NumbaInflow[calculate_inflow_numba<br/>@jit parallel=True]
        NumbaStorage[update_storage_numba<br/>@jit parallel=True]
        NumbaDiagnostics[calculate_flow_diagnostics_numba<br/>@jit parallel=True]
        NumbaManning[calculate_manning_flow_simple<br/>@jit]
    end
    
    PyPhysics --> NumbaFloodStage
    PyHydraulics --> NumbaOutflow
    PyHydraulics --> NumbaInflow
    NumbaOutflow --> NumbaManning
    PyStorage --> NumbaStorage
    PyDiagnostics --> NumbaDiagnostics
    
    style NumbaFloodStage fill:#e1f5fe
    style NumbaOutflow fill:#e1f5fe
    style NumbaInflow fill:#e1f5fe
    style NumbaStorage fill:#e1f5fe
    style NumbaDiagnostics fill:#e1f5fe
    style NumbaManning fill:#e1f5fe
```

## 配置管理流程图

```mermaid
graph TD
    Start[开始] --> LoadConfig[加载配置文件]
    LoadConfig --> ValidateConfig[验证配置]
    ValidateConfig --> ConfigValid{配置有效?}
    
    ConfigValid -->|是| CreateObjects[创建配置对象]
    ConfigValid -->|否| UseDefaults[使用默认配置]
    
    CreateObjects --> ModelConfig[模型配置<br/>adaptive_timestep<br/>bifurcation_flow]
    CreateObjects --> PhysicsConfig[物理配置<br/>manning_river<br/>manning_flood]
    CreateObjects --> TimeConfig[时间配置<br/>start/end时间<br/>timestep]
    CreateObjects --> IOConfig[I/O配置<br/>输入输出路径<br/>文件格式]
    
    UseDefaults --> CreateDefaults[创建默认配置对象]
    CreateDefaults --> ModelConfig
    
    ModelConfig --> InitializeModel[初始化模型]
    PhysicsConfig --> InitializeModel
    TimeConfig --> InitializeModel
    IOConfig --> InitializeModel
    
    InitializeModel --> End[结束]
```

## 错误处理和日志流程图

```mermaid
graph TD
    Operation[执行操作] --> Success{成功?}
    
    Success -->|是| LogInfo[记录信息日志]
    Success -->|否| CatchException[捕获异常]
    
    CatchException --> LogError[记录错误日志]
    LogError --> ErrorType{错误类型}
    
    ErrorType -->|配置错误| ConfigError[配置错误处理<br/>使用默认值]
    ErrorType -->|文件错误| FileError[文件错误处理<br/>跳过或重试]
    ErrorType -->|计算错误| CalcError[计算错误处理<br/>调整参数]
    ErrorType -->|系统错误| SystemError[系统错误处理<br/>终止程序]
    
    ConfigError --> Continue[继续执行]
    FileError --> Continue
    CalcError --> Continue
    SystemError --> Terminate[终止程序]
    
    LogInfo --> Continue
    Continue --> NextOperation[下一个操作]
    NextOperation --> Operation
    
    Terminate --> End[结束]
```

## 测试架构图

```mermaid
graph TB
    subgraph "测试层次"
        UnitTests[单元测试<br/>test_physics.py<br/>test_hydraulics.py]
        IntegrationTests[集成测试<br/>test_integration.py]
        PerformanceTests[性能测试<br/>大规模网格测试]
    end
    
    subgraph "被测试模块"
        CoreModules[核心模块<br/>physics, hydraulics, storage]
        DataModules[数据模块<br/>forcing, maps, io_manager]
        UtilModules[工具模块<br/>config, time_manager, diagnostics]
    end
    
    UnitTests --> CoreModules
    UnitTests --> DataModules
    UnitTests --> UtilModules
    
    IntegrationTests --> CoreModules
    IntegrationTests --> DataModules
    
    PerformanceTests --> CoreModules
    
    subgraph "测试工具"
        Pytest[pytest框架]
        NumbaTest[Numba函数测试]
        MockData[模拟数据生成]
    end
    
    UnitTests --> Pytest
    IntegrationTests --> Pytest
    PerformanceTests --> Pytest
    
    UnitTests --> NumbaTest
    IntegrationTests --> MockData
```
