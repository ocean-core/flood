# Flood 函数调用关系图

## 主程序调用流程

```mermaid
graph TD
    A[MAIN_cmf.F90] --> B[CMF_DRV_INPUT]
    A --> C[CMF_DRV_INIT]
    A --> D[时间循环]
    D --> E[CMF_FORCING_GET]
    D --> F[CMF_FORCING_PUT]
    D --> G[CMF_DRV_ADVANCE]
    A --> H[CMF_DRV_END]

    %% 初始化模块
    C --> C1[CMF_TIME_INIT]
    C --> C2[CMF_RIVMAP_INIT]
    C --> C3[CMF_TOPO_INIT]
    C --> C4[CMF_OUTPUT_INIT]
    C --> C5[CMF_FORCING_INIT]
    C --> C6[CMF_PROG_INIT]
    C --> C7[CMF_DIAG_INIT]
    C --> C8[CMF_PHYSICS_FLDSTG]

    %% 时间步进模块
    G --> G1[CMF_TIME_NEXT]
    G --> G2[CMF_PHYSICS_ADVANCE]
    G --> G3[CMF_OUTPUT_WRITE]
    G --> G4[CMF_RESTART_WRITE]
    G --> G5[CMF_TIME_UPDATE]

    %% 物理计算核心
    G2 --> P1[CMF_PHYSICS_FLDSTG]
    G2 --> P2[CMF_CALC_OUTFLW]
    G2 --> P3[CMF_CALC_PTHOUT]
    G2 --> P4[CMF_CALC_INFLOW]
    G2 --> P5[CMF_CALC_STONXT]
    G2 --> P6[CMF_DIAG_AVEMAX_ADPSTP]

    %% 洪水阶段计算
    P1 --> F1[CMF_CALC_FLDSTG_DEF]
    P1 --> F2[CMF_LEVEE_FLDSTG]
    P1 --> F3[CMF_OPT_FLDSTG_ES]

    %% 流量计算
    P2 --> O1[计算水面高程]
    P2 --> O2[计算河道流量]
    P2 --> O3[计算漫滩流量]
    P2 --> O4[Manning公式计算]

    %% 存储计算
    P5 --> S1[更新河道存储]
    P5 --> S2[更新漫滩存储]
    P5 --> S3[水量平衡检查]
```

## 核心模块功能分析

### 1. 主控制模块 (cmf_drv_control_mod.F90)
**位置**: `/src/cmf_drv_control_mod.F90`
**功能**: 模型初始化和终止控制
**关键函数**:
- `CMF_DRV_INPUT()` - 读取配置文件和参数
- `CMF_DRV_INIT()` - 初始化所有模块
- `CMF_DRV_END()` - 清理和终止

### 2. 时间步进模块 (cmf_drv_advance_mod.F90)
**位置**: `/src/cmf_drv_advance_mod.F90`
**功能**: 时间循环和物理过程调度
**关键函数**:
- `CMF_DRV_ADVANCE(KSTEPS)` - 主时间循环
**参数**: `KSTEPS` - 时间步数

### 3. 物理过程控制 (cmf_ctrl_physics_mod.F90)
**位置**: `/src/cmf_ctrl_physics_mod.F90`
**功能**: 物理计算过程协调
**关键函数**:
- `CMF_PHYSICS_ADVANCE()` - 物理过程主循环
- `CMF_PHYSICS_FLDSTG()` - 洪水阶段计算选择器
**参数**: 无

### 4. 流量计算模块 (cmf_calc_outflw_mod.F90)
**位置**: `/src/cmf_calc_outflw_mod.F90`
**功能**: 河道和漫滩流量计算
**关键函数**:
- `CMF_CALC_OUTFLW()` - 主流量计算
- `CMF_CALC_INFLOW()` - 入流计算
**参数**: 使用全局变量

### 5. 洪水阶段计算 (cmf_calc_fldstg_mod.F90)
**位置**: `/src/cmf_calc_fldstg_mod.F90`
**功能**: 根据存储量计算水深和洪水范围
**关键函数**:
- `CMF_CALC_FLDSTG_DEF()` - 默认洪水阶段计算
- `CMF_OPT_FLDSTG_ES()` - 向量处理器优化版本
**参数**: 使用全局存储变量

### 6. 存储更新模块 (cmf_calc_stonxt_mod.F90)
**位置**: `/src/cmf_calc_stonxt_mod.F90`
**功能**: 下一时间步存储量计算
**关键函数**:
- `CMF_CALC_STONXT()` - 存储量更新
**参数**: 使用全局流量和存储变量

### 7. 强迫数据模块 (cmf_ctrl_forcing_mod.F90)
**位置**: `/src/cmf_ctrl_forcing_mod.F90`
**功能**: 径流强迫数据处理
**关键函数**:
- `CMF_FORCING_GET(ZBUFF)` - 读取强迫数据
- `CMF_FORCING_PUT(ZBUFF)` - 插值和分配强迫数据
**参数**: `ZBUFF(NXIN,NYIN,2)` - 数据缓冲区

### 8. 输出控制模块 (cmf_ctrl_output_mod.F90)
**位置**: `/src/cmf_ctrl_output_mod.F90`
**功能**: 结果输出管理
**关键函数**:
- `CMF_OUTPUT_WRITE()` - 写入输出文件
**参数**: 使用全局输出变量

## 数据流向图

```mermaid
graph LR
    A[径流输入数据] --> B[强迫数据模块]
    B --> C[插值到河网格点]
    C --> D[河道入流计算]
    D --> E[流量计算]
    E --> F[存储更新]
    F --> G[洪水阶段计算]
    G --> H[诊断变量计算]
    H --> I[输出模块]
    
    J[地形数据] --> K[河网初始化]
    K --> L[拓扑关系]
    L --> E
    
    M[参数文件] --> N[物理参数]
    N --> E
```

## 关键变量说明

### 全局数组变量 (定义在 yos_*.F90 文件中)
- `P2RIVSTO(NSEQMAX,1)` - 河道存储量 [m³]
- `P2FLDSTO(NSEQMAX,1)` - 漫滩存储量 [m³]
- `D2RIVOUT(NSEQMAX,1)` - 河道出流 [m³/s]
- `D2FLDOUT(NSEQMAX,1)` - 漫滩出流 [m³/s]
- `D2RIVDPH(NSEQMAX,1)` - 河道水深 [m]
- `D2FLDDPH(NSEQMAX,1)` - 漫滩水深 [m]
- `D2RIVINF(NSEQMAX,1)` - 河道入流 [m³/s]

### 地形参数
- `D2RIVELV(NSEQMAX,1)` - 河床高程 [m]
- `D2RIVWTH(NSEQMAX,1)` - 河道宽度 [m]
- `D2RIVHGT(NSEQMAX,1)` - 河道深度 [m]
- `D2RIVLEN(NSEQMAX,1)` - 河道长度 [m]
- `D2NXTDST(NSEQMAX,1)` - 到下游距离 [m]

### 物理参数
- `PMANRIV` - 河道Manning系数
- `PMANFLD` - 漫滩Manning系数
- `DT` - 时间步长 [s]
- `PGRV` - 重力加速度 [m/s²]
