# 🧬 neutron-diffusion-solver

> 反应堆物理数值计算全栈项目 —— 从一维单群扩散到三维双群 + 点堆动力学 + 燃耗耦合

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-✓-013243.svg?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-✓-8CAAE6.svg?logo=scipy)](https://scipy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-✓-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-✓-11557c.svg)](https://matplotlib.org/)
[![Tests](https://github.com/buzhidao2006/neutron-diffusion-solver/actions/workflows/test.yml/badge.svg)](https://github.com/buzhidao2006/neutron-diffusion-solver/actions)

---

## 项目概览

这是一个从零开始构建的反应堆物理数值计算教学项目。一个月时间（2026.06.15 → 07.14），16 次提交，从 44 行的一维单群求解器，演进为覆盖**临界计算、燃耗分析、点堆动力学、三维双群扩散**的 4300+ 行代码，配套解析基准、单元测试与 GitHub Actions 持续集成。

**核心能力**：有限差分法 · 幂迭代 · Chebyshev 外推加速 · 双群/三维扩散 · 点堆动力学 · 临界搜索 · Bateman 燃耗 · Kronecker 积稀疏矩阵 · 解析验证

---

## 演进时间线

```
Day 1  ● 一维单群扩散求解器
Day 2  ● 双群扩散 + k 更新 bug 修复 + debug 方法论
       ● 临界尺寸扫描 + 临界硼搜索
Day 3  ● Streamlit 交互式可视化 + 求解器模块化重构
Day 4  ● Bateman 燃耗方程 + 燃耗-扩散耦合
Day 5  ● 二维双群 + Kronecker 积稀疏矩阵 + 解析验证
Day 6  ● 阶段性学习总结
Day 7  ● 幂迭代 + Chebyshev 外推加速
Wk 4   ● 点堆动力学求解器（6 群缓发中子）
       ● 47 个 pytest 单元测试
       ● Jupyter 教学版 notebook
       ● 三维双群扩散（Kronecker 积扩展到 3D）
       ● 统一 CLI 入口 main.py
       ● GitHub Actions 持续集成
```

| 日期 | 提交 | 内容 | 核心文件 |
|------|------|------|----------|
| 06-15 | `0deb49a` | 🚀 一维单群扩散，有限差分法，通量可视化 | `diffusion_1d.py` |
| 06-16 | `18871e0` | 🔬 双群（快/热）扩散，k 更新 bug 修复，debug 方法论 | `diffusion_2g.py`, `debug_demo.py`, `NOTES.md` |
| 06-16 | `41865bf` | ⚛️ 临界尺寸扫描 + 临界硼浓度搜索（二分法+嵌套迭代） | `critical_size.py`, `boron_search.py` |
| 06-17 | `baa38fc` | 🎨 Streamlit Web 应用 + solver 模块化重构 | `app.py`, `solver.py` |
| 06-17 | `2ffbeee` | ☢️ Bateman 衰变链 + 燃耗-扩散耦合求解器 | `bateman.py`, `burnup_solver.py` |
| 06-18 | `feab2cc` | 🌐 二维双群，Kronecker 积稀疏矩阵，解析特征值验证 | `diffusion_2d.py`, `solver_2d.py`, `validate_2d.py` |
| 06-22 | `8cfac52` | 📝 阶段性学习总结（1D→2D 物理与数值边界） | `2026-06-18-总结.md` |
| 06-22 | `8109540` | ⚡ 幂迭代 + Chebyshev 3-term 外推加速（3× 收敛加速） | `power_iteration.py`, `demo_chebyshev.py` |
| 07-14 | `efa1ef3` | 🧨 点堆动力学（6 群缓发中子）+ 交互式演示 + Streamlit 集成 | `point_kinetics.py`, `demo_kinetics.py` |
| 07-14 | `4c4c6d7` | ✅ 42 个 pytest 单元测试（扩散 + 点堆） | `tests/` |
| 07-14 | `ecffc31` | 📓 Jupyter 教学版 —— 从中子扩散到点堆动力学全链路 | `neutron_diffusion_tutorial.ipynb` |
| 07-14 | `40f5826` | 🧊 三维双群扩散 —— Kronecker 积扩展到 3D | `diffusion_3d.py`, `solver_3d.py` |
| 07-14 | `e8f231c` | 🎛️ 统一 CLI 入口（9 个子命令） | `main.py` |
| 07-14 | `3333604` | 🤖 GitHub Actions —— push/PR 自动跑 pytest | `.github/workflows/test.yml` |

---

## 项目结构

```
neutron-diffusion-solver/
├── main.py                  # 统一 CLI 入口（9 个子命令）
├── app.py                   # Streamlit Web 应用（交互式可视化）
│
├── solver.py                # 一维求解器核心模块（临界扫描 + 硼搜索）
├── solver_2d.py             # 二维求解器（Kronecker 积 + 稀疏矩阵）
├── solver_3d.py             # 三维求解器（Kronecker 积扩展到 3D）
│
├── diffusion_1d.py          # 一维单群扩散（入门脚本）
├── diffusion_2g.py          # 一维双群扩散（2N×2N 分块矩阵）
├── diffusion_2d.py          # 二维双群扩散 + 四联 heatmap 可视化
├── diffusion_3d.py          # 三维双群扩散
│
├── critical_size.py         # 临界尺寸扫描（二分法 keff→1）
├── boron_search.py          # 临界硼浓度搜索（外迭代+内迭代嵌套）
│
├── power_iteration.py       # 幂迭代 + Chebyshev 外推加速引擎
├── visualization.py          # 收敛历史与残差诊断图
├── result_export.py          # CSV 通量表与可复现 JSON 导出
├── resource_guards.py        # 2D/3D 规模估算与安全上限
├── demo_chebyshev.py        # 收敛行为对比 + ω 参数扫描
│
├── point_kinetics.py        # 点堆动力学（6 群缓发中子 + 倒时方程 + 反馈）
├── demo_kinetics.py         # 点堆动力学交互式演示
│
├── bateman.py               # Bateman 燃耗方程组（放射性衰变链）
├── burnup_solver.py         # 燃耗-扩散耦合求解器
│
├── debug_demo.py            # Debug 教学：小网格 + 解析特征值验证
├── validate_2d.py           # 二维解析验证（特征值 + buckling + 网格收敛）
├── neutron_diffusion_tutorial.ipynb  # Jupyter 教学版（扩散→点堆全链路）
│
├── tests/                   # 78 个 pytest 单元测试
│   ├── test_diffusion.py    # 扩散/临界/幂迭代/2D/3D 验证
│   └── test_kinetics.py     # 点堆动力学验证
├── docs/USAGE.md            # 安装、计算、导出与诊断使用指南
├── .github/workflows/test.yml  # GitHub Actions（lint + 测试）
├── requirements.txt         # 运行依赖
├── requirements-dev.txt     # 开发、测试与 lint 依赖
├── pyproject.toml           # Ruff 与 pytest 配置
└── NOTES.md                 # 完整物理推导笔记
```

### 学习总结

| 文件 | 内容 |
|------|------|
| `2026-06-16-总结.md` | 单群→双群、k 更新 bug、临界搜索、debug 方法论 |
| `2026-06-18-总结.md` | 1D→2D 扩散、稀疏矩阵、Kronecker 积、解析验证 |
| `2026-06-22-总结.md` | 幂迭代、Chebyshev 加速、ρ 估计、参数扫描方法论 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/buzhidao2006/neutron-diffusion-solver.git
cd neutron-diffusion-solver
pip install -r requirements.txt
```

推荐在独立虚拟环境中安装：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install --upgrade pip
```

若要运行测试或参与开发，安装开发依赖：

```bash
pip install -r requirements-dev.txt
```

### 统一 CLI 入口

```bash
python3 main.py 1d              # 一维单群扩散
python3 main.py 2g              # 一维双群扩散
python3 main.py critical-scan   # 临界尺寸扫描
python3 main.py boron-search    # 临界硼浓度搜索
python3 main.py 2d              # 二维双群扩散
python3 main.py 3d              # 三维双群扩散
python3 main.py kinetics        # 点堆动力学演示
python3 main.py benchmark       # 一维单群解析基准 + 网格收敛
python3 main.py test            # 运行测试套件
python3 main.py demo            # 运行所有演示
```

带参数示例：

```bash
python3 main.py 2g --L 300 --N 200
python3 main.py critical-scan --L-min 40 --L-max 400
python3 main.py 2d --Lx 200 --Nx 60
python3 main.py 3d --L 160 --N 20
```

### 直接运行单个脚本

```bash
python3 diffusion_1d.py      # 一维单群：通量 + k_eff
python3 diffusion_2g.py      # 一维双群：快/热中子通量
python3 critical_size.py     # 临界尺寸扫描
python3 boron_search.py      # 临界硼浓度搜索
python3 diffusion_2d.py      # 二维双群：2D 通量热力图
python3 demo_chebyshev.py    # Chebyshev 加速对比 + ω 参数扫描
python3 demo_kinetics.py     # 点堆动力学：阶跃/斜坡/弹棒场景
python3 debug_demo.py        # 解析特征值验证 demo
```

### 启动交互式 Web 应用

```bash
streamlit run app.py
```

打开浏览器访问 `http://localhost:8501`，七个标签页：
1. **双群扩散求解** — 一维快/热通量、热化比、收敛诊断与结果导出
2. **临界尺寸扫描** — `k_eff` 与尺寸关系及 Buckling 对比
3. **临界硼搜索** — 临界硼浓度与反应性价值
4. **二维扩散 (2D)** — 通量云图、中心线剖面、资源估算与结果导出
5. **三维扩散 (3D)** — 正交截面、资源保护、收敛诊断与结果导出
6. **燃耗耦合** — Bateman 衰变链与燃耗-扩散耦合
7. **点堆动力学** — 6 群缓发中子瞬态与反应性场景

完整的安装、CLI、Web 操作、结果导出和收敛判读说明见 [使用指南](docs/USAGE.md)。

### 运行测试

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

### 运行解析基准

```bash
python3 main.py benchmark
```

该命令使用均匀一维单群平板的基态解析解 ``sin(πx/L)``，对比矩阵计算、
离散 FDM 解析特征值与连续解析特征值。输出的相对误差应随网格加密单调下降，
用于验证矩阵组装和二阶空间收敛性。

---

## 技术栈

| 层面 | 技术 | 用途 |
|------|------|------|
| 数值计算 | NumPy, SciPy (sparse, spsolve, solve_ivp) | 矩阵构造、稀疏求解、ODE 积分 |
| 可视化 | Matplotlib | 通量分布、收敛曲线、heatmap |
| Web UI | Streamlit | 交互式参数调节 + 实时出图 |
| 算法 | 有限差分法、幂迭代、Chebyshev 外推、二分法、Runge-Kutta | 核心求解 |
| 工程化 | pytest, Ruff, GitHub Actions, argparse | 测试、静态检查、持续集成、命令行 |

---

## 物理覆盖

> **模型边界**：本项目用于教学和数值方法练习。默认截面与燃耗、反馈模型均为简化示例，不应用于反应堆设计、安全分析或运行决策。

- ✅ 单群/双群/三维中子扩散方程
- ✅ k_eff 特征值问题（Rayleigh 商）
- ✅ 临界尺寸确定（几何曲率 B²）
- ✅ 临界硼浓度搜索（PWR 反应性控制）
- ✅ Bateman 燃耗方程组（放射性衰变链）
- ✅ 燃耗-扩散耦合（时间-空间双物理过程）
- ✅ 点堆动力学（6 群缓发中子，Keepin 1965 数据）
- ✅ 倒时方程 + 渐近周期 + 瞬发跳变
- ✅ 弹棒事故等反应性场景 + 简化温度反馈
- ✅ 幂迭代 + Chebyshev 外推加速
- ✅ 收敛状态、末次 k_eff 变化量与严格失败诊断
- ✅ 求解入口参数校验：几何尺寸、网格、迭代控制与两群截面
- ✅ 2D/3D 规模估算与资源保护，阻止过大网格在矩阵分配前耗尽资源
- ✅ 1D/2D/3D 结果导出：CSV 通量表与包含参数、诊断和时间戳的可复现 JSON 记录
- ✅ 1D/2D/3D 迭代诊断图：k_eff 收敛历史、残差对数曲线与末次收缩率
- ✅ Web UI 第一轮产品化：统一结果区、参数操作流程提示与可读的求解错误反馈
- ✅ 解析特征值验证（debug 方法论）
- ✅ 一维单群平板解析基准 + 网格收敛性验证
- ✅ 1D → 2D → 3D 扩展（Kronecker 积构造法）
- ✅ 稀疏矩阵存储与求解（CSR 格式）

---

## 测试与持续集成

- **78 个单元测试**，覆盖扩散、点堆动力学与解析基准：
  - `test_diffusion.py`（41 个）— 扩散求解器、输入参数校验、资源保护、临界搜索、幂迭代、收敛失败诊断、2D/3D Laplacian 解析验证、跨模块物理一致性
  - `test_kinetics.py`（18 个）— 点堆方程、倒时方程、瞬发跳变、弹棒事故
  - `test_benchmarks.py`（3 个）— 一维平板解析基准与网格收敛
  - `test_visualization.py`（4 个）— 收敛诊断图与收缩率计算
  - `test_edge_cases.py`（8 个）— 资源上限、3D 导出、JSON 序列化与诊断边界条件
- **GitHub Actions**：push / PR 到 `master`/`main` 时，自动运行 Ruff 静态检查，并在 Python 3.10 和 3.12 下运行 `pytest tests/ -v`

---

## 考研衔接

- **初试（865 核能与核技术基础）**：扩散方程推导、k_eff 定义、临界条件、多群概念、点堆方程全覆盖
- **复试**：能讲清楚这个求解器的物理原理和数值方法，证明理解从方程到代码的完整链路
- **后续方向**：含缓发中子的时空动力学 → 功率分布重构 → 精细燃耗计算 → 堆芯优化，对应 NECP 软件（Bamboo-C、NECP-X）核心链路

---

## License

MIT — 教学用途，欢迎 fork 和改进。
