# 🧬 neutron-diffusion-solver

> 从单群一维扩散到二维双群 + 幂迭代 + Chebyshev 加速 —— 8 天全栈反应堆物理教学项目

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-✓-013243.svg?logo=numpy)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-✓-8CAAE6.svg?logo=scipy)](https://scipy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-✓-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-✓-11557c.svg)](https://matplotlib.org/)

---

## 项目概览

这是一个从零开始构建的反应堆物理数值计算教学项目。8 天时间（2026.06.15 → 06.22），9 次提交，从 44 行的一维单群求解器，演进为覆盖**临界计算、燃耗分析、特征值求解**的 3400+ 行准工业级代码。

**核心能力**：有限差分法 · 幂迭代 · Chebyshev 外推加速 · 双群扩散 · 临界搜索 · Bateman 燃耗 · Kronecker 积稀疏矩阵 · 解析验证

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

---

## 项目结构

```
neutron-diffusion-solver/
├── app.py                  # Streamlit Web 应用（五标签页交互式可视化）
├── solver.py               # 一维求解器核心模块（临界扫描 + 硼搜索）
│
├── diffusion_1d.py         # 一维单群扩散（入门脚本）
├── diffusion_2g.py         # 一维双群扩散（2N×2N 分块矩阵）
├── diffusion_2d.py         # 二维双群扩散 + 四联 heatmap 可视化
│
├── solver_2d.py            # 二维求解器模块（Kronecker 积 + 稀疏矩阵）
├── validate_2d.py          # 二维解析验证（特征值 + buckling + 网格收敛）
│
├── critical_size.py        # 临界尺寸扫描（二分法 keff→1）
├── boron_search.py         # 临界硼浓度搜索（外迭代+内迭代嵌套）
│
├── power_iteration.py      # 幂迭代 + Chebyshev 外推加速引擎
├── demo_chebyshev.py       # 收敛行为对比 + ω 参数扫描
├── debug_demo.py           # Debug 教学：小网格 + 解析特征值验证
│
├── bateman.py              # Bateman 燃耗方程组（放射性衰变链）
├── burnup_solver.py        # 燃耗-扩散耦合求解器
│
├── requirements.txt        # Python 依赖
└── NOTES.md                # 完整物理推导笔记（228 行）
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

### 运行单个脚本

```bash
python3 diffusion_1d.py      # 一维单群：通量 + k_eff
python3 diffusion_2g.py      # 一维双群：快/热中子通量
python3 critical_size.py     # 临界尺寸扫描
python3 boron_search.py      # 临界硼浓度搜索
python3 diffusion_2d.py      # 二维双群：2D 通量热力图
python3 demo_chebyshev.py    # Chebyshev 加速对比 + ω 参数扫描
python3 debug_demo.py        # 解析特征值验证 demo
```

### 启动交互式 Web 应用

```bash
streamlit run app.py
```

打开浏览器访问 `http://localhost:8501`，五个标签页：
1. **一维单群** — 参数滑块 + 实时通量图
2. **一维双群** — 快/热通量对比
3. **临界设计** — 临界尺寸 + 硼浓度搜索
4. **燃耗计算** — Bateman 衰变链 + 燃耗-扩散耦合
5. **二维扩散** — 2D 通量 heatmap + 收敛分析

---

## 技术栈

| 层面 | 技术 | 用途 |
|------|------|------|
| 数值计算 | NumPy, SciPy (sparse, spsolve) | 矩阵构造、稀疏求解 |
| 可视化 | Matplotlib | 通量分布、收敛曲线、heatmap |
| Web UI | Streamlit | 交互式参数调节 + 实时出图 |
| 算法 | 有限差分法、幂迭代、Chebyshev 外推、二分法 | 核心求解 |

---

## 物理覆盖

- ✅ 单群/双群中子扩散方程
- ✅ k_eff 特征值问题（Rayleigh 商）
- ✅ 临界尺寸确定（几何曲率 B²）
- ✅ 临界硼浓度搜索（PWR 反应性控制）
- ✅ Bateman 燃耗方程组（放射性衰变链）
- ✅ 燃耗-扩散耦合（时间-空间双物理过程）
- ✅ 幂迭代 + Chebyshev 外推加速
- ✅ 解析特征值验证（debug 方法论）
- ✅ 1D → 2D 扩展（Kronecker 积构造法）
- ✅ 稀疏矩阵存储与求解（CSR 格式）

---

## 考研衔接

- **初试（865 核反应堆物理）**：扩散方程推导、k_eff 定义、临界条件、多群概念全覆盖
- **复试**：能讲清楚这个求解器的物理原理和数值方法，证明理解从方程到代码的完整链路
- **后续方向**：功率分布重构 → 精细燃耗计算 → 堆芯优化，对应 NECP 软件（Bamboo-C、NECP-X）核心链路

---

## License

MIT — 教学用途，欢迎 fork 和改进。
