# 使用指南

本项目用于反应堆物理教学与数值方法练习；默认截面和反馈模型是简化示例，不能用于工程设计、安全分析或运行决策。

## 安装与验证

```bash
git clone https://github.com/buzhidao2006/neutron-diffusion-solver.git
cd neutron-diffusion-solver
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

`requirements-dev.txt` 包含运行、测试和 Ruff 所需的依赖。测试通过后，环境即可用于命令行或 Web 界面。

## 命令行计算

以下命令均在项目根目录运行：

```bash
python main.py 2g --L 300 --N 200
python main.py critical-scan --L-min 40 --L-max 400
python main.py boron-search --L 200 --N 150
python main.py 2d --Lx 200 --Nx 60
python main.py 3d --L 160 --N 20
python main.py benchmark
```

先从 `2g` 或 `benchmark` 开始。二维和三维计算的未知数分别为 `2 × Nx × Ny` 和 `2 × Nx × Ny × Nz`，应逐步加密网格而非直接使用大规模参数。

## Streamlit Web 界面

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 后：

1. 在左侧选择计算模块，并设置几何、网格和两群截面。
2. 点击对应的“求解”按钮，先阅读收敛状态和迭代诊断。
3. 查看通量分布、二维/三维截面或物理指标。
4. 需要复现实验时，下载 CSV 通量表和 JSON 结果记录。

JSON 含模型、输入参数、时间戳、`k_eff`、收敛状态和迭代历史；CSV 含注释形式的元数据以及每个网格点的坐标、快群和热群通量。

## 如何判断结果是否可信

- 显示“求解已收敛”后，才将 `k_eff` 和通量用于比较。
- 关注末次 `|Δk_eff|`、残差对数曲线和收缩率；收缩率小于 1 表示末次迭代仍在收缩。
- 若未收敛，增大最大迭代次数、适度放宽容差，或在 2D/3D 中选择 Chebyshev 加速。
- 2D/3D 界面在组装矩阵前会估算内存并拒绝超过安全上限的网格；应降低网格点数后重试。

## 开发检查

提交前运行：

```bash
python -m ruff check .
python -m pytest tests/ -v
```

GitHub Actions 会在 Python 3.10 和 3.12 上重复执行这些检查。
