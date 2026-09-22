# 快速开始：一次完整的双群计算

下面的流程从安装开始，运行一个一维双群扩散问题，并解释导出的结果。

## 1. 安装

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## 2. 运行计算

```bash
python main.py 2g --L 200 --N 150 --output diffusion_result.json
```

`L` 是平板长度（cm），`N` 是内部网格点数。网格点不包含零通量边界，边界位于 `x=0` 和 `x=L`。

## 3. 判断结果

命令行会输出 `k_eff`、临界状态、通量峰值和迭代次数。

- `k_eff > 1`：模型为超临界，裂变链反应倾向于增长。
- `k_eff < 1`：模型为次临界，裂变链反应倾向于衰减。
- 只有显示收敛的结果才适合用于比较或后续计算。
- 如果未收敛，增大 `--max-iter`、适度放宽 `--tol`，或改用 2D/3D 的 `--method chebyshev`。

## 4. 查看导出文件

JSON 包含模型名称、输入参数、`k_eff`、收敛状态、残差历史、网格坐标和两群通量；CSV 适合在表格软件或绘图工具中继续处理。

```bash
python -c "import json; d=json.load(open('diffusion_result.json', encoding='utf-8')); print(d['diagnostics']['k_eff'], d['diagnostics']['converged'])"
```

## 5. 模型边界

这是教学级有限差分双群扩散模型，默认使用均匀材料、零通量（Dirichlet）边界和简化群常数。它用于学习方程离散化、特征值迭代和结果诊断，不用于反应堆设计、安全分析或运行决策。
