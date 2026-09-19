"""一维单群平板扩散方程的解析与离散基准验证。

对于零通量边界的均匀平板，基态通量为 ``sin(pi*x/L)``。本模块比较：

* 矩阵特征值计算结果；
* 相同有限差分网格的离散解析特征值；
* 连续方程的解析特征值。

这将矩阵组装误差与有限网格的离散误差明确分开。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlabBenchmarkResult:
    """单个网格上的一维单群平板基准结果。"""

    n_nodes: int
    k_numeric: float
    k_discrete: float
    k_continuous: float
    relative_error: float


def _validate_inputs(length: float, n_nodes: int, diffusion: float,
                     absorption: float, nu_sigma_f: float) -> None:
    if length <= 0 or n_nodes < 1:
        raise ValueError("length must be positive and n_nodes must be at least 1.")
    if diffusion <= 0 or absorption < 0 or nu_sigma_f <= 0:
        raise ValueError("diffusion and nu_sigma_f must be positive; absorption cannot be negative.")


def run_slab_benchmark(
    n_nodes: int,
    length: float = 200.0,
    diffusion: float = 1.2,
    absorption: float = 0.08,
    nu_sigma_f: float = 0.105,
) -> SlabBenchmarkResult:
    """运行均匀一维单群平板的基态 ``k_eff`` 基准。

    ``n_nodes`` 是开区间 ``(0, length)`` 内的未知数数目，因此
    ``h = length / (n_nodes + 1)``。这与项目 1D/2D/3D 求解器的
    内部节点 Dirichlet 约定保持一致。
    """
    _validate_inputs(length, n_nodes, diffusion, absorption, nu_sigma_f)

    h = length / (n_nodes + 1)
    diagonal = 2.0 * np.ones(n_nodes)
    off_diagonal = -np.ones(n_nodes - 1)
    laplacian = np.diag(diagonal)
    if n_nodes > 1:
        laplacian += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

    loss_operator = diffusion * laplacian / h**2 + absorption * np.eye(n_nodes)
    smallest_eigenvalue = np.linalg.eigvalsh(loss_operator)[0]
    k_numeric = nu_sigma_f / smallest_eigenvalue

    lambda_discrete = 4.0 * np.sin(np.pi / (2.0 * (n_nodes + 1))) ** 2 / h**2
    k_discrete = nu_sigma_f / (absorption + diffusion * lambda_discrete)

    lambda_continuous = (np.pi / length) ** 2
    k_continuous = nu_sigma_f / (absorption + diffusion * lambda_continuous)

    return SlabBenchmarkResult(
        n_nodes=n_nodes,
        k_numeric=float(k_numeric),
        k_discrete=float(k_discrete),
        k_continuous=float(k_continuous),
        relative_error=float(abs(k_numeric - k_continuous) / k_continuous),
    )


def convergence_study(n_nodes_values: tuple[int, ...] = (20, 40, 80, 160)) -> list[SlabBenchmarkResult]:
    """在一组网格上运行基准，以展示二阶空间收敛。"""
    if len(n_nodes_values) < 2 or any(n < 1 for n in n_nodes_values):
        raise ValueError("Provide at least two positive grid sizes.")
    return [run_slab_benchmark(n_nodes) for n_nodes in n_nodes_values]


def format_convergence_study(results: list[SlabBenchmarkResult]) -> str:
    """将基准结果格式化为可直接放入实验记录的表格。"""
    lines = [
        " N       k_numeric      k_discrete     k_continuous   relative error",
        "-" * 74,
    ]
    for result in results:
        lines.append(
            f"{result.n_nodes:3d}  {result.k_numeric:14.8f}  {result.k_discrete:14.8f}  "
            f"{result.k_continuous:14.8f}  {result.relative_error:12.3e}"
        )
    return "\n".join(lines)
