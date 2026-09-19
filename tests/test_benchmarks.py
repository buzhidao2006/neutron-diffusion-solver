"""解析基准与网格收敛性测试。"""

import numpy as np

from benchmarks.slab_one_group import convergence_study, run_slab_benchmark


def test_matrix_solution_matches_discrete_analytic_eigenvalue():
    """矩阵特征值应与同一 FDM 网格的离散解析解一致。"""
    result = run_slab_benchmark(40)
    np.testing.assert_allclose(result.k_numeric, result.k_discrete, rtol=1e-12, atol=1e-14)


def test_continuous_solution_error_decreases_with_grid_refinement():
    """节点数加倍时，FDM 相对连续解析解的误差应下降。"""
    coarse = run_slab_benchmark(20)
    fine = run_slab_benchmark(40)
    assert fine.relative_error < coarse.relative_error / 3.5


def test_convergence_study_is_monotonic():
    """标准网格序列的误差应单调下降。"""
    results = convergence_study()
    errors = [result.relative_error for result in results]
    assert all(next_error < error for error, next_error in zip(errors, errors[1:]))
