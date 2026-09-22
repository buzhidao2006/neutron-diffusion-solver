"""
中子扩散方程核心求解器模块
提供：双群求解、临界尺寸扫描、临界硼搜索
"""
import numpy as np
from numbers import Real
from scipy.sparse import bmat, csr_matrix, diags, eye
from resource_guards import guard_problem_size

from power_iteration import power_iteration


# ============ 默认截面数据 (典型 PWR) ============
DEFAULTS = {
    # 群 1 — 快群
    'D1': 1.2,         # 扩散系数 (cm)
    'nu_Sf1': 0.003,   # νΣ_f (cm⁻¹)
    'Sa1': 0.008,      # 吸收截面 Σ_a1 (cm⁻¹)
    'Ss12': 0.020,     # 散射截面 Σ_{s,1→2} (cm⁻¹)
    # 群 2 — 热群
    'D2': 0.4,         # 扩散系数 (cm)
    'nu_Sf2': 0.105,   # νΣ_f (cm⁻¹)
    'Sa2': 0.08,       # 吸收截面 Σ_a2 (cm⁻¹)
    # 几何
    'L': 200.0,        # 平板半厚度 (cm)
    'N': 150,          # 网格点数
}


def merge_sections(sections=None):
    """Merge material data with defaults and reject unknown input names."""
    if sections is None:
        return DEFAULTS.copy()
    if not isinstance(sections, dict):
        raise ValueError("sections must be a mapping of material parameters.")
    unknown = sorted(set(sections) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown material parameter(s): {', '.join(unknown)}.")
    return {**DEFAULTS, **sections}


def _validate_positive_number(name, value):
    """Require a finite, strictly positive scalar model input."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


def _validate_nonnegative_number(name, value):
    """Require a finite scalar that is zero or positive."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")


def _validate_grid_count(name, value):
    """Require an integer count of interior grid nodes."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1.")


def validate_two_group_inputs(lengths, grids, sections):
    """Validate geometry, grid, and cross-section inputs shared by all solvers."""
    required = ('D1', 'D2', 'Sa1', 'Sa2', 'nu_Sf1', 'nu_Sf2', 'Ss12')
    missing = [name for name in required if name not in sections]
    if missing:
        raise ValueError(f"Missing material parameters: {', '.join(missing)}.")
    for name, value in lengths.items():
        _validate_positive_number(name, value)
    for name, value in grids.items():
        _validate_grid_count(name, value)

    positive_sections = ("D1", "D2", "Sa1", "Sa2")
    nonnegative_sections = ("nu_Sf1", "nu_Sf2", "Ss12")
    for name in positive_sections:
        _validate_positive_number(name, sections[name])
    for name in nonnegative_sections:
        value = sections[name]
        if (isinstance(value, bool) or not isinstance(value, Real) or
                not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite non-negative number.")
    if sections["nu_Sf1"] == 0 and sections["nu_Sf2"] == 0:
        raise ValueError("At least one fission production cross section must be positive.")


def validate_iteration_controls(max_iter, tol):
    """Validate iteration controls before assembling a potentially large system."""
    _validate_grid_count("max_iter", max_iter)
    _validate_positive_number("tol", tol)


def solve_two_group(L=None, N=None, sections=None, max_iter=300, tol=1e-10,
                    raise_on_nonconvergence=False):
    """
    求解双群一维中子扩散方程，返回 k_eff 和通量分布。

    Parameters
    ----------
    L : float, 平板半厚度 (cm)，默认 200
    N : int, 网格点数，默认 150
    sections : dict, 截面数据，可部分覆盖默认值
    max_iter : int, 最大幂迭代次数
    tol : float, k_eff 相对收敛容差
    raise_on_nonconvergence : bool, 为真时迭代耗尽将抛出 ConvergenceError

    Returns
    -------
    dict: {
        'k_eff': float,
        'x': np.ndarray,       # 网格坐标
        'phi1': np.ndarray,    # 快群通量
        'phi2': np.ndarray,    # 热群通量
        'phi': np.ndarray,     # 完整通量向量 (2N,)
        'n_iter': int,         # 实际迭代次数
        'converged': bool,     # 是否满足收敛容差
        'termination_reason': str,
    }
    """
    p = merge_sections(sections)
    L = p['L'] if L is None else L
    N = p['N'] if N is None else N
    validate_two_group_inputs({'L': L}, {'N': N}, p)
    validate_iteration_controls(max_iter, tol)
    guard_problem_size(1, N)
    # N unknowns are interior nodes; zero-flux Dirichlet boundaries are at
    # x=0 and x=L, so there are N+1 intervals.
    h = L / (N + 1)

    D1, nu_Sf1, Sa1, Ss12 = p['D1'], p['nu_Sf1'], p['Sa1'], p['Ss12']
    D2, nu_Sf2, Sa2 = p['D2'], p['nu_Sf2'], p['Sa2']
    Sr1 = Sa1 + Ss12

    coeff1 = D1 / (h * h)
    coeff2 = D2 / (h * h)

    L_mat = diags(
        [-np.ones(N - 1), 2.0 * np.ones(N), -np.ones(N - 1)],
        offsets=[-1, 0, 1], shape=(N, N), format='csr',
    )

    # 组装 2N×2N 分块矩阵
    A11 = coeff1 * L_mat + Sr1 * eye(N, format='csr')
    A12 = csr_matrix((N, N))
    A21 = -Ss12 * eye(N, format='csr')
    A22 = coeff2 * L_mat + Sa2 * eye(N, format='csr')
    A = bmat([[A11, A12], [A21, A22]], format='csr')

    F11 = nu_Sf1 * eye(N, format='csr')
    F12 = nu_Sf2 * eye(N, format='csr')
    F = bmat([[F11, F12], [csr_matrix((N, N)), csr_matrix((N, N))]], format='csr')

    result = power_iteration(
        A, F, np.ones(2 * N), max_iter=max_iter, tol=tol,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
    phi = result['phi'] / np.max(result['phi'])

    x = np.arange(1, N + 1) * h
    return {
        'k_eff': result['k_eff'],
        'x': x,
        'phi1': phi[:N],
        'phi2': phi[N:],
        'phi': phi,
        'n_iter': result['n_iter'],
        'k_history': result['k_history'],
        'residual': result['residual'],
        'converged': result['converged'],
        'delta_k': result['delta_k'],
        'termination_reason': result['termination_reason'],
    }


def solve_keff_with_boron(C_B, L=200.0, N=150, alpha=1.0e-5, sections=None,
                          raise_on_nonconvergence=False):
    """
    给定硼浓度 C_B (ppm)，返回 k_eff。
    硼增加热群吸收截面：Σ_a2 = Σ_a2_base + alpha * C_B
    """
    _validate_nonnegative_number('C_B', C_B)
    _validate_positive_number('alpha', alpha)
    p = merge_sections(sections)
    Sa2_eff = p['Sa2'] + alpha * C_B
    sec = {**p, 'Sa2': Sa2_eff}
    result = solve_two_group(
        L=L, N=N, sections=sec, max_iter=500, tol=1e-7,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
    return result['k_eff']


def scan_critical_size(L_min=40.0, L_max=400.0, n_points=36, N=150, sections=None):
    """
    扫描 k_eff 随平板厚度 L 的变化，找出临界尺寸。

    Returns
    -------
    dict: {
        'L_vals': np.ndarray,
        'k_vals': np.ndarray,
        'L_crit': float,         # k≈1 对应的临界尺寸
        'k_crit': float,
        'analytic': {            # buckling 解析近似
            'k_inf': float,
            'M2': float,         # 徙动面积
            'k_buckling': np.ndarray,
        }
    }
    """
    _validate_positive_number('L_min', L_min)
    _validate_positive_number('L_max', L_max)
    if L_min >= L_max:
        raise ValueError('L_min must be smaller than L_max.')
    _validate_grid_count('n_points', n_points)
    if n_points < 2:
        raise ValueError('n_points must be at least 2.')
    _validate_grid_count('N', N)
    L_vals = np.linspace(L_min, L_max, n_points)
    k_vals = []

    for L in L_vals:
        k = solve_two_group(
            L=L, N=N, sections=sections, max_iter=500, tol=1e-7,
            raise_on_nonconvergence=True,
        )['k_eff']
        k_vals.append(k)

    k_vals = np.array(k_vals)

    # 找临界尺寸：有夹点时在线性插值上估计 k=1，而不是返回最近扫描点。
    idx = np.argmin(np.abs(k_vals - 1.0))
    L_crit = L_vals[idx]
    k_crit = k_vals[idx]
    critical_bracketed = False
    for left in range(len(k_vals) - 1):
        if (k_vals[left] - 1.0) * (k_vals[left + 1] - 1.0) <= 0:
            right = left + 1
            if k_vals[right] != k_vals[left]:
                fraction = (1.0 - k_vals[left]) / (k_vals[right] - k_vals[left])
                L_crit = L_vals[left] + fraction * (L_vals[right] - L_vals[left])
                k_crit = 1.0
                critical_bracketed = True
            break

    # 解析 buckling 近似
    p = merge_sections(sections)
    Sr1 = p['Sa1'] + p['Ss12']
    k_inf = p['nu_Sf1'] / Sr1 + (p['nu_Sf2'] / p['Sa2']) * (p['Ss12'] / Sr1)
    L2 = p['D2'] / p['Sa2']
    tau = p['D1'] / Sr1
    M2 = L2 + tau
    B2 = (np.pi / L_vals) ** 2
    k_buckling = k_inf / (1 + M2 * B2)

    return {
        'L_vals': L_vals,
        'k_vals': k_vals,
        'L_crit': L_crit,
        'k_crit': k_crit,
        'critical_bracketed': critical_bracketed,
        'analytic': {
            'k_inf': k_inf,
            'M2': M2,
            'k_buckling': k_buckling,
        }
    }


def search_critical_boron(L=200.0, N=150, alpha=1.0e-5, C_range=(0, 3000), sections=None):
    """
    二分法搜索使 k=1 的临界硼浓度。

    Returns
    -------
    dict: {
        'C_crit': float,         # 临界硼浓度 (ppm)
        'k_final': float,
        'C_scan': np.ndarray,    # 扫描点
        'k_scan': np.ndarray,
        'boron_worth': float,    # 硼微分价值 (pcm/ppm)
    }
    """
    _validate_positive_number('L', L)
    _validate_grid_count('N', N)
    _validate_positive_number('alpha', alpha)
    if not isinstance(C_range, (tuple, list)) or len(C_range) != 2:
        raise ValueError('C_range must contain exactly two concentrations.')
    C_low, C_high = C_range
    _validate_nonnegative_number('C_range[0]', C_low)
    _validate_positive_number('C_range[1]', C_high)
    if C_low >= C_high:
        raise ValueError("C_range must satisfy C_low < C_high.")

    # 扫描
    C_scan = np.linspace(C_range[0], C_range[1], 11)
    k_scan = np.array([solve_keff_with_boron(
        C, L=L, N=N, alpha=alpha, sections=sections,
        raise_on_nonconvergence=True,
    )
                        for C in C_scan])

    # 二分法。二分法只有在区间两端夹住 k=1 时才有物理意义；若没有
    # 临界点，继续迭代会返回一个看似合理、实际上错误的浓度。
    C_low, C_high = float(C_range[0]), float(C_range[1])
    k_low = solve_keff_with_boron(
        C_low, L=L, N=N, alpha=alpha, sections=sections, raise_on_nonconvergence=True,
    )
    k_high = solve_keff_with_boron(
        C_high, L=L, N=N, alpha=alpha, sections=sections, raise_on_nonconvergence=True,
    )

    if not (k_low >= 1.0 >= k_high):
        raise ValueError(
            "C_range does not bracket a critical boron concentration: "
            f"k({C_low:g})={k_low:.6f}, k({C_high:g})={k_high:.6f}."
        )

    for _ in range(30):
        C_mid = (C_low + C_high) / 2
        k_mid = solve_keff_with_boron(
            C_mid, L=L, N=N, alpha=alpha, sections=sections,
            raise_on_nonconvergence=True,
        )
        if abs(k_mid - 1.0) < 1e-6:
            break
        if k_mid > 1.0:
            C_low = C_mid
        else:
            C_high = C_mid

    C_crit = (C_low + C_high) / 2
    k_final = solve_keff_with_boron(
        C_crit, L=L, N=N, alpha=alpha, sections=sections,
        raise_on_nonconvergence=True,
    )

    # 硼微分价值
    dC = 10.0
    k_plus = solve_keff_with_boron(
        C_crit + dC, L=L, N=N, alpha=alpha, sections=sections,
        raise_on_nonconvergence=True,
    )
    k_minus = solve_keff_with_boron(
        C_crit - dC, L=L, N=N, alpha=alpha, sections=sections,
        raise_on_nonconvergence=True,
    )
    rho_per_ppm = (k_minus - k_plus) / (2 * dC) / k_final ** 2 * 1e5

    return {
        'C_crit': C_crit,
        'k_final': k_final,
        'C_scan': C_scan,
        'k_scan': k_scan,
        'boron_worth': rho_per_ppm,
    }
