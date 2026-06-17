"""
中子扩散方程核心求解器模块
提供：双群求解、临界尺寸扫描、临界硼搜索
"""
import numpy as np


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


def solve_two_group(L=None, N=None, sections=None):
    """
    求解双群一维中子扩散方程，返回 k_eff 和通量分布。

    Parameters
    ----------
    L : float, 平板半厚度 (cm)，默认 200
    N : int, 网格点数，默认 150
    sections : dict, 截面数据，可部分覆盖默认值

    Returns
    -------
    dict: {
        'k_eff': float,
        'x': np.ndarray,       # 网格坐标
        'phi1': np.ndarray,    # 快群通量
        'phi2': np.ndarray,    # 热群通量
        'phi': np.ndarray,     # 完整通量向量 (2N,)
    }
    """
    p = {**DEFAULTS, **(sections or {})}
    L = L or p['L']
    N = N or p['N']
    h = L / N

    D1, nu_Sf1, Sa1, Ss12 = p['D1'], p['nu_Sf1'], p['Sa1'], p['Ss12']
    D2, nu_Sf2, Sa2 = p['D2'], p['nu_Sf2'], p['Sa2']
    Sr1 = Sa1 + Ss12

    coeff1 = D1 / (h * h)
    coeff2 = D2 / (h * h)

    main = 2.0 * np.ones(N)
    off = -1.0 * np.ones(N - 1)
    L_mat = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)

    # 组装 2N×2N 分块矩阵
    A11 = coeff1 * L_mat + Sr1 * np.eye(N)
    A12 = np.zeros((N, N))
    A21 = -Ss12 * np.eye(N)
    A22 = coeff2 * L_mat + Sa2 * np.eye(N)
    A = np.vstack([np.hstack([A11, A12]), np.hstack([A21, A22])])

    F11 = nu_Sf1 * np.eye(N)
    F12 = nu_Sf2 * np.eye(N)
    F = np.vstack([np.hstack([F11, F12]), np.zeros((N, 2 * N))])

    # 幂迭代
    phi = np.ones(2 * N)
    k_eff = 1.0
    for _ in range(100):
        source = F @ phi
        phi_new = np.linalg.solve(A, source)
        k_new = np.sum(F @ phi_new) / np.sum(source)
        phi = phi_new / np.max(phi_new)
        if abs(k_new - k_eff) < 1e-8:
            k_eff = k_new
            break
        k_eff = k_new

    x = np.linspace(h / 2, L - h / 2, N)
    return {
        'k_eff': k_eff,
        'x': x,
        'phi1': phi[:N],
        'phi2': phi[N:],
        'phi': phi,
    }


def solve_keff_with_boron(C_B, L=200.0, N=150, alpha=1.0e-5, sections=None):
    """
    给定硼浓度 C_B (ppm)，返回 k_eff。
    硼增加热群吸收截面：Σ_a2 = Σ_a2_base + alpha * C_B
    """
    p = {**DEFAULTS, **(sections or {})}
    Sa2_eff = p['Sa2'] + alpha * C_B
    sec = {**p, 'Sa2': Sa2_eff}
    return solve_two_group(L=L, N=N, sections=sec)['k_eff']


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
    L_vals = np.linspace(L_min, L_max, n_points)
    k_vals = []

    for L in L_vals:
        k = solve_two_group(L=L, N=N, sections=sections)['k_eff']
        k_vals.append(k)

    k_vals = np.array(k_vals)

    # 找临界尺寸（插值）
    idx = np.argmin(np.abs(k_vals - 1.0))
    L_crit = L_vals[idx]
    k_crit = k_vals[idx]

    # 解析 buckling 近似
    p = {**DEFAULTS, **(sections or {})}
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
    # 扫描
    C_scan = np.linspace(C_range[0], C_range[1], 11)
    k_scan = np.array([solve_keff_with_boron(C, L=L, N=N, alpha=alpha, sections=sections)
                        for C in C_scan])

    # 二分法
    C_low, C_high = float(C_range[0]), float(C_range[1])
    k_low = solve_keff_with_boron(C_low, L=L, N=N, alpha=alpha, sections=sections)
    k_high = solve_keff_with_boron(C_high, L=L, N=N, alpha=alpha, sections=sections)

    # 如果初始区间不对，调整
    if k_low < 1.0:
        C_low = max(0.0, C_low - 200)
        k_low = solve_keff_with_boron(C_low, L=L, N=N, alpha=alpha, sections=sections)
    if k_high > 1.0:
        C_high = min(5000.0, C_high + 1500)
        k_high = solve_keff_with_boron(C_high, L=L, N=N, alpha=alpha, sections=sections)

    for _ in range(30):
        C_mid = (C_low + C_high) / 2
        k_mid = solve_keff_with_boron(C_mid, L=L, N=N, alpha=alpha, sections=sections)
        if abs(k_mid - 1.0) < 1e-6:
            break
        if k_mid > 1.0:
            C_low = C_mid
        else:
            C_high = C_mid

    C_crit = (C_low + C_high) / 2
    k_final = solve_keff_with_boron(C_crit, L=L, N=N, alpha=alpha, sections=sections)

    # 硼微分价值
    dC = 10.0
    k_plus = solve_keff_with_boron(C_crit + dC, L=L, N=N, alpha=alpha, sections=sections)
    k_minus = solve_keff_with_boron(C_crit - dC, L=L, N=N, alpha=alpha, sections=sections)
    rho_per_ppm = (k_minus - k_plus) / (2 * dC) / k_final ** 2 * 1e5

    return {
        'C_crit': C_crit,
        'k_final': k_final,
        'C_scan': C_scan,
        'k_scan': k_scan,
        'boron_worth': rho_per_ppm,
    }
