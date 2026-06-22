"""
二维双群中子扩散方程求解器模块
使用 5 点有限差分 + 稀疏矩阵 + 幂迭代

与 solver.py (一维) 的区别：
  - 矩阵构造使用 Kronecker 积 + scipy.sparse
  - 线性求解使用 spsolve (直接稀疏求解器)
  - 通量返回 2D 数组，坐标返回 meshgrid
  - 双群块结构完全相同
"""
import numpy as np
from scipy.sparse import diags, kron, eye, bmat, csr_matrix
from scipy.sparse.linalg import spsolve

# 复用 solver.py 的默认截面数据
from solver import DEFAULTS
from power_iteration import power_iteration, power_iteration_chebyshev


def _build_2d_laplacian(Nx, Ny, hx, hy):
    """
    使用 Kronecker 积构造 2D 离散 Laplacian 矩阵（5 点格式）。

    L_2d = I_y ⊗ (L_x / hx²) + (L_y / hy²) ⊗ I_x

    其中 L_x, L_y 是各方向的 1D 三对角 Laplacian：
        L = diag(-1, 2, -1)

    Parameters
    ----------
    Nx, Ny : int, x/y 方向网格点数
    hx, hy : float, x/y 方向步长 (cm)

    Returns
    -------
    L_2d : (Nx*Ny, Nx*Ny) 稀疏矩阵 (CSR)
    """
    # 各方向的 1D Laplacian
    Lx = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(Nx, Nx))
    Ly = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(Ny, Ny))

    Ix = eye(Nx)
    Iy = eye(Ny)

    # 2D Laplacian：Kronecker 积直接给出 5 点格式
    L_2d = kron(Iy, Lx) / (hx * hx) + kron(Ly, Ix) / (hy * hy)
    return L_2d.tocsr()


def solve_two_group_2d(Lx=None, Ly=None, Nx=None, Ny=None, sections=None,
                        method='chebyshev', tol=1e-10, max_iter=200):
    """
    求解二维双群中子扩散方程，返回 k_eff 和 2D 通量分布。

    矩形区域 [0, Lx] × [0, Ly]，cell-centered 网格自动满足
    零通量（Dirichlet）边界条件。

    Parameters
    ----------
    Lx, Ly : float, x/y 方向边长 (cm)，默认均为 200
    Nx, Ny : int, x/y 方向网格点数，默认均为 60
    sections : dict, 截面数据，可部分覆盖 DEFAULTS
    method : str
        'power'     — 标准幂迭代（无加速）
        'chebyshev' — Chebyshev 多项式外推加速（默认）
    tol : float
        k_eff 收敛容忍度
    max_iter : int
        最大迭代次数

    Returns
    -------
    dict: {
        'k_eff': float,
        'X': np.ndarray,         # 2D meshgrid x 坐标 (Ny, Nx)
        'Y': np.ndarray,         # 2D meshgrid y 坐标 (Ny, Nx)
        'x': np.ndarray,         # 1D x 坐标 (Nx,)
        'y': np.ndarray,         # 1D y 坐标 (Ny,)
        'phi1': np.ndarray,      # 快群通量 2D (Ny, Nx)
        'phi2': np.ndarray,      # 热群通量 2D (Ny, Nx)
        'phi': np.ndarray,       # 完整通量向量 (2*Nx*Ny,)
        'n_iter': int,           # 实际迭代次数
        'k_history': list,       # k_eff 收敛历史
        'residual': list,        # 残差历史
    }
    """
    p = {**DEFAULTS, **(sections or {})}
    Lx = Lx or p.get('Lx', p['L'])
    Ly = Ly or p.get('Ly', p['L'])
    Nx = Nx or p.get('Nx', p['N'])
    Ny = Ny or p.get('Ny', p['N'])

    hx = Lx / Nx
    hy = Ly / Ny
    N_total = Nx * Ny

    D1, nu_Sf1, Sa1, Ss12 = p['D1'], p['nu_Sf1'], p['Sa1'], p['Ss12']
    D2, nu_Sf2, Sa2 = p['D2'], p['nu_Sf2'], p['Sa2']
    Sr1 = Sa1 + Ss12  # 快群移出截面

    # ---- 构造 2D Laplacian ----
    L_2d = _build_2d_laplacian(Nx, Ny, hx, hy)
    I_total = eye(N_total)

    # ---- 组装双群分块矩阵 ----
    # A = [[A11,   0],   F = [[F11, F12],
    #      [A21, A22]]        [  0,   0]]
    A11 = D1 * L_2d + Sr1 * I_total
    A12 = csr_matrix((N_total, N_total))
    A21 = -Ss12 * I_total
    A22 = D2 * L_2d + Sa2 * I_total
    A = bmat([[A11, A12], [A21, A22]], format='csr')

    F11 = nu_Sf1 * I_total
    F12 = nu_Sf2 * I_total
    Z = csr_matrix((N_total, N_total))
    F = bmat([[F11, F12], [Z, Z]], format='csr')

    # ---- 幂迭代 ----
    phi0 = np.ones(2 * N_total)

    if method == 'power':
        result = power_iteration(A, F, phi0, max_iter=max_iter, tol=tol)
    elif method == 'chebyshev':
        result = power_iteration_chebyshev(A, F, phi0, max_iter=max_iter, tol=tol, warmup=15)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'power' or 'chebyshev'.")

    phi = result['phi']
    k_eff = result['k_eff']

    # ---- 坐标与通量整形 ----
    x = np.linspace(hx / 2, Lx - hx / 2, Nx)  # cell centers
    y = np.linspace(hy / 2, Ly - hy / 2, Ny)
    X, Y = np.meshgrid(x, y)

    phi1_2d = phi[:N_total].reshape(Ny, Nx)
    phi2_2d = phi[N_total:].reshape(Ny, Nx)

    return {
        'k_eff': k_eff,
        'X': X, 'Y': Y,
        'x': x,  'y': y,
        'phi1': phi1_2d,
        'phi2': phi2_2d,
        'phi': phi,
        'n_iter': result['n_iter'],
        'k_history': result['k_history'],
        'residual': result['residual'],
    }


def scan_critical_size_2d(L_min=40.0, L_max=400.0, n_points=20, Nx=60, Ny=60,
                          sections=None):
    """
    扫描 k_eff 随方形堆芯尺寸 L (= Lx = Ly) 的变化，找出临界尺寸。

    Parameters
    ----------
    L_min, L_max : float, 扫描范围 (cm)
    n_points : int, 扫描点数
    Nx, Ny : int, 网格点数
    sections : dict, 截面数据

    Returns
    -------
    dict: {
        'L_vals': np.ndarray,
        'k_vals': np.ndarray,
        'L_crit': float,
        'k_crit': float,
        'analytic': {
            'k_inf': float,
            'M2': float,
            'k_buckling': np.ndarray,
        }
    }
    """
    L_vals = np.linspace(L_min, L_max, n_points)
    k_vals = []

    for L in L_vals:
        result = solve_two_group_2d(Lx=L, Ly=L, Nx=Nx, Ny=Ny, sections=sections)
        k_vals.append(result['k_eff'])

    k_vals = np.array(k_vals)

    # 找临界尺寸（插值附近）
    idx = np.argmin(np.abs(k_vals - 1.0))
    L_crit = L_vals[idx]
    k_crit = k_vals[idx]

    # 解析 buckling 近似 (2D 方形: B² = 2*(π/L)²)
    p = {**DEFAULTS, **(sections or {})}
    Sr1 = p['Sa1'] + p['Ss12']
    k_inf = p['nu_Sf1'] / Sr1 + (p['nu_Sf2'] / p['Sa2']) * (p['Ss12'] / Sr1)
    L2 = p['D2'] / p['Sa2']
    tau = p['D1'] / Sr1
    M2 = L2 + tau
    B2 = 2.0 * (np.pi / L_vals) ** 2  # 方形 2D 几何曲率
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
