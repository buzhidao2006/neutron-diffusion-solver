"""
三维双群中子扩散方程求解器模块
使用 7 点有限差分 + 稀疏矩阵 + 幂迭代

与 solver_2d.py 的区别：
  - 矩阵构造: L_3d = Iz⊗Iy⊗Lx/hx² + Iz⊗Ly/hy²⊗Ix + Lz/hz²⊗Iy⊗Ix
  - 通量返回 3D 数组 (Nz, Ny, Nx)
  - 双群块结构完全相同，只是维度更大了

Kronecker 积构造法:
  1D → 2D → 3D 扩展完全公式化:
    L_2d = I_y ⊗ L_x/hx² + L_y/hy² ⊗ I_x
    L_3d = I_z ⊗ I_y ⊗ L_x/hx² + I_z ⊗ L_y/hy² ⊗ I_x + L_z/hz² ⊗ I_y ⊗ I_x

注意: 3D 网格增长为 N³，建议 N ≤ 30 以保持可接受的计算速度。
"""

import numpy as np
from scipy.sparse import diags, kron, eye, bmat, csr_matrix
from solver import merge_sections, validate_iteration_controls, validate_two_group_inputs
from power_iteration import power_iteration, power_iteration_chebyshev
from resource_guards import guard_problem_size


def _build_3d_laplacian(Nx, Ny, Nz, hx, hy, hz):
    """
    使用 Kronecker 积构造 3D 离散 Laplacian 矩阵（7 点格式）。

    L_3d = Iz ⊗ Iy ⊗ (Lx/hx²) + Iz ⊗ (Ly/hy²) ⊗ Ix + (Lz/hz²) ⊗ Iy ⊗ Ix

    每个点 (i,j,k) 依赖 6 个邻居 (±x, ±y, ±z)，每行 7 个非零元。

    Parameters
    ----------
    Nx, Ny, Nz : int, 各方向网格点数
    hx, hy, hz : float, 各方向步长 (cm)

    Returns
    -------
    L_3d : (Nx*Ny*Nz, Nx*Ny*Nz) 稀疏矩阵 (CSR)
    """
    # 各方向的 1D Laplacian
    Lx = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(Nx, Nx))
    Ly = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(Ny, Ny))
    Lz = diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(Nz, Nz))

    Ix = eye(Nx)
    Iy = eye(Ny)
    Iz = eye(Nz)

    # 三层 Kronecker 积构造
    L_3d = (kron(Iz, kron(Iy, Lx)) / (hx * hx) +
            kron(Iz, kron(Ly, Ix)) / (hy * hy) +
            kron(Lz, kron(Iy, Ix)) / (hz * hz))

    return L_3d.tocsr()


def solve_two_group_3d(Lx=None, Ly=None, Lz=None, Nx=None, Ny=None, Nz=None,
                        sections=None, method='chebyshev', tol=1e-10, max_iter=200,
                        raise_on_nonconvergence=False):
    """
    求解三维双群中子扩散方程，返回 k_eff 和 3D 通量分布。

    长方体区域 [0,Lx]×[0,Ly]×[0,Lz]，使用内部节点网格，
    零通量（Dirichlet）边界条件自动满足。

    Parameters
    ----------
    Lx, Ly, Lz : float, 各方向边长 (cm)，默认均为 200
    Nx, Ny, Nz : int, 各方向网格点数，默认均为 25
    sections : dict, 截面数据，可部分覆盖 DEFAULTS
    method : str
        'power'     — 标准幂迭代
        'chebyshev' — Chebyshev 外推加速（默认）
    tol : float, k_eff 收敛容忍度
    max_iter : int, 最大迭代次数
    raise_on_nonconvergence : bool
        为真时未收敛将抛出 ConvergenceError，适用于依赖该结果的工作流。

    Returns
    -------
    dict: {
        'k_eff': float,
        'X': np.ndarray,     # 3D meshgrid (Nz, Ny, Nx)
        'Y': np.ndarray,
        'Z': np.ndarray,
        'x', 'y', 'z': 1D 坐标,
        'phi1': np.ndarray,  # 快群通量 3D (Nz, Ny, Nx)
        'phi2': np.ndarray,  # 热群通量 3D (Nz, Ny, Nx)
        'n_iter': int,
        'k_history': list,
    }
    """
    p = merge_sections(sections)
    Lx = p.get('Lx', p['L']) if Lx is None else Lx
    Ly = p.get('Ly', p['L']) if Ly is None else Ly
    Lz = p.get('Lz', p['L']) if Lz is None else Lz
    Nx = p.get('Nx', 25) if Nx is None else Nx
    Ny = p.get('Ny', 25) if Ny is None else Ny
    Nz = p.get('Nz', 25) if Nz is None else Nz
    validate_two_group_inputs(
        {'Lx': Lx, 'Ly': Ly, 'Lz': Lz}, {'Nx': Nx, 'Ny': Ny, 'Nz': Nz}, p,
    )
    validate_iteration_controls(max_iter, tol)
    if method not in ('power', 'chebyshev'):
        raise ValueError("method must be 'power' or 'chebyshev'.")

    # Nx/Ny/Nz are interior-node counts; the zero-flux boundaries are
    # excluded, giving N+1 intervals along each direction.
    hx = Lx / (Nx + 1)
    hy = Ly / (Ny + 1)
    hz = Lz / (Nz + 1)
    N_total = Nx * Ny * Nz
    guard_problem_size(3, Nx, Ny, Nz)

    D1, nu_Sf1, Sa1, Ss12 = p['D1'], p['nu_Sf1'], p['Sa1'], p['Ss12']
    D2, nu_Sf2, Sa2 = p['D2'], p['nu_Sf2'], p['Sa2']
    Sr1 = Sa1 + Ss12  # 快群移出截面

    # ---- 构造 3D Laplacian ----
    L_3d = _build_3d_laplacian(Nx, Ny, Nz, hx, hy, hz)
    I_total = eye(N_total)

    # ---- 组装双群分块矩阵 ----
    A11 = D1 * L_3d + Sr1 * I_total
    A12 = csr_matrix((N_total, N_total))
    A21 = -Ss12 * I_total
    A22 = D2 * L_3d + Sa2 * I_total
    A = bmat([[A11, A12], [A21, A22]], format='csr')

    F11 = nu_Sf1 * I_total
    F12 = nu_Sf2 * I_total
    Z = csr_matrix((N_total, N_total))
    F = bmat([[F11, F12], [Z, Z]], format='csr')

    # ---- 幂迭代 ----
    phi0 = np.ones(2 * N_total)

    if method == 'power':
        result = power_iteration(
            A, F, phi0, max_iter=max_iter, tol=tol,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )
    elif method == 'chebyshev':
        result = power_iteration_chebyshev(
            A, F, phi0, max_iter=max_iter, tol=tol, warmup=15,
            raise_on_nonconvergence=raise_on_nonconvergence,
        )
    phi = result['phi']
    k_eff = result['k_eff']

    # ---- 坐标与通量整形 ----
    x = np.arange(1, Nx + 1) * hx
    y = np.arange(1, Ny + 1) * hy
    z = np.arange(1, Nz + 1) * hz
    # Flux is stored as (z, y, x), so coordinate arrays must use that layout.
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    phi1_3d = phi[:N_total].reshape(Nz, Ny, Nx)
    phi2_3d = phi[N_total:].reshape(Nz, Ny, Nx)

    return {
        'k_eff': k_eff,
        'X': X, 'Y': Y, 'Z': Z,
        'x': x, 'y': y, 'z': z,
        'phi1': phi1_3d,
        'phi2': phi2_3d,
        'n_iter': result['n_iter'],
        'k_history': result['k_history'],
        'residual': result['residual'],
        'converged': result['converged'],
        'delta_k': result['delta_k'],
        'termination_reason': result['termination_reason'],
    }


def validate_3d_laplacian(Nx=6, Ny=6, Nz=6, Lx=200.0, Ly=200.0, Lz=200.0):
    """
    验证 3D Laplacian 的特征值与解析公式一致。

    3D 离散 Laplacian 的特征值：
        λ_{p,q,r} = 4 sin²(pπ/(2(Nx+1)))/hx² + 4 sin²(qπ/(2(Ny+1)))/hy²
                   + 4 sin²(rπ/(2(Nz+1)))/hz²

    用于验证矩阵构造正确性（debug_demo 方法论在 3D 的延续）。
    """
    hx, hy, hz = Lx / (Nx + 1), Ly / (Ny + 1), Lz / (Nz + 1)

    L_3d = _build_3d_laplacian(Nx, Ny, Nz, hx, hy, hz)
    L_dense = L_3d.toarray()

    lambda_num = np.sort(np.linalg.eigvalsh(L_dense))

    # 解析特征值
    p_idx = np.arange(1, Nx + 1)
    q_idx = np.arange(1, Ny + 1)
    r_idx = np.arange(1, Nz + 1)

    Lp = 4.0 * np.sin(p_idx * np.pi / (2 * (Nx + 1))) ** 2 / (hx * hx)
    Lq = 4.0 * np.sin(q_idx * np.pi / (2 * (Ny + 1))) ** 2 / (hy * hy)
    Lr = 4.0 * np.sin(r_idx * np.pi / (2 * (Nz + 1))) ** 2 / (hz * hz)

    lambda_analytic = np.sort(
        (Lp[:, None, None] + Lq[None, :, None] + Lr[None, None, :]).ravel()
    )

    max_err = np.max(np.abs(lambda_num - lambda_analytic))
    return {
        'max_error': max_err,
        'passed': max_err < 1e-12,
        'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
        'n_total': Nx * Ny * Nz,
    }
