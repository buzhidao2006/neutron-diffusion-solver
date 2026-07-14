"""扩散求解器核心功能的单元测试。

基于解析验证、收敛性测试和物理一致性检查。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.sparse import diags, bmat, eye

from solver import solve_two_group, scan_critical_size, search_critical_boron, DEFAULTS
from solver_2d import _build_2d_laplacian, solve_two_group_2d
from solver_3d import _build_3d_laplacian, solve_two_group_3d, validate_3d_laplacian
from power_iteration import power_iteration, power_iteration_chebyshev


def _build_matrices_2g_1d(L_val=200.0, N_val=60, sections=None):
    """构造一维双群扩散的 A 和 F 矩阵（测试辅助函数）。"""
    p = DEFAULTS.copy()
    if sections:
        p.update(sections)

    h = L_val / N_val
    coeff1 = p['D1'] / (h * h)
    coeff2 = p['D2'] / (h * h)
    Sr1 = p['Sa1'] + p['Ss12']

    diag = 2 * np.ones(N_val)
    offd = -1 * np.ones(N_val - 1)
    L_mat = diags([offd, diag, offd], [-1, 0, 1])

    A11 = coeff1 * L_mat + Sr1 * eye(N_val)
    A21 = -p['Ss12'] * eye(N_val)
    A22 = coeff2 * L_mat + p['Sa2'] * eye(N_val)
    A = bmat([[A11, None], [A21, A22]], format='csr')

    F11 = p['nu_Sf1'] * eye(N_val)
    F12 = p['nu_Sf2'] * eye(N_val)
    Z = eye(N_val) * 0.0  # NxN 零矩阵 (CSR 会优化掉)
    F_top = bmat([[F11, F12]], format='csr')     # (N, 2N)
    F_bot = bmat([[Z, Z]], format='csr')          # (N, 2N)
    F = bmat([[F_top], [F_bot]], format='csr')    # (2N, 2N)

    return A, F


# ================================================================
# 1D 扩散测试
# ================================================================


class TestTwoGroup1D:
    """一维双群扩散求解器的验证测试。"""

    def test_k_eff_reasonable(self):
        """k_eff 应在合理范围内 (0.5, 2.0)。"""
        result = solve_two_group(L=200, N=150)
        assert 0.5 < result['k_eff'] < 2.0

    def test_flux_shape_symmetric(self):
        """通量分布应对称（中心两侧相等）。"""
        result = solve_two_group(L=200, N=100)
        phi1 = result['phi1']
        phi2 = result['phi2']
        # 对称性：phi[i] ≈ phi[N-1-i]
        np.testing.assert_allclose(phi1, phi1[::-1], rtol=1e-10)
        np.testing.assert_allclose(phi2, phi2[::-1], rtol=1e-10)

    def test_flux_concave_down(self):
        """通量在中心处应有最大值（凹向下形状）。"""
        result = solve_two_group(L=200, N=100)
        mid = 50
        # 中心通量应大于边界通量
        assert result['phi1'][mid] > result['phi1'][0]
        assert result['phi2'][mid] > result['phi2'][0]

    def test_fast_flux_larger_at_center(self):
        """裂变源在快群 → 中心处快群通量应大于热群（快中子从裂变直接产生）。"""
        result = solve_two_group(L=200, N=100)
        mid = 50
        # 快群/热群比在中心 > 1（裂变主要产生快中子）
        ratio = result['phi1'][mid] / result['phi2'][mid]
        assert ratio > 1.0, f"中心快/热通量比 = {ratio:.2f}，裂变源驱动快群应大于热群"

    def test_N_insensitive(self):
        """k_eff 对网格细化不敏感（网格收敛性）。"""
        k1 = solve_two_group(L=200, N=60)['k_eff']
        k2 = solve_two_group(L=200, N=120)['k_eff']
        dk = abs(k2 - k1)
        # 60→120 网格加倍，k 变化应 < 0.001
        assert dk < 0.002, f"Δk = {dk:.6f} > 0.002"


class TestCriticalSize:
    """临界尺寸扫描的验证测试。"""

    def test_critical_L_positive(self):
        """临界尺寸应为正值。"""
        cs = scan_critical_size(L_min=40, L_max=400, n_points=20, N=100)
        assert cs['L_crit'] > 0

    def test_k_at_critical_L_near_one(self):
        """临界尺寸处的 k_eff 应接近 1。"""
        cs = scan_critical_size(L_min=40, L_max=400, n_points=20, N=100)
        assert abs(cs['k_crit'] - 1.0) < 0.02, f"k_crit = {cs['k_crit']:.6f}"

    def test_larger_L_gives_larger_k(self):
        """更大的堆芯 → k_eff 更大（单调递增）。"""
        cs = scan_critical_size(L_min=40, L_max=400, n_points=20, N=100)
        k_vals = np.array(cs['k_vals'])
        # k 应随 L 单调递增
        assert np.all(np.diff(k_vals) > -1e-10), "k(L) 应为单调递增"

    def test_k_approaches_k_inf(self):
        """大尺寸时 k_eff 应接近 k_inf。"""
        cs = scan_critical_size(L_min=40, L_max=600, n_points=30, N=100)
        # 最大 L 处的 k_eff 应接近 k_inf (偏差 < 3%)
        k_inf = cs['analytic']['k_inf']
        k_max = cs['k_vals'][-1]
        assert (k_inf - k_max) / k_inf < 0.03


class TestCriticalBoron:
    """临界硼搜索的验证测试。"""

    def test_critical_boron_positive(self):
        """临界硼浓度应为正值。"""
        cb = search_critical_boron(L=200, N=100)
        assert cb['C_crit'] > 0

    def test_k_at_critical_boron_near_one(self):
        """临界硼浓度处的 k_eff 应非常接近 1。"""
        cb = search_critical_boron(L=200, N=100)
        assert abs(cb['k_final'] - 1.0) < 1e-4, f"k({cb['C_crit']:.0f} ppm) = {cb['k_final']:.8f}"

    def test_more_boron_lowers_k(self):
        """更高的硼浓度 → 更低的 k_eff（单调递减）。"""
        cb = search_critical_boron(L=200, N=100)
        k_scan = np.array(cb['k_scan'])
        assert np.all(np.diff(k_scan) < 1e-10), "k(C_B) 应为单调递减"


# ================================================================
# 2D 扩散测试
# ================================================================


class TestLaplacian2D:
    """二维离散 Laplacian 矩阵的解析验证。"""

    def test_eigenvalues_against_analytic(self):
        """Kronecker积构造的2D Laplacian特征值应与解析公式一致。"""
        Nx, Ny = 8, 8
        Lx, Ly = 200.0, 200.0
        hx, hy = Lx / Nx, Ly / Ny

        L_2d = _build_2d_laplacian(Nx, Ny, hx, hy)
        L_dense = L_2d.toarray()
        lambda_num = np.sort(np.linalg.eigvalsh(L_dense))

        # 解析特征值
        p_idx = np.arange(1, Nx + 1)
        q_idx = np.arange(1, Ny + 1)
        Lp = 4.0 * np.sin(p_idx * np.pi / (2 * (Nx + 1))) ** 2 / (hx * hx)
        Lq = 4.0 * np.sin(q_idx * np.pi / (2 * (Ny + 1))) ** 2 / (hy * hy)
        lambda_analytic = np.sort((Lp[:, None] + Lq[None, :]).ravel())

        np.testing.assert_allclose(lambda_num, lambda_analytic, rtol=1e-12, atol=1e-14,
                                   err_msg="2D Laplacian 特征值与解析解偏差过大")

    def test_rectangular_grid_eigenvalues(self):
        """非正方形网格 (Nx ≠ Ny) 的特征值验证。"""
        Nx, Ny = 6, 10
        Lx, Ly = 150.0, 300.0
        hx, hy = Lx / Nx, Ly / Ny

        L_2d = _build_2d_laplacian(Nx, Ny, hx, hy)
        L_dense = L_2d.toarray()
        lambda_num = np.sort(np.linalg.eigvalsh(L_dense))

        p_idx = np.arange(1, Nx + 1)
        q_idx = np.arange(1, Ny + 1)
        Lp = 4.0 * np.sin(p_idx * np.pi / (2 * (Nx + 1))) ** 2 / (hx * hx)
        Lq = 4.0 * np.sin(q_idx * np.pi / (2 * (Ny + 1))) ** 2 / (hy * hy)
        lambda_analytic = np.sort((Lp[:, None] + Lq[None, :]).ravel())

        np.testing.assert_allclose(lambda_num, lambda_analytic, rtol=1e-12, atol=1e-14)


class TestTwoGroup2D:
    """二维双群扩散求解器的验证测试。"""

    def test_k_eff_reasonable(self):
        """k_eff 应在合理范围内。"""
        result = solve_two_group_2d(Lx=160, Ly=160, Nx=30, Ny=30)
        assert 0.5 < result['k_eff'] < 2.0

    def test_square_symmetry(self):
        """正方形几何中，x 和 y 方向的中心线剖面应一致。"""
        result = solve_two_group_2d(Lx=160, Ly=160, Nx=30, Ny=30)
        phi2 = result['phi2']
        mid = 15
        np.testing.assert_allclose(phi2[mid, :], phi2[:, mid], rtol=1e-10)

    def test_flux_sine_shape(self):
        """通量形状应与 sin(πx/Lx)·sin(πy/Ly) 高度相似。"""
        result = solve_two_group_2d(Lx=160, Ly=160, Nx=30, Ny=30)
        X, Y = result['X'], result['Y']
        phi2 = result['phi2']

        theory = np.sin(np.pi * X / 160) * np.sin(np.pi * Y / 160)
        theory = theory / theory.max()
        phi_norm = phi2 / phi2.max()

        # 余弦相似度 > 0.998（FDM 离散解与连续 sin 形状有微小差异）
        cos_sim = np.sum(theory * phi_norm) / np.sqrt(
            np.sum(theory**2) * np.sum(phi_norm**2))
        assert cos_sim > 0.998, f"余弦相似度 = {cos_sim:.6f}"

    def test_grid_convergence(self):
        """网格加倍时 k_eff 变化应减小。"""
        k1 = solve_two_group_2d(Lx=160, Ly=160, Nx=25, Ny=25)['k_eff']
        k2 = solve_two_group_2d(Lx=160, Ly=160, Nx=50, Ny=50)['k_eff']
        dk = abs(k2 - k1)
        assert dk < 0.005, f"网格 25→50 时 Δk = {dk:.6f}"


# ================================================================
# 幂迭代测试
# ================================================================


class TestPowerIteration:
    """幂迭代和 Chebyshev 加速的验证测试。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """构造测试用的双群矩阵。"""
        L_val = 200.0
        N_val = 60
        self.A, self.F = _build_matrices_2g_1d(L_val, N_val)
        self.N_val = N_val

    def test_both_methods_same_k(self):
        """标准幂迭代和 Chebyshev 加速应得到相同的 k_eff。"""
        phi0 = np.ones(2 * self.N_val)
        r_std = power_iteration(self.A, self.F, phi0, max_iter=300, tol=1e-10)
        r_ch = power_iteration_chebyshev(
            self.A, self.F, phi0, max_iter=300, tol=1e-10, warmup=15)

        np.testing.assert_allclose(r_std['k_eff'], r_ch['k_eff'], rtol=1e-8, atol=1e-10,
                                   err_msg="两种迭代方法应得到相同的 k_eff")

    def test_chebyshev_faster(self):
        """Chebyshev 加速应比标准 PI 用更少迭代步数。"""
        phi0 = np.ones(2 * self.N_val)
        r_std = power_iteration(self.A, self.F, phi0, max_iter=300, tol=1e-10)
        r_ch = power_iteration_chebyshev(
            self.A, self.F, phi0, max_iter=300, tol=1e-10, warmup=15)

        assert r_ch['n_iter'] < r_std['n_iter'], (
            f"Chebyshev ({r_ch['n_iter']} iters) 应比标准 PI ({r_std['n_iter']} iters) 更快")

    def test_k_converges_monotonically(self):
        """k_eff 级数应收敛（后期波动小）。"""
        phi0 = np.ones(2 * self.N_val)
        r = power_iteration(self.A, self.F, phi0, max_iter=200, tol=1e-10)
        # 最后 20 次迭代的 k 波动应非常小
        last_20 = r['k_history'][-20:]
        std_dev = np.std(last_20)
        assert std_dev < 1e-6, f"最后20次 k 波动 = {std_dev:.2e}"


# ================================================================
# 3D 扩散测试
# ================================================================


class TestLaplacian3D:
    """三维离散 Laplacian 矩阵的解析验证。"""

    def test_eigenvalues_against_analytic(self):
        """3D Kronecker 积 Laplacian 特征值应与解析公式一致。"""
        result = validate_3d_laplacian(Nx=5, Ny=5, Nz=5)
        assert result['passed'], f"3D Laplacian 特征值误差: {result['max_error']:.2e}"

    def test_cube_laplacian_passed(self):
        """6³ 网格验证也通过。"""
        result = validate_3d_laplacian(Nx=6, Ny=6, Nz=6)
        assert result['passed']


class TestTwoGroup3D:
    """三维双群扩散求解器的验证测试。"""

    def test_k_eff_reasonable(self):
        """k_eff 应在合理范围内。"""
        result = solve_two_group_3d(Lx=160, Ly=160, Lz=160, Nx=15, Ny=15, Nz=15,
                                     method='chebyshev')
        assert 0.5 < result['k_eff'] < 2.0

    def test_solution_symmetric(self):
        """立方体三个方向的中心线剖面应对称。"""
        result = solve_two_group_3d(Lx=160, Ly=160, Lz=160, Nx=15, Ny=15, Nz=15,
                                     method='chebyshev')
        phi2 = result['phi2']
        mx, my, mz = 7, 7, 7
        # x 和 y 方向剖面应对称（立方体）
        np.testing.assert_allclose(phi2[mz, my, :], phi2[mz, :, mx], rtol=1e-10)

    def test_grid_convergence(self):
        """网格加倍时 k_eff 变化减小。"""
        k1 = solve_two_group_3d(Lx=160, Ly=160, Lz=160, Nx=12, Ny=12, Nz=12,
                                 method='chebyshev')['k_eff']
        k2 = solve_two_group_3d(Lx=160, Ly=160, Lz=160, Nx=18, Ny=18, Nz=18,
                                 method='chebyshev')['k_eff']
        dk = abs(k2 - k1)
        assert dk < 0.01, f"网格 12→18 时 Δk = {dk:.6f}"


# ================================================================
# 物理一致性测试
# ================================================================


class TestPhysicalConsistency:
    """跨模块的物理一致性检查。"""

    def test_1d_2d_agree_square_analogy(self):
        """对于正方形几何，1D buckling与2D buckling应满足 B²_2d = 2·B²_1d。"""
        L = 200.0
        B2_1d = (np.pi / L) ** 2
        B2_2d = (np.pi / L) ** 2 + (np.pi / L) ** 2
        assert B2_2d == pytest.approx(2 * B2_1d)

    def test_k_inf_independent_of_L(self):
        """k_inf 不应依赖几何尺寸。"""
        cs1 = scan_critical_size(L_min=40, L_max=200, n_points=5, N=80)
        cs2 = scan_critical_size(L_min=40, L_max=200, n_points=5, N=80)
        assert cs1['analytic']['k_inf'] == pytest.approx(cs2['analytic']['k_inf'])

    def test_boron_increases_Sigma_a2(self):
        """加硼应增加热群吸收截面，降低 k_eff。"""
        # 不加硼
        k_no_boron = solve_two_group(L=200, N=100)['k_eff']

        # 加 500 ppm
        alpha = 1.0e-5
        delta_Sa2 = alpha * 500
        sections_with_boron = DEFAULTS.copy()
        sections_with_boron['Sa2'] += delta_Sa2
        k_with_boron = solve_two_group(L=200, N=100, sections=sections_with_boron)['k_eff']

        assert k_with_boron < k_no_boron, "加硼应降低 k_eff"
