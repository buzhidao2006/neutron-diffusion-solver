"""
二维扩散求解器 — 解析验证脚本

验证内容：
  1. 2D 离散 Laplacian 特征值与解析公式对比（矩阵构造正确性）
  2. k_eff 与 buckling 近似公式对比
  3. 通量形状与 sin(πx/Lx)·sin(πy/Ly) 的相似度

这是 debug_demo.py 的二维对应版本。
"""
import numpy as np
from scipy.sparse.linalg import eigsh
from solver_2d import _build_2d_laplacian, solve_two_group_2d
from solver import DEFAULTS

print("=" * 60)
print("  二维扩散求解器 — 解析验证")
print("=" * 60)

# ================================================================
# 验证 1：2D Laplacian 特征值
# ================================================================
print("\n[验证 1] 2D 离散 Laplacian 特征值")

# 用小网格可以直接算 dense 特征值
Nx, Ny = 8, 8
Lx, Ly = 200.0, 200.0
hx, hy = Lx / Nx, Ly / Ny

L_2d = _build_2d_laplacian(Nx, Ny, hx, hy)
L_dense = L_2d.toarray()

# 数值特征值
lambda_num = np.sort(np.linalg.eigvalsh(L_dense))

# 解析特征值 (2D)
# λ_{p,q} = 4 sin²(pπ/(2(Nx+1))) / hx² + 4 sin²(qπ/(2(Ny+1))) / hy²
p_idx = np.arange(1, Nx + 1)
q_idx = np.arange(1, Ny + 1)
lambda_p = 4.0 * np.sin(p_idx * np.pi / (2 * (Nx + 1))) ** 2 / (hx * hx)
lambda_q = 4.0 * np.sin(q_idx * np.pi / (2 * (Ny + 1))) ** 2 / (hy * hy)
lambda_analytic = np.sort(
    (lambda_p[:, None] + lambda_q[None, :]).ravel()
)

max_err = np.max(np.abs(lambda_num - lambda_analytic))
print(f"  网格 {Nx}×{Ny} (共 {Nx*Ny} 个特征值)")
print(f"  特征值范围: [{lambda_analytic[0]:.6f}, {lambda_analytic[-1]:.6f}]")
print(f"  最大误差: {max_err:.2e}")
if max_err < 1e-12:
    print("  ✅ 通过 — 2D Laplacian 矩阵构造完全正确")
else:
    print(f"  ❌ 失败 — 误差过大 ({max_err:.2e})")
    exit(1)

# 显示前几个特征值（验证趋势）
print(f"\n  前 5 个特征值对比:")
print(f"  {'解析':>14s}  {'数值':>14s}  {'误差':>14s}")
print(f"  {'-'*44}")
for k in range(min(5, Nx*Ny)):
    print(f"  {lambda_analytic[k]:14.6f}  {lambda_num[k]:14.6f}  "
          f"{abs(lambda_analytic[k] - lambda_num[k]):14.2e}")

# ================================================================
# 验证 2：单群特征值与特征向量
# ================================================================
print("\n[验证 2] 单群扩散矩阵特征值")

D = DEFAULTS['D1']
Sigma_a = DEFAULTS['Sa1']
A_sg = D * L_2d + Sigma_a * np.eye(Nx * Ny)
A_sg_dense = A_sg.toarray() if hasattr(A_sg, 'toarray') else A_sg
alpha_num = np.sort(np.linalg.eigvalsh(A_sg_dense))
alpha_analytic = np.sort(D * lambda_analytic + Sigma_a)

max_err_alpha = np.max(np.abs(alpha_num - alpha_analytic))
print(f"  单群矩阵 = D·L_2d + Σa·I")
print(f"  特征值范围: [{alpha_analytic[0]:.6f}, {alpha_analytic[-1]:.6f}]")
print(f"  最大误差: {max_err_alpha:.2e}")
if max_err_alpha < 1e-12:
    print("  ✅ 通过")
else:
    print(f"  ❌ 失败 ({max_err_alpha:.2e})")
    exit(1)

# 验证最小特征值对应的特征向量形状
eigvals, eigvecs = np.linalg.eigh(A_sg_dense)
min_idx = np.argmin(eigvals)
phi_fund = eigvecs[:, min_idx].reshape(Ny, Nx)

# 理论基模形状：sin(πx/Lx)·sin(πy/Ly)
x_cells = np.linspace(hx / 2, Lx - hx / 2, Nx)
y_cells = np.linspace(hy / 2, Ly - hy / 2, Ny)
Xc, Yc = np.meshgrid(x_cells, y_cells)
phi_theory = np.sin(np.pi * Xc / Lx) * np.sin(np.pi * Yc / Ly)
phi_theory = phi_theory / np.max(phi_theory)

# 归一化后比较（eigh 返回 L2=1 的向量，需统一到 max=1；转为 1D array 避免 matrix 类型问题）
phi_fund_norm = np.abs(phi_fund) / np.max(np.abs(phi_fund))
phi_fund_flat = np.asarray(phi_fund_norm).ravel()
phi_theory_flat = phi_theory.ravel()

# 余弦相似度
dot = np.dot(phi_fund_flat, phi_theory_flat)
norm_fund = np.linalg.norm(phi_fund_flat)
norm_theory = np.linalg.norm(phi_theory_flat)
cos_sim = dot / (norm_fund * norm_theory)

print(f"\n  基本模态与 sin·sin 的余弦相似度: {cos_sim:.8f}")
if cos_sim > 0.9999:
    print("  ✅ 通过 — 通量形状与理论一致")
else:
    print(f"  ⚠️  相似度偏低 ({cos_sim:.6f})，检查网格或边界")

# ================================================================
# 验证 3：双群 k_eff 与 buckling 近似
# ================================================================
print("\n[验证 3] 双群 k_eff vs Buckling 近似")

# 用适中的网格跑 2D 双群求解
result = solve_two_group_2d(Lx=200, Ly=200, Nx=60, Ny=60)
k_num = result['k_eff']

# Buckling 近似
p = DEFAULTS
Sr1 = p['Sa1'] + p['Ss12']
k_inf = p['nu_Sf1'] / Sr1 + (p['nu_Sf2'] / p['Sa2']) * (p['Ss12'] / Sr1)
L2 = p['D2'] / p['Sa2']
tau = p['D1'] / Sr1
M2 = L2 + tau
B2 = (np.pi / 200)**2 + (np.pi / 200)**2  # 方形 200×200
k_buckling = k_inf / (1 + M2 * B2)

rel_err = abs(k_num - k_buckling) / k_num * 100
print(f"  k_eff (FDM 数值):     {k_num:.6f}")
print(f"  k_eff (Buckling):     {k_buckling:.6f}")
print(f"  相对偏差:             {rel_err:.4f}%")
print(f"  k_inf = {k_inf:.4f},  M² = {M2:.1f} cm²,  B² = {B2:.6f}")

if rel_err < 5.0:
    print("  ✅ 通过 — 二维双群求解器物理合理")
else:
    print(f"  ⚠️  偏差 > 5%，检查扩散系数或网格收敛性")

# ================================================================
# 验证 4：网格收敛性
# ================================================================
print("\n[验证 4] 网格收敛性测试")

grids = [(30, 30), (40, 40), (50, 50), (60, 60)]
k_results = []
for Nx, Ny in grids:
    r = solve_two_group_2d(Lx=200, Ly=200, Nx=Nx, Ny=Ny)
    k_results.append(r['k_eff'])
    print(f"  {Nx}×{Ny} → k = {r['k_eff']:.8f}  "
          f"(Δk from finest = {abs(r['k_eff'] - k_results[-1]) if len(k_results) > 1 else 0:.2e})")

# 检查是否收敛（最后两个网格的 k 差应该很小）
if len(k_results) >= 2:
    dk = abs(k_results[-1] - k_results[-2])
    if dk < 1e-3:
        print(f"  ✅ 收敛良好 (Δk = {dk:.2e})")
    else:
        print(f"  ⚠️  收敛较慢 (Δk = {dk:.2e})，建议增大网格")

print("\n" + "=" * 60)
print("  验证完成！")
print("=" * 60)
