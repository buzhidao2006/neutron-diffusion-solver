"""
二维双群中子扩散方程 — 独立运行脚本
输出 k_eff 和 2D 通量分布图 (flux_2d.png)

用法：python diffusion_2d.py
"""
import numpy as np
import matplotlib.pyplot as plt
from solver_2d import solve_two_group_2d

# ===== 默认参数 =====
Lx = 200.0     # x 方向边长 (cm)
Ly = 200.0     # y 方向边长 (cm)
Nx = 60        # x 方向网格点
Ny = 60        # y 方向网格点

print("=" * 55)
print("  二维双群中子扩散方程求解器")
print("  矩形几何 · 5 点有限差分 · 幂迭代")
print("=" * 55)
print(f"\n  边长: Lx = {Lx:.0f} cm,  Ly = {Ly:.0f} cm")
print(f"  网格: Nx = {Nx},   Ny = {Ny}")
print(f"  未知数: {2 * Nx * Ny} 个 (双群)")

# ===== 求解 =====
print("\n  幂迭代中...")
result = solve_two_group_2d(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)

k_eff = result['k_eff']
X, Y = result['X'], result['Y']
phi1 = result['phi1']
phi2 = result['phi2']

print(f"\n  k_eff = {k_eff:.6f}")
print(f"  快群通量峰值: {np.max(phi1):.4f}")
print(f"  热群通量峰值: {np.max(phi2):.4f}")
print(f"  热/快比 (平均): {np.mean(phi2 / (phi1 + 1e-12)):.2f}")

# ===== 解析 buckling 对比 =====
from solver import DEFAULTS
p = DEFAULTS
Sr1 = p['Sa1'] + p['Ss12']
k_inf = p['nu_Sf1'] / Sr1 + (p['nu_Sf2'] / p['Sa2']) * (p['Ss12'] / Sr1)
L2 = p['D2'] / p['Sa2']
tau = p['D1'] / Sr1
M2 = L2 + tau
B2_sq = (np.pi / Lx)**2 + (np.pi / Ly)**2
k_buckling = k_inf / (1 + M2 * B2_sq)
print(f"\n  Buckling 近似: k ≈ {k_buckling:.6f}  "
      f"(k_inf={k_inf:.4f}, M²={M2:.1f} cm², B²={B2_sq:.6f})")
print(f"  相对偏差: {abs(k_eff - k_buckling) / k_eff * 100:.2f}%")

# ===== 绘图 =====
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 图 1: 快群通量 heatmap
ax = axes[0, 0]
im1 = ax.contourf(X, Y, phi1, levels=20, cmap='Reds')
ax.set_xlabel('x (cm)', fontsize=11)
ax.set_ylabel('y (cm)', fontsize=11)
ax.set_title(f'Fast Flux (Group 1)  |  peak={np.max(phi1):.3f}', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
plt.colorbar(im1, ax=ax, shrink=0.8)

# 图 2: 热群通量 heatmap
ax = axes[0, 1]
im2 = ax.contourf(X, Y, phi2, levels=20, cmap='Blues')
ax.set_xlabel('x (cm)', fontsize=11)
ax.set_ylabel('y (cm)', fontsize=11)
ax.set_title(f'Thermal Flux (Group 2)  |  peak={np.max(phi2):.3f}', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
plt.colorbar(im2, ax=ax, shrink=0.8)

# 图 3: 热/快比 contour
ax = axes[1, 0]
ratio = phi2 / (phi1 + 1e-12)
levels_ratio = np.linspace(np.min(ratio), np.max(ratio), 20)
im3 = ax.contourf(X, Y, ratio, levels=levels_ratio, cmap='RdYlBu_r')
ax.set_xlabel('x (cm)', fontsize=11)
ax.set_ylabel('y (cm)', fontsize=11)
ax.set_title(f'Thermal / Fast Ratio  '
             f'(min={np.min(ratio):.2f}, max={np.max(ratio):.2f})', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
plt.colorbar(im3, ax=ax, shrink=0.8)

# 图 4: 中心线剖面 (y = Ly/2)
ax = axes[1, 1]
mid_y_idx = Ny // 2
x_1d = result['x']
ax.plot(x_1d, phi1[mid_y_idx, :], '#e74c3c', linewidth=2, label='Fast (y = Ly/2)')
ax.plot(x_1d, phi2[mid_y_idx, :], '#3498db', linewidth=2, label='Thermal (y = Ly/2)')
ax.set_xlabel('x (cm)', fontsize=11)
ax.set_ylabel('Normalized Flux', fontsize=11)
ax.set_title(f'Centerline Profile (y = {Ly/2:.0f} cm)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.25)

plt.suptitle(f'2D Two-Group Neutron Diffusion  |  '
             f'k_eff = {k_eff:.6f}  |  '
             f'L = {Lx:.0f}×{Ly:.0f} cm²  |  '
             f'Grid = {Nx}×{Ny}',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('flux_2d.png', dpi=150, bbox_inches='tight')
print(f"\n  图已保存: flux_2d.png")
