"""
三维双群中子扩散 — 演示与可视化脚本

直接运行: python3 diffusion_3d.py
"""
import numpy as np
import matplotlib.pyplot as plt
from solver_3d import solve_two_group_3d, validate_3d_laplacian

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


def demo_3d_solution(Lx=160, Ly=160, Lz=160, Nx=25, Ny=25, Nz=25):
    """三维求解 + 多切片可视化。"""
    print(f"  求解 3D 双群扩散: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz} 网格点 "
          f"(双群共 {2*Nx*Ny*Nz} 未知数)...")

    result = solve_two_group_3d(Lx=Lx, Ly=Ly, Lz=Lz,
                                Nx=Nx, Ny=Ny, Nz=Nz, method='chebyshev')

    print(f"  k_eff = {result['k_eff']:.6f}  ({result['n_iter']} iterations)")
    return result


def plot_3d_slices(result):
    """绘制三个正交中心切片的通量 heatmap + 中心线剖面。"""
    phi1 = result['phi1']
    phi2 = result['phi2']
    x, y, z = result['x'], result['y'], result['z']

    Nx, Ny, Nz = len(x), len(y), len(z)
    mx, my, mz = Nx // 2, Ny // 2, Nz // 2

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ---- 上排: 快群 ----
    im = axes[0, 0].contourf(x, y, phi1[mz, :, :], levels=20, cmap='Reds')
    axes[0, 0].set_title(f'Fast Flux: z={z[mz]:.0f} cm slice')
    axes[0, 0].set_xlabel('x (cm)'); axes[0, 0].set_ylabel('y (cm)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    im = axes[0, 1].contourf(x, z, phi1[:, my, :], levels=20, cmap='Reds')
    axes[0, 1].set_title(f'Fast Flux: y={y[my]:.0f} cm slice')
    axes[0, 1].set_xlabel('x (cm)'); axes[0, 1].set_ylabel('z (cm)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    im = axes[0, 2].contourf(y, z, phi1[:, :, mx], levels=20, cmap='Reds')
    axes[0, 2].set_title(f'Fast Flux: x={x[mx]:.0f} cm slice')
    axes[0, 2].set_xlabel('y (cm)'); axes[0, 2].set_ylabel('z (cm)')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)

    # ---- 下排: 热群 ----
    im = axes[1, 0].contourf(x, y, phi2[mz, :, :], levels=20, cmap='Blues')
    axes[1, 0].set_title(f'Thermal Flux: z={z[mz]:.0f} cm slice')
    axes[1, 0].set_xlabel('x (cm)'); axes[1, 0].set_ylabel('y (cm)')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    im = axes[1, 1].contourf(x, z, phi2[:, my, :], levels=20, cmap='Blues')
    axes[1, 1].set_title(f'Thermal Flux: y={y[my]:.0f} cm slice')
    axes[1, 1].set_xlabel('x (cm)'); axes[1, 1].set_ylabel('z (cm)')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

    im = axes[1, 2].contourf(y, z, phi2[:, :, mx], levels=20, cmap='Blues')
    axes[1, 2].set_title(f'Thermal Flux: x={x[mx]:.0f} cm slice')
    axes[1, 2].set_xlabel('y (cm)'); axes[1, 2].set_ylabel('z (cm)')
    axes[1, 2].set_aspect('equal')
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

    fig.suptitle(
        f'3D Two-Group Diffusion Flux  |  k_eff = {result["k_eff"]:.6f}',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    plt.savefig('flux_3d.png', bbox_inches='tight', facecolor='white')
    print("  保存: flux_3d.png")
    plt.close()


def plot_3d_centerline(result):
    """绘制穿过中心的三个方向的通量剖面，与理论 sin 曲线对比。"""
    phi2 = result['phi2']
    x, y, z = result['x'], result['y'], result['z']
    Nx, Ny, Nz = len(x), len(y), len(z)
    mx, my, mz = Nx // 2, Ny // 2, Nz // 2
    Lx, Ly, Lz = x[-1] - x[0] + x[1] - x[0], y[-1] - y[0] + y[1] - y[0], z[-1] - z[0] + z[1] - z[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, coord, data, half_range, label, L_val in [
        (axes[0], x, phi2[mz, my, :], mx, 'x', Lx),
        (axes[1], y, phi2[mz, :, mx], my, 'y', Ly),
        (axes[2], z, phi2[:, my, mx], mz, 'z', Lz),
    ]:
        ax.plot(coord, data, 'b-', linewidth=2, label='FDM (thermal)')
        # 叠加理论 sin 形状
        theory = np.sin(np.pi * coord / (coord[-1] - coord[0] + coord[1] - coord[0]))
        theory = theory / theory.max() * data.max()
        ax.plot(coord, theory, 'r--', linewidth=1.5, alpha=0.7, label='sin(pi*coord/L)')
        ax.set_xlabel(f'{label} (cm)')
        ax.set_ylabel('Thermal Flux')
        ax.set_title(f'Centerline profile along {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('3D Flux Profiles — Comparison with sin Shape', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('flux_3d_profiles.png', bbox_inches='tight', facecolor='white')
    print("  保存: flux_3d_profiles.png")
    plt.close()


def demo_grid_convergence():
    """网格收敛性测试: N=15, 20, 25 的 k_eff 变化。"""
    print("\n  网格收敛性测试 (cube 160 cm, N=15→20→25):")
    grids = [(15, 15, 15), (20, 20, 20), (25, 25, 25)]
    prev_k = None
    for Nx, Ny, Nz in grids:
        result = solve_two_group_3d(Lx=160, Ly=160, Lz=160,
                                    Nx=Nx, Ny=Ny, Nz=Nz, method='chebyshev')
        delta = f"Δk = {result['k_eff'] - prev_k:+.6f}" if prev_k else ""
        print(f"    N={Nx}³: k_eff={result['k_eff']:.6f} "
              f"({result['n_iter']:3d} iter) {delta}")
        prev_k = result['k_eff']


# ============ 主程序 ============

if __name__ == '__main__':
    print("=" * 60)
    print("  三维双群中子扩散求解器")
    print("=" * 60)

    # 验证 1: Laplacian 特征值
    print("\n[验证] 3D Laplacian 特征值...")
    val = validate_3d_laplacian(Nx=5, Ny=5, Nz=5)
    print(f"  网格 {val['Nx']}x{val['Ny']}x{val['Nz']} ({val['n_total']} points)")
    print(f"  最大误差: {val['max_error']:.2e}")
    print(f"  {'✅ 通过' if val['passed'] else '❌ 失败'} — "
          f"3D Laplacian 矩阵构造{'正确' if val['passed'] else '有误'}")

    # 验证 2: 求解 + 可视化
    print("\n[求解] 3D 双群扩散...")
    result = demo_3d_solution(Lx=160, Ly=160, Lz=160, Nx=25, Ny=25, Nz=25)
    plot_3d_slices(result)
    plot_3d_centerline(result)

    # 验证 3: 网格收敛性
    demo_grid_convergence()

    # Buckling 对比
    p = {'D1': 1.2, 'nu_Sf1': 0.003, 'Sa1': 0.008, 'Ss12': 0.020,
         'D2': 0.4, 'nu_Sf2': 0.105, 'Sa2': 0.08}
    Sr1 = p['Sa1'] + p['Ss12']
    k_inf = p['nu_Sf1'] / Sr1 + (p['nu_Sf2'] / p['Sa2']) * (p['Ss12'] / Sr1)
    M2 = p['D2'] / p['Sa2'] + p['D1'] / Sr1
    B2_3d = 3.0 * (np.pi / 160)**2
    k_buckling = k_inf / (1 + M2 * B2_3d)
    print(f"\n  Buckling 近似: k = k_inf/(1+M²B²) = {k_buckling:.6f}")
    print(f"  (FDM 数值解: {result['k_eff']:.6f})")
    print(f"  偏差: {abs(result['k_eff'] - k_buckling) / result['k_eff'] * 100:.2f}%")
    print("\n完成!")
