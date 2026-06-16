"""
临界尺寸扫描：k_eff vs 平板厚度 L

对给定材料，找出使 k=1 的临界尺寸。
同时对比解析 buckling 公式，加深对临界条件的理解。
"""
import numpy as np
import matplotlib.pyplot as plt

# ====== 双群截面（固定，即"给定材料"）======
D1, nu_Sf1, Sa1, Ss12 = 1.2, 0.003, 0.008, 0.020
Sr1 = Sa1 + Ss12
D2, nu_Sf2, Sa2 = 0.4, 0.105, 0.08

# ====== 扫描参数 ======
L_vals = np.linspace(40, 400, 36)   # 不同平板厚度
N = 150                              # 每个厚度用 150 个网格
k_numerical = []

for L in L_vals:
    h = L / N
    coeff1 = D1 / (h * h)
    coeff2 = D2 / (h * h)

    # 拉普拉斯矩阵
    main = 2.0 * np.ones(N)
    off  = -1.0 * np.ones(N - 1)
    L_mat = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)

    # 组装 2N×2N 矩阵
    A11 = coeff1 * L_mat + Sr1 * np.eye(N)
    A12 = np.zeros((N, N))
    A21 = -Ss12 * np.eye(N)
    A22 = coeff2 * L_mat + Sa2 * np.eye(N)
    A = np.vstack([np.hstack([A11, A12]), np.hstack([A21, A22])])

    F11 = nu_Sf1 * np.eye(N)
    F12 = nu_Sf2 * np.eye(N)
    F = np.vstack([np.hstack([F11, F12]), np.zeros((N, 2*N))])

    # 幂迭代
    phi = np.ones(2 * N)
    for _ in range(100):
        source = F @ phi
        phi_new = np.linalg.solve(A, source)
        k_new = np.sum(F @ phi_new) / np.sum(source)
        phi = phi_new / np.max(phi_new)

    k_numerical.append(k_new)

k_numerical = np.array(k_numerical)

# ====== 解析 buckling 公式（对比用）======
# 几何曲率: B² = (π/L̃)²，用 L 近似（忽略外推距离）
B2 = (np.pi / L_vals) ** 2

# k_inf — 无穷大介质的增殖因子
k_inf = nu_Sf1 / Sr1 + (nu_Sf2 / Sa2) * (Ss12 / Sr1)

# 徙动面积 M² = L² + τ（扩散面积 + 年龄）
L2 = D2 / Sa2        # 热群扩散面积
tau = D1 / Sr1       # 费米年龄
M2 = L2 + tau        # 徙动面积

# 临界方程: k_eff = k_inf / (1 + M²B²)
k_buckling = k_inf / (1 + M2 * B2)

# ====== 找临界尺寸 ======
# 插值找 k=1 对应的 L
idx = np.argmin(np.abs(k_numerical - 1.0))
L_crit = L_vals[idx]
print(f"k_inf = {k_inf:.4f}")
print(f"徙动面积 M² = {M2:.1f} cm²")
print(f"临界半厚度（数值）≈ {L_crit:.1f} cm")
print(f"临界半厚度（解析）≈ {np.pi * np.sqrt(M2 / (k_inf - 1)):.1f} cm  (仅当 k_inf>1)")

# ====== 画图 ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 左图：k_eff vs L
ax1.plot(L_vals, k_numerical, 'b.-', linewidth=2, markersize=8, label='FDM 数值解')
ax1.plot(L_vals, k_buckling, 'r--', linewidth=2, label='Buckling 近似')
ax1.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='k=1 临界线')
ax1.axvline(x=L_crit, color='gray', linestyle=':', linewidth=1)
ax1.set_xlabel('Slab Thickness L (cm)')
ax1.set_ylabel('k_eff')
ax1.set_title('Critical Size Scan')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：误差分析
error = np.abs(k_numerical - k_buckling) / k_numerical * 100
ax2.semilogy(L_vals, error, 'g.-', linewidth=2, markersize=8)
ax2.set_xlabel('Slab Thickness L (cm)')
ax2.set_ylabel('Relative Error (%)')
ax2.set_title('FDM vs Buckling Approximation — 误差来源')
ax2.grid(True, alpha=0.3)
# 标注误差来源
ax2.annotate('小 L: 网格粗\n(h 大, 离散误差大)',
             xy=(60, error[0]), fontsize=9, color='gray')
ax2.annotate('大 L: buckling 近似\n(无外推修正)',
             xy=(350, error[-1]), fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('critical_scan.png', dpi=150)
print("图片已保存到 critical_scan.png")

# ====== 关键物理讨论 ======
print(f"""
物理要点：
  k_inf = {k_inf:.4f}  — 无穷大介质的增殖因子
    若 k_inf < 1 → 任何有限尺寸都次临界（不可能达临界）
    若 k_inf > 1 → 存在有限临界尺寸

  临界条件: k_inf / (1 + M²B²) = 1
            → B² = (k_inf - 1)/M² = ({k_inf:.4f} - 1)/{M2:.1f} = {(k_inf-1)/M2:.6f}
            → L_crit = π / sqrt(B²) = {np.pi/np.sqrt((k_inf-1)/M2):.1f} cm

  泄漏的物理含义：
    小尺寸 → 大 B² → 大泄漏 → k 小（次临界）
    增大尺寸 → 减小 B² → 减少泄漏 → k 增大 → 达临界
""")
