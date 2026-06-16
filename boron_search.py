"""
临界硼浓度搜索 — PWR 反应堆控制核心算法

问题：给定堆芯尺寸 L，求使 k=1 的硼浓度 C_B

算法：外迭代（二分法）+ 内迭代（幂迭代）
     外循环：二分法搜索 C_B，使 k(C_B) - 1 = 0
     内循环：给定 C_B，幂迭代算 k_eff

物理：硼 (B-10) 是热中子强吸收体，溶解在冷却剂中
     通过调节硼浓度来控制反应性
"""
import numpy as np
import matplotlib.pyplot as plt

# ====== 双群截面 ======
D1, nu_Sf1, Sa1, Ss12 = 1.2, 0.003, 0.008, 0.020
Sr1 = Sa1 + Ss12
D2, nu_Sf2, Sa2_base = 0.4, 0.105, 0.08

# ====== 几何 ======
L = 200.0          # 固定堆芯尺寸 (cm)
N = 150
h = L / N

# ====== 硼参数 ======
# 硼的微观吸收截面 ~3837 barns (B-10), 天然硼含 19.9% B-10
# 1 ppm 天然硼 ≈ 1e-5 cm⁻¹ 的热群宏观吸收截面增量
alpha = 1.0e-5     # d(Σ_a2)/dC_B  [(cm⁻¹)/ppm]


def solve_keff(Sa2_eff):
    """给定热群吸收截面，返回 k_eff"""
    coeff1 = D1 / (h * h)
    coeff2 = D2 / (h * h)

    main = 2.0 * np.ones(N)
    off = -1.0 * np.ones(N - 1)
    L_mat = np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)

    A11 = coeff1 * L_mat + Sr1 * np.eye(N)
    A12 = np.zeros((N, N))
    A21 = -Ss12 * np.eye(N)
    A22 = coeff2 * L_mat + Sa2_eff * np.eye(N)
    A = np.vstack([np.hstack([A11, A12]), np.hstack([A21, A22])])

    F11 = nu_Sf1 * np.eye(N)
    F12 = nu_Sf2 * np.eye(N)
    F = np.vstack([np.hstack([F11, F12]), np.zeros((N, 2 * N))])

    phi = np.ones(2 * N)
    for _ in range(80):
        source = F @ phi
        phi_new = np.linalg.solve(A, source)
        k_new = np.sum(F @ phi_new) / np.sum(source)
        phi = phi_new / np.max(phi_new)
    return k_new


# ====== 二分法搜索临界硼浓度 ======
print("临界硼浓度搜索")
print("=" * 50)

# 先扫描一遍看 k 随 C_B 的变化
C_scan = np.linspace(0, 3000, 11)
k_scan = []
print("扫描 k(C_B):")
for C in C_scan:
    k = solve_keff(Sa2_base + alpha * C)
    k_scan.append(k)
    print(f"  C_B = {C:5.0f} ppm  →  k = {k:.6f}")

k_scan = np.array(k_scan)

# 用二分法精确找 k=1
# 前提：k(C_B) 单调递减（硼越多，吸收越多，k 越小）
# 区间 [C_low, C_high] 满足 k(C_low) > 1 > k(C_high)
C_low, C_high = 200.0, 1500.0

if solve_keff(Sa2_base + alpha * C_low) < 1:
    C_low = 0.0
if solve_keff(Sa2_base + alpha * C_high) > 1:
    C_high = 3000.0

print(f"\n二分法: 初始区间 [{C_low:.0f}, {C_high:.0f}] ppm")
print(f"  k({C_low:.0f}) = {solve_keff(Sa2_base + alpha * C_low):.6f}")
print(f"  k({C_high:.0f}) = {solve_keff(Sa2_base + alpha * C_high):.6f}")
print()

for it in range(30):
    C_mid = (C_low + C_high) / 2
    k_mid = solve_keff(Sa2_base + alpha * C_mid)

    if abs(k_mid - 1.0) < 1e-6:
        break

    if k_mid > 1.0:
        C_low = C_mid
    else:
        C_high = C_mid

    if it < 10 or it % 5 == 4:
        print(f"  iter {it:2d}: C_B ∈ [{C_low:8.2f}, {C_high:8.2f}], "
              f"中点 k={k_mid:.8f}")

C_crit = (C_low + C_high) / 2
k_final = solve_keff(Sa2_base + alpha * C_crit)

print(f"\n结果:")
print(f"  临界硼浓度 C_B* = {C_crit:.1f} ppm")
print(f"  对应 k_eff        = {k_final:.8f}")
print(f"  Σ_a2(0)          = {Sa2_base:.4f} cm⁻¹")
print(f"  Σ_a2(C_B*)       = {Sa2_base + alpha*C_crit:.4f} cm⁻¹")
print(f"  ΔΣ_a2            = {alpha*C_crit:.4f} cm⁻¹")

# ====== 硼价值 ======
# 微分硼价值: dρ/dC_B ≈ (dk/k²)/dC_B
dC = 10.0  # ppm
k_plus = solve_keff(Sa2_base + alpha * (C_crit + dC))
k_minus = solve_keff(Sa2_base + alpha * (C_crit - dC))
# 反应性 ρ = (k-1)/k ≈ Δk/k
rho_per_ppm = (k_minus - k_plus) / (2 * dC) / k_final**2 * 1e5
print(f"\n  硼微分价值 ≈ {rho_per_ppm:.1f} pcm/ppm")
print(f"  (每增加 1 ppm 硼, 反应性减少 {rho_per_ppm:.1f} pcm)")

# ====== 可视化 ======
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(C_scan, k_scan, 'b.-', linewidth=2, markersize=10, label='k(C_B)')
ax1.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='k=1')
ax1.axvline(x=C_crit, color='r', linestyle='--', linewidth=1.5,
            label=f'C_B* = {C_crit:.0f} ppm')
ax1.set_xlabel('Boron Concentration (ppm)')
ax1.set_ylabel('k_eff')
ax1.set_title('Critical Boron Search — PWR Reactivity Control')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右轴：反应性
ax2 = ax1.twinx()
rho_scan = (k_scan - 1) / k_scan * 1e5   # pcm
ax2.plot(C_scan, rho_scan, 'g--', linewidth=1, alpha=0.5)
ax2.set_ylabel('Reactivity (pcm)', color='g')
ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)

plt.tight_layout()
plt.savefig('boron_search.png', dpi=150)
print("\n图片已保存到 boron_search.png")
