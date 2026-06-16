"""
双群一维中子扩散方程求解器
Two-group 1D neutron diffusion solver using FDM + power iteration.
"""
import numpy as np
import matplotlib.pyplot as plt

# ====== 几何和网格 ======
L = 200.0          # 平板长度 (cm)
N = 200            # 内部网格点数 (加密网格)
h = L / N          # 网格间距
x = np.linspace(h/2, L - h/2, N)

# ====== 双群截面 (典型 PWR 数据) ======
# 群 1 - 快群
D1 = 1.2           # 扩散系数 (cm)
nu_Sf1 = 0.003     # νΣ_f (cm⁻¹)
Sa1 = 0.008        # 吸收截面 Σ_a1 (cm⁻¹)
Ss12 = 0.020       # 散射截面 Σ_{s,1→2} (cm⁻¹)
Sr1 = Sa1 + Ss12   # 移出截面 Σ_r1

# 群 2 - 热群
D2 = 0.4           # 扩散系数 (cm)
nu_Sf2 = 0.105     # νΣ_f (cm⁻¹) — 略增使系统近临界
Sa2 = 0.08         # 吸收截面 Σ_a2 (cm⁻¹)

# ====== 构造离散拉普拉斯矩阵 L (N×N) ======
# 二阶中心差分: ∇²φ ≈ (φ_{i-1} - 2φ_i + φ_{i+1}) / h²
coeff1 = D1 / (h * h)
coeff2 = D2 / (h * h)

main_diag = 2.0 * np.ones(N)
off_diag  = -1.0 * np.ones(N - 1)
L_mat = (np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1))

# ====== 组装 2N×2N 分块矩阵 ======
# A · φ = (1/k) · F · φ
# 块排列: 前 N 行 → 群1, 后 N 行 → 群2

# ─ A 矩阵 ─
A11 = coeff1 * L_mat + Sr1 * np.eye(N)     # 快群: 泄漏 + 移出
A12 = np.zeros((N, N))                      # 群1不受群2直接影响
A21 = -Ss12 * np.eye(N)                    # 热群源项 (散射耦合)
A22 = coeff2 * L_mat + Sa2 * np.eye(N)     # 热群: 泄漏 + 吸收

A_top = np.hstack([A11, A12])
A_bot = np.hstack([A21, A22])
A = np.vstack([A_top, A_bot])

# ─ F 矩阵 (裂变源) ─
F11 = nu_Sf1 * np.eye(N)
F12 = nu_Sf2 * np.eye(N)
F21 = np.zeros((N, N))
F22 = np.zeros((N, N))

F_top = np.hstack([F11, F12])
F_bot = np.hstack([F21, F22])
F = np.vstack([F_top, F_bot])

# ====== 幂迭代 ======
phi = np.ones(2 * N)    # 初始通量 (前N=群1, 后N=群2)

for it in range(200):
    source = F @ phi                             # 当前裂变源
    phi_new = np.linalg.solve(A, source)         # 求解新通量
    k_new = np.sum(F @ phi_new) / np.sum(source)  # k = 新/旧 裂变中子数
    phi = phi_new / np.max(phi_new)              # 归一化
    if it % 20 == 0:
        print(f"  iter {it}: k = {k_new:.6f}")

# ====== 拆分通量 ======
phi1 = phi[:N]          # 快群通量
phi2 = phi[N:]          # 热群通量

print(f"k_eff = {k_new:.6f}")
print(f"快群通量峰值: {np.max(phi1):.4f}  at x = {x[np.argmax(phi1)]:.1f} cm")
print(f"热群通量峰值: {np.max(phi2):.4f}  at x = {x[np.argmax(phi2)]:.1f} cm")

# ====== 可视化 ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, phi1, 'r-', linewidth=2, label='Fast (Group 1)')
ax1.plot(x, phi2, 'b-', linewidth=2, label='Thermal (Group 2)')
ax1.set_xlabel('Position (cm)')
ax1.set_ylabel('Neutron Flux (normalized)')
ax1.set_title(f'2-Group Flux Distribution, k_eff = {k_new:.6f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x, phi2 / phi1, 'g-', linewidth=2)
ax2.set_xlabel('Position (cm)')
ax2.set_ylabel('Thermal / Fast Ratio')
ax2.set_title(r'Flux Ratio $\phi_2 / \phi_1$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('flux_2g.png', dpi=150)
print("图片已保存到 flux_2g.png")
