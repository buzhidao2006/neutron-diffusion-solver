"""
调试演示：为什么"小网格 + 解析特征值"能定位 bug

问题背景：
  幂迭代跑出的 k_eff 和理论值差了 100 倍。
  可能的 bug 来源：
    A) 矩阵 A 或 F 构造错了？
    B) 幂迭代公式错了？
    C) 两者都错了？

方法：
  用小网格 (N=6) 手动算特征值 → 排除矩阵问题 → 锁定迭代公式
"""
import numpy as np

# ====== 参数 ======
L = 200.0
N = 6              # 故意用小网格，方便手算验证
h = L / N
D, nu_Sf, Sa = 1.0, 0.01, 0.01
coeff = D / (h * h)

# ====== 构造矩阵 A 和 F ======
main_diag = 2 * coeff + Sa
off_diag  = -coeff
A = np.diag(main_diag * np.ones(N)) \
    + np.diag(off_diag * np.ones(N-1), k=1) \
    + np.diag(off_diag * np.ones(N-1), k=-1)
F = nu_Sf * np.eye(N)

print("=" * 55)
print("第 1 步：验证矩阵 A 是否正确")
print("=" * 55)
print(f"A 的形状: {A.shape}")
print(f"A 的三对角结构 (前3行):")
print(A[:3, :3])
print()

# ─── 解析特征值 ───
# 对于 N×N 三对角矩阵 (2, -1, -1)：
#   特征向量: sin(k*π*j/(N+1))  j=1,...,N
#   特征值:   4*sin²(k*π/(2*(N+1)))  k=1,...,N
#
# 我们的 A = coeff * L_mat + Sa * I，所以：
#   α_k = coeff * 4*sin²(k*π/(2*(N+1))) + Sa

k_idx = np.arange(1, N+1)
alpha_analytic = 4 * coeff * np.sin(k_idx * np.pi / (2 * (N + 1)))**2 + Sa
alpha_numeric  = np.sort(np.linalg.eigvalsh(A))  # eigvalsh 适用于对称矩阵

print("A 的特征值对比（解析 vs 数值）：")
for i, (ana, num) in enumerate(zip(alpha_analytic, alpha_numeric)):
    match = "✓" if abs(ana - num) < 1e-12 else "✗ BUG!"
    print(f"  λ{i+1}: 解析={ana:.10f}  数值={num:.10f}  {match}")
print("→ 矩阵 A 构造正确\n")

# ====== 解析算 k_eff ======
# 特征值问题: A φ = (1/k) F φ
# F = νΣ_f * I 是对角阵
# → A φ = (νΣ_f/k) φ
# → k = νΣ_f / α  其中 α 是 A 的特征值
# 基模对应最小 α（最平坦的分布），k 最大
#
# 但注意！对于临界问题，k > 0 且取最大特征值
# A^{-1}F φ = k φ  →  k = 最大的特征值 of A^{-1}F

AF = np.linalg.solve(A, F)
k_analytic = np.sort(np.linalg.eigvals(AF))[::-1]  # 降序排列
print("=" * 55)
print("第 2 步：解析计算真实 k_eff")
print("=" * 55)
print(f"解析 k_eff = {k_analytic[0].real:.6f}")
print(f"次要模态 k  = {k_analytic[1].real:.6f}")
print(f"            = {k_analytic[2].real:.6f}")
print()

# ====== 幂迭代（错误公式） ======
print("=" * 55)
print("第 3 步：对比两种迭代公式")
print("=" * 55)

# 公式 A（原代码的错误写法）
phi = np.ones(N)
k_old = 1.0
print("公式 A [错误]: k_new = k_old * sum(φ_new) / sum(φ)")
for it in range(6):
    source = F @ phi
    phi_new = np.linalg.solve(A, source)
    k_new = k_old * np.sum(phi_new) / np.sum(phi)
    phi = phi_new / np.max(phi_new)
    print(f"  iter {it}: k = {k_new:.6f}")
    k_old = k_new
print(f"  → 漂移到 {k_new:.6f}，一直在下降，永远不收敛到 {k_analytic[0].real:.6f}")

# 为什么？数学推导：
#   收敛时 φ_new = A^{-1}F φ = k_true * φ
#   代入：k_new = k_old * sum(k_true * φ) / sum(φ) = k_old * k_true
#   要求 k_new = k_old → k_old = k_old * k_true → k_true = 1
#   但实际 k_true ≠ 1，所以 k 每步乘以 k_true ≈ 0.98
#   → k_n ≈ 1.0 * (0.98)^n → 指数衰减到 0！
print()

# 公式 B（正确）
phi = np.ones(N)
print(f"公式 B [正确]: k_new = sum(F·φ_new) / sum(F·φ_old)")
for it in range(6):
    source = F @ phi                          # 当前裂变中子源
    phi_new = np.linalg.solve(A, source)      # 下一代谢出的通量
    k_new = np.sum(F @ phi_new) / np.sum(source)  # 新裂变 / 旧裂变
    phi = phi_new / np.max(phi_new)
    print(f"  iter {it}: k = {k_new:.6f}")
print(f"  → 已经收敛到 {k_analytic[0].real:.6f} ✓")

# 为什么正确？
#   物理含义：k = 下一代裂变中子数 / 当前裂变中子数
#   收敛时 φ_new = A^{-1}F φ = k_true * φ
#   k_new = sum(F·(k_true*φ)) / sum(F·φ) = k_true * sum(F·φ) / sum(F·φ) = k_true ✓

print()
print("=" * 55)
print("总结")
print("=" * 55)
print("""
调试三步法：
  1. 小网格 + 解析特征值 → 确认矩阵 A 无 bug
  2. 算 A^{-1}F 的特征值 → 得到"标准答案"
  3. 跑迭代看是否收敛到标准答案 → 不一致则迭代公式有 bug

如果跳过第 1 步直接改迭代公式，
万一是矩阵构造错了，改了也白改。
这就是"隔离变量"的 debug 思路。""")
