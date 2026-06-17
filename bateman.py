"""
Bateman 燃耗方程求解器 — 简化三核素链

核素链:
    ²³⁵U  ──σ_a·φ──→  FP
    ²³⁸U  ──σ_c·φ──→  ²³⁹Pu ──σ_a·φ──→  FP

数值方法: 欧拉步进 (固定时间步长)
物理量: 浓度 N_i (原子/cm³), 燃耗深度 BU (MWd/kgU)
"""
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt


# ====== 物理常数 ======
AVOGADRO = 0.6022     # 阿伏伽德罗常数 (×10²⁴ atoms/mol)
# 燃料: UO₂, 密度 ~10.5 g/cm³, 分子量 ~270
N_UO2 = 0.0234        # UO₂ 数密度 (×10²⁴ molecules/cm³)
N_U_TOTAL = N_UO2     # 初始铀总核子密度

# ====== 默认截面 (barns) ======
# 反应堆物理惯例: N 用 ×10²⁴ atoms/cm³, σ 用 barns (10⁻²⁴ cm²)
# 则 Σ = N × σ 直接得到 cm⁻¹, 单位自动抵消
SIGMA_A_U235 = 680.0    # U235 微观吸收截面 (barns) — 裂变+俘获
SIGMA_C_U238 = 2.7      # U238 微观俘获截面 (barns)
SIGMA_A_PU239 = 1010.0  # Pu239 微观吸收截面 (barns)

# ====== 能量换算 ======
# 每次裂变释放 ~200 MeV = 3.2×10⁻¹¹ J
# 1 MWd = 10⁶ W × 86400 s = 8.64×10¹⁰ J
# 1 MWd / (3.2×10⁻¹¹ J/fission) = 2.7×10²¹ fissions/MWd
FISSIONS_PER_MWD = 2.7e21    # 裂变数 / MWd
# 1 kgU ≈ N_U_TOTAL × 10²⁴ atoms/cm³ × (1 / ρ_UO₂) × 1000 cm³ × (238/270) ...
# 简化: 1 cm³ 燃料含 N_U_TOTAL×10²⁴ 个铀原子
# 1 cm³ UO₂ ≈ 10.5 g, 铀质量占比 238/270 ≈ 0.88 → 约 9.3 gU/cm³
GRAMS_U_PER_CM3 = 9.3


def solve_bateman(initial_enrichment=0.03,    # 初始 U235 富集度
                  flux=1e14,                  # 平均中子通量 (n/cm²/s)
                  total_burnup=50.0,          # 总燃耗深度 (MWd/kgU)
                  n_steps=200):
    """
    求解 Bateman 燃耗方程。

    Parameters
    ----------
    initial_enrichment : float
        初始 U235 富集度 (0.03 = 3%)
    flux : float
        平均中子通量 (n/cm²/s)
    total_burnup : float
        总燃耗深度 (MWd/kgU) — PWR 典型值 40-60
    n_steps : int
        时间步数

    Returns
    -------
    dict: {
        'burnup':  燃耗点序列 (MWd/kgU),
        'time':    时间点序列 (天),
        'N_U235':  U235 浓度序列,
        'N_U238':  U238 浓度序列,
        'N_Pu239': Pu239 浓度序列,
        'N_FP':    裂变产物浓度序列,
        'k_inf_trend':  k_inf 趋势 (简化估算),
    }
    """
    # 初始浓度 (×10²⁴ atoms/cm³)
    N_U235_0 = N_U_TOTAL * initial_enrichment
    N_U238_0 = N_U_TOTAL * (1 - initial_enrichment)
    N_Pu239_0 = 0.0
    N_FP_0 = 0.0

    # 总铀质量 (gU/cm³) — 用于燃耗换算
    # 1 MWd/kgU → 需要多少裂变
    # 1 cm³ 含 GRAMS_U_PER_CM3 gU = GRAMS_U_PER_CM3/1000 kgU
    kgU_per_cm3 = GRAMS_U_PER_CM3 / 1000.0

    # 每步的燃耗增量 (MWd/kgU)
    # 功率 = flux × Σ_f × 能量/裂变
    # 宏观裂变截面 Σ_f = N_U235 × σ_f_U235 + N_Pu239 × σ_f_Pu239
    # 作为简化，用吸收截面近似（裂变占比约 85%）
    # 功率密度 (MW/cm³) = flux × Σ_f × (1/FISSIONS_PER_MWD) × (1/1e6 秒/天)
    # 实际上我们反向推导 Δt 使得燃耗步进均匀

    # 直接用平均功率推导时间步长
    # 典型 PWR: 功率密度 ~100 W/cm³ = 1e-4 MW/cm³
    # 我们反过来算: 给定 total_burnup, n_steps → 每步 burnup_step
    burnup_step = total_burnup / n_steps  # MWd/kgU per step

    # 每步消耗的裂变数
    fissions_per_step = burnup_step * kgU_per_cm3 * FISSIONS_PER_MWD  # fissions/cm³ per step

    # 时间步长: Δt = fissions / (flux × Σ_f)
    # 这个 Δt 会随核素变化而变化，我们先近似
    # ====== 截面转换 ======
    # 微观截面用 barns, 算宏观 Σ = N(×10²⁴) × σ(barns) → cm⁻¹
    # 反应率 R = φ × σ(cm²) = φ × σ(barns) × 1e-24 → s⁻¹
    sigma_a_cm2 = {
        'U235':  SIGMA_A_U235 * 1e-24,
        'U238':  SIGMA_C_U238 * 1e-24,
        'Pu239': SIGMA_A_PU239 * 1e-24,
    }

    # 存储
    burnup_vals = []
    time_vals = []
    N_U235_vals = []
    N_U238_vals = []
    N_Pu239_vals = []
    N_FP_vals = []
    k_inf_vals = []

    N_U235 = N_U235_0
    N_U238 = N_U238_0
    N_Pu239 = N_Pu239_0
    N_FP = N_FP_0
    total_time = 0.0
    cumulative_burnup = 0.0

    # 初始状态
    burnup_vals.append(0.0)
    time_vals.append(0.0)
    N_U235_vals.append(N_U235)
    N_U238_vals.append(N_U238)
    N_Pu239_vals.append(N_Pu239)
    N_FP_vals.append(N_FP)
    k_inf_vals.append(_estimate_k_inf(N_U235, N_U238, N_Pu239, N_FP))

    for step in range(n_steps):
        # 当前宏观截面 (cm⁻¹) — N 是 ×10²⁴ 单位, σ 是 barns, 乘积直接是 cm⁻¹
        Sigma_f = N_U235 * SIGMA_A_U235 * 0.85 + N_Pu239 * SIGMA_A_PU239 * 0.70
        if Sigma_f < 1e-10:
            break

        # 从燃耗步长反算时间
        dt = fissions_per_step / (flux * Sigma_f)

        # 反应率 = φ × σ_cm² (单位: s⁻¹)
        R_U235 = flux * sigma_a_cm2['U235']    # U235 消耗率
        R_U238 = flux * sigma_a_cm2['U238']    # U238→Pu239 转换率
        R_Pu239 = flux * sigma_a_cm2['Pu239']  # Pu239 消耗率

        # 欧拉步进
        dN_U235 = -R_U235 * N_U235 * dt
        dN_U238 = -R_U238 * N_U238 * dt
        dN_Pu239 = (R_U238 * N_U238 - R_Pu239 * N_Pu239) * dt
        dN_FP = (R_U235 * N_U235 + R_Pu239 * N_Pu239) * dt

        N_U235 += dN_U235
        N_U238 += dN_U238
        N_Pu239 += dN_Pu239
        N_FP += dN_FP

        total_time += dt
        cumulative_burnup += burnup_step

        burnup_vals.append(cumulative_burnup)
        time_vals.append(total_time / 86400.0)  # 秒 → 天
        N_U235_vals.append(N_U235)
        N_U238_vals.append(N_U238)
        N_Pu239_vals.append(N_Pu239)
        N_FP_vals.append(N_FP)
        k_inf_vals.append(_estimate_k_inf(N_U235, N_U238, N_Pu239, N_FP))

        if N_U235 < 1e-8:
            break

    return {
        'burnup': np.array(burnup_vals),
        'time': np.array(time_vals),
        'N_U235': np.array(N_U235_vals),
        'N_U238': np.array(N_U238_vals),
        'N_Pu239': np.array(N_Pu239_vals),
        'N_FP': np.array(N_FP_vals),
        'k_inf_trend': np.array(k_inf_vals),
    }


def _estimate_k_inf(N_U235, N_U238, N_Pu239, N_FP):
    """
    简化 k_inf 估算。
    k_inf = ν·Σ_f / Σ_a

    用双群典型数据做近似：
    - 快群裂变贡献小，热群是主力
    - Σ_f ≈ N_U235·σ_f_U235 + N_Pu239·σ_f_Pu239
    - Σ_a ≈ 裂变产物吸收 + 核素吸收 + 结构材料(常数)
    """
    # 裂变截面
    sigma_f_U235 = SIGMA_A_U235 * 0.85
    sigma_f_Pu239 = SIGMA_A_PU239 * 0.70

    Sigma_f = N_U235 * sigma_f_U235 + N_Pu239 * sigma_f_Pu239

    # 吸收截面 (包括 FP 的吸收)
    sigma_a_FP = 50.0        # FP 的平均微观吸收截面 (barns)
    sigma_const = 0.02       # 结构材料 + 冷却剂的宏观吸收 (cm⁻¹) 近似

    Sigma_a = (N_U235 * SIGMA_A_U235
               + N_U238 * SIGMA_C_U238
               + N_Pu239 * SIGMA_A_PU239
               + N_FP * sigma_a_FP
               + sigma_const)

    nu = 2.43  # 每次裂变释放的平均中子数
    if Sigma_a < 1e-10:
        return 0.0
    return nu * Sigma_f / Sigma_a


# ====== 可视化 ======
if __name__ == "__main__":
    # 运行计算
    result = solve_bateman(
        initial_enrichment=0.04,   # 4% 富集度
        flux=3e14,                  # 高通量
        total_burnup=60.0,          # 60 MWd/kgU — 典型卸料燃耗
        n_steps=300,
    )

    bu = result['burnup']
    N_U235 = result['N_U235']
    N_U238 = result['N_U238']
    N_Pu239 = result['N_Pu239']
    N_FP = result['N_FP']
    k_inf = result['k_inf_trend']
    time_days = result['time']

    # 打印关键数据
    print("=" * 60)
    print("Bateman 燃耗计算 — 三核素简化链")
    print("=" * 60)
    print(f"初始 U235 富集度: 4.0%")
    print(f"中子通量: 3×10¹⁴ n/cm²/s")
    print(f"燃耗时间: {time_days[-1]:.0f} 天 ({time_days[-1]/365:.1f} 年)")
    print(f"最终燃耗: {bu[-1]:.1f} MWd/kgU")
    print()
    print(f"最终核素浓度 (×10²⁴ atoms/cm³):")
    print(f"  U235:  {N_U235[-1]:.6f}  (初始: {N_U235[0]:.6f})")
    print(f"  U238:  {N_U238[-1]:.6f}  (初始: {N_U238[0]:.6f})")
    print(f"  Pu239: {N_Pu239[-1]:.6f}  (初始: 0)")
    print(f"  FP:    {N_FP[-1]:.6f}")
    print(f"  守恒检查: ΣN = {(N_U235[-1] + N_U238[-1] + N_Pu239[-1] + N_FP[-1]):.6f}")
    print(f"          初始 ΣN = {(N_U235[0] + N_U238[0] + N_Pu239[0] + N_FP[0]):.6f}")
    print()
    print(f"初始 k_inf: {k_inf[0]:.4f}")
    print(f"最终 k_inf: {k_inf[-1]:.4f}")
    idx_k1 = np.argmin(np.abs(np.array(k_inf) - 1.0))
    print(f"k_inf 降至 1 的燃耗: {bu[idx_k1]:.1f} MWd/kgU (约 {time_days[idx_k1]:.0f} 天)")
    print()

    # ====== 四联图 ======
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 图 1: 核素浓度 vs 燃耗
    ax = axes[0, 0]
    ax.plot(bu, N_U235 / N_U_TOTAL * 100, '#e74c3c', linewidth=2,
            label=r'$^{235}$U')
    ax.plot(bu, N_U238 / N_U_TOTAL * 100, '#3498db', linewidth=2,
            label=r'$^{238}$U')
    ax.plot(bu, N_Pu239 / N_U_TOTAL * 100, '#2ecc71', linewidth=2,
            label=r'$^{239}$Pu')
    ax.plot(bu, N_FP / N_U_TOTAL * 100, '#95a5a6', linewidth=2,
            label='FP')
    ax.set_xlabel('燃耗深度 (MWd/kgU)', fontsize=11)
    ax.set_ylabel('核素份额 (%)', fontsize=11)
    ax.set_title('核素浓度随燃耗变化', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

    # 图 2: Pu239 累积 (放大)
    ax = axes[0, 1]
    ax.plot(bu, N_Pu239 * 1e24, '#2ecc71', linewidth=2)
    ax.fill_between(bu, 0, N_Pu239 * 1e24, color='#2ecc71', alpha=0.1)
    ax.set_xlabel('燃耗深度 (MWd/kgU)', fontsize=11)
    ax.set_ylabel(r'$^{239}$Pu 浓度 (atoms/cm³)', fontsize=11)
    ax.set_title(r'$^{239}$Pu 累积曲线 — U238 转换产物', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    # 标注峰值
    idx_max = np.argmax(N_Pu239)
    ax.annotate(f'峰值: {N_Pu239[idx_max]*1e24:.2e} atoms/cm³\n'
                f'燃耗: {bu[idx_max]:.1f} MWd/kgU',
                xy=(bu[idx_max], N_Pu239[idx_max] * 1e24),
                xytext=(bu[idx_max] + 8, N_Pu239[idx_max] * 1e24 * 0.85),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9)

    # 图 3: k_inf 趋势
    ax = axes[1, 0]
    ax.plot(bu, k_inf, '#e67e22', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='k=1 临界线')
    ax.fill_between(bu, 0, k_inf, color='#e67e22', alpha=0.08)
    ax.set_xlabel('燃耗深度 (MWd/kgU)', fontsize=11)
    ax.set_ylabel('k_inf (估算)', fontsize=11)
    ax.set_title('k_inf 随燃耗下降 — 反应性消耗', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    # 标注 k=1 点
    if idx_k1 < len(bu):
        ax.axvline(x=bu[idx_k1], color='gray', linestyle='--', linewidth=1)
        ax.annotate(f'k=1 @ {bu[idx_k1]:.1f} MWd/kgU',
                    xy=(bu[idx_k1], 1.0),
                    xytext=(bu[idx_k1] + 5, 1.05),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9)

    # 图 4: 时间线
    ax = axes[1, 1]
    ax.plot(time_days / 365, N_U235 / N_U_TOTAL * 100, '#e74c3c', linewidth=2,
            label=r'$^{235}$U')
    ax.plot(time_days / 365, N_Pu239 / N_U_TOTAL * 100, '#2ecc71', linewidth=2,
            label=r'$^{239}$Pu')
    ax.set_xlabel('时间 (年)', fontsize=11)
    ax.set_ylabel('核素份额 (%)', fontsize=11)
    ax.set_title('时间尺度 — 燃料在堆芯中的"寿命"', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig('bateman.png', dpi=150)
    print("图片已保存到 bateman.png")
