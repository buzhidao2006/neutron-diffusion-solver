"""
燃耗-扩散耦合求解器
将 Bateman 燃耗方程与双群中子扩散方程耦合

循环:
    1. 给定核素浓度 → 计算宏观截面 → 双群扩散求 φ, k_eff
    2. 用 φ 驱动 Bateman 步进 → 更新核素浓度
    3. 重复直到 k_eff < 1 (次临界, 无法维持链式反应)
"""
import numpy as np
import matplotlib.pyplot as plt
from solver import solve_two_group, DEFAULTS
from bateman import (
    N_U_TOTAL, GRAMS_U_PER_CM3,
    SIGMA_A_U235, SIGMA_C_U238, SIGMA_A_PU239,
    FISSIONS_PER_MWD,
)


def run_burnup_coupled(initial_enrichment=0.04,   # 初始 U235 富集度
                        L=200.0,                    # 堆芯半厚度 (cm)
                        N_grid=150,                 # 扩散网格点数
                        total_burnup=60.0,          # 总燃耗 (MWd/kgU)
                        n_burnup_steps=50,          # 燃耗步数
                        base_sections=None,         # 基础截面 (不含燃料贡献)
                        ):
    """
    燃耗-扩散耦合主循环。

    每个燃耗步:
      1. 根据当前核素浓度更新宏观截面
      2. 双群扩散求解 → 得到 k_eff, φ₁, φ₂
      3. 用平均通量驱动 Bateman 步进
      4. 记录 k_eff, 核素浓度, 燃耗

    Returns
    -------
    dict: 完整燃耗历史
    """
    # 基础截面（不含燃料的宏观截面部分）
    if base_sections is None:
        base_sections = {
            'D1': DEFAULTS['D1'],
            'D2': DEFAULTS['D2'],
            'Ss12': DEFAULTS['Ss12'],
        }

    # 初始核素浓度 (×10²⁴ atoms/cm³)
    N_U235 = N_U_TOTAL * initial_enrichment
    N_U238 = N_U_TOTAL * (1 - initial_enrichment)
    N_Pu239 = 0.0
    N_FP = 0.0

    # 燃耗步长
    burnup_step = total_burnup / n_burnup_steps  # MWd/kgU per step
    kgU_per_cm3 = GRAMS_U_PER_CM3 / 1000.0

    # 存储历史
    history = {
        'burnup': [],
        'time_days': [],
        'k_eff': [],
        'flux_fast': [],       # 平均快群通量
        'flux_thermal': [],    # 平均热群通量
        'N_U235': [],
        'N_U238': [],
        'N_Pu239': [],
        'N_FP': [],
        'Sigma_a1': [],
        'Sigma_a2': [],
        'Sigma_f2': [],
    }

    total_time = 0.0
    cumulative_burnup = 0.0

    for step in range(n_burnup_steps + 1):
        # === Step 1: 从核素浓度计算宏观截面 ===
        # 核素的宏观截面贡献 (cm⁻¹)
        # 快群: U235 和 Pu239 都有快裂变贡献，但相对热群小
        # 简化处理: 快群吸收主要来自核素 + 结构材料
        Sigma_a1_fuel = (N_U235 * SIGMA_A_U235 * 0.15  # 快群吸收份额
                         + N_U238 * SIGMA_C_U238 * 0.10
                         + N_Pu239 * SIGMA_A_PU239 * 0.20)
        Sigma_s12_fuel = N_U238 * SIGMA_C_U238 * 0.05  # 快群散射到热群的额外贡献（很小）

        # 热群: 这里是裂变和吸收的主力
        Sigma_a2_fuel = (N_U235 * SIGMA_A_U235 * 0.85  # U235 热吸收
                         + N_U238 * SIGMA_C_U238 * 0.90  # U238 共振吸收主要贡献热群
                         + N_Pu239 * SIGMA_A_PU239 * 0.80  # Pu239 热吸收
                         + N_FP * 50.0)  # FP 吸收截面 ~50 barns
        Sigma_f2 = (N_U235 * SIGMA_A_U235 * 0.85 * 0.85   # νΣ_f 热群 (×ν/吸收比)
                    + N_Pu239 * SIGMA_A_PU239 * 0.80 * 0.70)

        # 合并基础截面（结构材料、冷却剂等）
        Sa1_total = base_sections.get('Sa1_struct', 0.005) + Sigma_a1_fuel
        Sa2_total = base_sections.get('Sa2_struct', 0.02) + Sigma_a2_fuel
        Ss12_total = base_sections['Ss12'] + Sigma_s12_fuel

        # νΣ_f 需要单独处理
        nu_Sf1_eff = N_U235 * SIGMA_A_U235 * 0.15 * 0.85 * 2.43  # 快群 νΣ_f
        nu_Sf2_eff = Sigma_f2 * 2.43  # 热群 νΣ_f (Σ_f × ν)

        sections = {
            'D1': base_sections['D1'],
            'D2': base_sections['D2'],
            'Sa1': Sa1_total,
            'Sa2': Sa2_total,
            'Ss12': Ss12_total,
            'nu_Sf1': max(nu_Sf1_eff, 1e-6),
            'nu_Sf2': max(nu_Sf2_eff, 1e-6),
        }

        # === Step 2: 双群扩散求解 ===
        result = solve_two_group(L=L, N=N_grid, sections=sections)
        k_eff = result['k_eff']
        phi1 = result['phi1']  # 快群通量分布
        phi2 = result['phi2']  # 热群通量分布

        # 平均通量 (用于 Bateman 步进)
        avg_flux_fast = np.mean(phi1)   # 归一化通量, 需要缩放
        avg_flux_thermal = np.mean(phi2)

        # === Step 3: 功率标定 ===
        # 归一化通量需要标定到实际物理通量
        # 典型 PWR: 平均热通量 ~3×10¹³ n/cm²/s
        # 这里用功率密度反标: P = Σ_f × φ × (能量/裂变)
        # 简化: 设定目标功率密度，标定通量幅度
        target_power_density = 100.0  # W/cm³ — 典型 PWR 功率密度
        energy_per_fission_J = 3.2e-11  # J/fission
        power_per_fission_W = energy_per_fission_J  # W·s / fission → wrong unit

        # 功率密度 (W/cm³) = Σ_f(cm⁻¹) × φ(n/cm²/s) × E_f(J/fission)
        #  = Σ_f × φ × 3.2e-11 W/(n/cm²·s)? No, let's be more careful.
        # Actually: Power density = Σ_f × φ × E_f
        # W/cm³ = cm⁻¹ × (n/cm²/s) × J = J/(cm³·s)
        # So: φ_scale = P_target / (Σ_f × E_f)
        # But our φ₁, φ₂ from the solver are normalized (max=1)
        # We need to scale φ₂ to get the actual thermal flux

        current_Sigma_f = max(Sigma_f2, 1e-8)
        # 目标通量幅度: φ₂_max × scale = P / (Σ_f × E_f)
        scale_factor = target_power_density / (current_Sigma_f * energy_per_fission_J) / np.max(phi2) if np.max(phi2) > 0 else 1e13

        # 实际物理通量
        phi1_physical = phi1 * scale_factor * 0.5   # 快通量约热通量的一半
        phi2_physical = phi2 * scale_factor

        avg_flux = np.mean(phi2_physical)   # 热群平均通量作为 Bateman 驱动

        # === Step 4: 记录当前状态 ===
        history['burnup'].append(cumulative_burnup)
        history['time_days'].append(total_time / 86400.0)
        history['k_eff'].append(k_eff)
        history['flux_fast'].append(np.mean(phi1_physical))
        history['flux_thermal'].append(np.mean(phi2_physical))
        history['N_U235'].append(N_U235)
        history['N_U238'].append(N_U238)
        history['N_Pu239'].append(N_Pu239)
        history['N_FP'].append(N_FP)
        history['Sigma_a1'].append(Sa1_total)
        history['Sigma_a2'].append(Sa2_total)
        history['Sigma_f2'].append(Sigma_f2)

        if step >= n_burnup_steps:
            break

        # === Step 5: Bateman 步进 ===
        # 反应率 (s⁻¹) = φ × σ
        flux_val = max(avg_flux, 1e10)  # 防止通量为 0

        R_U235 = flux_val * SIGMA_A_U235 * 1e-24
        R_U238 = flux_val * SIGMA_C_U238 * 1e-24
        R_Pu239 = flux_val * SIGMA_A_PU239 * 1e-24

        # 计算达到目标燃耗步长所需的时间
        # 燃耗步长 (MWd/kgU) → 需要的裂变数 → Δt
        fissions_needed = burnup_step * kgU_per_cm3 * FISSIONS_PER_MWD
        if Sigma_f2 > 1e-10:
            dt = fissions_needed / (flux_val * max(Sigma_f2, 1e-8))
        else:
            dt = 86400.0  # 默认 1 天

        dt = min(dt, 365 * 86400.0)  # 最大步长 1 年

        # 欧拉步进
        dN_U235 = -R_U235 * N_U235 * dt
        dN_U238 = -R_U238 * N_U238 * dt
        dN_Pu239 = (R_U238 * N_U238 - R_Pu239 * N_Pu239) * dt
        dN_FP = (R_U235 * N_U235 + R_Pu239 * N_Pu239) * dt

        N_U235 = max(N_U235 + dN_U235, 0.0)
        N_U238 = max(N_U238 + dN_U238, 0.0)
        N_Pu239 = max(N_Pu239 + dN_Pu239, 0.0)
        N_FP = max(N_FP + dN_FP, 0.0)

        total_time += dt
        cumulative_burnup += burnup_step

        # k_eff < 0.95 → 次临界，可以提前结束
        if k_eff < 0.95 and cumulative_burnup > 10:
            print(f"  [Step {step}] k_eff={k_eff:.4f} < 0.95, 深度次临界, 结束计算")
            break

    # 转换为 numpy 数组
    for key in history:
        history[key] = np.array(history[key])

    return history


# ====== 可视化 ======
if __name__ == "__main__":
    print("=" * 60)
    print("燃耗-扩散耦合计算")
    print("=" * 60)

    hist = run_burnup_coupled(
        initial_enrichment=0.04,
        L=200.0,
        N_grid=150,
        total_burnup=60.0,
        n_burnup_steps=50,
    )

    bu = hist['burnup']
    k_eff = hist['k_eff']
    time_years = hist['time_days'] / 365

    # 找 k=1 对应的燃耗
    idx_k1 = np.argmin(np.abs(k_eff - 1.0))
    bu_k1 = bu[idx_k1] if idx_k1 < len(bu) else bu[-1]

    print(f"初始 k_eff: {k_eff[0]:.4f}")
    print(f"k=1 燃耗: {bu_k1:.1f} MWd/kgU ({time_years[idx_k1]:.2f} 年)")
    print(f"最终 k_eff: {k_eff[-1]:.4f} @ {bu[-1]:.1f} MWd/kgU")
    print(f"辐照时间: {hist['time_days'][-1]:.0f} 天 ({time_years[-1]:.2f} 年)")
    print()
    print("最终核素份额 (%):")
    print(f"  U235:  {hist['N_U235'][-1] / N_U_TOTAL * 100:.2f}%")
    print(f"  U238:  {hist['N_U238'][-1] / N_U_TOTAL * 100:.2f}%")
    print(f"  Pu239: {hist['N_Pu239'][-1] / N_U_TOTAL * 100:.2f}%")
    print(f"  FP:    {hist['N_FP'][-1] / N_U_TOTAL * 100:.2f}%")

    # ====== 四联图 ======
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 图 1: k_eff vs burnup
    ax = axes[0, 0]
    ax.plot(bu, k_eff, '#e74c3c', linewidth=2.5)
    ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, label='k=1 (critical)')
    ax.axvline(x=bu_k1, color='gray', linestyle='--', linewidth=1)
    ax.fill_between(bu, 0, k_eff, color='#e74c3c', alpha=0.06)
    ax.set_xlabel('Burnup (MWd/kgU)', fontsize=11)
    ax.set_ylabel('k_eff', fontsize=11)
    ax.set_title('k_eff vs Burnup — Reactivity Depletion', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.annotate(f'k=1 @ {bu_k1:.1f} MWd/kgU',
                xy=(bu_k1, 1.0), xytext=(bu_k1 + 5, 1.08),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

    # 图 2: 核素浓度
    ax = axes[0, 1]
    ax.plot(bu, hist['N_U235'] / N_U_TOTAL * 100, '#e74c3c', linewidth=2, label='U-235')
    ax.plot(bu, hist['N_U238'] / N_U_TOTAL * 100, '#3498db', linewidth=2, label='U-238')
    ax.plot(bu, hist['N_Pu239'] / N_U_TOTAL * 100, '#2ecc71', linewidth=2, label='Pu-239')
    ax.plot(bu, hist['N_FP'] / N_U_TOTAL * 100, '#95a5a6', linewidth=2, label='FP')
    ax.set_xlabel('Burnup (MWd/kgU)', fontsize=11)
    ax.set_ylabel('Nuclide Fraction (%)', fontsize=11)
    ax.set_title('Nuclide Evolution — Coupled with Diffusion', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # 图 3: 通量演化
    ax = axes[1, 0]
    ax.plot(bu, hist['flux_thermal'] / 1e13, '#e67e22', linewidth=2, label='Thermal flux')
    ax.plot(bu, hist['flux_fast'] / 1e13, '#9b59b6', linewidth=2, label='Fast flux')
    ax.set_xlabel('Burnup (MWd/kgU)', fontsize=11)
    ax.set_ylabel('Flux (x10^13 n/cm^2/s)', fontsize=11)
    ax.set_title('Neutron Flux Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # 图 4: 宏观截面演化
    ax = axes[1, 1]
    ax.plot(bu, hist['Sigma_a2'], '#c0392b', linewidth=2, label='Sigma_a2 (absorption)')
    ax.plot(bu, hist['Sigma_f2'], '#27ae60', linewidth=2, label='Sigma_f2 (fission)')
    ax.set_xlabel('Burnup (MWd/kgU)', fontsize=11)
    ax.set_ylabel('Macroscopic XS (cm^-1)', fontsize=11)
    ax.set_title('Cross Section Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig('burnup_coupled.png', dpi=150)
    print("\n图片已保存到 burnup_coupled.png")
