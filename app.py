"""
中子扩散方程 — 交互式可视化
Streamlit 应用，支持双群求解、临界尺寸扫描、临界硼搜索
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from solver import solve_two_group, scan_critical_size, search_critical_boron, DEFAULTS

st.set_page_config(page_title="Neutron Diffusion Solver", page_icon="⚛️", layout="wide")
st.title("⚛️ 一维中子扩散方程求解器")
st.caption("双群 · 有限差分法 · 幂迭代  |  核工程交互式学习工具")

# ===== 侧边栏 =====
st.sidebar.header("⚙️ 参数设置")

tab = st.sidebar.radio("📐 选择模块", ["双群扩散求解", "临界尺寸扫描", "临界硼搜索"])

# 通用几何参数
N = st.sidebar.slider("网格点数 N", 30, 300, 150, 10,
                       help="越大越精确，但计算更慢")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 截面数据")

with st.sidebar.expander("快群 (Group 1)", expanded=False):
    D1 = st.number_input("D₁ 扩散系数", 0.1, 5.0, DEFAULTS['D1'], 0.1)
    nu_Sf1 = st.number_input("νΣf₁", 0.0001, 0.1, DEFAULTS['nu_Sf1'], 0.0001, format="%.4f")
    Sa1 = st.number_input("Σa₁ 吸收", 0.001, 0.1, DEFAULTS['Sa1'], 0.001, format="%.4f")
    Ss12 = st.number_input("Σs₁₂ 散射", 0.001, 0.1, DEFAULTS['Ss12'], 0.001, format="%.4f")

with st.sidebar.expander("热群 (Group 2)", expanded=False):
    D2 = st.number_input("D₂ 扩散系数", 0.1, 5.0, DEFAULTS['D2'], 0.1)
    nu_Sf2 = st.number_input("νΣf₂", 0.001, 0.5, DEFAULTS['nu_Sf2'], 0.001, format="%.3f")
    Sa2 = st.number_input("Σa₂ 吸收", 0.001, 0.5, DEFAULTS['Sa2'], 0.001, format="%.3f")

sections = {
    'D1': D1, 'nu_Sf1': nu_Sf1, 'Sa1': Sa1, 'Ss12': Ss12,
    'D2': D2, 'nu_Sf2': nu_Sf2, 'Sa2': Sa2,
}

st.sidebar.markdown("---")
st.sidebar.caption("GitHub: [neutron-diffusion-solver](https://github.com/buzhidao2006/neutron-diffusion-solver)")

# ===== 双群扩散求解 =====
if tab == "双群扩散求解":
    L = st.sidebar.slider("平板半厚度 L (cm)", 20.0, 500.0, 200.0, 10.0)

    if st.sidebar.button("🔬 求解", type="primary", use_container_width=True):
        with st.spinner("幂迭代中..."):
            result = solve_two_group(L=L, N=N, sections=sections)

        k_eff = result['k_eff']
        x = result['x']
        phi1 = result['phi1']
        phi2 = result['phi2']

        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta = "🟢 超临界" if k_eff > 1.001 else ("🔴 次临界" if k_eff < 0.999 else "🟡 临界")
            st.metric("k_eff", f"{k_eff:.6f}", delta=delta)
        with col2:
            st.metric("快群通量峰值", f"{np.max(phi1):.4f}")
        with col3:
            st.metric("热群通量峰值", f"{np.max(phi2):.4f}")
        with col4:
            st.metric("热/快比 (平均)", f"{np.mean(phi2/phi1):.2f}")

        # 通量分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(x, phi1, '#e74c3c', linewidth=2, label='快群 (Group 1)')
        ax1.plot(x, phi2, '#3498db', linewidth=2, label='热群 (Group 2)')
        ax1.fill_between(x, 0, phi1, color='#e74c3c', alpha=0.08)
        ax1.fill_between(x, 0, phi2, color='#3498db', alpha=0.08)
        ax1.set_xlabel('位置 (cm)', fontsize=11)
        ax1.set_ylabel('中子通量 (归一化)', fontsize=11)
        ax1.set_title(f'双群通量分布  |  k_eff = {k_eff:.6f}', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.25)

        ratio = phi2 / (phi1 + 1e-10)
        ax2.plot(x, ratio, '#2ecc71', linewidth=2)
        ax2.fill_between(x, 0, ratio, color='#2ecc71', alpha=0.08)
        ax2.set_xlabel('位置 (cm)', fontsize=11)
        ax2.set_ylabel('热/快通量比', fontsize=11)
        ax2.set_title('热化程度 φ₂/φ₁', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.25)

        # 标注
        center_idx = N // 2
        ax2.annotate(f'中心: {ratio[center_idx]:.2f}',
                     xy=(x[center_idx], ratio[center_idx]),
                     xytext=(x[center_idx] + 30, ratio[center_idx] * 1.1),
                     arrowprops=dict(arrowstyle='->', color='gray'),
                     fontsize=9, color='gray')

        st.pyplot(fig)

        # 物理解释
        with st.expander("📖 物理含义", expanded=False):
            st.markdown(f"""
            | 指标 | 值 | 含义 |
            |------|-----|------|
            | k_eff | {k_eff:.6f} | {'**超临界** — 中子数逐代增长，功率上升' if k_eff > 1.001 else ('**次临界** — 中子数逐代减少，链式反应无法维持' if k_eff < 0.999 else '**临界** — 自持链式反应')} |
            | 通量形状 | 余弦分布 | 边界处通量最低，中心最高（对称） |
            | 热/快比 | 中心 > 边界 | 中心区域热化更充分，边界快中子泄漏多 |
            | Sr1 | {DEFAULTS['Sa1'] + DEFAULTS['Ss12']:.4f} cm⁻¹ | 快群移出截面 = Σa₁ + Σs₁₂ |
            """)

# ===== 临界尺寸扫描 =====
elif tab == "临界尺寸扫描":
    L_min = st.sidebar.slider("最小 L (cm)", 10.0, 200.0, 40.0, 10.0)
    L_max = st.sidebar.slider("最大 L (cm)", 100.0, 600.0, 400.0, 10.0)
    n_pts = st.sidebar.slider("扫描点数", 10, 50, 36, 2)

    if st.sidebar.button("🔬 扫描", type="primary", use_container_width=True):
        with st.spinner("扫描不同尺寸..."):
            cs = scan_critical_size(L_min=L_min, L_max=L_max, n_points=n_pts, N=N, sections=sections)

        L_crit = cs['L_crit']
        k_crit = cs['k_crit']
        analytic = cs['analytic']

        # 指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("k_inf (无穷大介质)", f"{analytic['k_inf']:.4f}")
        with col2:
            st.metric("徙动面积 M²", f"{analytic['M2']:.1f} cm²")
        with col3:
            st.metric("临界尺寸 L_crit", f"{L_crit:.0f} cm",
                      delta=f"数值解 k ≈ {k_crit:.4f}")

        # 图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(cs['L_vals'], cs['k_vals'], 'b.-', linewidth=2, markersize=8, label='FDM 数值解')
        ax1.plot(cs['L_vals'], analytic['k_buckling'], 'r--', linewidth=2, label='Buckling 近似')
        ax1.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='k=1')
        ax1.axvline(x=L_crit, color='gray', linestyle=':', linewidth=1,
                    label=f'L_crit ≈ {L_crit:.0f} cm')
        ax1.set_xlabel('平板厚度 L (cm)')
        ax1.set_ylabel('k_eff')
        ax1.set_title('临界尺寸扫描')
        ax1.legend()
        ax1.grid(True, alpha=0.25)

        error = np.abs(cs['k_vals'] - analytic['k_buckling']) / cs['k_vals'] * 100
        ax2.semilogy(cs['L_vals'], error, 'g.-', linewidth=2, markersize=8)
        ax2.set_xlabel('平板厚度 L (cm)')
        ax2.set_ylabel('相对误差 (%)')
        ax2.set_title('FDM vs Buckling — 误差分析')
        ax2.grid(True, alpha=0.25)

        st.pyplot(fig)

        with st.expander("📖 临界条件公式", expanded=False):
            st.markdown(f"""
            **临界方程** (考研必考)：
            $$k_{{eff}} = \\frac{{k_\\infty}}{{1 + M^2 B^2}} = 1$$

            其中：
            - $k_\\infty = {analytic['k_inf']:.4f}$ — 无穷大介质的增殖因子
            - $M^2 = {analytic['M2']:.1f}$ cm² — 徙动面积
            - $B^2 = (\\pi/L)^2$ — 几何曲率

            反解临界尺寸：$L_{{crit}} = \\pi \\sqrt{{\\frac{{M^2}}{{k_\\infty - 1}}}} = {np.pi * np.sqrt(analytic['M2'] / (analytic['k_inf'] - 1)):.1f}$ cm
            """)

# ===== 临界硼搜索 =====
elif tab == "临界硼搜索":
    L = st.sidebar.slider("堆芯半厚度 L (cm)", 50.0, 500.0, 200.0, 10.0,
                          key="boron_L")
    alpha = st.sidebar.number_input("硼灵敏度 α (cm⁻¹/ppm)", 1e-7, 1e-3,
                                     DEFAULTS.get('alpha', 1.0e-5), 1e-6, format="%.1e")

    if st.sidebar.button("🔬 搜索", type="primary", use_container_width=True):
        with st.spinner("二分法搜索临界硼浓度..."):
            cb = search_critical_boron(L=L, N=N, alpha=alpha, sections=sections)

        # 指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("临界硼浓度 C_B*", f"{cb['C_crit']:.1f} ppm")
        with col2:
            st.metric("硼微分价值", f"{cb['boron_worth']:.1f} pcm/ppm",
                      delta="每 ppm 硼 ≈ -10.8 pcm")
        with col3:
            st.metric("k(C_B*)", f"{cb['k_final']:.8f}")

        # 图
        fig, ax1 = plt.subplots(figsize=(9, 5))

        ax1.plot(cb['C_scan'], cb['k_scan'], 'b.-', linewidth=2, markersize=10,
                 label='k(C_B) 扫描')
        ax1.axhline(y=1.0, color='k', linestyle=':', linewidth=1, label='k=1')
        ax1.axvline(x=cb['C_crit'], color='r', linestyle='--', linewidth=2,
                    label=f"C_B* = {cb['C_crit']:.0f} ppm")
        ax1.set_xlabel('硼浓度 C_B (ppm)', fontsize=11)
        ax1.set_ylabel('k_eff', fontsize=11)
        ax1.set_title('临界硼搜索 — PWR 反应性控制', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.25)

        # 右轴：反应性
        ax2 = ax1.twinx()
        rho_scan = (np.array(cb['k_scan']) - 1) / np.array(cb['k_scan']) * 1e5
        ax2.plot(cb['C_scan'], rho_scan, 'g--', linewidth=1, alpha=0.4)
        ax2.set_ylabel('反应性 (pcm)', color='g')
        ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)

        st.pyplot(fig)

        with st.expander("📖 PWR 硼控制原理", expanded=False):
            st.markdown(f"""
            **硼酸化学补偿控制**是 PWR 的核心运行策略：

            1. **新燃料** → 硼浓度高 (~1200 ppm) → 补偿过剩反应性
            2. **燃料消耗** → 逐渐稀释硼 → 维持 k=1
            3. **换料时** → 重新加硼 → 开始新循环

            | 参数 | 值 | 意义 |
            |------|-----|------|
            | 临界硼浓度 | {cb['C_crit']:.0f} ppm | 给定尺寸下达临界的硼量 |
            | 硼微分价值 | {cb['boron_worth']:.1f} pcm/ppm | 每 ppm 硼抑制的反应性 |
            | 对应的 Σa₂ 增量 | {alpha * cb['C_crit']:.4f} cm⁻¹ | 硼对热群吸收的贡献 |

            **算法**：外迭代 (二分法) + 内迭代 (幂迭代) — 典型的嵌套迭代结构。
            """)

# ===== 底部 =====
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **提示**：修改参数后点击对应模块的按钮重新计算。\n\n"
    "截面数据默认使用典型 PWR 值，可展开修改。"
)
