"""
中子扩散方程 — 交互式可视化
Streamlit 应用，支持双群求解、临界尺寸扫描、临界硼搜索、燃耗耦合、点堆动力学
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from solver import solve_two_group, scan_critical_size, search_critical_boron, DEFAULTS
from solver_2d import solve_two_group_2d
from solver_3d import solve_two_group_3d
from burnup_solver import run_burnup_coupled, N_U_TOTAL
from point_kinetics import (
    solve_point_kinetics, KEEPIN_U235,
    reactivity_step, reactivity_ramp, reactivity_sinusoidal,
    reactivity_rod_ejection, prompt_jump, asymptotic_period,
)

st.set_page_config(page_title="Neutron Diffusion Solver", page_icon="⚛️", layout="wide")
st.title("⚛️ 中子扩散方程求解器")
st.caption("一维 & 二维 · 双群 · 有限差分法 · 幂迭代 · 点堆动力学  |  核工程交互式学习工具")

# ===== 侧边栏 =====
st.sidebar.header("⚙️ 参数设置")

tab = st.sidebar.radio("📐 选择模块",
                       ["双群扩散求解", "临界尺寸扫描", "临界硼搜索",
                        "🟦 二维扩散 (2D)", "🧊 三维扩散 (3D)", "🔥 燃耗耦合", "⏱️ 点堆动力学"])

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

# ===== 二维扩散 (2D) =====
elif tab == "🟦 二维扩散 (2D)":
    st.sidebar.markdown("### 📐 二维几何")

    col_L1, col_L2 = st.sidebar.columns(2)
    with col_L1:
        Lx = st.slider("Lx (cm)", 50.0, 400.0, 200.0, 10.0, help="x 方向边长")
    with col_L2:
        Ly = st.slider("Ly (cm)", 50.0, 400.0, 200.0, 10.0, help="y 方向边长")

    col_N1, col_N2 = st.sidebar.columns(2)
    with col_N1:
        Nx = st.slider("Nx 网格", 20, 100, 60, 5, help="x 方向网格点数")
    with col_N2:
        Ny = st.slider("Ny 网格", 20, 100, 60, 5, help="y 方向网格点数")

    grid_info = st.sidebar.caption(
        f"未知数: {2 * Nx * Ny} 个 (双群 × {Nx}×{Ny})"
    )

    if st.sidebar.button("🔬 二维求解", type="primary", use_container_width=True):
        with st.spinner(f"稀疏矩阵求解中... ({Nx}×{Ny} 网格, {2*Nx*Ny} 未知数)"):
            result = solve_two_group_2d(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, sections=sections)

        k_eff = result['k_eff']
        X, Y = result['X'], result['Y']
        phi1 = result['phi1']
        phi2 = result['phi2']
        ratio = phi2 / (phi1 + 1e-12)

        # Buckling 对比
        p = {**DEFAULTS, **sections}
        Sr1 = p['Sa1'] + p['Ss12']
        k_inf = p['nu_Sf1'] / Sr1 + (p['nu_Sf2'] / p['Sa2']) * (p['Ss12'] / Sr1)
        L2_mig = p['D2'] / p['Sa2']
        tau = p['D1'] / Sr1
        M2 = L2_mig + tau
        B2 = (np.pi / Lx)**2 + (np.pi / Ly)**2
        k_buckling = k_inf / (1 + M2 * B2)

        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta = "🟢 超临界" if k_eff > 1.001 else ("🔴 次临界" if k_eff < 0.999 else "🟡 临界")
            st.metric("k_eff (FDM)", f"{k_eff:.6f}", delta=delta)
        with col2:
            st.metric("k_eff (Buckling)", f"{k_buckling:.6f}",
                     delta=f"偏差 {abs(k_eff-k_buckling)/k_eff*100:.2f}%")
        with col3:
            st.metric("快群通量峰值", f"{np.max(phi1):.4f}")
        with col4:
            st.metric("热群通量峰值", f"{np.max(phi2):.4f}")

        # 四联图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 图 1: 快群 heatmap
        ax = axes[0, 0]
        im1 = ax.contourf(X, Y, phi1, levels=20, cmap='Reds')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Fast Flux (Group 1)', fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im1, ax=ax, shrink=0.8)

        # 图 2: 热群 heatmap
        ax = axes[0, 1]
        im2 = ax.contourf(X, Y, phi2, levels=20, cmap='Blues')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Thermal Flux (Group 2)', fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im2, ax=ax, shrink=0.8)

        # 图 3: 中心线剖面
        ax = axes[1, 0]
        mid_y = Ny // 2
        x_1d = result['x']
        ax.plot(x_1d, phi1[mid_y, :], '#e74c3c', linewidth=2, label=f'Fast (y=L/2)')
        ax.plot(x_1d, phi2[mid_y, :], '#3498db', linewidth=2, label=f'Thermal (y=L/2)')
        # 叠加 sin 形状
        theory = np.sin(np.pi * x_1d / Lx)
        theory = theory / theory.max()
        ax.plot(x_1d, theory * np.max(phi1[mid_y, :]), 'k--', linewidth=1, alpha=0.5,
                label='sin(πx/L) 参考')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('Normalized Flux')
        ax.set_title('Centerline Profile (y = Ly/2)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # 图 4: 热/快比
        ax = axes[1, 1]
        im3 = ax.contourf(X, Y, ratio, levels=20, cmap='RdYlBu_r')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_title(f'Thermal / Fast Ratio  '
                     f'({np.min(ratio):.2f} – {np.max(ratio):.2f})', fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im3, ax=ax, shrink=0.8)

        plt.suptitle(f'2D Two-Group Diffusion  |  k_eff = {k_eff:.6f}  '
                     f'|  {Lx:.0f}×{Ly:.0f} cm  |  Grid {Nx}×{Ny}',
                     fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)

        # 物理解释
        with st.expander("📖 二维扩散原理", expanded=False):
            st.markdown(f"""
            ### 从 1D 到 2D

            **1D slab** → **2D 矩形** 的关键变化：

            | 项目 | 1D | 2D |
            |------|-----|-----|
            | 几何 | 平板 (仅 x) | 矩形 (x, y) |
            | Laplacian | 3 点模板 | 5 点模板 |
            | 矩阵尺寸 | 2N × 2N | 2(Nx·Ny) × 2(Nx·Ny) |
            | 矩阵类型 | 三对角 + 块 | 块稀疏 (每行 5 非零元) |
            | 求解器 | `np.linalg.solve` | `scipy.sparse.linalg.spsolve` |

            **Buckling 公式 (2D 方形)**：
            $$B^2 = \\left(\\frac{{\\pi}}{{L_x}}\\right)^2 + \\left(\\frac{{\\pi}}{{L_y}}\\right)^2$$
            $$k_{{eff}} = \\frac{{k_\\infty}}{{1 + M^2 B^2}}$$

            **当前结果**：
            - k_eff (FDM) = {k_eff:.6f}
            - k_eff (Buckling) = {k_buckling:.6f}
            - k_inf = {k_inf:.4f},  M² = {M2:.1f} cm²
            - B² = {B2:.6f} cm⁻²
            - 通量形状：二维 sin(πx/Lx)·sin(πy/Ly) 分布，中心最高，边界为零
            """)

# ===== 三维扩散 (3D) =====
elif tab == "🧊 三维扩散 (3D)":
    st.sidebar.markdown("### 📐 三维几何")

    col_L1, col_L2, col_L3 = st.sidebar.columns(3)
    with col_L1:
        Lx_3d = st.slider("Lx (cm)", 50.0, 300.0, 160.0, 10.0, key="3d_Lx")
    with col_L2:
        Ly_3d = st.slider("Ly (cm)", 50.0, 300.0, 160.0, 10.0, key="3d_Ly")
    with col_L3:
        Lz_3d = st.slider("Lz (cm)", 50.0, 300.0, 160.0, 10.0, key="3d_Lz")

    N_3d = st.sidebar.slider("网格点数 (每方向)", 10, 30, 20, 2,
                             help="N³ 增长很快，建议 ≤25")
    grid_info_3d = st.sidebar.caption(
        f"未知数: {2 * N_3d**3} 个 (双群 × {N_3d}³ = {N_3d**3} 节点)"
    )

    if st.sidebar.button("🧊 三维求解", type="primary", use_container_width=True):
        with st.spinner(f"3D 稀疏矩阵求解中... ({N_3d}³ = {N_3d**3} 节点, "
                        f"{2*N_3d**3} 未知数)"):
            result_3d = solve_two_group_3d(
                Lx=Lx_3d, Ly=Ly_3d, Lz=Lz_3d,
                Nx=N_3d, Ny=N_3d, Nz=N_3d,
                sections=sections, method='chebyshev',
            )

        k_eff = result_3d['k_eff']
        X, Y, Z = result_3d['X'], result_3d['Y'], result_3d['Z']
        phi1, phi2 = result_3d['phi1'], result_3d['phi2']
        N_pts = N_3d

        # Buckling 对比 (3D: B² = 3(π/L)²)
        p = {**DEFAULTS, **sections}
        Sr1 = p['Sa1'] + p['Ss12']
        k_inf = p['nu_Sf1'] / Sr1 + (p['nu_Sf2'] / p['Sa2']) * (p['Ss12'] / Sr1)
        M2 = p['D2'] / p['Sa2'] + p['D1'] / Sr1
        L_avg = (Lx_3d + Ly_3d + Lz_3d) / 3
        B2_3d = 3.0 * (np.pi / L_avg)**2
        k_buckling = k_inf / (1 + M2 * B2_3d)

        # 指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta = "🟢 超临界" if k_eff > 1.001 else ("🔴 次临界" if k_eff < 0.999 else "🟡 临界")
            st.metric("k_eff (3D FDM)", f"{k_eff:.6f}", delta=delta)
        with col2:
            st.metric("k_eff (Buckling)", f"{k_buckling:.6f}",
                     delta=f"偏差 {abs(k_eff-k_buckling)/k_eff*100:.2f}%")
        with col3:
            st.metric("迭代次数", f"{result_3d['n_iter']}")
        with col4:
            st.metric("系统规模", f"{2*N_pts**3} 未知数")

        # 三截面视图: z-center, y-center, x-center
        mz, my, mx = N_pts // 2, N_pts // 2, N_pts // 2
        x, y, z = result_3d['x'], result_3d['y'], result_3d['z']

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        for row, (flux_data, label, cmap) in enumerate([
            (phi1, 'Fast Flux (Group 1)', 'Reds'),
            (phi2, 'Thermal Flux (Group 2)', 'Blues'),
        ]):
            im = axes[row, 0].contourf(x, y, flux_data[mz, :, :], levels=20, cmap=cmap)
            axes[row, 0].set_title(f'{label} — z={z[mz]:.0f} cm slice')
            axes[row, 0].set_xlabel('x (cm)'); axes[row, 0].set_ylabel('y (cm)')
            axes[row, 0].set_aspect('equal')
            plt.colorbar(im, ax=axes[row, 0], shrink=0.8)

            im = axes[row, 1].contourf(x, z, flux_data[:, my, :], levels=20, cmap=cmap)
            axes[row, 1].set_title(f'{label} — y={y[my]:.0f} cm slice')
            axes[row, 1].set_xlabel('x (cm)'); axes[row, 1].set_ylabel('z (cm)')
            axes[row, 1].set_aspect('equal')
            plt.colorbar(im, ax=axes[row, 1], shrink=0.8)

            im = axes[row, 2].contourf(y, z, flux_data[:, :, mx], levels=20, cmap=cmap)
            axes[row, 2].set_title(f'{label} — x={x[mx]:.0f} cm slice')
            axes[row, 2].set_xlabel('y (cm)'); axes[row, 2].set_ylabel('z (cm)')
            axes[row, 2].set_aspect('equal')
            plt.colorbar(im, ax=axes[row, 2], shrink=0.8)

        plt.suptitle(f'3D Two-Group Diffusion  |  k_eff = {k_eff:.6f}  '
                     f'|  {N_pts}³ grid  |  {2*N_pts**3} unknowns',
                     fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("📖 三维扩散原理", expanded=False):
            st.markdown(f"""
            ### 从 2D 到 3D

            **Kronecker 积构造的 3D Laplacian**:
            $$L_{{3D}} = I_z \\otimes I_y \\otimes \\frac{{L_x}}{{h_x^2}} +
                          I_z \\otimes \\frac{{L_y}}{{h_y^2}} \\otimes I_x +
                          \\frac{{L_z}}{{h_z^2}} \\otimes I_y \\otimes I_x$$

            | 项目 | 2D | 3D |
            |------|-----|-----|
            | Laplacian 模板 | 5 点 (N, S, E, W) | 7 点 (+ 上, 下) |
            | 每行非零元 | 5 | 7 |
            | 节点数 | N² | N³ |
            | 双群未知数 | 2N² | 2N³ |

            **Buckling 公式 (3D 立方体)**：
            $$B^2 = 3\\left(\\frac{{\\pi}}{{L}}\\right)^2$$
            $$k_{{eff}} = \\frac{{k_\\infty}}{{1 + M^2 B^2}}$$

            **当前结果**：
            - k_eff (FDM) = {k_eff:.6f}
            - k_eff (Buckling) = {k_buckling:.6f}
            - k_inf = {k_inf:.4f}, M² = {M2:.1f} cm²
            - 矩阵规模 = {2*N_pts**3} × {2*N_pts**3}
            - 稀疏度 = 99.98%（仅 7/{2*N_pts**3} 非零每行）
            """)

# ===== 燃耗耦合 =====
elif tab == "🔥 燃耗耦合":
    st.sidebar.markdown("### 🏭 燃耗参数")

    enrichment = st.sidebar.slider("初始 U235 富集度 (%)", 1.0, 10.0, 4.0, 0.5) / 100
    L_burn = st.sidebar.slider("堆芯半厚度 L (cm)", 50.0, 500.0, 200.0, 10.0, key="burn_L")
    total_bu = st.sidebar.slider("总燃耗 (MWd/kgU)", 10.0, 80.0, 50.0, 5.0)
    n_steps_burn = st.sidebar.slider("燃耗步数", 10, 100, 40, 5,
                                      help="步数越多越精细，计算越慢")

    if st.sidebar.button("🔥 开始耦合计算", type="primary", use_container_width=True):
        with st.spinner(f"燃耗-扩散耦合计算中... ({n_steps_burn} 步, 可能需要几十秒)"):
            hist = run_burnup_coupled(
                initial_enrichment=enrichment,
                L=L_burn,
                N_grid=N,
                total_burnup=total_bu,
                n_burnup_steps=n_steps_burn,
            )

        bu = hist['burnup']
        k_eff = hist['k_eff']
        time_years = hist['time_days'] / 365

        idx_k1 = np.argmin(np.abs(k_eff - 1.0))
        bu_k1 = bu[idx_k1]

        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("初始 k_eff", f"{k_eff[0]:.4f}")
        with col2:
            st.metric("k=1 燃耗", f"{bu_k1:.1f} MWd/kgU",
                      delta=f"{time_years[idx_k1]:.1f} 年")
        with col3:
            st.metric("最终 k_eff", f"{k_eff[-1]:.4f}",
                      delta=f"@{bu[-1]:.1f} MWd/kgU")
        with col4:
            u235_final = hist['N_U235'][-1] / N_U_TOTAL * 100
            st.metric("U235 剩余", f"{u235_final:.2f}%",
                      delta=f"初始 {enrichment*100:.1f}%")

        # 四联图
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # 图 1: k_eff vs burnup
        ax = axes[0, 0]
        ax.plot(bu, k_eff, '#e74c3c', linewidth=2.5)
        ax.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, label='k=1')
        ax.axvline(x=bu_k1, color='gray', linestyle='--', linewidth=1)
        ax.fill_between(bu, 0, k_eff, color='#e74c3c', alpha=0.06)
        ax.set_xlabel('Burnup (MWd/kgU)')
        ax.set_ylabel('k_eff')
        ax.set_title('Reactivity Depletion', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.25)

        # 图 2: 核素演化
        ax = axes[0, 1]
        ax.plot(bu, hist['N_U235'] / N_U_TOTAL * 100, '#e74c3c', linewidth=2, label='U-235')
        ax.plot(bu, hist['N_U238'] / N_U_TOTAL * 100, '#3498db', linewidth=2, label='U-238')
        ax.plot(bu, hist['N_Pu239'] / N_U_TOTAL * 100, '#2ecc71', linewidth=2, label='Pu-239')
        ax.plot(bu, hist['N_FP'] / N_U_TOTAL * 100, '#95a5a6', linewidth=2, label='FP')
        ax.set_xlabel('Burnup (MWd/kgU)')
        ax.set_ylabel('Nuclide Fraction (%)')
        ax.set_title('Nuclide Evolution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.25)

        # 图 3: 通量演化
        ax = axes[1, 0]
        ax.plot(bu, np.array(hist['flux_thermal']) / 1e13, '#e67e22', linewidth=2, label='Thermal')
        ax.plot(bu, np.array(hist['flux_fast']) / 1e13, '#9b59b6', linewidth=2, label='Fast')
        ax.set_xlabel('Burnup (MWd/kgU)')
        ax.set_ylabel('Avg Flux (x10^13 n/cm^2/s)')
        ax.set_title('Flux Evolution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.25)

        # 图 4: 截面演化
        ax = axes[1, 1]
        ax.plot(bu, hist['Sigma_a2'], '#c0392b', linewidth=2, label='Sigma_a2')
        ax.plot(bu, hist['Sigma_f2'], '#27ae60', linewidth=2, label='Sigma_f2')
        ax.set_xlabel('Burnup (MWd/kgU)')
        ax.set_ylabel('Macroscopic XS (cm^-1)')
        ax.set_title('Cross Section Evolution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("📖 燃耗耦合原理", expanded=False):
            st.markdown(f"""
            ### 耦合计算流程

            每个燃耗步执行三步：

            **1. 核素 → 截面**：由当前核素浓度计算宏观截面
            - Σa₂ = N_U235·σa_U235 + N_U238·σc_U238 + N_Pu·σa_Pu + N_FP·σa_FP + Σ_struct
            - Σf₂ = N_U235·σf_U235 + N_Pu·σf_Pu

            **2. 截面 → 扩散**：双群扩散求解 k_eff 和中子通量分布

            **3. 通量 → 燃耗**：Bateman 方程欧拉步进更新核素浓度

            | 关键结果 | 值 |
            |----------|-----|
            | 初始 k_eff | {k_eff[0]:.4f} |
            | k=1 燃耗 (卸料) | {bu_k1:.1f} MWd/kgU |
            | 辐照时间 | {hist['time_days'][-1]:.0f} 天 ({time_years[-1]:.1f} 年) |
            | U235 消耗 | {enrichment*100:.1f}% → {u235_final:.2f}% |
            | Pu239 峰值 | {np.max(hist['N_Pu239']) / N_U_TOTAL * 100:.2f}% |
            """)

# ===== 点堆动力学 =====
elif tab == "⏱️ 点堆动力学":
    st.sidebar.markdown("### ⏱️ 瞬态场景")

    scenario = st.sidebar.selectbox(
        "反应性引入方式",
        ["小阶跃 (+100 pcm)", "负阶跃 (−500 pcm, 停堆)",
         "瞬发超临界 (+1000 pcm)", "线性提棒",
         "弹棒事故", "正弦振荡"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 场景参数")

    if scenario == "小阶跃 (+100 pcm)":
        rho_pcm = st.sidebar.slider("反应性 (pcm)", 10, 600, 100, 10)
        t_span_val = st.sidebar.slider("仿真时间 (s)", 5, 120, 40, 5)

    elif scenario == "负阶跃 (−500 pcm, 停堆)":
        rho_pcm = -st.sidebar.slider("|反应性| (pcm)", 50, 1000, 500, 50)
        t_span_val = st.sidebar.slider("仿真时间 (s)", 10, 300, 100, 10)

    elif scenario == "瞬发超临界 (+1000 pcm)":
        rho_pcm = st.sidebar.slider("反应性 (pcm)", 700, 2000, 1000, 50)
        t_span_val = st.sidebar.slider("仿真时间 (s)", 0.05, 1.0, 0.15, 0.05)

    elif scenario == "线性提棒":
        rho_rate = st.sidebar.slider("提棒速率 (pcm/s)", 0.5, 20.0, 3.0, 0.5)
        rho_pcm = 0  # 不使用
        t_span_val = st.sidebar.slider("仿真时间 (s)", 20, 300, 150, 10)

    elif scenario == "弹棒事故":
        rho_pcm = st.sidebar.slider("最大反应性 (pcm)", 500, 2000, 1000, 50)
        tau_eject = st.sidebar.slider("弹棒时间常数 (ms)", 10, 200, 30, 10)
        t_span_val = st.sidebar.slider("仿真时间 (s)", 0.1, 2.0, 0.4, 0.1)

    elif scenario == "正弦振荡":
        amp = st.sidebar.slider("振幅 (pcm)", 10, 200, 50, 10)
        period_osc = st.sidebar.slider("振荡周期 (s)", 5, 60, 20, 5)
        rho_pcm = 0  # 不使用
        t_span_val = st.sidebar.slider("仿真时间 (s)", 20, 200, 80, 10)

    if st.sidebar.button("⏱️ 计算瞬态", type="primary", use_container_width=True):
        with st.spinner("求解点堆动力学方程..."):

            # 构造反应性函数
            if scenario == "小阶跃 (+100 pcm)":
                rho_func = lambda t: reactivity_step(t, rho_pcm * 1e-5, t_insert=1.0)
            elif scenario == "负阶跃 (−500 pcm, 停堆)":
                rho_func = lambda t: reactivity_step(t, rho_pcm * 1e-5, t_insert=1.0)
            elif scenario == "瞬发超临界 (+1000 pcm)":
                rho_func = lambda t: reactivity_step(t, rho_pcm * 1e-5, t_insert=0.0)
                use_log = True
                P_max_val = 1e4
            elif scenario == "线性提棒":
                rho_func = lambda t: reactivity_ramp(t, rho_rate * 1e-5, t_start=5.0)
            elif scenario == "弹棒事故":
                rho_func = lambda t: reactivity_rod_ejection(
                    t, rho_pcm * 1e-5, t_eject=0.0, tau=tau_eject / 1000)
            elif scenario == "正弦振荡":
                rho_func = lambda t: reactivity_sinusoidal(t, amp * 1e-5, period_osc)

            # 判断是否需要用 log-space
            use_log = scenario in ["瞬发超临界 (+1000 pcm)", "弹棒事故", "线性提棒"]
            P_max_val = 1e4 if use_log else 1e8

            result = solve_point_kinetics(
                rho_func,
                t_span=(0, t_span_val),
                use_log=use_log,
                P_max=P_max_val,
            )

        beta = KEEPIN_U235['beta']
        beta_pcm = beta * 1e5

        # 指标卡片
        P_final = result.P[-1]
        P_max_val = np.max(result.P)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最终功率 P/P₀", f"{P_final:.4f}",
                      delta=f"峰值 {P_max_val:.4f}" if P_max_val > P_final else None)
        with col2:
            rho_final = result.rho[-1]
            st.metric("最终反应性", f"{rho_final * 1e5:.0f} pcm",
                      delta=f"{rho_final / beta:.1f} $")
        with col3:
            T_asym = asymptotic_period(rho_final)
            st.metric("渐近周期", f"{T_asym:.1f} s" if abs(T_asym) < 1e6 else "∞")
        with col4:
            if abs(rho_final) > 0 and abs(rho_final) < beta:
                Pj = prompt_jump(1.0, rho_final)
                st.metric("瞬发跳变理论值", f"{Pj:.3f} P₀")
            else:
                st.metric("状态", "瞬发临界!" if rho_final >= beta else "深次临界")

        # 双图：功率 + 反应性
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 功率
        ax1.plot(result.t, result.P, 'b-', linewidth=2.0)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='初始稳态')
        if abs(rho_final) > 0 and abs(rho_final) < beta:
            Pj = prompt_jump(1.0, rho_final)
            if 0 < Pj < 20:
                ax1.axhline(y=Pj, color='orange', linestyle=':', alpha=0.7,
                            label=f'瞬发跳变 = {Pj:.2f}')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('归一化功率 P/P₀')
        ax1.set_title(f'功率响应 — {scenario}', fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25)
        if P_max_val > 100:
            ax1.set_yscale('log')

        # 反应性
        ax2.plot(result.t, result.rho * 1e5, 'r-', linewidth=2.0)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=beta_pcm, color='orange', linestyle=':', alpha=0.7,
                    label=f'β = {beta_pcm:.0f} pcm (瞬发临界)')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('反应性 ρ (pcm)')
        ax2.set_title('反应性引入历史', fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig)

        # 周期
        fig2, ax = plt.subplots(figsize=(13, 4))
        period = result.period
        mask = (np.abs(period) < 1e4) & (period != 0)
        ax.plot(result.t[mask], period[mask], 'g-', linewidth=2.0)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('反应堆周期 T (s)')
        ax.set_title('反应堆周期', fontweight='bold')
        ax.grid(True, alpha=0.25)
        st.pyplot(fig2)

        # 物理解释
        with st.expander("📖 点堆动力学原理", expanded=False):
            st.markdown(f"""
            ### 点堆动力学方程

            6 群缓发中子先驱核方程：

            $$\\frac{{dP}}{{dt}} = \\frac{{\\rho(t) - \\beta}}{{\\Lambda}} P(t) + \\sum_{{i=1}}^{{6}} \\lambda_i C_i(t)$$

            $$\\frac{{dC_i}}{{dt}} = \\frac{{\\beta_i}}{{\\Lambda}} P(t) - \\lambda_i C_i(t)$$

            ### 关键物理概念

            | 概念 | 值 | 含义 |
            |------|-----|------|
            | **缓发中子份额 β** | {beta_pcm:.0f} pcm (0.7%) | 裂变中子中由先驱核衰变产生的份额 |
            | **中子代时间 Λ** | 2×10⁻⁵ s | 热中子从产生到引起下次裂变的平均时间 |
            | **瞬发临界** | ρ > β | 仅靠瞬发中子就能维持链式反应 → 周期 ~0.01s |
            | **缓发临界** | 0 < ρ < β | 需要缓发中子参与 → 周期 ~10–100s (可控) |
            | **1 元 ($)** | = β = {beta_pcm:.0f} pcm | 反应性的美元单位 |

            ### 为什么缓发中子让反应堆可控？

            如果没有缓发中子（β=0），中子代时间 Λ=2×10⁻⁵ s 会让 **任何正反应性** 都导致功率 ~10⁻⁴ s 级指数爆发。
            缓发中子将有效代时间延长到 ~0.1s，给了控制棒、硼酸等机械/化学控制手段足够的响应时间。

            ### 瞬发跳变

            引入反应性 ρ 后，在 ~10⁻⁴ s 内功率跳变到：
            $$P_{{jump}} = P_0 \\cdot \\frac{{\\beta}}{{\\beta - \\rho}}$$

            跳变后，功率以渐近周期变化，周期由 **倒时方程** 决定。

            ### 当前场景分析

            | 指标 | 值 |
            |------|-----|
            | 最终反应性 | {rho_final * 1e5:.0f} pcm = {rho_final / beta:.2f} $ |
            | 最终功率 | {P_final:.4f} P₀ |
            | 渐近周期 | {T_asym:.1f} s |
            | 求解器信息 | {result.info} |
            """)

# ===== 底部 =====
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **提示**：修改参数后点击对应模块的按钮重新计算。\n\n"
    "截面数据默认使用典型 PWR 值，可展开修改。"
)
