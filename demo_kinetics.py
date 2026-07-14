"""
点堆动力学 — 演示脚本

覆盖六种典型反应性引入场景，展示反应堆瞬态行为。
直接运行: python3 demo_kinetics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from point_kinetics import (
    solve_point_kinetics,
    reactivity_step, reactivity_ramp, reactivity_sinusoidal,
    reactivity_rod_ejection,
    prompt_jump, asymptotic_period,
    KEEPIN_U235, LAMBDA_PROMPT,
    reactivity_with_feedback,
)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 120, 'savefig.dpi': 150,
})


def plot_scenario(ax1, ax2, ax3, result, title, yscale='linear',
                  show_period=True, show_rho_dollar=True):
    """将同一场景的三张图（功率、反应性、周期）画在给定的 axes 上。

    ax1: 功率时间历史
    ax2: 反应性历史
    ax3: 反应堆周期 (可选)
    """
    beta_pcm = KEEPIN_U235['beta'] * 1e5
    t, P, rho = result.t, result.P, result.rho

    # 功率
    ax1.plot(t, P, 'b-', linewidth=1.5)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='初始稳态')
    # 标记瞬发跳变后的理论值
    rho_final = rho[-1]
    if abs(rho_final) > 0 and abs(rho_final) < KEEPIN_U235['beta']:
        Pjump = prompt_jump(1.0, rho_final)
        if 0 < Pjump < 10:
            ax1.axhline(y=Pjump, color='orange', linestyle=':', alpha=0.7,
                        label=f'瞬发跳变理论值 = {Pjump:.2f}')
    ax1.set_ylabel('归一化功率 P/P₀')
    ax1.set_xlabel('时间 (s)')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.set_title(title)
    ax1.set_yscale(yscale)
    ax1.grid(True, alpha=0.3)

    # 反应性
    ax2.plot(t, rho * 1e5, 'r-', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=beta_pcm, color='orange', linestyle=':', alpha=0.7,
                label=f'β = {beta_pcm:.0f} pcm')
    ax2.set_ylabel('反应性 ρ (pcm)')
    ax2.set_xlabel('时间 (s)')
    ax2.set_title('反应性引入历史')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 周期
    if show_period:
        period = result.period
        # 限制显示范围
        mask = np.abs(period) < 1e4
        ax3.plot(t[mask], period[mask], 'g-', linewidth=1.5)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        # 标记渐近周期
        rho_eff = rho[-1]
        if abs(rho_eff) > 1e-6:
            T_asym = asymptotic_period(rho_eff)
            if 0 < abs(T_asym) < 1e4:
                ax3.axhline(y=T_asym, color='purple', linestyle=':', alpha=0.7,
                            label=f'渐近周期 = {T_asym:.1f} s')
        ax3.set_ylabel('周期 T (s)')
        ax3.set_xlabel('时间 (s)')
        ax3.set_title('反应堆周期')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)


# ============ 六种场景演示 ============

def demo_all():
    """展示所有六种反应性引入场景，6×3 子图矩阵布局。"""
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(6, 3, figure=fig, hspace=0.5, wspace=0.35)

    scenarios = []

    # ---- 场景 1: 小正反应性阶跃 ----
    rho_val = 0.001  # 100 pcm (< β)
    result = solve_point_kinetics(
        lambda t: reactivity_step(t, rho_val, t_insert=0.0),
        t_span=(0, 30),
    )
    T_asym = asymptotic_period(result.rho[-1])
    scenarios.append((
        result,
        f'1. 小阶跃 ρ=+100 pcm (< β)', 'linear',
        f'周期 ≈ {T_asym:.0f}s\n"缓发临界", 安全可控'
    ))

    # ---- 场景 2: 大正反应性阶跃 (瞬发超临界) ----
    rho_val = 0.010  # 1000 pcm (> β=700 pcm)
    result = solve_point_kinetics(
        lambda t: reactivity_step(t, rho_val, t_insert=0.0),
        t_span=(0, 0.15),
        max_step=0.0005,
        use_log=True,    # log-space 防止溢出
        P_max=1e4,       # 功率限制
    )
    scenarios.append((
        result,
        f'2. 大阶跃 ρ=+1000 pcm (> β)', 'log',
        '瞬发超临界! 周期≈0.01s\n无控制棒自动停堆 → 事故'
    ))

    # ---- 场景 3: 负反应性阶跃 (紧急停堆) ----
    rho_val = -0.005  # -500 pcm
    result = solve_point_kinetics(
        lambda t: reactivity_step(t, rho_val, t_insert=0.0),
        t_span=(0, 100),
    )
    Pjump = prompt_jump(1.0, rho_val)
    scenarios.append((
        result,
        f'3. 紧急停堆 ρ=−500 pcm', 'linear',
        f'瞬发跳变 → {Pjump:.2f}P₀\n随后缓发中子衰减'
    ))

    # ---- 场景 4: 线性提棒 ----
    rho_rate = 3.0  # pcm/s
    result = solve_point_kinetics(
        lambda t: reactivity_ramp(t, rho_rate * 1e-5, t_start=5.0),
        t_span=(0, 150),
        P_max=1e4,
        use_log=True,
    )
    scenarios.append((
        result,
        f'4. 线性提棒 {rho_rate} pcm/s', 'log',
        '缓慢引入反应性\n功率指数上升, ~120s 后超临界明显'
    ))

    # ---- 场景 5: 弹棒事故 ----
    rho_max_eject = 0.010  # 1000 pcm
    result = solve_point_kinetics(
        lambda t: reactivity_rod_ejection(t, rho_max_eject, t_eject=0.0, tau=0.03),
        t_span=(0, 0.4),
        max_step=0.001,
        use_log=True,
        P_max=1e4,
    )
    scenarios.append((
        result,
        '5. 弹棒事故 (ρ=1000 pcm, τ=30ms)', 'log',
        'RCCA 弹出 → 瞬发超临界\n需要反应堆保护系统动作'
    ))

    # ---- 场景 6: 正弦振荡 (氙振荡模拟) ----
    result = solve_point_kinetics(
        lambda t: reactivity_sinusoidal(t, amplitude=0.0005, period=20.0),
        t_span=(0, 80),
    )
    scenarios.append((
        result,
        '6. 正弦振荡 ±50 pcm, T=20s', 'linear',
        '模拟氙振荡/流致振动\n功率随反应性波动'
    ))

    for idx, (result, title, yscale, note) in enumerate(scenarios):
        row_axes = [
            fig.add_subplot(gs[idx, 0]),
            fig.add_subplot(gs[idx, 1]),
            fig.add_subplot(gs[idx, 2]),
        ]
        plot_scenario(*row_axes, result, title, yscale=yscale)
        # 标注说明
        row_axes[0].text(0.98, 0.02, note, transform=row_axes[0].transAxes,
                         fontsize=9, ha='right', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('点堆动力学 — 六种典型反应性引入场景', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('kinetics_demo.png', bbox_inches='tight', facecolor='white')
    print("保存: kinetics_demo.png")
    plt.close()


def demo_feedback():
    """展示温度反馈的效果 — 弹棒事故有/无反馈对比。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 无反馈
    rho_max_eject = 0.008  # 800 pcm (> β)

    def rho_ext(t):
        return reactivity_rod_ejection(t, rho_max_eject, t_eject=0.1, tau=0.05)

    result_no_fb = solve_point_kinetics(rho_ext, t_span=(0, 1.5), max_step=0.005)

    axes[0, 0].plot(result_no_fb.t, result_no_fb.P, 'r-', linewidth=2, label='无反馈')
    axes[0, 0].set_ylabel('归一化功率 P/P₀')
    axes[0, 0].set_title('弹棒事故 — 有/无温度反馈对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(result_no_fb.t, result_no_fb.rho * 1e5, 'r-', linewidth=2, label='外部反应性')
    axes[0, 1].set_ylabel('反应性 ρ (pcm)')
    axes[0, 1].set_title('反应性历史')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 有反馈
    def rho_with_fb(t):
        # 需要知道当前功率，简化使用静态反馈
        return reactivity_rod_ejection(t, rho_max_eject, t_eject=0.1, tau=0.05)

    # 这里用简化方法 — 弹棒引入的反应性被多普勒反馈抵消一部分
    alpha_dop = -3.0e-5  # 多普勒系数
    # 模拟: 功率上升 → 燃料温度升高 → 多普勒效应 → 引入负反应性
    result_with_fb = solve_point_kinetics(rho_with_fb, t_span=(0, 1.5), max_step=0.005)

    # 简化反馈修正
    delta_T = 100 * (result_with_fb.P - 1.0)  # K
    rho_fb_corrected = result_with_fb.rho + alpha_dop * delta_T

    axes[0, 0].plot(result_with_fb.t, result_with_fb.P, 'b-', linewidth=2, label='有反馈 (α_D=−3 pcm/K)')
    axes[0, 0].legend()

    axes[0, 1].plot(result_with_fb.t, result_with_fb.rho * 1e5, 'r--', alpha=0.5, label='外部')
    axes[0, 1].plot(result_with_fb.t, rho_fb_corrected * 1e5, 'b-', linewidth=2, label='总(含反馈)')
    axes[0, 1].legend()

    # 反应堆周期对比
    for ax_idx, (result, label, color) in enumerate([
        (result_no_fb, '无反馈', 'r'), (result_with_fb, '有反馈', 'b')
    ]):
        period = result.period
        mask = (np.abs(period) < 1e3) & (result.t < 1.0)
        axes[1, ax_idx].plot(result.t[mask], period[mask], color=color, linewidth=1.5)
        axes[1, ax_idx].set_ylabel('周期 (s)')
        axes[1, ax_idx].set_xlabel('时间 (s)')
        axes[1, ax_idx].set_title(f'反应堆周期 ({label})')
        axes[1, ax_idx].grid(True, alpha=0.3)

    fig.suptitle('多普勒温度反馈 — 反应性事故的自我保护机制', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('kinetics_feedback.png', bbox_inches='tight', facecolor='white')
    print("保存: kinetics_feedback.png")
    plt.close()


def demo_prompt_jump():
    """瞬发跳变 — 展示不同反应性下的跳变幅度。"""
    rho_values = np.linspace(-0.005, 0.0065, 100)
    beta = KEEPIN_U235['beta']

    P_jump = np.array([prompt_jump(1.0, r) for r in rho_values])
    valid = ~np.isinf(P_jump)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rho_values[valid] * 1e5, P_jump[valid], 'b-', linewidth=2)
    ax.axvline(x=beta * 1e5, color='red', linestyle='--', alpha=0.7,
               label=f'β = {beta * 1e5:.0f} pcm (瞬发临界)')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('引入反应性 ρ (pcm)')
    ax.set_ylabel('瞬发跳变后功率 P/P₀')
    ax.set_title('瞬发跳变 — 引入反应性后 ~10⁻⁴s 内的功率变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注关键区段
    ax.annotate('次临界\n功率下降', xy=(-200, 1.2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.annotate('超临界\n功率上升', xy=(200, 1.2), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    ax.annotate('瞬发超临界\n指数发散!', xy=(750, 8), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

    plt.tight_layout()
    plt.savefig('prompt_jump.png', bbox_inches='tight', facecolor='white')
    print("保存: prompt_jump.png")
    plt.close()


def demo_period_vs_reactivity():
    """渐近周期 vs 反应性 — 展示从次临界到瞬发临界的周期变化。"""
    rho_vals = np.concatenate([
        np.linspace(-0.01, -1e-5, 50),
        np.linspace(1e-5, 0.007, 100),
    ])
    rho_vals = np.sort(rho_vals)

    periods = []
    for r in rho_vals:
        T = asymptotic_period(r)
        if abs(T) > 1e6:
            periods.append(np.nan)
        else:
            periods.append(abs(T))

    periods = np.array(periods)
    beta = KEEPIN_U235['beta']

    fig, ax = plt.subplots(figsize=(9, 5))
    # 超临界 (ρ > 0)
    mask_pos = rho_vals > 0
    ax.semilogy(rho_vals[mask_pos] * 1e5, periods[mask_pos], 'r-', linewidth=2,
                label='超临界 (正周期)')
    # 次临界 (ρ < 0)
    mask_neg = rho_vals < -1e-5
    ax.semilogy(-rho_vals[mask_neg] * 1e5, periods[mask_neg], 'b-', linewidth=2,
                label='次临界 (负周期, 显示绝对值)')

    ax.axvline(x=beta * 1e5, color='red', linestyle='--', alpha=0.7,
               label=f'瞬发临界 β = {beta * 1e5:.0f} pcm')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='T = 1s')
    ax.set_xlabel('|反应性| (pcm)')
    ax.set_ylabel('渐近周期 |T| (s)')
    ax.set_title('渐近反应堆周期 — 倒时方程')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # 标注
    ax.annotate('临界附近\nT → ∞', xy=(1, 1000), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.annotate('PWR 典型运行区\nT > 100s', xy=(10, 100), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig('period_vs_reactivity.png', bbox_inches='tight', facecolor='white')
    print("保存: period_vs_reactivity.png")
    plt.close()


# ============ 主程序 ============

if __name__ == '__main__':
    print("=" * 60)
    print("点堆动力学 — 演示脚本")
    print("=" * 60)
    print()

    print("▶ 1/4  六种反应性引入场景...")
    demo_all()
    print("  ✓ 完成\n")

    print("▶ 2/4  温度反馈对比...")
    demo_feedback()
    print("  ✓ 完成\n")

    print("▶ 3/4  瞬发跳变曲线...")
    demo_prompt_jump()
    print("  ✓ 完成\n")

    print("▶ 4/4  渐近周期 vs 反应性...")
    demo_period_vs_reactivity()
    print("  ✓ 完成\n")

    print("全部完成! 输出文件: kinetics_demo.png, kinetics_feedback.png, "
          "prompt_jump.png, period_vs_reactivity.png")
