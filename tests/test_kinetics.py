"""点堆动力学求解器的单元测试。"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from point_kinetics import (
    solve_point_kinetics, KEEPIN_U235, LAMBDA_PROMPT,
    reactivity_step, reactivity_ramp, reactivity_sinusoidal,
    reactivity_rod_ejection,
    prompt_jump, asymptotic_period, inhour_equation,
)


class TestPromptJump:
    """瞬发跳变公式的测试。"""

    def test_negative_reactivity_below_one(self):
        """负反应性下功率应跳变到 < 1。"""
        Pj = prompt_jump(1.0, -0.005)
        assert Pj < 1.0

    def test_positive_reactivity_above_one(self):
        """正反应性下功率应跳变到 > 1。"""
        Pj = prompt_jump(1.0, 0.001)
        assert Pj > 1.0

    def test_zero_reactivity_unchanged(self):
        """零反应性下功率不变。"""
        Pj = prompt_jump(1.0, 0.0)
        assert Pj == pytest.approx(1.0)

    def test_superprompt_critical_diverges(self):
        """ρ ≥ β 时瞬发超临界 — 返回 inf。"""
        Pj = prompt_jump(1.0, KEEPIN_U235['beta'] + 0.001)
        assert np.isinf(Pj)


class TestInhourEquation:
    """倒时方程测试。"""

    def test_root_at_zero_reactivity(self):
        """ρ=0 时，ω=0 应是根。"""
        residual = inhour_equation(0.0, 0.0)
        assert abs(residual) < 1e-15

    def test_positive_rho_positive_omega(self):
        """正反应性：存在正根 ω>0。"""
        rho = 0.001  # 100 pcm
        omega = np.linspace(0.001, 0.1, 100)
        residuals = [inhour_equation(w, rho) for w in omega]
        # 应存在区间内变号的点
        sign_changes = sum(
            1 for i in range(len(residuals) - 1)
            if residuals[i] * residuals[i + 1] < 0
        )
        assert sign_changes >= 1, "应至少有一个根"

    def test_asymptotic_period_finite(self):
        """渐近周期对于非零反应性应有限。"""
        T = asymptotic_period(0.001)  # 100 pcm
        assert 0 < T < 1e6, f"周期 T = {T}"

    def test_period_decreases_with_higher_reactivity(self):
        """更高反应性 → 更短周期。"""
        T1 = asymptotic_period(0.0005)  # 50 pcm
        T2 = asymptotic_period(0.001)   # 100 pcm
        assert T2 < T1, f"T(50pcm)={T1:.1f}s, T(100pcm)={T2:.1f}s"


class TestPointKineticsSolver:
    """点堆方程求解器测试。"""

    def test_step_insertion_runs(self):
        """阶跃反应性引入应成功求解。"""
        result = solve_point_kinetics(
            lambda t: reactivity_step(t, 0.001, t_insert=0.0),
            t_span=(0, 10),
        )
        assert len(result.t) > 10
        assert result.P[0] == pytest.approx(1.0)

    def test_power_increases_with_positive_rho(self):
        """正反应性：功率应上升。"""
        result = solve_point_kinetics(
            lambda t: reactivity_step(t, 0.001, t_insert=0.0),
            t_span=(0, 10),
        )
        assert result.P[-1] > 1.0

    def test_power_decreases_with_negative_rho(self):
        """负反应性：功率应下降。"""
        result = solve_point_kinetics(
            lambda t: reactivity_step(t, -0.005, t_insert=0.0),
            t_span=(0, 50),
        )
        assert result.P[-1] < 1.0

    def test_steady_state_with_zero_rho(self):
        """零反应性：功率应保持稳态。"""
        result = solve_point_kinetics(
            lambda t: 0.0,
            t_span=(0, 5),
        )
        # 稳态保持（允许很小的数值漂移）
        assert abs(result.P[-1] - 1.0) < 1e-6

    def test_rho_history_matches_function(self):
        """返回的反应性历史应与输入函数一致。"""
        rho_val = 0.002
        result = solve_point_kinetics(
            lambda t: reactivity_step(t, rho_val, t_insert=0.0),
            t_span=(0, 5),
        )
        # 初始 ρ=0（t_insert=0，所以ρ直接生效）
        assert result.rho[0] == pytest.approx(rho_val)

    def test_ramp_rho_monotonic(self):
        """斜坡引入：反应性应单调增加。"""
        result = solve_point_kinetics(
            lambda t: reactivity_ramp(t, 3e-5, t_start=1.0),
            t_span=(0, 20),
        )
        drho = np.diff(result.rho)
        # 反应性应单调不减
        assert np.all(drho >= -1e-16)

    def test_sinusoidal_rho_bounded(self):
        """正弦振荡：反应性应在 ±振幅 范围内。"""
        amp = 0.0005
        result = solve_point_kinetics(
            lambda t: reactivity_sinusoidal(t, amp, period=10.0),
            t_span=(0, 20),
        )
        assert np.max(result.rho) <= amp * 1.01
        assert np.min(result.rho) >= -amp * 1.01

    def test_result_period_property(self):
        """周期属性应返回合理值。"""
        result = solve_point_kinetics(
            lambda t: reactivity_step(t, 0.001, t_insert=0.0),
            t_span=(0, 20),
        )
        period = result.period
        assert len(period) == len(result.t)
        # 稳定状态下周期应为正值
        assert period[-1] > 0

    def test_result_reactivity_dollar(self):
        """反应性美元单位转换。"""
        result = solve_point_kinetics(
            lambda t: reactivity_step(t, KEEPIN_U235['beta'], t_insert=0.0),
            t_span=(0, 5),
        )
        # ρ = β → 1$
        assert result.reactivity_dollar[-1] == pytest.approx(1.0, rel=1e-6)


class TestRodEjection:
    """弹棒事故场景测试。"""

    def test_rho_exponential_approach(self):
        """弹棒场景：反应性应以指数形式接近最大值。"""
        rho_max = 0.008
        tau = 0.05
        t_start = 0.0
        # 解析：t ≥ t_start 时，ρ = ρ_max * (1 - exp(-(t - t_start)/τ))
        for t in [0.01, 0.05, 0.1]:
            rho = reactivity_rod_ejection(t, rho_max, t_eject=t_start, tau=tau)
            expected = rho_max * (1.0 - np.exp(-t / tau))
            assert rho == pytest.approx(expected, rel=1e-10)
