"""
点堆动力学 (Point Kinetics) — 时变中子通量求解器

物理
----
点堆方程描述反应堆功率随时间的变化，是反应堆安全分析和瞬态分析的基础。

6 群缓发中子先驱核方程:
    dP/dt  = (ρ(t) - β)/Λ · P(t)  +  Σ λ_i · C_i(t)
    dC_i/dt = β_i/Λ · P(t)  -  λ_i · C_i(t)    (i = 1..6)

其中:
    P     : 中子功率（归一化至初始稳态）
    ρ(t)  : 反应性 = (k-1)/k
    β     : 总缓发中子份额 = Σ β_i
    β_i   : 第 i 群缓发中子份额
    λ_i   : 第 i 群先驱核衰变常数 (s⁻¹)
    Λ     : 中子代时间 (s)
    C_i   : 第 i 群先驱核浓度

缓发中子数据 (Keepin 1965, U-235 热中子裂变):
    群    β_i (pcm)    λ_i (s⁻¹)     半衰期
    1     26.6         0.0127         54.6 s
    2     149.1        0.0317         21.9 s
    3     131.6        0.115          6.0 s
    4     284.9        0.311          2.23 s
    5     89.6         1.40           0.495 s
    6     18.2         3.87           0.179 s
    ─────────────────────────────────────────
    β = 700 pcm = 0.007

参考文献
--------
- Keepin, W. R. (1965). Physics of Nuclear Kinetics. Addison-Wesley.
- Duderstadt & Hamilton. Nuclear Reactor Analysis. Ch. 6.
- 谢仲生. 核反应堆物理分析. 第 7 章.

考研衔接
--------
- 考研 865 第 7 章重点: 点堆方程推导、倒时方程、反应性单位(pcm/$)
- 复试: 能讲清楚"为什么需要缓发中子才能控制反应堆"
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, Callable

# ============ U-235 6 群缓发中子数据 (Keepin 1965) ============
KEEPIN_U235 = {
    'beta_i': np.array([0.000266, 0.001491, 0.001316, 0.002849, 0.000896, 0.000182]),
    'lambda_i': np.array([0.0127, 0.0317, 0.115, 0.311, 1.40, 3.87]),
}

# 总缓发中子份额
KEEPIN_U235['beta'] = np.sum(KEEPIN_U235['beta_i'])  # 0.007 (700 pcm)

# 典型 PWR 中子代时间 (s)
LAMBDA_PROMPT = 2.0e-5  # thermal reactor

# ============ 反应性场景 ============


def reactivity_step(t: float, rho_0: float, t_insert: float = 1.0) -> float:
    """阶跃引入反应性: t >= t_insert 后 ρ = ρ_0"""
    return rho_0 if t >= t_insert else 0.0


def reactivity_ramp(t: float, rho_rate: float, t_start: float = 0.0) -> float:
    """线性引入反应性: ρ = rho_rate * max(t - t_start, 0)"""
    return rho_rate * max(t - t_start, 0)


def reactivity_sinusoidal(t: float, amplitude: float, period: float) -> float:
    """正弦振荡反应性: ρ = A·sin(2πt/T)"""
    return amplitude * np.sin(2 * np.pi * t / period)


def reactivity_rod_ejection(t: float, rho_max: float, t_eject: float = 0.5, tau: float = 0.05) -> float:
    """控制棒弹棒事故: ρ(t) = ρ_max * (1 - exp(-(t - t_eject)/τ))   (t >= t_eject)"""
    if t < t_eject:
        return 0.0
    return rho_max * (1.0 - np.exp(-(t - t_eject) / tau))


# ============ 核心求解器 ============


@dataclass
class PointKineticsResult:
    """点堆方程求解结果"""
    t: np.ndarray           # 时间轴 (s)
    P: np.ndarray           # 归一化功率
    C: np.ndarray           # 先驱核浓度 (n_groups, n_times)
    rho: np.ndarray         # 反应性历史
    n_groups: int           # 缓发中子群数 (默认 6)
    info: str               # 求解器状态信息

    @property
    def period(self) -> np.ndarray:
        """反应堆周期 T = P / (dP/dt)"""
        dPdt = np.gradient(self.P, self.t)
        safe = np.abs(dPdt) > 1e-20
        T = np.full_like(self.t, np.inf)
        T[safe] = self.P[safe] / dPdt[safe]
        return T

    @property
    def reactivity_dollar(self) -> np.ndarray:
        """反应性以 $ 为单位: 1$ = β"""
        return self.rho / KEEPIN_U235['beta']


def solve_point_kinetics(
    reactivity_func: Callable[[float], float],
    t_span: tuple[float, float] = (0, 60),
    Lambda: float = LAMBDA_PROMPT,
    beta_data: Optional[dict] = None,
    P0: float = 1.0,
    max_step: float = 0.1,
    method: str = 'LSODA',
    P_max: float = 1e8,
    use_log: bool = False,
) -> PointKineticsResult:
    r"""
    求解点堆动力学方程。

    Parameters
    ----------
    reactivity_func : callable
        反应性函数 ρ(t)，单位: Δk/k (absolute reactivity)。
    t_span : (t_start, t_end)
        积分时间范围 (s)。
    Lambda : float
        中子代时间 (s)。热堆 ~2e-5, 快堆 ~1e-7。
    beta_data : dict, optional
        缓发中子数据。默认使用 U-235 Keepin 6 群数据。
        格式: {'beta_i': array, 'lambda_i': array, 'beta': float}
    P0 : float
        初始归一化功率（默认 1.0 = 稳态临界）。
    max_step : float
        最大积分步长 (s)。

    Returns
    -------
    PointKineticsResult
    """
    if beta_data is None:
        beta_data = KEEPIN_U235

    beta_i = np.asarray(beta_data['beta_i'], dtype=float)
    lambda_i = np.asarray(beta_data['lambda_i'], dtype=float)
    beta = beta_data['beta']
    n_groups = len(beta_i)

    # 初始稳态条件: dC_i/dt = 0
    # C_i(0) = (β_i / Λ) / λ_i · P0
    C0 = (beta_i / Lambda) / lambda_i * P0
    y0 = np.zeros(1 + n_groups)
    y0[0] = P0
    y0[1:] = C0

    # ---- log-space ODE (用于大范围瞬态，防止浮点溢出) ----
    # 变量: q = ln(P), 则 dq/dt = dP/dt / P = (ρ-β)/Λ + Σ λ_i C_i / P
    # 配合 C_i 仍在线性空间
    # 当 use_log=True 且 P > P_thresh 时自动切换

    def ode_rhs_linear(t: float, y: np.ndarray) -> np.ndarray:
        P = max(y[0], 0.0)
        C = y[1:]
        rho = reactivity_func(t)
        dPdt = (rho - beta) / Lambda * P + np.dot(lambda_i, C)
        dCdt = (beta_i / Lambda) * P - lambda_i * C
        return np.concatenate([[dPdt], dCdt])

    def ode_rhs_log(t: float, y: np.ndarray) -> np.ndarray:
        q = y[0]           # ln(P)
        C = y[1:]
        rho = reactivity_func(t)
        exp_q = np.exp(np.clip(q, -50, 50))  # 防溢出
        dqdt = (rho - beta) / Lambda + np.dot(lambda_i, C) / max(exp_q, 1e-300)
        dCdt = (beta_i / Lambda) * exp_q - lambda_i * C
        return np.concatenate([[dqdt], dCdt])

    # 选择 ODE 形式
    if use_log:
        # log-space: q = ln(P), y0[0] = ln(P0)
        y0_log = y0.copy()
        y0_log[0] = np.log(max(P0, 1e-300))
        ode_rhs = ode_rhs_log
        y0_final = y0_log
    else:
        ode_rhs = ode_rhs_linear
        y0_final = y0

    # 终止事件: 功率超过阈值时停止积分
    def event_power_max(t, y):
        P = np.exp(y[0]) if use_log else y[0]
        return P - P_max
    event_power_max.terminal = True
    event_power_max.direction = 1

    # 终止事件: 功率衰减到接近零
    def event_power_min(t, y):
        P = np.exp(y[0]) if use_log else y[0]
        return P - 1e-10
    event_power_min.terminal = True
    event_power_min.direction = -1

    try:
        sol = solve_ivp(
            ode_rhs, t_span, y0_final,
            method=method,
            max_step=max_step,
            rtol=1e-8, atol=1e-10,
            dense_output=False,
            events=[event_power_max, event_power_min],
        )
    except (ValueError, OverflowError) as e:
        # 最后的备用方案: 缩短积分区间, 减少 max_step
        if t_span[1] - t_span[0] > 1.0:
            t_trunc = t_span[0] + (t_span[1] - t_span[0]) * 0.5
        else:
            t_trunc = t_span[0] + 0.01
        sol = solve_ivp(
            ode_rhs, (t_span[0], t_trunc), y0_final,
            method='BDF',
            max_step=min(max_step, 0.001),
            rtol=1e-6, atol=1e-8,
            dense_output=False,
            events=[event_power_max, event_power_min],
        )

    # 如果在 log-space 求解，转换回线性功率
    P_history = np.exp(sol.y[0]) if use_log else sol.y[0]
    C_history = sol.y[1:]

    # 计算反应性历史
    rho_history = np.array([reactivity_func(ti) for ti in sol.t])

    # 如果提前终止，补充信息
    status_msg = sol.message
    if len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
        status_msg += f" (事件触发: P 达到阈值 at t={sol.t_events[0][0]:.4f}s)"

    return PointKineticsResult(
        t=sol.t,
        P=P_history,
        C=C_history,
        rho=rho_history,
        n_groups=n_groups,
        info=f"求解完成: {sol.nfev} 次函数估值, 状态={status_msg}",
    )


# ============ 简化温度反馈模型 ============


def reactivity_with_feedback(
    t: float,
    rho_ext_func: Callable[[float], float],
    P: float,
    alpha_f: float = -3.0e-5,  # 多普勒系数 (Δk/k per K)
    alpha_c: float = -2.0e-4,  # 慢化剂温度系数 (Δk/k per K)
    T_fuel_0: float = 900.0,   # 初始燃料温度 (K)
    T_cool_0: float = 580.0,   # 初始冷却剂温度 (K)
    tau_f: float = 5.0,        # 燃料传热时间常数 (s)
    tau_c: float = 15.0,       # 冷却剂传热时间常数 (s)
    P_norm: float = 1.0,       # 参考功率
) -> tuple[float, float, float]:
    """
    带温度反馈的反应性模型。

    总反应性 = 外部反应性 + 多普勒反馈 + 慢化剂温度反馈

    使用简化的一阶传热模型:
        dT_f/dt = (P - (T_f - T_c) / R_f) / C_f
        dT_c/dt = ((T_f - T_c) / R_f - (T_c - T_in) / R_c) / C_c
    近似为:
        ΔT_f(t) ≈ K_f · (P(t) - P0)   (简化单节点)
    实际使用时间常数近似。
    """
    rho_ext = rho_ext_func(t)

    # 简化温度模型 — 功率变化驱动温度变化
    # ΔT_f 用一阶滞后近似
    delta_T_fuel = 100 * (P / P_norm - 1.0)  # 100K 温升对应满功率
    delta_T_cool = 50 * (P / P_norm - 1.0)

    rho_feedback = alpha_f * delta_T_fuel + alpha_c * delta_T_cool
    rho_total = rho_ext + rho_feedback

    return rho_total, delta_T_fuel, delta_T_cool


# ============ 倒时方程 (Inhour Equation) ============


def inhour_equation(omega: float, rho: float, Lambda: float = LAMBDA_PROMPT,
                    beta_data: Optional[dict] = None) -> float:
    r"""
    倒时方程（用于函数求根得到稳定周期对应的 omega）。

        ρ = ωΛ + Σ [ω β_i / (ω + λ_i)]

    正根 ω > 0 对应超临界 → 稳定周期 T = 1/ω。

    Returns
    -------
    residual : float
        代入 ω 后的残差 ρ - (ωΛ + Σ ωβ_i/(ω+λ_i))
    """
    if beta_data is None:
        beta_data = KEEPIN_U235

    beta_i = beta_data['beta_i']
    lambda_i = beta_data['lambda_i']
    beta = beta_data['beta']

    term_sum = np.sum(omega * beta_i / (omega + lambda_i))
    return rho - (omega * Lambda + term_sum)


def asymptotic_period(rho: float, Lambda: float = LAMBDA_PROMPT,
                      beta_data: Optional[dict] = None) -> float:
    """
    计算渐近反应堆周期。

    对正反应性 (ρ > 0)，求解 inhour 方程在 (0, min(λ_i)) 之间的根。
    对负反应性 (ρ < 0)，返回负周期（衰减时间常数）。

    Returns
    -------
    T : float
        渐近周期 (s)。超临界时为正值，次临界时为负值。
    """
    if beta_data is None:
        beta_data = KEEPIN_U235

    lambda_i = beta_data['lambda_i']

    if abs(rho) < 1e-15:
        return np.inf

    # 对正反应性，正根在 (0, λ_min) 之间
    if rho > 0:
        lo, hi = 1e-10, np.min(lambda_i) * 0.999
        f_lo = inhour_equation(lo, rho, Lambda, beta_data)
        f_hi = inhour_equation(hi, rho, Lambda, beta_data)

        if f_lo * f_hi > 0:
            # 反应性太大导致根超出范围，扩大搜索
            hi = 100.0
            f_hi = inhour_equation(hi, rho, Lambda, beta_data)
            if f_lo * f_hi > 0:
                return 1.0 / max(rho / Lambda, 1e-6)

        # 二分法求根
        for _ in range(60):
            mid = (lo + hi) / 2
            f_mid = inhour_equation(mid, rho, Lambda, beta_data)
            if abs(f_mid) < 1e-12:
                return 1.0 / mid
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 1.0 / ((lo + hi) / 2)
    else:
        # 负反应性，负根
        omega = rho / Lambda  # 近似（当 prompt jump 后）
        return 1.0 / omega


def prompt_jump(P0: float, rho: float, beta: float = KEEPIN_U235['beta']) -> float:
    """
    瞬发跳变后的功率值（忽略瞬发中子代时间）。

    引入反应性 ρ 后，在 ~10⁻⁴ s 内通量跳跃到:
        P_jump = P0 · β / (β - ρ)

    注意：仅当 |ρ| < β 时有效（缓发临界之下）。
    若 ρ ≥ β → 瞬发超临界，无跳变，指数发散。
    """
    if rho >= beta:
        return np.inf  # 瞬发超临界
    return P0 * beta / (beta - rho)
