"""Reusable numerical-diagnostics figures for diffusion solver results."""

from plotting import configure_matplotlib_fonts

configure_matplotlib_fonts()

import matplotlib.pyplot as plt
import numpy as np


def build_convergence_figure(result):
    """Plot eigenvalue and residual histories from a solver result.

    The helper accepts short histories as well, so it remains useful when a
    loose tolerance converges in only one or two iterations.
    """
    k_history = np.asarray(result.get("k_history", []), dtype=float)
    residual = np.asarray(result.get("residual", []), dtype=float)
    iterations = np.arange(1, len(k_history) + 1)

    fig, (k_axis, residual_axis) = plt.subplots(1, 2, figsize=(12, 4.2))

    if k_history.size:
        k_axis.plot(iterations, k_history, color="#6c5ce7", marker="o", markersize=3,
                    linewidth=1.8)
        k_axis.axhline(k_history[-1], color="#636e72", linestyle="--", linewidth=1,
                       label=f"最终 k_eff = {k_history[-1]:.6f}")
        k_axis.legend(fontsize=9)
    else:
        k_axis.text(0.5, 0.5, "没有 k_eff 迭代历史", ha="center", va="center",
                    transform=k_axis.transAxes)
    k_axis.set_title("k_eff 收敛历史", fontweight="bold")
    k_axis.set_xlabel("迭代次数")
    k_axis.set_ylabel("k_eff")
    k_axis.grid(True, alpha=0.25)

    if residual.size:
        residual_axis.semilogy(np.arange(1, len(residual) + 1),
                               np.maximum(np.abs(residual), np.finfo(float).tiny),
                               color="#00b894", marker="o", markersize=3, linewidth=1.8)
    else:
        residual_axis.text(0.5, 0.5, "没有残差历史", ha="center", va="center",
                           transform=residual_axis.transAxes)
    residual_axis.set_title("通量残差（对数坐标）", fontweight="bold")
    residual_axis.set_xlabel("迭代次数")
    residual_axis.set_ylabel("相对残差")
    residual_axis.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    return fig


def final_convergence_rate(result):
    """Return the final |Δk| contraction factor, or ``None`` if unavailable."""
    k_history = np.asarray(result.get("k_history", []), dtype=float)
    if len(k_history) < 3:
        return None
    changes = np.abs(np.diff(k_history))
    if changes[-2] == 0:
        return None
    return float(changes[-1] / changes[-2])
