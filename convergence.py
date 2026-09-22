"""Shared convergence diagnostics for CLI, Web UI, and tests."""

import math


def convergence_summary(result):
    """Normalize solver convergence fields into a user-facing diagnostic record."""
    converged = bool(result.get("converged", False))
    n_iter = int(result.get("n_iter", 0))
    delta_k = float(result.get("delta_k", math.inf))
    reason = str(result.get("termination_reason", "unknown termination reason"))
    recommendation = (
        ""
        if converged
        else "增大最大迭代次数、适度放宽容差，或改用 Chebyshev 加速后再使用结果。"
    )
    return {
        "converged": converged,
        "n_iter": n_iter,
        "delta_k": delta_k,
        "termination_reason": reason,
        "recommendation": recommendation,
    }


def require_converged(result):
    """Raise a clear error when a downstream workflow receives an invalid result."""
    summary = convergence_summary(result)
    if not summary["converged"]:
        raise RuntimeError(
            f"Solver did not converge after {summary['n_iter']} iterations "
            f"(|delta_k|={summary['delta_k']:.3e}): {summary['termination_reason']}"
        )
    return summary
