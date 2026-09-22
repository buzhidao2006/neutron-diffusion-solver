"""Tests for shared convergence diagnostics used by the Web UI."""

import pytest

from convergence import convergence_summary, require_converged


def test_unconverged_summary_contains_actionable_fields():
    summary = convergence_summary({
        'converged': False,
        'n_iter': 12,
        'delta_k': 1e-4,
        'termination_reason': 'iteration budget exhausted',
    })

    assert summary['n_iter'] == 12
    assert summary['delta_k'] == pytest.approx(1e-4)
    assert '迭代次数' in summary['recommendation']


def test_require_converged_rejects_downstream_use():
    with pytest.raises(RuntimeError, match='did not converge'):
        require_converged({
            'converged': False, 'n_iter': 1, 'delta_k': 0.1,
            'termination_reason': 'budget exhausted',
        })


def test_require_converged_returns_summary_for_valid_result():
    summary = require_converged({
        'converged': True, 'n_iter': 10, 'delta_k': 1e-11,
        'termination_reason': 'converged',
    })

    assert summary['converged'] is True
