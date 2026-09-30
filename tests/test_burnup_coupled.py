"""Regression checks for the complete teaching-model burnup workflow."""

import numpy as np
import pytest

from bateman import N_U_TOTAL
from burnup_solver import find_critical_burnup, run_burnup_coupled


def _run(**kwargs):
    return run_burnup_coupled(N_grid=30, **kwargs)


def test_burnup_history_conserves_tracked_nuclides_and_reaches_target():
    history = _run(total_burnup=20, n_burnup_steps=10)
    total = sum(history[key] for key in ('N_U235', 'N_U238', 'N_Pu239', 'N_FP'))

    assert len(history['burnup']) == 11
    assert history['burnup'][-1] == pytest.approx(20)
    np.testing.assert_allclose(total, N_U_TOTAL, rtol=1e-12)
    assert all(np.all(np.isfinite(values)) for values in history.values())
    assert np.all(np.diff(history['burnup']) > 0)
    assert np.all(np.diff(history['time_days']) > 0)


def test_burnup_step_refinement_has_a_stable_final_keff():
    coarse = _run(total_burnup=20, n_burnup_steps=10)
    fine = _run(total_burnup=20, n_burnup_steps=20)

    assert coarse['burnup'][-1] == fine['burnup'][-1] == 20
    assert abs(coarse['k_eff'][-1] - fine['k_eff'][-1]) < 0.01


def test_zero_burnup_returns_only_the_initial_state():
    history = _run(total_burnup=0, n_burnup_steps=8)

    assert len(history['burnup']) == 1
    assert history['burnup'][0] == history['time_days'][0] == 0
    assert history['N_U235'][0] == pytest.approx(0.04 * N_U_TOTAL)


def test_deep_subcritical_run_stops_after_recording_triggering_state():
    history = _run(total_burnup=60, n_burnup_steps=12)

    assert 10 < history['burnup'][-1] < 60
    assert history['k_eff'][-1] < 0.95
    assert len(history['burnup']) == len(history['time_days'])
    assert find_critical_burnup(history['burnup'], history['k_eff']) is not None


@pytest.mark.parametrize('kwargs', [
    {'total_burnup': -1},
    {'n_burnup_steps': 0},
    {'initial_enrichment': 0},
])
def test_burnup_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        _run(**kwargs)
