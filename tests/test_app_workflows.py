"""Streamlit result-page regressions using small deterministic solver histories."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from bateman import N_U_TOTAL
from point_kinetics import PointKineticsResult


APP = Path(__file__).resolve().parents[1] / 'app.py'


def _burnup_history(k_values):
    count = len(k_values)
    burnup = np.arange(count, dtype=float) * 5
    return {
        'burnup': burnup,
        'time_days': burnup * 3,
        'k_eff': np.asarray(k_values, dtype=float),
        'flux_fast': np.full(count, 1e13),
        'flux_thermal': np.full(count, 2e13),
        'N_U235': np.full(count, 0.04 * N_U_TOTAL),
        'N_U238': np.full(count, 0.96 * N_U_TOTAL),
        'N_Pu239': np.zeros(count),
        'N_FP': np.zeros(count),
        'Sigma_a1': np.full(count, 0.01),
        'Sigma_a2': np.full(count, 0.02),
        'Sigma_f2': np.full(count, 0.03),
    }


def _run_burnup_page(history, capture_downloads=False):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.radio[0].set_value(app.radio[0].options[-2]).run()
    with (patch('burnup_solver.run_burnup_coupled', return_value=history),
          patch('streamlit.download_button') as downloads):
        app.button[0].click().run()
    assert not app.exception
    return (app, downloads.call_args_list) if capture_downloads else app


def _kinetics_result(final_time):
    return PointKineticsResult(
        t=np.array([0.0, final_time / 2, final_time]),
        P=np.array([1.0, 1.1, 1.2]),
        C=np.ones((6, 3)),
        rho=np.array([0.0, 0.001, 0.001]),
        n_groups=6,
        info='deterministic regression result',
    )


def _run_kinetics_page(result, capture_downloads=False):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.radio[0].set_value(app.radio[0].options[-1]).run()
    with (patch('point_kinetics.solve_point_kinetics', return_value=result),
          patch('streamlit.download_button') as downloads):
        app.button[0].click().run()
    assert not app.exception
    return (app, downloads.call_args_list) if capture_downloads else app


def test_burnup_page_does_not_label_an_unbracketed_point_as_critical():
    app = _run_burnup_page(_burnup_history([1.2, 1.1, 1.05]))

    assert any(metric.label == 'k=1 燃耗' and metric.value == '未达到' for metric in app.metric)
    assert app.success
    assert not app.warning


def test_burnup_page_reports_early_stop_and_real_crossing():
    app = _run_burnup_page(_burnup_history([1.1, 1.01, 0.9]))

    assert any('提前结束' in warning.value for warning in app.warning)
    assert any(metric.label == 'k=1 燃耗' and metric.value != '未达到'
               for metric in app.metric)


def test_burnup_page_offers_complete_downloads_without_rerunning():
    app, calls = _run_burnup_page(_burnup_history([1.2, 1.1, 1.05]), True)

    assert app.success
    assert len(calls) == 2
    assert all(call.kwargs['on_click'] == 'ignore' for call in calls)
    assert 'burnup_MWd_per_kgU' in calls[0].args[1].decode('utf-8')
    record = json.loads(calls[1].args[1])
    assert len(record['history']['burnup']) == 3
    assert record['inputs']['initial_enrichment'] == 0.04


@pytest.mark.parametrize('final_time,stopped', [(40.0, False), (2.0, True)])
def test_kinetics_page_distinguishes_completion_from_protection_stop(final_time, stopped):
    app = _run_kinetics_page(_kinetics_result(final_time))

    assert bool(app.warning) is stopped
    assert bool(app.success) is not stopped
    assert any(metric.label == '最终归一化功率' for metric in app.metric)
    if stopped:
        assert any('计算提前停止' in warning.value for warning in app.warning)


def test_kinetics_page_offers_complete_downloads_without_rerunning():
    app, calls = _run_kinetics_page(_kinetics_result(40.0), True)

    assert app.success
    assert len(calls) == 2
    assert all(call.kwargs['on_click'] == 'ignore' for call in calls)
    assert 'time_s,normalized_power' in calls[0].args[1].decode('utf-8')
    record = json.loads(calls[1].args[1])
    assert record['summary']['n_time_points'] == 3
    assert record['inputs']['scenario'] == '小阶跃 (+100 pcm)'
