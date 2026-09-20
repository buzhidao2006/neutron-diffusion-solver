"""Tests for reproducible solver-result exports."""

import csv
from io import StringIO
import json

import numpy as np

from result_export import (
    build_burnup_record,
    burnup_history_to_csv_bytes,
    burnup_history_to_json_bytes,
    build_result_record,
    result_to_csv_bytes,
    result_to_json_bytes,
)


def _sample_result():
    return {
        'k_eff': 1.0123,
        'n_iter': 12,
        'converged': True,
        'delta_k': 1e-11,
        'termination_reason': 'converged',
        'k_history': [1.0, 1.0123],
        'residual': [0.1, 1e-9],
        'x': np.array([10.0, 20.0]),
        'phi1': np.array([0.4, 1.0]),
        'phi2': np.array([0.6, 0.9]),
    }


def test_json_export_preserves_inputs_diagnostics_and_fluxes():
    payload = result_to_json_bytes(
        _sample_result(), 'two_group_diffusion_1d', {'N': 2}, '2026-01-01T00:00:00+00:00',
    )
    exported = json.loads(payload)

    assert exported['inputs'] == {'N': 2}
    assert exported['diagnostics']['converged'] is True
    assert exported['flux']['phi1'] == [0.4, 1.0]
    assert exported['coordinates']['x'] == [10.0, 20.0]


def test_csv_export_contains_metadata_header_and_flux_rows():
    payload = result_to_csv_bytes(
        _sample_result(), 'two_group_diffusion_1d', {'N': 2}, '2026-01-01T00:00:00+00:00',
    ).decode('utf-8')
    lines = payload.splitlines()
    rows = list(csv.reader(StringIO('\n'.join(lines[4:]))))

    assert lines[0] == '# neutron-diffusion-solver export; model=two_group_diffusion_1d'
    assert rows[0] == ['x', 'phi_fast', 'phi_thermal']
    assert rows[1] == ['10.0', '0.4', '0.6']


def test_record_converts_numpy_values_to_standard_python_types():
    record = build_result_record(
        _sample_result(), 'two_group_diffusion_1d', {'N': np.int64(2)},
        '2026-01-01T00:00:00+00:00',
    )

    assert record['inputs']['N'] == 2
    assert isinstance(record['diagnostics']['k_eff'], float)


def test_csv_export_flattens_2d_coordinate_meshes_with_fluxes():
    result = _sample_result() | {
        'x': np.array([10.0, 20.0]),
        'y': np.array([30.0, 40.0]),
        'X': np.array([[10.0, 20.0], [10.0, 20.0]]),
        'Y': np.array([[30.0, 30.0], [40.0, 40.0]]),
        'phi1': np.ones((2, 2)),
        'phi2': np.full((2, 2), 2.0),
    }
    payload = result_to_csv_bytes(result, 'two_group_diffusion_2d', {},
                                  '2026-01-01T00:00:00+00:00').decode('utf-8')
    rows = list(csv.reader(StringIO('\n'.join(payload.splitlines()[4:]))))

    assert rows[0] == ['x', 'y', 'phi_fast', 'phi_thermal']
    assert rows[1] == ['10.0', '30.0', '1.0', '2.0']
    assert len(rows) == 5


def test_burnup_exports_preserve_inputs_and_each_history_step():
    history = {
        'burnup': np.array([0.0, 5.0]),
        'time_days': np.array([0.0, 42.0]),
        'k_eff': np.array([1.05, 0.99]),
        'flux_fast': np.array([1.0e13, 0.9e13]),
        'flux_thermal': np.array([2.0e13, 1.8e13]),
        'N_U235': np.array([0.04, 0.03]),
    }
    inputs = {'initial_enrichment': 0.04, 'N_grid': 100}
    timestamp = '2026-01-01T00:00:00+00:00'

    record = build_burnup_record(history, inputs, timestamp)
    exported_json = json.loads(burnup_history_to_json_bytes(history, inputs, timestamp))
    rows = list(csv.reader(StringIO(
        burnup_history_to_csv_bytes(history, inputs, timestamp).decode('utf-8').split(
            '# inputs=', 1,
        )[1].split('\n', 1)[1],
    )))

    assert record['model'] == 'burnup_coupled_diffusion'
    assert exported_json['history']['burnup'] == [0.0, 5.0]
    assert rows[0][:3] == ['burnup_MWd_per_kgU', 'time_days', 'k_eff']
    assert rows[2][2] == '0.99'
