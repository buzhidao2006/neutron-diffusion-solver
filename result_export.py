"""Portable CSV and JSON exports for reproducible diffusion calculations."""

import csv
from datetime import datetime, timezone
from io import StringIO
import json

import numpy as np


def _to_builtin(value):
    """Convert NumPy values recursively into JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def build_result_record(result, model, inputs, generated_at=None):
    """Create a self-contained, reproducible record of a solver calculation."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    diagnostics = {
        key: result[key]
        for key in ('k_eff', 'n_iter', 'converged', 'delta_k', 'termination_reason',
                    'k_history', 'residual')
        if key in result
    }
    return _to_builtin({
        'schema_version': 1,
        'generated_at': timestamp,
        'model': model,
        'inputs': inputs,
        'diagnostics': diagnostics,
    })


def result_to_json_bytes(result, model, inputs, generated_at=None):
    """Export metadata, diagnostics, coordinates, and fluxes as UTF-8 JSON."""
    record = build_result_record(result, model, inputs, generated_at)
    record['coordinates'] = {
        key: result[key] for key in ('x', 'y', 'z') if key in result
    }
    record['flux'] = {
        key: result[key] for key in ('phi1', 'phi2') if key in result
    }
    return json.dumps(_to_builtin(record), ensure_ascii=False, indent=2).encode('utf-8')


def result_to_csv_bytes(result, model, inputs, generated_at=None):
    """Export flux data to CSV with comment-prefixed reproducibility metadata."""
    record = build_result_record(result, model, inputs, generated_at)
    stream = StringIO(newline='')
    stream.write(f"# neutron-diffusion-solver export; model={record['model']}\n")
    stream.write(f"# generated_at={record['generated_at']}\n")
    stream.write(f"# inputs={json.dumps(record['inputs'], ensure_ascii=False, sort_keys=True)}\n")
    stream.write(f"# diagnostics={json.dumps(record['diagnostics'], ensure_ascii=False)}\n")

    dimensions = [key for key in ('x', 'y', 'z') if key in result]
    if 'X' in result:
        coordinate_arrays = [result[key.upper()] for key in dimensions]
    else:
        coordinate_arrays = [result['x']]
    phi1, phi2 = np.broadcast_arrays(result['phi1'], result['phi2'])
    writer = csv.writer(stream)
    writer.writerow([*dimensions, 'phi_fast', 'phi_thermal'])
    for row in zip(*[array.ravel() for array in (*coordinate_arrays, phi1, phi2)]):
        writer.writerow(row)
    return stream.getvalue().encode('utf-8')


_BURNUP_COLUMNS = (
    ('burnup', 'burnup_MWd_per_kgU'),
    ('time_days', 'time_days'),
    ('k_eff', 'k_eff'),
    ('flux_fast', 'avg_fast_flux_n_per_cm2_s'),
    ('flux_thermal', 'avg_thermal_flux_n_per_cm2_s'),
    ('N_U235', 'U235_atoms_per_cm3_x1e24'),
    ('N_U238', 'U238_atoms_per_cm3_x1e24'),
    ('N_Pu239', 'Pu239_atoms_per_cm3_x1e24'),
    ('N_FP', 'fission_products_atoms_per_cm3_x1e24'),
    ('Sigma_a1', 'Sigma_a1_cm_minus_1'),
    ('Sigma_a2', 'Sigma_a2_cm_minus_1'),
    ('Sigma_f2', 'Sigma_f2_cm_minus_1'),
)


def build_burnup_record(history, inputs, generated_at=None):
    """Create a reproducible record for a coupled burnup history."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    return _to_builtin({
        'schema_version': 1,
        'generated_at': timestamp,
        'model': 'burnup_coupled_diffusion',
        'inputs': inputs,
        'history': {key: history[key] for key, _ in _BURNUP_COLUMNS if key in history},
    })


def burnup_history_to_json_bytes(history, inputs, generated_at=None):
    """Export the complete coupled burnup history as UTF-8 JSON."""
    return json.dumps(
        build_burnup_record(history, inputs, generated_at), ensure_ascii=False, indent=2,
    ).encode('utf-8')


def burnup_history_to_csv_bytes(history, inputs, generated_at=None):
    """Export each burnup step in a flat CSV with reproducibility metadata."""
    record = build_burnup_record(history, inputs, generated_at)
    stream = StringIO(newline='')
    stream.write(f"# neutron-diffusion-solver export; model={record['model']}\n")
    stream.write(f"# generated_at={record['generated_at']}\n")
    stream.write(f"# inputs={json.dumps(record['inputs'], ensure_ascii=False, sort_keys=True)}\n")
    columns = [(key, label) for key, label in _BURNUP_COLUMNS if key in history]
    writer = csv.writer(stream)
    writer.writerow([label for _, label in columns])
    for row in zip(*(np.asarray(history[key]).ravel() for key, _ in columns)):
        writer.writerow(row)
    return stream.getvalue().encode('utf-8')


def _finite_series(values):
    """Convert a numeric series to JSON-safe values, representing infinities as null."""
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values).ravel()]


def point_kinetics_record(result, inputs, generated_at=None):
    """Build a reproducible point-kinetics record and complete time history."""
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    precursor = np.asarray(result.C)
    history = {
        'time_s': _finite_series(result.t),
        'normalized_power': _finite_series(result.P),
        'reactivity': _finite_series(result.rho),
        'reactor_period_s': _finite_series(result.period),
    }
    for group_index, values in enumerate(precursor, start=1):
        history[f'precursor_group_{group_index}'] = _finite_series(values)
    peak_index = int(np.argmax(result.P))
    return {
        'schema_version': 1,
        'generated_at': timestamp,
        'model': 'point_kinetics',
        'inputs': _to_builtin(inputs),
        'solver_info': result.info,
        'summary': {
            'n_groups': int(result.n_groups),
            'n_time_points': int(len(result.t)),
            'final_time_s': float(result.t[-1]),
            'final_power': float(result.P[-1]),
            'peak_power': float(result.P[peak_index]),
            'peak_time_s': float(result.t[peak_index]),
            'final_reactivity': float(result.rho[-1]),
        },
        'history': history,
    }


def point_kinetics_to_json_bytes(result, inputs, generated_at=None):
    """Export point-kinetics inputs, solver status, and histories as UTF-8 JSON."""
    return json.dumps(
        point_kinetics_record(result, inputs, generated_at),
        ensure_ascii=False, indent=2, allow_nan=False,
    ).encode('utf-8')


def point_kinetics_to_csv_bytes(result, inputs, generated_at=None):
    """Export point-kinetics time series with reproducibility metadata as CSV."""
    record = point_kinetics_record(result, inputs, generated_at)
    stream = StringIO(newline='')
    stream.write(f"# neutron-diffusion-solver export; model={record['model']}\n")
    stream.write(f"# generated_at={record['generated_at']}\n")
    stream.write(f"# inputs={json.dumps(record['inputs'], ensure_ascii=False, sort_keys=True)}\n")
    stream.write(f"# summary={json.dumps(record['summary'], ensure_ascii=False)}\n")
    stream.write(f"# solver_info={record['solver_info']}\n")
    writer = csv.writer(stream)
    columns = list(record['history'])
    writer.writerow(columns)
    writer.writerows(zip(*(record['history'][key] for key in columns)))
    return stream.getvalue().encode('utf-8')
