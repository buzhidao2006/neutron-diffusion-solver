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
