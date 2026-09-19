"""Boundary and regression tests for exports, resource guards, and diagnostics."""

import csv
from io import StringIO
import json

import numpy as np
import pytest

from resource_guards import guard_problem_size
from result_export import build_result_record, result_to_csv_bytes, result_to_json_bytes
from visualization import build_convergence_figure, final_convergence_rate


def test_resource_guard_accepts_grid_exactly_at_2d_limit():
    estimate = guard_problem_size(2, 250, 400)

    assert estimate["spatial_nodes"] == 100_000
    assert estimate["unknowns"] == 200_000


def test_resource_guard_rejects_grid_just_above_3d_limit_with_shape():
    with pytest.raises(ValueError, match=r"3D grid 30×25×41 has 30,750 spatial nodes"):
        guard_problem_size(3, 30, 25, 41)


def test_resource_guard_rejects_unknown_dimension():
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        guard_problem_size(1, 10)


def test_json_export_omits_missing_optional_diagnostics():
    payload = result_to_json_bytes(
        {"k_eff": np.float64(1.01), "x": np.array([1.0]), "phi1": [0.5], "phi2": [0.7]},
        "one_dimension", {}, "2026-01-01T00:00:00+00:00",
    )
    exported = json.loads(payload)

    assert exported["diagnostics"] == {"k_eff": 1.01}
    assert exported["flux"] == {"phi1": [0.5], "phi2": [0.7]}


def test_csv_export_flattens_3d_mesh_and_scalar_flux():
    x, y, z = np.meshgrid([1.0, 2.0], [3.0], [4.0], indexing="xy")
    result = {"x": np.array([1.0, 2.0]), "y": np.array([3.0]), "z": np.array([4.0]),
              "X": x, "Y": y, "Z": z, "phi1": np.ones_like(x), "phi2": 2.0}
    payload = result_to_csv_bytes(result, "three_dimension", {}, "2026-01-01T00:00:00+00:00")
    rows = list(csv.reader(StringIO(payload.decode("utf-8").split("# diagnostics=", 1)[1].split("\n", 1)[1])))

    assert rows[0] == ["x", "y", "z", "phi_fast", "phi_thermal"]
    assert rows[1] == ["1.0", "3.0", "4.0", "1.0", "2.0"]
    assert len(rows) == 3


def test_record_converts_nested_tuple_and_dictionary_values():
    record = build_result_record({}, "model", {"shape": (np.int64(2), {"value": np.float64(3.5)})},
                                 "2026-01-01T00:00:00+00:00")

    assert record["inputs"] == {"shape": [2, {"value": 3.5}]}


def test_final_convergence_rate_returns_none_after_zero_change():
    assert final_convergence_rate({"k_history": [1.0, 1.1, 1.1, 1.1]}) is None


def test_convergence_figure_accepts_zero_residual():
    figure = build_convergence_figure({"k_history": [1.0], "residual": [0.0]})

    assert figure.axes[1].get_yscale() == "log"
