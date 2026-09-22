"""Boundary and regression tests for exports, resource guards, and diagnostics."""

import csv
from io import StringIO
import json

import numpy as np
import pytest

from resource_guards import estimate_two_group_resources, guard_problem_size
from solver import scan_critical_size, search_critical_boron, solve_two_group
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
    with pytest.raises(ValueError, match="dimension must be 1, 2, or 3"):
        guard_problem_size(4, 10)


def test_resource_guard_validates_dimension_and_grid_shape():
    assert estimate_two_group_resources(1, 100)['spatial_nodes'] == 100
    with pytest.raises(ValueError, match=r"requires 2 grid count\(s\)"):
        estimate_two_group_resources(2, 10)
    with pytest.raises(ValueError, match="positive integers"):
        estimate_two_group_resources(3, 10, 0, 10)


def test_one_dimensional_guard_rejects_dangerous_dense_input():
    with pytest.raises(ValueError, match="exceeding the safe limit"):
        solve_two_group(N=20_001)


def test_scan_interpolates_a_bracketed_critical_size():
    result = scan_critical_size(L_min=40, L_max=400, n_points=8, N=40)

    assert result['critical_bracketed'] is True
    assert result['k_crit'] == 1.0


def test_scan_and_boron_search_reject_invalid_ranges():
    with pytest.raises(ValueError, match="L_min must be smaller"):
        scan_critical_size(L_min=100, L_max=100, n_points=4, N=20)
    with pytest.raises(ValueError, match="non-negative"):
        search_critical_boron(C_range=(-1, 3000), N=20)


def test_unknown_material_parameter_is_rejected_in_all_dimensions():
    with pytest.raises(ValueError, match="Unknown material parameter"):
        solve_two_group(sections={'not_a_cross_section': 1.0}, N=10)


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
