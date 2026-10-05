"""Real-browser regressions for default burnup and point-kinetics downloads."""

import csv
import json
import math
import re

from playwright.sync_api import expect


BURNUP_COLUMNS = {
    "burnup": "burnup_MWd_per_kgU",
    "time_days": "time_days",
    "k_eff": "k_eff",
    "flux_fast": "avg_fast_flux_n_per_cm2_s",
    "flux_thermal": "avg_thermal_flux_n_per_cm2_s",
    "N_U235": "U235_atoms_per_cm3_x1e24",
    "N_U238": "U238_atoms_per_cm3_x1e24",
    "N_Pu239": "Pu239_atoms_per_cm3_x1e24",
    "N_FP": "fission_products_atoms_per_cm3_x1e24",
    "Sigma_a1": "Sigma_a1_cm_minus_1",
    "Sigma_a2": "Sigma_a2_cm_minus_1",
    "Sigma_f2": "Sigma_f2_cm_minus_1",
}


def _download(page, button_name, destination):
    with page.expect_download() as pending:
        page.get_by_role("button", name=button_name).click()
    download = pending.value
    assert download.suggested_filename == destination.name
    download.save_as(str(destination))
    assert destination.stat().st_size > 0


def _assert_csv_matches_json(csv_path, record, columns):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))

    history = record["history"]
    assert len(rows) == len(next(iter(history.values()))) > 2
    assert set(rows[0]) == set(columns.values())
    for key, column in columns.items():
        assert len(history[key]) == len(rows)
        for row, value in zip(rows, history[key]):
            if value is None:
                assert row[column] == ""
            else:
                assert math.isclose(float(row[column]), value, rel_tol=1e-10, abs_tol=1e-12)
    return rows


def test_default_burnup_early_stop_and_downloads(page, tmp_path):
    page.get_by_text("🔥 燃耗耦合", exact=True).click()
    page.get_by_role("button", name="🔥 开始耦合计算").click()

    expect(page.get_by_text("燃耗-扩散耦合计算结果", exact=True)).to_be_visible(timeout=180000)
    expect(page.get_by_text(re.compile(r"计算在第 \d+ 个燃耗步提前结束"))).to_be_visible()

    csv_path = tmp_path / "burnup_history.csv"
    json_path = tmp_path / "burnup_history.json"
    _download(page, "下载燃耗历史 CSV", csv_path)
    _download(page, "下载可复现记录 JSON", json_path)
    expect(page.get_by_text("燃耗-扩散耦合计算结果", exact=True)).to_be_visible()

    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["model"] == "burnup_coupled_diffusion"
    assert record["inputs"] == {
        "initial_enrichment": 0.04,
        "L_cm": 200.0,
        "N_grid": 150,
        "total_burnup_MWd_per_kgU": 50.0,
        "n_burnup_steps": 40,
    }
    rows = _assert_csv_matches_json(csv_path, record, BURNUP_COLUMNS)
    burnup = record["history"]["burnup"]
    times = record["history"]["time_days"]
    assert burnup[0] == times[0] == 0
    assert 0 < burnup[-1] < 50
    assert times[-1] > 0
    assert record["history"]["k_eff"][-1] < 0.95
    assert float(rows[0]["burnup_MWd_per_kgU"]) == burnup[0]
    assert float(rows[-1]["burnup_MWd_per_kgU"]) == burnup[-1]


def test_default_kinetics_completion_and_downloads(page, tmp_path):
    page.get_by_text("⏱️ 点堆动力学", exact=True).click()
    page.get_by_role("button", name="⏱️ 计算瞬态").click()

    expect(page.get_by_text("点堆动力学计算结果", exact=True)).to_be_visible(timeout=120000)
    expect(page.get_by_text(re.compile(r"瞬态计算完成：.*仿真至 40 s"))).to_be_visible()

    csv_path = tmp_path / "point_kinetics_history.csv"
    json_path = tmp_path / "point_kinetics_history.json"
    _download(page, "下载瞬态历史 CSV", csv_path)
    _download(page, "下载可复现记录 JSON", json_path)
    expect(page.get_by_text("点堆动力学计算结果", exact=True)).to_be_visible()

    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["model"] == "point_kinetics"
    assert record["inputs"]["scenario"] == "小阶跃 (+100 pcm)"
    assert record["inputs"]["simulation_end_s"] == 40
    assert record["inputs"]["reactivity_pcm"] == 100
    assert record["summary"]["n_groups"] == 6
    assert record["summary"]["n_time_points"] == len(record["history"]["time_s"])
    assert math.isclose(record["summary"]["final_time_s"], 40, abs_tol=1e-6)
    assert record["summary"]["final_power"] > 1
    columns = {key: key for key in record["history"]}
    rows = _assert_csv_matches_json(csv_path, record, columns)
    assert float(rows[0]["time_s"]) == 0
    assert math.isclose(float(rows[-1]["time_s"]), 40, abs_tol=1e-6)
