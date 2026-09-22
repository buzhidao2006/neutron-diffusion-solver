"""Regression tests for the unified command-line entry point."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

import main


def test_1d_command_runs_the_demo_script_once(monkeypatch):
    calls = []

    def fake_run(command, cwd):
        calls.append((command, cwd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    main.cmd_1d(SimpleNamespace())

    assert len(calls) == 1
    command, cwd = calls[0]
    assert command[0] == main.sys.executable
    assert command[1].endswith("diffusion_1d.py")
    assert cwd == str(main.PROJECT_DIR)


def test_diffusion_export_helper_writes_json():
    result = {
        'k_eff': 1.01,
        'x': np.array([1.0]),
        'phi1': np.array([0.4]),
        'phi2': np.array([1.0]),
    }
    output = main.PROJECT_DIR / '.test-result-export.json'
    try:
        main._export_diffusion_result(result, 'test_model', {'N': 1}, str(output))

        exported = json.loads(output.read_text(encoding='utf-8'))
        assert exported['model'] == 'test_model'
        assert exported['flux']['phi2'] == [1.0]
    finally:
        output.unlink(missing_ok=True)


def test_export_helper_rejects_unknown_file_extension():
    with pytest.raises(ValueError, match=r"\.json or \.csv"):
        main._export_diffusion_result(
            {}, 'model', {}, str(main.PROJECT_DIR / '.test-result-export.txt'),
        )
