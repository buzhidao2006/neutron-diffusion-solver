"""End-to-end checks for the public command-line interface."""

import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_2g_cli_exports_a_reproducible_json_record():
    output = PROJECT_ROOT / ".test-cli-diffusion.json"
    try:
        completed = subprocess.run(
            [
                sys.executable, "main.py", "2g", "--N", "12", "--tol", "1e-7",
                "--max-iter", "500", "--output", str(output),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert completed.returncode == 0, completed.stderr
        exported = json.loads(output.read_text(encoding="utf-8"))
        assert exported["model"] == "two_group_diffusion_1d"
        assert exported["diagnostics"]["converged"] is True
        assert "phi1" in exported["flux"]
    finally:
        output.unlink(missing_ok=True)


def test_cli_rejects_invalid_grid_with_nonzero_status():
    completed = subprocess.run(
        [sys.executable, "main.py", "2g", "--N", "0"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 2
    assert "N must be an integer" in completed.stderr
