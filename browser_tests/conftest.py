"""Run the real Streamlit app in Chromium and retain diagnostics on failure."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    setattr(item, f"rep_{call.when}", outcome.get_result())


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    configured = os.environ.get("BROWSER_TEST_ARTIFACTS")
    path = Path(configured).resolve() if configured else tmp_path_factory.mktemp("browser-artifacts")
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def browser_server(artifacts):
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]

    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address", "127.0.0.1",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    with (artifacts / "streamlit.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Streamlit exited early with code {process.returncode}")
                try:
                    with urlopen(f"{url}/_stcore/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, URLError):
                    time.sleep(0.25)
            else:
                raise TimeoutError("Streamlit did not become healthy within 60 seconds")
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


@pytest.fixture
def page(request, browser_server, artifacts):
    with sync_playwright() as playwright:
        channel = os.environ.get("BROWSER_TEST_CHANNEL")
        browser = playwright.chromium.launch(headless=True, channel=channel)
        try:
            context = browser.new_context(
                accept_downloads=True, viewport={"width": 1440, "height": 1000},
            )
            try:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)
                page = context.new_page()
                page.set_default_timeout(30000)
                page.goto(browser_server, wait_until="domcontentloaded")
                yield page
            finally:
                failed = getattr(request.node, "rep_call", None)
                if failed is not None and failed.failed:
                    stem = request.node.name
                    try:
                        page.screenshot(path=str(artifacts / f"{stem}.png"), full_page=True)
                    except Exception:
                        pass  # The trace and server log still diagnose a closed page.
                    trace_path = str(artifacts / f"{stem}.zip")
                else:
                    trace_path = None
                try:
                    context.tracing.stop(path=trace_path)
                finally:
                    context.close()
        finally:
            browser.close()
