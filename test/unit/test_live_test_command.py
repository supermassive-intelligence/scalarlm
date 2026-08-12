"""Regression tests for the live profile's shell orchestration."""

import os
import subprocess
from pathlib import Path

TEST_COMMAND = Path(__file__).resolve().parents[2] / "cmd" / "test_command.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(0o755)


def test_live_profile_stops_when_image_build_fails(tmp_path):
    repo = tmp_path / "repo"
    (repo / "cmd").mkdir(parents=True)
    (repo / "test" / "live").mkdir(parents=True)
    (repo / "cmd" / "test_command.sh").write_text(TEST_COMMAND.read_text())

    _write_executable(repo / "scalarlm", "#!/bin/bash\nexit 23\n")
    _write_executable(
        repo / "test" / "live" / "run_live_server_tests.sh",
        '#!/bin/bash\ntouch "$HARNESS_MARKER"\n',
    )

    runner = tmp_path / "run-test-command.sh"
    _write_executable(
        runner,
        """#!/bin/bash
inspect_args() { :; }
red_bold() { printf '%s\\n' "$*"; }
green_bold() { printf '%s\\n' "$*"; }
blue_bold() { printf '%s\\n' "$*"; }
declare -A args=(
  [test-path]=""
  [--level]="live"
  [--tag]="cray:latest"
  [--coverage-path]="/tmp/coverage"
  [--verbose]="no"
  [--workers]="1"
  [--no-build]="no"
  [--model]="tiny-random/gemma-4-dense"
  [--live-target]="cpu"
  [--live-timeout]="30"
)
source "$1"
""",
    )

    harness_marker = tmp_path / "harness-ran"
    env = os.environ.copy()
    env["HARNESS_MARKER"] = str(harness_marker)
    result = subprocess.run(
        ["bash", str(runner), str(repo / "cmd" / "test_command.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert not harness_marker.exists()
    assert "Live server (cpu, tiny-random/gemma-4-dense) FAILED" in result.stdout
