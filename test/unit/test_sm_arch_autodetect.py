"""
Unit tests for the nvidia-smi sm_arch auto-detect logic used by the `up` and
`build-image` bashly commands.

Regression coverage for #221: the auto-detect line must space-join and
dedupe *all* GPUs' compute capabilities into TORCH_CUDA_ARCH_LIST, not just
the first line of `nvidia-smi` output. An earlier version wrapped the
command substitution in `(...)`, which silently collapsed it to a bash
array and, on unindexed expansion, discarded every GPU past the first.

These tests extract the actual assignment line out of the cmd/*.sh sources
(rather than reimplementing it) so a future regression in the real script
fails the test.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

SM_ARCH_FILES = [
    REPO_ROOT / "cmd" / "up_command.sh",
    REPO_ROOT / "cmd" / "build_image_command.sh",
]

SM_ARCH_LINE_RE = re.compile(r"^\s*sm_arch=\$\(nvidia-smi\b.*\)\s*$", re.MULTILINE)


def _extract_sm_arch_assignment(path: Path) -> str:
    match = SM_ARCH_LINE_RE.search(path.read_text())
    assert match, f"no `sm_arch=$(nvidia-smi ...)` assignment found in {path}"
    return match.group(0).strip()


def _fake_nvidia_smi(tmp_path: Path, compute_caps: list[str]) -> Path:
    script = tmp_path / "nvidia-smi"
    body = "".join(f"echo '{cap}'\n" for cap in compute_caps)
    script.write_text(f"#!/bin/sh\n{body}")
    script.chmod(0o755)
    return tmp_path


def _run_sm_arch_assignment(path: Path, compute_caps: list[str], tmp_path: Path) -> str:
    bin_dir = _fake_nvidia_smi(tmp_path, compute_caps)
    assignment = _extract_sm_arch_assignment(path)

    result = subprocess.run(
        ["bash", "-c", f'{assignment}; echo "$sm_arch"'],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("path", SM_ARCH_FILES, ids=lambda p: p.name)
def test_sm_arch_autodetect_joins_and_dedupes_all_gpus(path, tmp_path):
    output = _run_sm_arch_assignment(path, ["8.9", "8.0", "8.9"], tmp_path)
    assert output == "8.0 8.9"


@pytest.mark.parametrize("path", SM_ARCH_FILES, ids=lambda p: p.name)
def test_sm_arch_autodetect_single_gpu(path, tmp_path):
    output = _run_sm_arch_assignment(path, ["9.0"], tmp_path)
    assert output == "9.0"


def test_build_image_command_passes_multi_gpu_arch_list_as_one_docker_arg(tmp_path):
    """
    build_image_command.sh assembles its docker invocation as a single
    string and runs it through `eval`, unlike up_command.sh (which passes
    TORCH_CUDA_ARCH_LIST via an env-var prefix, safe from word-splitting).
    Under `eval`, an unquoted multi-GPU value like "8.0 8.9" splits into two
    separate arguments, so `docker build` sees a stray extra positional arg.
    This exercises the real eval path end-to-end with a stub `docker` that
    records exactly how it was invoked.
    """
    bin_dir = _fake_nvidia_smi(tmp_path, ["8.9", "8.0", "8.9"])

    docker_log = tmp_path / "docker_args.log"
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        "#!/bin/sh\n"
        f'for a in "$@"; do echo "$a" >> "{docker_log}"; done\n'
    )
    docker_stub.chmod(0o755)

    harness = tmp_path / "run.sh"
    harness.write_text(
        "inspect_args() { :; }\n"
        'green_bold() { echo "$@"; }\n'
        'declare -A args=( [target]="nvidia" [sm_arch]="auto" )\n'
        f'source "{REPO_ROOT / "cmd" / "build_image_command.sh"}"\n'
    )

    subprocess.run(
        ["bash", str(harness)],
        cwd=tmp_path,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )

    docker_args = docker_log.read_text().splitlines()
    assert "TORCH_CUDA_ARCH_LIST=8.0 8.9" in docker_args, docker_args
