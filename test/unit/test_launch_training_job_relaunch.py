"""
Unit tests for the auto-relaunch additions to launch_training_job.py.

Covers training-lifecycle.md §3.2 (--signal flag, per-slice --time
semantics) and §3.4 / §5.4 (resubmit.sh writer). The slurm-side and
status.json-side relaunch behavior is covered separately by
test_stop_flag.py and (forthcoming) test_training_entrypoint.py.
"""

import os
import shlex
import subprocess

import pytest

from cray_infra.training import launch_training_job as mod


# ---- get_train_time_limit: per-SLICE cap, not user total -----------------


def test_get_train_time_limit_caps_at_max_train_time(monkeypatch):
    # User asks for 10 days, server allows max 1 day per slice. The
    # returned walltime is the per-slice cap + buffer — TimeoutCallback
    # enforces the 10-day total separately by accumulating across
    # slices (training-lifecycle.md §5.4).
    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: {"max_train_time": 86400, "extra_training_seconds": 300},
    )
    out = mod.get_train_time_limit({"timeout": 10 * 86400})
    assert out == "1-00:05:00"


def test_get_train_time_limit_uses_user_value_when_below_cap(monkeypatch):
    # Short jobs run as a single slice: --time = user timeout + buffer.
    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: {"max_train_time": 86400, "extra_training_seconds": 300},
    )
    out = mod.get_train_time_limit({"timeout": 600})
    assert out == "0-00:15:00"  # 600 + 300 = 900 seconds = 15 min


def test_get_train_time_limit_no_timeout_uses_max(monkeypatch):
    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: {"max_train_time": 86400, "extra_training_seconds": 300},
    )
    out = mod.get_train_time_limit({})
    assert out == "1-00:05:00"


# ---- --signal=B:TERM@N flag injection -------------------------------------


def _stub_scontrol(monkeypatch, *, gpus_per_node=0, nodes=1, cpus_per_node=8):
    """Replace `scontrol show nodes` with deterministic output so the
    command builder doesn't have to actually shell out."""
    lines = []
    for i in range(nodes):
        name = f"node{i}"
        gres = f" Gres=gpu:{gpus_per_node}" if gpus_per_node > 0 else ""
        lines.append(f"NodeName={name} CPUTot={cpus_per_node}{gres}")
    fake = "\n".join(lines).encode()
    monkeypatch.setattr(
        subprocess, "check_output", lambda cmd: fake if "scontrol" in cmd[0] else b""
    )


def test_create_slurm_run_command_emits_signal_flag(tmp_path, monkeypatch):
    _stub_scontrol(monkeypatch)
    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: {
            "max_train_time": 3600,
            "extra_training_seconds": 0,
            "signal_grace_seconds": 90,
            "train_job_entrypoint": str(tmp_path / "entrypoint.sh"),
        },
    )
    (tmp_path / "entrypoint.sh").write_text("#!/bin/bash\necho stub\n")
    job_dir = tmp_path / "job-abc"
    job_dir.mkdir()
    (job_dir / "config.yaml").write_text("llm_name: x\n")

    cmd = mod.create_slurm_run_command(
        {"job_directory": str(job_dir), "timeout": 600, "gpus": 0, "nodes": 1}
    )

    signal_args = [a for a in cmd if a.startswith("--signal=")]
    assert signal_args == ["--signal=B:TERM@90"], cmd
    # B: prefix is mandatory — without it, slurm sends the signal to
    # mpirun (job step) rather than the batch shell that needs to call
    # sbatch on relaunch.
    assert "B:" in signal_args[0]


def test_create_slurm_run_command_signal_grace_defaults_to_120(tmp_path, monkeypatch):
    _stub_scontrol(monkeypatch)
    # Config omits signal_grace_seconds entirely; builder falls back to 120.
    monkeypatch.setattr(
        mod,
        "get_config",
        lambda: {
            "max_train_time": 3600,
            "extra_training_seconds": 0,
            "train_job_entrypoint": str(tmp_path / "entrypoint.sh"),
        },
    )
    (tmp_path / "entrypoint.sh").write_text("#!/bin/bash\n")
    job_dir = tmp_path / "job-xyz"
    job_dir.mkdir()
    (job_dir / "config.yaml").write_text("llm_name: x\n")

    cmd = mod.create_slurm_run_command(
        {"job_directory": str(job_dir), "timeout": 600, "gpus": 0, "nodes": 1}
    )

    assert "--signal=B:TERM@120" in cmd


# ---- write_resubmit_script ------------------------------------------------


def test_write_resubmit_script_writes_executable(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    mod.write_resubmit_script(
        ["sbatch", "--time=0-01:00:00", "/path/to/entry.sh"],
        {"job_directory": str(job_dir)},
    )
    script = job_dir / "resubmit.sh"
    assert script.is_file()
    # Executable bit set so `bash resubmit.sh` AND a direct exec both work.
    assert os.access(script, os.X_OK)


def test_write_resubmit_script_contains_verbatim_sbatch_invocation(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = ["sbatch", "--ntasks-per-node=1", "--time=0-01:00:00", "/x/entry.sh"]
    mod.write_resubmit_script(argv, {"job_directory": str(job_dir)})
    content = (job_dir / "resubmit.sh").read_text()
    for token in argv:
        assert shlex.quote(token) in content, f"missing token {token!r} in {content!r}"
    assert content.startswith("#!/bin/bash\n")
    assert "exec" in content


@pytest.mark.parametrize(
    "tricky_path",
    [
        "/tmp/job with spaces",
        "/tmp/job'with'quotes",
        "/tmp/job\"with\"doublequotes",
        "/tmp/job$with$dollars",
        "/tmp/job\\with\\backslashes",
    ],
)
def test_write_resubmit_script_quotes_tricky_paths(tmp_path, tricky_path):
    # The entrypoint shells out via `bash resubmit.sh`. shlex.quote on
    # every argv token means a path with spaces, quotes, $, or
    # backslashes doesn't shatter into multiple shell words. Verify by
    # re-parsing the script with shlex.split and recovering the original
    # argv.
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    argv = ["sbatch", "--output=" + tricky_path + "/slurm.out", tricky_path + "/x.sh"]
    mod.write_resubmit_script(argv, {"job_directory": str(job_dir)})
    script_text = (job_dir / "resubmit.sh").read_text()
    # bash -n parse check
    parse = subprocess.run(
        ["bash", "-n", str(job_dir / "resubmit.sh")],
        capture_output=True,
    )
    assert parse.returncode == 0, parse.stderr.decode()
    # Recover the argv from the `exec ...` line.
    exec_line = next(
        line for line in script_text.splitlines() if line.startswith("exec ")
    )
    recovered = shlex.split(exec_line[len("exec ") :])
    assert recovered == argv, (recovered, argv)
