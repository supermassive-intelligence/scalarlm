"""
Component-level tests for launch_training_job against a real slurmctld.

The unit tests in test/unit/test_launch_training_job_relaunch.py stub
out subprocess and config; these run the real flow: an actual `sbatch`
call against the slurmctld that the test runner starts, with a tiny
shell-only entrypoint so we don't pull in torch/mpirun. The point is to
catch breakage in the boundary the unit tests can't see: that the
constructed sbatch argv parses, that slurm accepts the --signal flag,
that resubmit.sh lands alongside the job, and that the recorded job_id
matches what squeue reports.

Per docs/test-plan.md §5.5 / §5.16 — the new auto-relaunch surface
(training-lifecycle.md §5.4) needs at least one slurm-backed test
exercising the launch path end-to-end.
"""

import asyncio
import os
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import time

import pytest


def _scancel(job_id: str) -> None:
    if not job_id:
        return
    subprocess.run(["scancel", job_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _job_in_squeue(job_id: str) -> bool:
    out = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%i"],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and out.stdout.strip() == job_id


def _write_noop_entrypoint(path: str) -> None:
    """Tiny stand-in for scripts/train_job_entrypoint.sh. Just exits 0
    after a short sleep — enough for the test to observe the job in
    squeue before it completes, without needing python/torch/mpirun.
    The launch_training_job copies this verbatim into the job dir and
    string-substitutes REPLACE_CONFIG_PATH (which we ignore here)."""
    with open(path, "w") as f:
        f.write(textwrap.dedent("""\
            #!/bin/bash
            # CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH
            echo "[noop-entrypoint] started"
            sleep 2
            echo "[noop-entrypoint] done"
        """))
    os.chmod(path, 0o755)


@pytest.fixture
def job_workspace(tmp_path, monkeypatch):
    """A tmp jobs-dir + a stub entrypoint script, both pointed at via
    SCALARLM_* env vars so get_config() picks them up. Cleans up any
    queued slurm jobs at teardown so a flaky test doesn't leave detritus
    in squeue."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    entrypoint = tmp_path / "noop_entrypoint.sh"
    _write_noop_entrypoint(str(entrypoint))

    monkeypatch.setenv("SCALARLM_TRAINING_JOB_DIRECTORY", str(jobs_dir))
    monkeypatch.setenv("SCALARLM_TRAIN_JOB_ENTRYPOINT", str(entrypoint))
    # Keep slices short so squeue interactions don't drag the test out.
    monkeypatch.setenv("SCALARLM_MAX_TRAIN_TIME", "60")
    monkeypatch.setenv("SCALARLM_EXTRA_TRAINING_SECONDS", "0")
    monkeypatch.setenv("SCALARLM_SIGNAL_GRACE_SECONDS", "10")

    submitted_jobs = []
    yield jobs_dir, submitted_jobs
    for jid in submitted_jobs:
        _scancel(jid)


def _read_status(job_dir):
    import json
    path = os.path.join(str(job_dir), "status.json")
    with open(path) as f:
        return json.load(f)


def test_launch_writes_resubmit_and_signal_flag_then_sbatch_accepts(
    slurm_running, job_workspace
):
    """End-to-end: real sbatch runs, slurm queues the job, status.json
    records job_id, resubmit.sh lands on disk with the verbatim sbatch
    invocation (including --signal=B:TERM@N), and the recorded job_id
    matches squeue."""
    from cray_infra.training.launch_training_job import launch_training_job

    jobs_dir, submitted = job_workspace
    job_hash = "test-component-launch"
    train_args = {
        "job_directory": str(jobs_dir / job_hash),
        "training_data_path": "ignored-by-noop-entrypoint",
        "dataset_hash": "deadbeef",
        "timeout": 30,
        "gpus": 0,
        "nodes": 1,
    }

    result = asyncio.run(launch_training_job(train_args))
    job_id = str(result["job_id"])
    submitted.append(job_id)

    # ---- resubmit.sh contract (training-lifecycle.md §3.4) -----------
    resubmit_path = os.path.join(train_args["job_directory"], "resubmit.sh")
    assert os.path.isfile(resubmit_path), "resubmit.sh not written"
    assert os.access(resubmit_path, os.X_OK), "resubmit.sh not executable"
    parse = subprocess.run(["bash", "-n", resubmit_path], capture_output=True)
    assert parse.returncode == 0, parse.stderr.decode()
    body = open(resubmit_path).read()
    # The --signal flag is the new bit; failure here means a sbatch
    # template regression (training-lifecycle.md §3.2).
    assert "--signal=B:TERM@10" in body, body
    # The exact entrypoint we stubbed in must be the last argv token.
    assert os.environ["SCALARLM_TRAIN_JOB_ENTRYPOINT"] not in body  # copied per-job
    assert os.path.join(train_args["job_directory"], "train_job_entrypoint.sh") in body

    # ---- status.json (training-lifecycle.md §3.4) --------------------
    status = _read_status(train_args["job_directory"])
    assert status["status"] == "QUEUED"
    assert status["job_id"] == job_id
    assert "start_time" in status

    # ---- slurm-side: job_id is real, squeue has heard of it ---------
    # The job may have run and exited by the time we poll if slurm is
    # very fast, but it must have been accepted (job_id parsed from
    # sbatch's stdout) — that already proved sbatch did NOT reject the
    # --signal flag with EINVAL.
    deadline = time.time() + 5
    seen_in_squeue = False
    while time.time() < deadline:
        if _job_in_squeue(job_id):
            seen_in_squeue = True
            break
        time.sleep(0.2)
    # squeue may miss very short jobs; accept either branch as long as
    # sacct or the absence is consistent with a real accepted job.
    sacct = subprocess.run(
        ["sacct", "-n", "-j", job_id, "-o", "JobID"],
        capture_output=True,
        text=True,
    )
    if not seen_in_squeue:
        assert job_id in sacct.stdout, (
            f"job_id {job_id} not in squeue nor sacct — sbatch likely "
            f"didn't actually queue. sacct out: {sacct.stdout!r}"
        )


def test_relaunch_via_resubmit_sh_queues_a_new_job_id(
    slurm_running, job_workspace
):
    """Simulates what the Python entrypoint does on slurm timeout: run
    the resubmit.sh script and verify a new slurm job lands with a
    different ID. This is the closest we can get to the auto-relaunch
    chain (training-lifecycle.md §5.4) without spawning a real training
    process — it proves the resubmit.sh round-trip is wire-compatible
    with slurm."""
    from cray_infra.training.launch_training_job import launch_training_job

    jobs_dir, submitted = job_workspace
    train_args = {
        "job_directory": str(jobs_dir / "test-component-relaunch"),
        "training_data_path": "ignored",
        "dataset_hash": "cafebabe",
        "timeout": 30,
        "gpus": 0,
        "nodes": 1,
    }
    first = asyncio.run(launch_training_job(train_args))
    first_id = str(first["job_id"])
    submitted.append(first_id)

    resubmit_path = os.path.join(train_args["job_directory"], "resubmit.sh")
    second = subprocess.run(["bash", resubmit_path], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    # `sbatch` prints "Submitted batch job <id>"; extract the new id.
    import re
    m = re.search(r"Submitted batch job (\d+)", second.stdout)
    assert m is not None, f"resubmit.sh did not return a slurm job id: {second.stdout!r}"
    second_id = m.group(1)
    submitted.append(second_id)

    assert second_id != first_id, "relaunch must produce a fresh slurm job id"
