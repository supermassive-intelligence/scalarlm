"""
Unit tests for cray_infra.training.launch_publish_job.

The launcher is the API pod's entry point into the publish-to-HF SLURM
flow (ui/docs/publish-to-hf.md). It must:

  - validate inputs and refuse to clobber an in-flight publish,
  - resolve the right checkpoint (or fall back to the latest one),
  - build an sbatch argv that does NOT contain the HF token on argv,
  - export the token via env so the SLURM job can read it from
    HF_TOKEN / HUGGING_FACE_HUB_TOKEN,
  - write a status.json the polling endpoint can find immediately.
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from cray_infra.training import launch_publish_job as mod


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / "deadbeef"
    d.mkdir()
    (d / "config.yaml").write_text("llm_name: foo/bar\n")
    (d / "checkpoint_5.pt").write_bytes(b"")
    (d / "checkpoint_100.pt").write_bytes(b"")
    return d


@pytest.fixture
def resolve_dir(job_dir):
    with patch(
        "cray_infra.training.launch_publish_job.get_job_directory_for_hash",
        return_value=str(job_dir),
    ):
        yield


def _ok(stdout="Submitted batch job 4242\n"):
    return subprocess.CompletedProcess(args=["sbatch"], returncode=0, stdout=stdout, stderr="")


def _fail(stderr="boom"):
    return subprocess.CompletedProcess(args=["sbatch"], returncode=1, stdout="", stderr=stderr)


# ---- happy path ----------------------------------------------------------


def test_submits_with_latest_checkpoint(job_dir, resolve_dir):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        return _ok()

    with patch("subprocess.run", side_effect=fake_run):
        result = mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="myorg/my-model",
            private=False,
            hf_token="hf_secret_xyz",
        )

    assert result["publish_job_id"] == "4242"
    publish_dir = Path(result["publish_dir"])
    assert publish_dir.parent == job_dir
    assert publish_dir.name.startswith("publish_")

    # Initial status.json was written before sbatch was called.
    status = json.loads((publish_dir / "status.json").read_text())
    assert status["mode"] == "merged"
    assert status["repo_id"] == "myorg/my-model"
    assert status["phase"] == "queued"
    assert status["publish_job_id"] == "4242"

    cli_args = captured["cmd"]
    # Token is NOT on argv.
    assert "hf_secret_xyz" not in " ".join(cli_args)
    # `--export=ALL,HF_TOKEN,HUGGING_FACE_HUB_TOKEN` rides through env.
    assert any(a.startswith("--export=") and "HF_TOKEN" in a for a in cli_args)
    # Token IS in the env we pass to sbatch.
    assert captured["env"]["HF_TOKEN"] == "hf_secret_xyz"
    assert captured["env"]["HUGGING_FACE_HUB_TOKEN"] == "hf_secret_xyz"
    # The newest checkpoint is the one passed.
    assert "--checkpoint" in cli_args
    ckpt_idx = cli_args.index("--checkpoint") + 1
    assert cli_args[ckpt_idx].endswith("checkpoint_100.pt")


def test_submits_with_explicit_checkpoint(job_dir, resolve_dir):
    with patch("subprocess.run", return_value=_ok()):
        result = mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="myorg/my-model",
            private=False,
            hf_token="t",
            checkpoint="checkpoint_5.pt",
        )
    assert result["publish_job_id"] == "4242"


def test_passes_through_optional_flags(job_dir, resolve_dir):
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _ok()

    with patch("subprocess.run", side_effect=fake_run):
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="myorg/my-model",
            private=True,
            hf_token="t",
            lora_alpha=64,
            commit_message="release v1",
        )
    cmd = captured["cmd"]
    assert "--private" in cmd
    assert "--lora-alpha" in cmd
    assert cmd[cmd.index("--lora-alpha") + 1] == "64"
    assert "--commit-message" in cmd
    assert cmd[cmd.index("--commit-message") + 1] == "release v1"


# ---- guard rails ---------------------------------------------------------


def test_rejects_unknown_mode(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="bogus",
            repo_id="x/y",
            private=False,
            hf_token="t",
        )
    assert exc.value.status_code == 400


def test_rejects_missing_token(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="x/y",
            private=False,
            hf_token="",
        )
    assert exc.value.status_code == 400


def test_rejects_repo_id_without_slash(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="bad",
            private=False,
            hf_token="t",
        )
    assert exc.value.status_code == 400


def test_rejects_checkpoint_path_traversal(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="x/y",
            private=False,
            hf_token="t",
            checkpoint="../../../etc/passwd",
        )
    assert exc.value.status_code == 400


def test_rejects_unknown_checkpoint(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="x/y",
            private=False,
            hf_token="t",
            checkpoint="checkpoint_999.pt",
        )
    assert exc.value.status_code == 404


def test_rejects_when_publish_already_in_flight(job_dir, resolve_dir):
    in_flight = job_dir / "publish_111"
    in_flight.mkdir()
    (in_flight / "status.json").write_text(
        json.dumps({"phase": "uploading", "publish_job_id": "9001"})
    )
    with pytest.raises(HTTPException) as exc:
        mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="x/y",
            private=False,
            hf_token="t",
        )
    assert exc.value.status_code == 409


def test_terminal_status_does_not_block_new_publish(job_dir, resolve_dir):
    # A previous publish that finished (phase=done) must not block new ones.
    done = job_dir / "publish_001"
    done.mkdir()
    (done / "status.json").write_text(
        json.dumps({"phase": "done", "publish_job_id": "1"})
    )
    with patch("subprocess.run", return_value=_ok()):
        result = mod.launch_publish_job(
            "deadbeef",
            mode="merged",
            repo_id="x/y",
            private=False,
            hf_token="t",
        )
    assert result["publish_job_id"] == "4242"


def test_sbatch_failure_writes_error_and_raises(job_dir, resolve_dir):
    with patch("subprocess.run", return_value=_fail("queue is full")):
        with pytest.raises(HTTPException) as exc:
            mod.launch_publish_job(
                "deadbeef",
                mode="merged",
                repo_id="x/y",
                private=False,
                hf_token="t",
            )
    assert exc.value.status_code == 500

    # The publish_dir's status.json was updated to reflect the error.
    publish_dirs = list(job_dir.glob("publish_*"))
    assert len(publish_dirs) == 1
    state = json.loads((publish_dirs[0] / "status.json").read_text())
    assert state["phase"] == "error"
    assert "queue is full" in state["error"]


# ---- get_publish_status -------------------------------------------------


def test_get_publish_status_returns_latest(job_dir, resolve_dir):
    older = job_dir / "publish_100"
    newer = job_dir / "publish_200"
    older.mkdir()
    newer.mkdir()
    (older / "status.json").write_text(json.dumps({"phase": "done"}))
    (newer / "status.json").write_text(json.dumps({"phase": "uploading"}))

    state = mod.get_publish_status("deadbeef")
    assert state["phase"] == "uploading"
    assert state["publish_dir"] == str(newer)


def test_get_publish_status_404_when_none(job_dir, resolve_dir):
    with pytest.raises(HTTPException) as exc:
        mod.get_publish_status("deadbeef")
    assert exc.value.status_code == 404
