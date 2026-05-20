"""
Unit tests for ml/cray_megatron/training_entrypoint.py.

Focused on handle_relaunch: it reads {job_dir}/status.json, checks the
relaunch_requested key written by TrainingLoop._finalize_slice
(training-lifecycle.md §5.4), and on True shells out to resubmit.sh.
This is the only logic worth testing in isolation — the mpirun spawn /
signal-forward path needs a real subprocess.
"""

import json
import os
from pathlib import Path

import pytest

from cray_megatron import training_entrypoint as mod


# ---- handle_relaunch -----------------------------------------------------


def test_handle_relaunch_no_op_when_status_missing(tmp_path, capsys):
    # Fresh job directory with nothing in it. handle_relaunch should
    # quietly return — never crash on a missing status.json.
    mod.handle_relaunch(tmp_path)
    # No subprocess was started; capsys is clean of "queuing next slice".
    out = capsys.readouterr().out
    assert "queuing next slice" not in out


def test_handle_relaunch_no_op_when_relaunch_requested_false(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "COMPLETED", "relaunch_requested": False})
    )
    called = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: called.append((a, kw)))

    mod.handle_relaunch(tmp_path)

    assert called == []  # never shelled out to bash resubmit.sh


def test_handle_relaunch_no_op_when_key_absent(tmp_path, monkeypatch):
    # status.json exists but has no relaunch_requested key at all.
    # status.get(key) is None / falsy → no relaunch. Guards against an
    # older job dir that predates this feature.
    (tmp_path / "status.json").write_text(json.dumps({"status": "COMPLETED"}))
    called = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: called.append((a, kw)))

    mod.handle_relaunch(tmp_path)

    assert called == []


def test_handle_relaunch_runs_resubmit_when_true(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "QUEUED", "relaunch_requested": True})
    )
    # The script must exist on disk for handle_relaunch to invoke it
    # (the missing-resubmit branch is covered by the next test).
    resubmit = tmp_path / "resubmit.sh"
    resubmit.write_text("#!/bin/bash\nexit 0\n")
    resubmit.chmod(0o755)

    invocations = []
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda cmd, *a, **kw: invocations.append(cmd)
        or _CompletedProcessShim(returncode=0),
    )

    mod.handle_relaunch(tmp_path)

    assert invocations == [["bash", str(resubmit)]]


def test_handle_relaunch_warns_when_resubmit_missing(tmp_path, monkeypatch, capsys):
    # relaunch flag is set but resubmit.sh is not on disk (e.g. the
    # operator deleted it or the API server failed to write it). Must
    # log a warning, not raise — the slice has already exited, raising
    # here would just make the slurm-{id}.out noisier without helping.
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "QUEUED", "relaunch_requested": True})
    )
    called = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: called.append((a, kw)))

    mod.handle_relaunch(tmp_path)

    assert called == []
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "cannot relaunch" in out


def test_handle_relaunch_tolerates_corrupt_status_json(tmp_path, monkeypatch, capsys):
    # Invalid JSON in status.json must not crash the entrypoint. The
    # trainer may have died mid-write; the entrypoint should log and
    # bail rather than letting an exception kill the slurm step
    # uncleanly.
    (tmp_path / "status.json").write_text("not valid json{")
    called = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: called.append((a, kw)))

    mod.handle_relaunch(tmp_path)

    assert called == []
    out = capsys.readouterr().out
    assert "could not read" in out


# ---- support ------------------------------------------------------------


class _CompletedProcessShim:
    def __init__(self, returncode):
        self.returncode = returncode
