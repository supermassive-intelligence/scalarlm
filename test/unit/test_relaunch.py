"""
Unit tests for ml/cray_megatron/relaunch.py.

handle_relaunch_if_needed is called by main.py on the main rank after
trainer.train() returns. It reads status.json, checks the
relaunch_requested key written by TrainingLoop._finalize_slice
(training-lifecycle.md §5.4), and on True shells out to resubmit.sh.

Lives in its own module specifically so importing it doesn't execute
main() at import time, which would happen if the logic stayed in main.py.
"""

import json
from pathlib import Path

import pytest

from cray_megatron import relaunch as mod


def test_no_op_when_status_missing(tmp_path, capsys):
    # Fresh job directory with nothing in it. Must quietly return —
    # never crash on a missing status.json.
    mod.handle_relaunch_if_needed(tmp_path)
    out = capsys.readouterr().out
    assert "queuing next slice" not in out


def test_no_op_when_relaunch_requested_false(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "COMPLETED", "relaunch_requested": False})
    )
    called = []
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **kw: called.append((a, kw))
    )

    mod.handle_relaunch_if_needed(tmp_path)

    assert called == []  # never shelled out


def test_no_op_when_key_absent(tmp_path, monkeypatch):
    # status.json exists but has no relaunch_requested key at all.
    # status.get(key) is None / falsy → no relaunch. Guards against an
    # older job dir that predates this feature.
    (tmp_path / "status.json").write_text(json.dumps({"status": "COMPLETED"}))
    called = []
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **kw: called.append((a, kw))
    )

    mod.handle_relaunch_if_needed(tmp_path)

    assert called == []


def test_runs_resubmit_when_true(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "QUEUED", "relaunch_requested": True})
    )
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

    mod.handle_relaunch_if_needed(tmp_path)

    assert invocations == [["bash", str(resubmit)]]


def test_warns_when_resubmit_missing(tmp_path, monkeypatch, capsys):
    # relaunch flag is set but resubmit.sh is not on disk. Log a
    # warning, don't raise — the slice has already exited, raising
    # would just make slurm-{id}.out noisier without helping.
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "QUEUED", "relaunch_requested": True})
    )
    called = []
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **kw: called.append((a, kw))
    )

    mod.handle_relaunch_if_needed(tmp_path)

    assert called == []
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "cannot relaunch" in out


def test_tolerates_corrupt_status_json(tmp_path, monkeypatch, capsys):
    # Invalid JSON in status.json must not crash. The trainer may have
    # died mid-write; better to log and bail than let an exception
    # kill the slurm step uncleanly.
    (tmp_path / "status.json").write_text("not valid json{")
    called = []
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **kw: called.append((a, kw))
    )

    mod.handle_relaunch_if_needed(tmp_path)

    assert called == []
    out = capsys.readouterr().out
    assert "could not read" in out


def test_accepts_str_or_path(tmp_path, monkeypatch):
    # Caller passes a job_directory string from get_job_config(); we
    # accept both strings and Path objects.
    (tmp_path / "status.json").write_text(json.dumps({"status": "COMPLETED"}))
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **kw: _CompletedProcessShim(0)
    )
    mod.handle_relaunch_if_needed(str(tmp_path))
    mod.handle_relaunch_if_needed(Path(tmp_path))


class _CompletedProcessShim:
    def __init__(self, returncode):
        self.returncode = returncode
