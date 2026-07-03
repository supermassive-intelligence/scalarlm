import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

# Load the script module by path (it is not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "run_finetune_sweep",
    Path(__file__).resolve().parents[1] / "finetune_sweep" / "run_finetune_sweep.py",
)
rfs = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rfs
_SPEC.loader.exec_module(rfs)


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"{}"


def test_cancel_training_posts_to_cancel_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        return _FakeResp()

    monkeypatch.setattr(rfs.urllib.request, "urlopen", fake_urlopen)

    rfs.cancel_training("http://api:8000", "deadbeef")

    assert captured["url"] == "http://api:8000/v1/megatron/cancel/deadbeef"
    assert captured["method"] == "POST"


def test_cancel_training_is_best_effort_on_error(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(rfs.urllib.request, "urlopen", boom)

    # Must not raise — a failed cancel can never be allowed to abort the sweep.
    rfs.cancel_training("http://api:8000", "deadbeef")


def test_cancel_training_swallows_http_error(monkeypatch):
    import urllib.error

    def http_500(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, "boom", {}, None)

    monkeypatch.setattr(rfs.urllib.request, "urlopen", http_500)

    rfs.cancel_training("http://api:8000", "deadbeef")


class _Completed:
    def __init__(self, stdout):
        self.stdout = stdout


def test_list_stale_job_hashes_parses_last_line(monkeypatch):
    # Container may emit a banner line before the JSON payload.
    monkeypatch.setattr(
        rfs.subprocess, "run",
        lambda *a, **k: _Completed('some banner\n["aaa", "bbb"]\n'),
    )
    assert rfs.list_stale_job_hashes("cray-spark") == ["aaa", "bbb"]


def test_list_stale_job_hashes_empty_output(monkeypatch):
    monkeypatch.setattr(rfs.subprocess, "run", lambda *a, **k: _Completed("\n"))
    assert rfs.list_stale_job_hashes("cray-spark") == []


def test_list_stale_job_hashes_exec_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise rfs.subprocess.CalledProcessError(1, "docker")

    monkeypatch.setattr(rfs.subprocess, "run", boom)
    assert rfs.list_stale_job_hashes("cray-spark") == []


def test_cleanup_stale_jobs_cancels_each(monkeypatch):
    cancelled = []
    monkeypatch.setattr(rfs, "list_stale_job_hashes", lambda svc, **k: ["h1", "h2", "h3"])
    monkeypatch.setattr(rfs, "cancel_training",
                        lambda api, h, log=None: cancelled.append(h))

    n = rfs.cleanup_stale_jobs("http://api:8000", "cray-spark")

    assert n == 3
    assert cancelled == ["h1", "h2", "h3"]


def test_cleanup_stale_jobs_noop_without_compose_service(monkeypatch):
    # k8s targets pass compose_service=None; must not exec or cancel anything.
    monkeypatch.setattr(rfs, "list_stale_job_hashes",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not enumerate")))
    assert rfs.cleanup_stale_jobs("http://api:8000", None) == 0
