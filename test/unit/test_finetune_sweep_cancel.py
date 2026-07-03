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
