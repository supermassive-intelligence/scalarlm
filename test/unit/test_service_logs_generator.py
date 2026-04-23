"""
Unit tests for service_logs_generator — the backend that feeds the
Metrics-page log panel. The UI needs three behaviors from it:

- `starting_line_number=N` resumes a live tail after an EOF reconnect.
- `tail=N` jumps to the last N lines on first load so we don't have
  to dump a 100k-line history to the client.
- `limit=M` bounds a one-shot fetch for scrollback.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from cray_infra.api.fastapi.health import service_logs_generator as gen_mod


@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "api.log"
    p.write_text("\n".join(f"line-{i}" for i in range(100)) + "\n")
    return p


@pytest.fixture
def resolve(log_file):
    with patch(
        "cray_infra.api.fastapi.health.service_logs_generator.get_service_log_file",
        return_value=str(log_file),
    ):
        yield


def _drain(agen) -> list[dict]:
    async def pump():
        out = []
        async for chunk in agen:
            out.append(json.loads(chunk))
        return out
    return asyncio.get_event_loop().run_until_complete(pump())


# ---- starting_line_number (existing behavior) ----------------------------


def test_resumes_from_starting_line(resolve):
    records = _drain(
        gen_mod.service_logs_generator("api", starting_line_number=97)
    )
    assert [r["line_number"] for r in records] == [97, 98, 99]
    assert records[0]["line"] == "line-97"


def test_zero_start_yields_full_file(resolve):
    records = _drain(gen_mod.service_logs_generator("api"))
    assert len(records) == 100
    assert records[0]["line_number"] == 0
    assert records[-1]["line_number"] == 99


# ---- tail (jump-to-end on initial load) ----------------------------------


def test_tail_yields_last_n_lines(resolve):
    records = _drain(gen_mod.service_logs_generator("api", tail=5))
    assert [r["line_number"] for r in records] == [95, 96, 97, 98, 99]


def test_tail_larger_than_file_returns_everything(resolve):
    records = _drain(gen_mod.service_logs_generator("api", tail=10_000))
    assert len(records) == 100
    assert records[0]["line_number"] == 0


def test_tail_overrides_starting_line_number(resolve):
    # When both are passed, tail wins — client hit "jump to end" so we
    # ignore whatever resume point it might have had.
    records = _drain(
        gen_mod.service_logs_generator(
            "api", starting_line_number=50, tail=3
        )
    )
    assert [r["line_number"] for r in records] == [97, 98, 99]


def test_tail_zero_or_negative_is_ignored(resolve):
    # tail=0 is treated as "no tail" so the request still returns
    # something deterministic rather than an empty stream.
    records = _drain(
        gen_mod.service_logs_generator("api", starting_line_number=97, tail=0)
    )
    assert [r["line_number"] for r in records] == [97, 98, 99]


# ---- limit (bounded backfill) --------------------------------------------


def test_limit_caps_yield_count(resolve):
    records = _drain(
        gen_mod.service_logs_generator("api", starting_line_number=10, limit=5)
    )
    assert [r["line_number"] for r in records] == [10, 11, 12, 13, 14]


def test_limit_plus_tail_combines(resolve):
    # Client requested "the last 20 lines, but only send me the first 7
    # of them" — used for scrollback windowing.
    records = _drain(gen_mod.service_logs_generator("api", tail=20, limit=7))
    assert [r["line_number"] for r in records] == [80, 81, 82, 83, 84, 85, 86]


def test_limit_larger_than_remaining_is_fine(resolve):
    records = _drain(
        gen_mod.service_logs_generator("api", starting_line_number=98, limit=10)
    )
    assert [r["line_number"] for r in records] == [98, 99]


# ---- count_lines ---------------------------------------------------------


def test_count_lines_matches_fixture(log_file):
    assert gen_mod.count_lines(str(log_file)) == 100


def test_count_lines_empty_file(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("")
    assert gen_mod.count_lines(str(p)) == 0


def test_count_lines_no_trailing_newline(tmp_path):
    p = tmp_path / "noeol.log"
    p.write_text("a\nb\nc")  # three lines, last missing \n
    assert gen_mod.count_lines(str(p)) == 3
