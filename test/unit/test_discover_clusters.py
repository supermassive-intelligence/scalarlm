"""
Unit tests for the node-memory-reporting bits of discover_clusters.py:
``get_memory_mb`` and ``write_node_config``.
"""

from unittest.mock import patch

from cray_infra.slurm.discovery import discover_clusters as dc


def _fake_meminfo(tmp_path, contents):
    path = tmp_path / "meminfo"
    path.write_text(contents)
    return str(path)


def test_get_memory_mb_parses_mem_total(tmp_path):
    path = _fake_meminfo(
        tmp_path,
        "MemTotal:       131072000 kB\nMemFree:        1000 kB\n",
    )
    real_open = open
    with patch("builtins.open", side_effect=lambda *_a, **_kw: real_open(path)):
        assert dc.get_memory_mb() == 131072000 // 1024


def test_get_memory_mb_returns_none_when_file_missing():
    with patch("builtins.open", side_effect=FileNotFoundError):
        assert dc.get_memory_mb() is None


def test_get_memory_mb_returns_none_on_permission_error():
    """A restricted container/sandbox can deny reading /proc/meminfo
    with PermissionError rather than FileNotFoundError -- both must be
    treated as "can't determine memory", not propagate as a crash."""
    with patch("builtins.open", side_effect=PermissionError):
        assert dc.get_memory_mb() is None


def test_get_memory_mb_returns_none_on_unparseable_content(tmp_path):
    path = _fake_meminfo(tmp_path, "MemTotal:       not-a-number kB\n")
    real_open = open
    with patch("builtins.open", side_effect=lambda *_a, **_kw: real_open(path)):
        assert dc.get_memory_mb() is None


def _node(**overrides):
    node = {
        "hostname": "node1",
        "cpu_count": 64,
        "gpu_count": 0,
    }
    node.update(overrides)
    return node


def _fake_get_config(**overrides):
    config = {"max_gpus_per_node": 8}
    config.update(overrides)
    return config


def test_write_node_config_includes_real_memory_when_present():
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=257723))
    assert "RealMemory=257723" in line
    assert line.startswith("NodeName=node1 CPUs=64 RealMemory=257723")


def test_write_node_config_omits_real_memory_when_missing():
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=None))
    assert "RealMemory=" not in line


def test_write_node_config_no_double_spaces_when_fields_missing():
    """Both RealMemory (no memory_mb) and Gres (no GPUs) can be absent
    at once; the rendered line must not have doubled-up whitespace from
    joining empty fields."""
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=None, gpu_count=0))
    assert "  " not in line
    assert line == "NodeName=node1 CPUs=64 State=UNKNOWN\n"


def test_write_node_config_includes_gres_when_gpus_present():
    with patch.object(dc, "get_config", return_value=_fake_get_config()):
        line = dc.write_node_config(_node(memory_mb=257723, gpu_count=6))
    assert line == "NodeName=node1 CPUs=64 RealMemory=257723 Gres=gpu:6 State=UNKNOWN\n"
