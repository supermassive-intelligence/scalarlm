"""Exercise ScalarLM's launcher against the installed vLLM fork."""

import pytest

pytest.importorskip("vllm.config")

from cray_infra.one_server import create_vllm
from cray_infra.util.default_config import Config


def test_tokenformer_launcher_builds_serving_routes(monkeypatch):
    monkeypatch.delenv("SCALARLM_VLLM_ARGS", raising=False)
    config = Config(model="Qwen/Qwen3-0.6B").model_dump()
    config.update(enable_lora=False, enable_tokenformer=True)
    parser = create_vllm.make_arg_parser(create_vllm.FlexibleArgumentParser())
    args = parser.parse_args(create_vllm.build_vllm_cli_args(config))
    args.model = config["model"]

    assert args.enable_tokenformer
    assert not args.enable_lora
    app = create_vllm.build_app(args, ("generate",))
    paths = {route.path for route in app.routes}
    assert {"/health", "/v1/completions", "/v1/chat/completions"} <= paths
