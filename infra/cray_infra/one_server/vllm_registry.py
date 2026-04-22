"""Process-wide registry for vLLM's already-initialised serving objects.

scalarlm and vLLM each run a FastAPI app in the same Python process on
different ports (8000 and 8001 respectively). Normally scalarlm's openai
proxy talks to vLLM over localhost HTTP. Phase 6 collapses that hop by
calling vLLM's Python API directly — which needs a path from the openai
proxy to the serving instances that ``init_app_state`` attached to
vLLM's FastAPI ``app.state``.

This module is the meeting point. ``create_vllm.py`` stashes the handles
here after ``init_app_state``; ``openai_v1_router.py`` reads them when
forwarding a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VLLMServings:
    engine_client: Any
    model_config: Any
    openai_serving_completion: Any
    openai_serving_chat: Any
    openai_serving_models: Any


_servings: Optional[VLLMServings] = None


def set_vllm_servings(app_state) -> None:
    """Called from create_vllm.run_server_worker after init_app_state.

    Accepts vLLM's ``app.state`` (a Starlette ``State`` with the attributes
    init_app_state sets) and copies the references we need into a plain
    dataclass. A missing attribute means vLLM's version shifted its
    layout; fail loudly so we notice at startup, not on the first request.
    """
    global _servings
    _servings = VLLMServings(
        engine_client=app_state.engine_client,
        model_config=app_state.model_config,
        openai_serving_completion=app_state.openai_serving_completion,
        openai_serving_chat=app_state.openai_serving_chat,
        openai_serving_models=app_state.openai_serving_models,
    )


def get_vllm_servings() -> Optional[VLLMServings]:
    """Returns None when called before ``set_vllm_servings`` has run — i.e.
    during startup, or when the scalarlm proxy is deployed against an
    external vLLM via the legacy HTTP-hop code path."""
    return _servings


def _reset_for_tests() -> None:
    """Test hook — production code should not call this."""
    global _servings
    _servings = None
