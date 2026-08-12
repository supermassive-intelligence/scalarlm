"""Queue-backed chat requests must retain vLLM's structured semantics."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.responses import Response

from cray_infra.one_server import create_generate_worker as worker


@pytest.mark.asyncio
async def test_chat_worker_rehydrates_request_and_preserves_structure(monkeypatch):
    captured = {}

    async def fake_create_chat_completion(request, raw_request):
        captured["request"] = request
        captured["path"] = raw_request.scope["path"]
        return Response(
            content=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning": "I need a tool.",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"Helsinki"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 7,
                        "total_tokens": 19,
                    },
                }
            ),
            media_type="application/json",
        )

    monkeypatch.setattr(worker, "create_chat_completion", fake_create_chat_completion)

    app = MagicMock()
    app.state.engine_client.check_health = AsyncMock()
    app.state.engine_client.model_config = MagicMock()
    queued = {
        "request_id": "req-1",
        "request_type": "chat_completions",
        "prompt": "pre-rendered prompt used for accounting only",
        "chat_request": {
            "model": "muse",
            "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 128,
            "include_reasoning": False,
            "reasoning_effort": "low",
            "top_k": 64,
            "chat_template_kwargs": {"reasoning_strength": "low"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        },
    }

    result = await worker.async_generate_task(queued, app)

    request = captured["request"]
    assert captured["path"] == "/v1/chat/completions"
    assert request.messages == [{"role": "user", "content": "weather?"}]
    assert request.tools[0].function.name == "get_weather"
    assert request.include_reasoning is False
    assert request.reasoning_effort == "low"
    assert request.top_k == 64
    assert request.chat_template_kwargs == {"reasoning_strength": "low"}
    assert result["response"] == ""
    assert result["reasoning"] == "I need a tool."
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["finish_reason"] == "tool_calls"
    assert result["prompt_tokens"] == 12
    assert result["completion_tokens"] == 7
    assert result["token_count"] == 19
    app.state.engine_client.check_health.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_worker_preserves_vllm_error_message(monkeypatch):
    async def fake_create_chat_completion(request, raw_request):
        return Response(
            content=json.dumps(
                {"error": {"message": "invalid tool choice", "type": "BadRequest"}}
            ),
            media_type="application/json",
            status_code=400,
        )

    monkeypatch.setattr(worker, "create_chat_completion", fake_create_chat_completion)

    app = MagicMock()
    app.state.engine_client.check_health = AsyncMock()
    queued = {
        "request_id": "req-error",
        "request_type": "chat_completions",
        "prompt": "accounting prompt",
        "chat_request": {
            "model": "model-1",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": False,
        },
    }

    result = await worker.async_generate_task(queued, app)

    assert result == {
        "request_id": "req-error",
        "error": "invalid tool choice",
    }
    app.state.engine_client.check_health.assert_awaited_once()
