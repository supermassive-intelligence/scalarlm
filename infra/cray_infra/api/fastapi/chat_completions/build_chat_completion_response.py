"""
Wrap the worker's flat result dict into the OpenAI ChatCompletion
shape the `AsyncOpenAI` SDK expects.

The worker (see `create_generate_worker.async_completion_task`)
returns:

    {
      "request_id": "<id>",
      "response": "<text>",            # present on success
      "error": "<message>",            # present on failure
      "token_count": <int>,            # present when usage was reported
      "is_acked": True,                # added by update_and_ack
      ...                              # original request fields
    }

The OpenAI ChatCompletion shape is:

    {
      "id": "chatcmpl-<id>",
      "object": "chat.completion",
      "created": <unix_ts>,
      "model": "<model>",
      "choices": [{"index": 0,
                   "message": {"role": "assistant", "content": "<text>"},
                   "finish_reason": "stop",
                   "logprobs": None}],
      "usage": {"prompt_tokens": 0, "completion_tokens": <n>,
                "total_tokens": <n>}
    }

We don't have the prompt-token vs completion-token split available at
this layer (the worker returned the union as `token_count`), so we put
all of it into `completion_tokens`. That keeps `total_tokens` correct
— operators reading metrics get the right number — at the cost of a
slightly inaccurate split. If we need the precise split later, the
worker has access to `response_data["usage"]` and can pass through.
"""

import time
from typing import Any


CHAT_COMPLETION_OBJECT = "chat.completion"


def build_chat_completion_response(
    *, result: dict[str, Any], model: str
) -> dict[str, Any]:
    """
    `result` is the dict the handler's heartbeat streamer is about to
    JSON-dump — i.e. the worker output enriched by `update_and_ack`.
    `model` is captured from the original request so we don't have to
    rely on the worker echoing it back.

    The future resolution path *should* hand us the per-request dict
    (`{is_acked, response, ...}`), but the on-disk `_response.json`
    matches a group-level shape (`{results: {<id>: per_request_dict},
    current_index, total_requests, ...}`) and field-testing showed
    empty content unless we tolerate both. `_unwrap_group_dict`
    is a no-op on the per-request shape, so this defends without
    regressing the documented contract.
    """
    result = _unwrap_group_dict(result)

    error = result.get("error")
    content = "" if error else (result.get("response") or "")

    chat_id = _build_chat_id(result.get("request_id"))
    finish_reason = "stop" if not error else "error"
    completion_tokens = int(result.get("token_count") or 0)

    response: dict[str, Any] = {
        "id": chat_id,
        "object": CHAT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
    }

    # OpenAI's error shape is a top-level `error` field; the SDK will
    # still parse the body if `choices` is present, so we preserve both
    # to give operators the diagnostic and the SDK a parseable result.
    if error:
        response["error"] = {"message": error, "type": "worker_error"}

    return response


def _build_chat_id(request_id: Any) -> str:
    """OpenAI ids are `chatcmpl-<token>`. Reuse the worker's request_id
    when it's a string so the inference browser can correlate; fall
    back to a timestamp-based id otherwise."""
    if isinstance(request_id, str) and request_id:
        return f"chatcmpl-{request_id}"
    return f"chatcmpl-{int(time.time() * 1000)}"


def _unwrap_group_dict(result: Any) -> dict[str, Any]:
    """
    If `result` is a group-level dict (has `results` mapping
    request_id → per_request_dict), pick the first per-request entry.

    The chat completions queue path packs requests one-per-row today
    (the coalescer batches at the file level, but each row that
    finishes resolves its own correlation_id), so picking the first
    entry is correct. If we ever get multi-prompt rows resolving a
    single cid, this needs to merge instead.

    Hands non-dicts straight back so the caller's `.get(...)` calls
    raise their natural AttributeError rather than silently returning
    empty content.
    """
    if not isinstance(result, dict):
        return result
    nested = result.get("results")
    if not isinstance(nested, dict) or not nested:
        return result
    # Mark the choice deterministic so two concurrent calls on the
    # same group don't disagree about which entry is "first".
    first_key = next(iter(sorted(nested.keys())))
    candidate = nested[first_key]
    if not isinstance(candidate, dict):
        return result
    return candidate
