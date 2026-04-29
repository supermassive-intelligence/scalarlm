#!/usr/bin/env python3
"""Apply ScalarLM's vLLM-fork patches in-place.

Invoked from both:
  - scripts/build-copy-vllm.sh at Docker build time (production path)
  - Kubernetes pod `command:` at startup (dev/bench iteration)

Adds new patches here. Each patch is a function that takes the vLLM tree
root, reads the target file, asserts the exact anchor it expects, and
writes the transformed file back. The assertions are load-bearing — they
fail loudly when a fork rebase drifts the source rather than silently
producing a mis-patched image.

Usage: apply_patches.py <path/to/vllm/root>
"""

from __future__ import annotations

import sys
from pathlib import Path


def patch_output_handler_metrics_offload(vllm_root: Path) -> None:
    """Phase 6.5: move `logger_ref[0].record(...)` out of async_llm's output
    loop onto a background consumer coroutine fed by a bounded asyncio.Queue.

    Follows the TODO vLLM itself left on `vllm/v1/engine/async_llm.py:700`.
    Profile: the synchronous record call was 20.3 %% of main-thread time at
    N=100 and the consumer-offload pattern recovered +15 %% throughput in
    the pilot A/B. See enhance-openai-api.md § "Phase 6.5".
    """
    target = vllm_root / "vllm" / "v1" / "engine" / "async_llm.py"
    src = target.read_text()

    # --- Anchor 1: the inline record call we're replacing. ---
    anchor_record = (
        "                    # 4) Logging.\n"
        "                    # TODO(rob): make into a coroutine and launch it in\n"
        "                    # background thread once Prometheus overhead is non-trivial.\n"
        "                    if logger_ref[0]:\n"
        "                        logger_ref[0].record(\n"
        "                            engine_idx=outputs.engine_index,\n"
        "                            scheduler_stats=outputs.scheduler_stats,\n"
        "                            iteration_stats=iteration_stats,\n"
        "                            mm_cache_stats=renderer.stat_mm_cache(),\n"
        "                        )\n"
    )
    assert anchor_record in src, (
        "async_llm.py anchor missing: the inline `logger_ref[0].record(...)` "
        "block expected at the end of output_handler has drifted. Rebase of "
        "the vLLM fork requires re-reading scripts/vllm_patches/apply_patches.py "
        "to re-anchor this patch."
    )

    replacement_record = (
        "                    # 4) Logging — offloaded to background consumer.\n"
        "                    #    ScalarLM patch (Phase 6.5). vLLM's own TODO\n"
        "                    #    on this spot suggested a background thread;\n"
        "                    #    an asyncio task + bounded queue yields the\n"
        "                    #    same shape with less machinery.\n"
        "                    if logger_ref[0] is not None:\n"
        "                        try:\n"
        "                            metrics_queue_ref[0].put_nowait((\n"
        "                                outputs.engine_index,\n"
        "                                outputs.scheduler_stats,\n"
        "                                iteration_stats,\n"
        "                                renderer.stat_mm_cache(),\n"
        "                            ))\n"
        "                        except asyncio.QueueFull:\n"
        "                            # Metrics are statistical — drop the\n"
        "                            # oldest rather than block the output\n"
        "                            # loop. The engine must keep pulling.\n"
        "                            try:\n"
        "                                metrics_queue_ref[0].get_nowait()\n"
        "                                metrics_queue_ref[0].put_nowait((\n"
        "                                    outputs.engine_index,\n"
        "                                    outputs.scheduler_stats,\n"
        "                                    iteration_stats,\n"
        "                                    renderer.stat_mm_cache(),\n"
        "                                ))\n"
        "                            except (asyncio.QueueEmpty, asyncio.QueueFull):\n"
        "                                pass\n"
    )

    # --- Anchor 2: the ``async def output_handler()`` signature — we hang
    #     the metrics queue + consumer task off this spot, one level up
    #     from the handler body, so they share closure state. ---
    anchor_handler_def = "        async def output_handler():\n"
    assert src.count(anchor_handler_def) == 1, (
        "async_llm.py anchor missing: expected exactly one "
        "`async def output_handler():` inside `_run_output_handler`. "
        "Fork has drifted."
    )

    consumer_preamble = (
        "        # ScalarLM Phase 6.5: bounded queue + consumer task that\n"
        "        # drains the record() calls off the output_handler hot\n"
        "        # path. 1024 items is ~seconds of engine iterations at our\n"
        "        # scale; metric samples that overflow are dropped (see\n"
        "        # output_handler for the drop-oldest path).\n"
        "        metrics_queue_ref: list = [asyncio.Queue(maxsize=1024)]\n"
        "\n"
        "        async def _metrics_consumer():\n"
        "            q = metrics_queue_ref[0]\n"
        "            while True:\n"
        "                try:\n"
        "                    engine_idx, scheduler_stats, iteration_stats, mm_cache_stats = (\n"
        "                        await q.get()\n"
        "                    )\n"
        "                except asyncio.CancelledError:\n"
        "                    return\n"
        "                try:\n"
        "                    if logger_ref[0] is not None:\n"
        "                        logger_ref[0].record(\n"
        "                            engine_idx=engine_idx,\n"
        "                            scheduler_stats=scheduler_stats,\n"
        "                            iteration_stats=iteration_stats,\n"
        "                            mm_cache_stats=mm_cache_stats,\n"
        "                        )\n"
        "                except Exception:\n"
        "                    logger.exception(\"Background metrics record failed.\")\n"
        "\n"
        "        async def output_handler():\n"
    )

    # --- Anchor 3: the ``self.output_handler = asyncio.create_task(...)``
    #     line where the handler task is actually scheduled. We schedule
    #     the consumer next to it so the two have matching lifecycles. ---
    anchor_schedule = (
        "        self.output_handler = asyncio.create_task(output_handler())\n"
    )
    assert anchor_schedule in src, (
        "async_llm.py anchor missing: expected the "
        "`self.output_handler = asyncio.create_task(output_handler())` line."
    )
    replacement_schedule = (
        "        self.output_handler = asyncio.create_task(output_handler())\n"
        "        self._metrics_consumer_task = asyncio.create_task(_metrics_consumer())\n"
    )

    # Apply in order; each assertion above already guaranteed uniqueness.
    patched = (
        src
        .replace(anchor_record, replacement_record)
        .replace(anchor_handler_def, consumer_preamble)
        .replace(anchor_schedule, replacement_schedule)
    )

    # Sanity: the patch should change the file.
    assert patched != src, "patch produced identical output — something's wrong"
    # And it should still parse.
    compile(patched, str(target), "exec")

    target.write_text(patched)
    print(f"[vllm_patches] Applied output_handler metrics offload to {target}")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm-root>", file=sys.stderr)
        return 2
    vllm_root = Path(sys.argv[1]).resolve()
    if not (vllm_root / "vllm" / "v1" / "engine" / "async_llm.py").exists():
        print(f"[vllm_patches] async_llm.py not found under {vllm_root}", file=sys.stderr)
        return 3

    patch_output_handler_metrics_offload(vllm_root)
    print("[vllm_patches] All patches applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
