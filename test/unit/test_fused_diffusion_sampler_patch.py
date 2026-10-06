"""Unit tests for the fused DiffusionGemma sampler patches.

Run the patches against minimal stand-ins for vllm-fork's diffusion_gemma.py and
v1/worker/gpu/model_runner.py carrying the exact anchors. These tests cover the source
transformation and run without a GPU; the kernels themselves are tested on hardware in
test_fused_diffusion_sampler_kernels.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "vllm_patches"))

from apply_patches import (  # noqa: E402
    FUSED_DIFFUSION_SAMPLER_SRC,
    patch_diffusion_gemma_fused_sampler,
    patch_model_runner_fused_softcap,
)

DIFFUSION_GEMMA = '''\
class DiffusionGemmaForConditionalGeneration:
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits


class DiffusionSampler:
    def __call__(self, logits, input_batch, num_decode, CL, max_num_logprobs, device):
        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)
        group = max(num_decode, 1)
        if num_decode > 0:
            free, _ = current_platform.mem_get_info()
            # ~10 transient fp32 copies of [group * CL, vocab] inside the step
            # (eager peaks at ~8; pad for allocator overhead and small tensors).
            bytes_per_req = CL * self.vocab_size * 4 * 10
        for start_req in range(0, num_decode, group):
            end_req = min(start_req + group, num_decode)
            scaled = _compiled_sample_step(
                logits[start_req * CL : end_req * CL],
                tp_group_name=self.tp_group_name,
            )
'''

MODEL_RUNNER = '''\
class GPUModelRunner:
    def sample(self, hidden_states, input_batch, grammar_output):
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        return self.sampler(logits, input_batch)
'''


def _tree(tmp_path, dg=DIFFUSION_GEMMA, mr=MODEL_RUNNER):
    models = tmp_path / "vllm" / "model_executor" / "models"
    models.mkdir(parents=True)
    (models / "diffusion_gemma.py").write_text(dg)
    runner = tmp_path / "vllm" / "v1" / "worker" / "gpu"
    runner.mkdir(parents=True)
    (runner / "model_runner.py").write_text(mr)
    return models, runner


def test_adds_the_dispatch_the_module_and_the_raw_logits_path(tmp_path):
    models, _ = _tree(tmp_path)
    patch_diffusion_gemma_fused_sampler(tmp_path)
    src = (models / "diffusion_gemma.py").read_text()
    assert "(fused_sample_step if use_fused else _compiled_sample_step)(" in src
    assert "max_num_logprobs < 0" in src                      # logprobs keep the original path
    assert '"SCALARLM_FUSED_DIFFUSION_SAMPLER", "1") != "0"' in src
    assert "(2 * 2 if use_fused else 4 * 10)" in src
    assert "def scalarlm_compute_sample_logits(" in src
    assert 'raw_softcap = getattr(logits, "_scalarlm_softcap", 0.0)' in src
    assert "logits = _softcap_logits(logits, raw_softcap)" in src   # exact fallback softcap
    assert "**step_kwargs," in src
    assert (models / "scalarlm_fused_sampler.py").read_text() == FUSED_DIFFUSION_SAMPLER_SRC
    compile(src, "diffusion_gemma.py", "exec")


def test_model_runner_uses_the_raw_logits_method_only_when_present(tmp_path):
    _, runner = _tree(tmp_path)
    patch_model_runner_fused_softcap(tmp_path)
    src = (runner / "model_runner.py").read_text()
    assert 'getattr(self.model, "scalarlm_compute_sample_logits", None)' in src
    assert "logits = self.model.compute_logits(sample_hidden_states)" in src   # other models
    ns: dict = {}
    exec(src, ns)

    class Plain:                                    # a model without the method
        def compute_logits(self, h):
            return ("softcapped", h)

    class Diffusion(Plain):
        def scalarlm_compute_sample_logits(self, h):
            return ("raw", h)

    class Batch:
        logits_indices = slice(None)

    r = ns["GPUModelRunner"]()
    r.sampler = lambda logits, batch: logits
    r.model = Plain()
    assert r.sample([1], Batch(), None)[0] == "softcapped"
    r.model = Diffusion()
    assert r.sample([1], Batch(), None)[0] == "raw"


def test_embedded_module_is_valid_python_with_the_entry_point():
    compile(FUSED_DIFFUSION_SAMPLER_SRC, "scalarlm_fused_sampler.py", "exec")
    assert "def fused_sample_step(" in FUSED_DIFFUSION_SAMPLER_SRC


def test_gumbel_noise_uses_the_torch_rand_distribution():
    # tl.rand rounds to nearest, which halves the mass of the top fp32 level and thins the
    # Gumbel tail relative to the original's torch.rand_like; _uniform24 reproduces it.
    assert "u = _uniform24(seed, row.to(tl.int64) * V + idx)" in FUSED_DIFFUSION_SAMPLER_SRC
    assert "tl.rand(" not in FUSED_DIFFUSION_SAMPLER_SRC


def test_column_pruning_is_embedded_and_off_by_default():
    src = FUSED_DIFFUSION_SAMPLER_SRC
    assert "def pruned_soft_embeds(" in src and "def _col_max(" in src and "def _row_probs_cols(" in src
    assert 'os.environ.get("SCALARLM_FUSED_SAMPLER_SC_COLS", "0")' in src
    assert "pruned = 0 < k_cols < (sc_vocab_end - sc_vocab_start) and tp_size == 1" in src


def test_applying_twice_is_a_no_op(tmp_path):
    models, runner = _tree(tmp_path)
    for _ in range(2):
        patch_diffusion_gemma_fused_sampler(tmp_path)
        patch_model_runner_fused_softcap(tmp_path)
    once = ((models / "diffusion_gemma.py").read_text(), (runner / "model_runner.py").read_text())
    patch_diffusion_gemma_fused_sampler(tmp_path)
    patch_model_runner_fused_softcap(tmp_path)
    assert ((models / "diffusion_gemma.py").read_text(), (runner / "model_runner.py").read_text()) == once


def test_drifted_source_fails_loudly(tmp_path):
    _tree(tmp_path, dg=DIFFUSION_GEMMA.replace("4 * 10", "4 * 12"),
          mr=MODEL_RUNNER.replace("sample_hidden_states)", "sample_hidden_states, None)"))
    with pytest.raises(AssertionError):
        patch_diffusion_gemma_fused_sampler(tmp_path)
    with pytest.raises(AssertionError):
        patch_model_runner_fused_softcap(tmp_path)


def test_missing_files_are_skipped(tmp_path):
    patch_diffusion_gemma_fused_sampler(tmp_path)
    patch_model_runner_fused_softcap(tmp_path)
