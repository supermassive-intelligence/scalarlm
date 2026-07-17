#!/usr/bin/env python3
"""Offline NaRA memorize eval for DiffusionGemma.

Serving NaRA is deferred (ADR 0012): the fork's static .pt adapter path can't
reconstruct dW(t)=B·C(t)·A. This runs the model's OWN native block-diffusion
`generate()` decode in-process with the NaRA adapter injected, hooking the shared
mapper into every denoising step (noise level t = fraction of not-yet-accepted
canvas positions, exactly training's corruption fraction), then scores the decoded
output against the golden string.

Runs three decodes from the SAME seed (so the only variable is the adapter):
  BASE          - no adapter
  LORA_ONLY     - NaRA adapter, mapper forced to identity (Ceff=I) => plain B·A
  NARA          - NaRA adapter, mapper active per denoising step
Reports difflib longest-block and total-match scores vs golden for each.

Usage (inside the cray container):
  python nara_offline_eval.py --checkpoint /app/cray/.../checkpoint_449.pt \
      --model google/diffusiongemma-26B-A4B-it \
      --prompt "My bank account's balance is" \
      --golden aaaf6f8ae738dfc6577e63dda6daf9cc --seed 42
"""
import argparse, sys, time, difflib, json

sys.path.insert(0, "/app/cray/ml")
sys.path.insert(0, "/app/cray/infra")

import torch


def block_scores(sample: str, golden: str, prompt: str = ""):
    """Sweep-reference scorer (~/block.py): longest contiguous match + total matched
    chars via difflib, GOLDEN first. The sweep scored the generated hex only, so strip
    the prompt prefix. autojunk=False so a long canvas decode isn't mangled (block.py's
    inputs were short enough that autojunk never triggered — equivalent on those)."""
    if prompt and sample.startswith(prompt):
        sample = sample[len(prompt):].lstrip()
    sm = difflib.SequenceMatcher(None, golden, sample, autojunk=False)
    m = sm.find_longest_match(0, len(golden), 0, len(sample))
    total = sum(b.size for b in sm.get_matching_blocks())
    return m.size, total


def load_base(model_id):
    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map={"": 0},
    )
    model.eval()
    print(f"[load] base model+tokenizer in {time.time()-t0:.1f}s", flush=True)
    return model, tok


def inject_and_load(model, ckpt_path):
    """Inject NaRA into exactly the modules the checkpoint adapted, then load the
    lora + mapper weights. Returns (context, report)."""
    from adapters.nara_prototype import inject_nara, NaRAConfig, find_nara_context

    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    lora_keys = [k for k in sd if ".lora_A" in k]
    # backbone-relative module paths -> outer-model paths (outer.model.<path>)
    mod_paths = sorted({k.rsplit(".lora_A", 1)[0] for k in lora_keys})
    outer_targets = ["model." + p for p in mod_paths]
    r = sd[lora_keys[0]].shape[0]
    mapper_keys = [k for k in sd if "nara_context.mapper" in k]

    cfg = NaRAConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, c_scale=0.1)
    inject_nara(model, outer_targets, cfg)

    # checkpoint keys are backbone-relative => load into the backbone (model.model)
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    # inject_nara created the adapter (lora_A/B) + mapper params on CPU in float32;
    # the base is on GPU in bf16 (device_map load). Training moves everything with
    # model.to(device) AFTER inject — mirror that so the adapter matches the base
    # device+dtype (else: 'mat2 is on cpu, other tensors on cuda').
    base_p = next(model.parameters())
    model.model.to(device=base_p.device, dtype=base_p.dtype)
    loaded_lora = [k for k in sd if ".lora_" in k]
    # sanity: adapter B must be non-zero after load (trained), mapper present+loaded
    some_B = [model.model.state_dict()[k] for k in sd if k.endswith(".lora_B")][:50]
    b_nonzero = sum(float(b.abs().sum() > 0) for b in some_B)
    report = {
        "adapted_modules": len(mod_paths), "rank": r,
        "lora_tensors_loaded": len(loaded_lora), "mapper_tensors": len(mapper_keys),
        "unexpected_keys": [k for k in unexpected], "b_nonzero_of_50": b_nonzero,
    }
    ctx = find_nara_context(model)
    return ctx, report


def make_mapper_hook(model, ctx, eps):
    """Wrap _denoising_step so the shared mapper sees the current noise level
    (fraction of not-yet-accepted canvas positions) before each decoder forward."""
    orig = model._denoising_step
    steps_seen = {"n": 0, "t_first": None, "t_last": None}

    def find_sampler(args, kwargs):
        s = kwargs.get("sampler")
        if s is not None:
            return s
        for a in args:
            if a.__class__.__name__ == "EntropyBoundSampler":
                return a
        return None

    def wrapped(*args, **kwargs):
        s = find_sampler(args, kwargs)
        am = getattr(s, "accepted_token_mask", None) if s is not None else None
        if am is None:
            t = 1.0
        else:
            t = float((~am).float().mean().item())
        t = max(t, eps)
        ctx.set_noise_level(torch.tensor([t], dtype=torch.float32))
        steps_seen["n"] += 1
        if steps_seen["t_first"] is None:
            steps_seen["t_first"] = t
        steps_seen["t_last"] = t
        return orig(*args, **kwargs)

    model._denoising_step = wrapped
    return steps_seen


def decode(model, tok, prompt, seed, gen_kwargs):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                             **gen_kwargs)
    seq = out.sequences if hasattr(out, "sequences") else out
    text = tok.decode(seq[0], skip_special_tokens=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--prompt", default="My bank account's balance is")
    ap.add_argument("--golden", default="aaaf6f8ae738dfc6577e63dda6daf9cc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--max-denoising-steps", type=int, default=0,
                    help="override generation_config.max_denoising_steps (0=keep)")
    args = ap.parse_args()

    model, tok = load_base(args.model)

    gcfg = model.generation_config
    if args.max_denoising_steps > 0:
        gcfg.max_denoising_steps = args.max_denoising_steps
    print("[gen] max_denoising_steps=%s sampler_config=%s" % (
        getattr(gcfg, "max_denoising_steps", "?"), getattr(gcfg, "sampler_config", "?")), flush=True)
    gen_kwargs = dict(generation_config=gcfg)

    results = {}

    # 1) BASE (no adapter)
    t0 = time.time()
    base_txt = decode(model, tok, args.prompt, args.seed, gen_kwargs)
    lo, to = block_scores(base_txt, args.golden, args.prompt)
    results["BASE"] = {"block": lo, "total": to, "sample": base_txt[:80]}
    print(f"[BASE] block={lo}/{len(args.golden)} total={to} ({time.time()-t0:.1f}s) :: {base_txt[:80]!r}", flush=True)

    # inject adapter
    ctx, rep = inject_and_load(model, args.checkpoint)
    print("[adapter] " + json.dumps(rep), flush=True)
    assert ctx is not None, "no NaRAContext after injection"
    assert rep["mapper_tensors"] > 0, "checkpoint has no mapper — persistence fix missing?"

    # 2) LORA_ONLY (mapper forced to identity)
    ctx.set_training_stage(1)   # Ceff = I every step => plain B·A
    t0 = time.time()
    lora_txt = decode(model, tok, args.prompt, args.seed, gen_kwargs)
    lo, to = block_scores(lora_txt, args.golden, args.prompt)
    results["LORA_ONLY"] = {"block": lo, "total": to, "sample": lora_txt[:80]}
    print(f"[LORA_ONLY] block={lo}/{len(args.golden)} total={to} ({time.time()-t0:.1f}s) :: {lora_txt[:80]!r}", flush=True)

    # 3) NARA (mapper active per denoising step)
    ctx.set_training_stage(2)
    seen = make_mapper_hook(model, ctx, args.eps)
    t0 = time.time()
    nara_txt = decode(model, tok, args.prompt, args.seed, gen_kwargs)
    lo, to = block_scores(nara_txt, args.golden, args.prompt)
    results["NARA"] = {"block": lo, "total": to, "sample": nara_txt[:80]}
    print(f"[NARA] block={lo}/{len(args.golden)} total={to} ({time.time()-t0:.1f}s) "
          f"steps={seen['n']} t:{seen['t_first']:.3f}->{seen['t_last']:.3f} :: {nara_txt[:80]!r}", flush=True)

    print("\n=== SUMMARY (golden=%s) ===" % args.golden, flush=True)
    for k in ("BASE", "LORA_ONLY", "NARA"):
        r = results[k]
        print(f"  {k:10s} block={r['block']:2d}/{len(args.golden)} total={r['total']:2d}", flush=True)
    print("RESULTS_JSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
