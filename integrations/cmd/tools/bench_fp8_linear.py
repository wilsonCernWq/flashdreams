"""FP8/FP4 vs bf16 per-Linear benchmark at real CMD shapes.

Runs from either worktree. On a branch without flashdreams.core.quant (or on a
GPU that can't do torch._scaled_mm) it says so and exits instead of pretending.

  uv run --project <repo> python3 bench/bench_fp8.py            # eager + compiled
  uv run --project <repo> python3 bench/bench_fp8.py --eager-only
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

# Real CMD/Cosmos geometry: model_channels=2048, num_heads=16 (head_dim=128),
# mlp_ratio=4.0 -> d_ff=8192, crossattn_emb_channels=1024.
D, D_FF, SEQ = 2048, 8192, 4096

LAYERS = [
    ("self_attn.q_proj", D, D),
    ("self_attn.k_proj", D, D),
    ("self_attn.v_proj", D, D),
    ("self_attn.output_proj", D, D),
    ("cross_attn.q_proj", D, D),
    ("cross_attn.output_proj", D, D),
    ("mlp.layer1", D, D_FF),
    ("mlp.layer2", D_FF, D),
]

# torch.compile is where FP8 actually wins, so cover both GEMM aspect ratios.
COMPILED = [("q_proj-like", D, D), ("mlp1-like", D, D_FF)]


def bench(mod, x, iters=50, warmup=15) -> float:
    for _ in range(warmup):
        mod(x)
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        mod(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager-only", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1

    try:
        from flashdreams.core.quant import (
            TorchScaledMMFP4Linear,
            TorchScaledMMFP8Linear,
        )
    except ImportError:
        print(
            "flashdreams.core.quant not found.\n"
            "This branch does not carry the torch._scaled_mm route "
            "(it landed on dev/cmd-fp8-scaled-mm). Nothing to benchmark here."
        )
        return 1

    cap = torch.cuda.get_device_capability()
    print(f"device : {torch.cuda.get_device_name()}  sm_{cap[0]}{cap[1]}")
    print(f"torch  : {torch.__version__}")
    print(f"shapes : D={D} D_FF={D_FF} seq={SEQ}, bf16 baseline\n")

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16

    print("EAGER (no torch.compile)")
    hdr = f"{'layer':<24}{'in':>6}{'out':>6}{'bf16 ms':>9}{'fp8 ms':>8}{'fp8x':>7}{'cos':>7}{'fp4 ms':>9}{'fp4x':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, in_f, out_f in LAYERS:
        lin = nn.Linear(in_f, out_f, bias=False).to(device=dev, dtype=dt)
        x = torch.randn(SEQ, in_f, device=dev, dtype=dt)
        ref = lin(x)

        base = bench(lin, x)
        m8 = TorchScaledMMFP8Linear.from_linear(lin, out_dtype=dt)
        t8 = bench(m8, x)
        m4 = TorchScaledMMFP4Linear.from_linear(lin, out_dtype=dt)
        t4 = bench(m4, x)
        print(
            f"{name:<24}{in_f:>6}{out_f:>6}{base:>9.3f}{t8:>8.3f}"
            f"{base / t8:>7.2f}{cos(m8(x), ref):>7.4f}{t4:>9.3f}{base / t4:>7.2f}"
        )

    if args.eager_only:
        return 0

    print("\nCOMPILED (mode='max-autotune-no-cudagraphs') — this is the real number")
    hdr2 = f"{'shape':<16}{'bf16 ms':>9}{'fp8 ms':>8}{'fp8x':>7}{'fp4 ms':>9}{'fp4x':>7}"
    print(hdr2)
    print("-" * len(hdr2))
    mode = "max-autotune-no-cudagraphs"
    for label, in_f, out_f in COMPILED:
        lin = nn.Linear(in_f, out_f, bias=False).to(device=dev, dtype=dt)
        x = torch.randn(SEQ, in_f, device=dev, dtype=dt)
        base = bench(torch.compile(lin, mode=mode), x)
        t8 = bench(
            torch.compile(TorchScaledMMFP8Linear.from_linear(lin, out_dtype=dt), mode=mode), x
        )
        t4 = bench(
            torch.compile(TorchScaledMMFP4Linear.from_linear(lin, out_dtype=dt), mode=mode), x
        )
        print(
            f"{label:<16}{base:>9.3f}{t8:>8.3f}{base / t8:>7.2f}{t4:>9.3f}{base / t4:>7.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
