<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CMD: FP8/FP4 weight quantization (design doc, not yet implemented)

Status: **deferred design**. Nothing in this document is implemented. Companion to
`sage_attention_plan.md` (attention-kernel choice); this doc covers the separate concern that
doc's Phase 2 stub gestured at: GEMM/weight quantization.

## Two things both called "omnidreams' quantization" — only one is portable

Grepped both candidate sources in full before writing this doc:

**`omnidreams_singleview`'s native FP8 path** (`cosmos_fp8_utils.py`,
`src/dit_streaming/kernels/block_quant.cu`/`.cuh`, `src/dit_streaming/pyext/sage3_fp4_quant_shim.cu`,
driven from `optimized_dit.py`'s `_ensure_fp8_runtime`/`_uses_fp8_dit`). The scale/quantize math in
`cosmos_fp8_utils.py` (`quantize_fp8_per_out_channel`, per-out-channel E4M3, dynamic/calibrated/static
activation scales) is plain tensor math and genuinely portable on its own. But nothing in
`optimized_dit.py` ever executes a quantized GEMM or attention op through PyTorch — every FP8 codepath
routes through `self._native_extension` (custom CUTLASS kernels in the files above, JIT-built, targeting
`omnidreams`'s own separate transformer implementation, not the shared `flashdreams` one CMD uses).
Grepped the whole `omnidreams_singleview` tree for `_scaled_mm`/`torchao`: zero hits — there is no
PyTorch-native fallback anywhere in this path. Porting it means porting the CUTLASS kernels and JIT
toolchain too, which is exactly what `sage_attention_plan.md` already concluded is a "materially larger,
separate undertaking." That conclusion stands; this doc does not revisit it.

**`sana/sana_wm/quant.py`** — a different, much smaller pattern already living in this same repo, using
only `torch._scaled_mm` (PyTorch's native scaled-matmul op) and Triton for FP4 packing. No custom CUDA
extension, no JIT native build. `TorchScaledMMFP8Linear`/`TorchScaledMMFP4Linear` are drop-in `nn.Linear`
replacements (quantize weight once at conversion time, quantize activations per forward call);
`replace_linear_with_quant(module, recipe, params_dtype, skip_patterns, include_patterns)` recursively
walks `named_children()` and swaps eligible `nn.Linear` instances — completely model-agnostic.

This is what actually matches "structure should be roughly the same, shouldn't need much code": CMD's
transformer already uses plain `nn.Linear`, and the naming lines up almost exactly with omnidreams'
own FP8 weight-key scheme, which is itself evidence both derive from the same upstream Cosmos
architecture:

| omnidreams `_COSMOS_BLOCK_FP8_LINEAR_KEYS`   | flashdreams Cosmos `Block` (`impl/modules.py`)          |
|-----------------------------------------------|----------------------------------------------------------|
| `self_attn.q_proj.weight` / `k_proj` / `v_proj` | `Block.self_attn.{q,k,v}_proj` (`MultiHeadAttention`, :259-261) |
| `self_attn.output_proj.weight`                | `Block.self_attn.output_proj` (:262)                      |
| `cross_attn.q_proj.weight`                    | `Block.cross_attn.q_proj` (`CrossAttention` reuses `MultiHeadAttention`) |
| `cross_attn.output_proj.weight`               | `Block.cross_attn.output_proj`                            |
| `mlp.layer1.weight` / `mlp.layer2.weight`     | `Block.mlp.layer1` / `layer2` (`GPT2FeedForward`, :37-38)  |

`Block` itself lives in `flashdreams/flashdreams/recipes/cosmos/transformer/impl/modules.py:471+`
(`self.self_attn`/`self.cross_attn`/`self.mlp`, :491/503/513) — shared code, not CMD-specific. CMD's
`CMDTransformerBlock` (`integrations/cmd/flashdreams_cmd/transformer/modules.py:28`) adds exactly one
extra `nn.Linear`: `self.self_attn.cam_encoder` (:37-40), and touches nothing else about the Linear
inventory. So this is the rest of the doc: port the SANA-WM *pattern* (not omnidreams' kernels).

## Real edit sites

1. `flashdreams/flashdreams/recipes/cosmos/transformer/impl/modules.py` — no change needed; existing
   `Block`/`MultiHeadAttention`/`GPT2FeedForward` classes already have the right `nn.Linear` shape/naming.
2. `flashdreams/flashdreams/recipes/cosmos/transformer/__init__.py:181-243` (`CosmosTransformer.__init__`)
   — the exact insertion point, mirroring where SANA-WM calls `_prepare_stage1_quant()` (right after
   `load_state_dict` + dtype cast, before first use): between line 219
   (`self.network.update_parameters_after_loading_checkpoint()`) and line 221
   (`if config.compile_network: self.network = compile_module(self.network)`). This is also precisely
   the site `sage_attention_plan.md`'s Phase 2 stub already earmarked ("applied ... after checkpoint
   load, before `compile_module()`") — confirmed real, not aspirational; `CosmosTransformerConfig` is a
   real class at `__init__.py:101` with `dtype`/`checkpoint_path`/`state_dict_transform`/
   `compile_network` fields already there to extend.
3. New module, e.g. `flashdreams/flashdreams/core/quant.py` (or `infra/`, per the `core -> infra ->
   recipes` dependency rule — this is generic enough to belong below `recipes`, unlike attention backend
   selection which stayed in `core/attention`) — port `TorchScaledMMFP8Linear`, `TorchScaledMMFP4Linear`,
   and `replace_linear_with_quant` from `integrations/sana/sana_wm/quant.py:260-712` essentially
   verbatim; nothing in that file is SANA-specific.
4. `CosmosTransformerConfig` (`__init__.py:101+`) — add `weight_quantization: Literal["none","fp8","fp4"]
   = "none"` next to `compile_network`, following the same plumbing precedent
   `sage_attention_plan.md` used for `attention_backend` (new literal field, opt-in default).
5. Skip-pattern list — must include `cam_encoder` for CMD specifically (small, camera-specific,
   `in_features`/`out_features` not guaranteed divisible by 16 the way `_replace_linear_with_quant`
   requires at `quant.py:632`) alongside the same categories SANA-WM already skips: embedders,
   time/AdaLN-modulation projections, final output layer (`_STAGE1_QUANT_SKIP_DEFAULTS`,
   `transformer.py:64-72`, generalizes directly — Cosmos has the same shape of auxiliary layers:
   `x_embedder`, `t_embedder`, `adaln_modulation_*`, `final_layer`).

No change needed to `integrations/cmd/flashdreams_cmd/transformer/network.py` or `modules.py` beyond the
skip-pattern addition — `CMDDiTNetworkConfig` inherits `CosmosDiTNetworkConfig` (not
`CosmosTransformerConfig`, note the different class — the transformer/network split already exists;
`weight_quantization` belongs on `CosmosTransformerConfig` since that's the class that owns
`checkpoint_path`/`compile_network`, i.e. the load lifecycle) automatically. Because the insertion point
is in shared `recipes/cosmos`, every other Cosmos-based integration gets this for free too — matches the
"generic config slot, not a model-specific branch" rule in `CLAUDE.md`.

## Phase 0 — hardware spike (**done**, 2026-08-31)

Ran on the actual dev box: `nvidia-smi` reports `NVIDIA GB10`, compute capability `12.1` (sm_121),
driver `580.173.02`; installed `torch==2.12.1+cu130` (newer than the `2.11.0+cu130` figure in
`sage_attention_plan.md` — re-check that doc's pin if acting on it later). Results, most conservative
first:

1. `hasattr(torch, "_scaled_mm")`, `torch.float8_e4m3fn`, `torch.float4_e2m1fn_x2` — all present.
2. Raw `torch._scaled_mm` FP8 call (E4M3, row-wise scales, `float32` scale dtype — the earlier
   `bfloat16`-scale attempt correctly raised `RuntimeError: Invalid scaling configuration`, matching the
   dtype contract `sana_wm/quant.py` already encodes): **works**. `cos_sim=0.9993` vs. a `bf16` reference
   matmul.
3. `sana_wm.quant.quantize_nvfp4_swizzled` (the Triton packing kernel) + raw `torch._scaled_mm` FP4 call:
   **works** — Triton compiled and ran for sm_121 with no codegen failure, contradicting
   `sage_attention_plan.md`'s "possibly Triton itself lacks sm_121 support yet" concern (at least for
   this kernel). `cos_sim=0.9910` vs. `bf16` reference (lower than FP8, expected — 4-bit mantissa).
4. The real `sana_wm.quant.TorchScaledMMFP8Linear`/`TorchScaledMMFP4Linear` classes (not just the raw
   ops) end-to-end on an `nn.Linear(256, 256)`: **work**. `cos_sim=0.9993` (FP8) / `0.9892` (FP4) vs.
   `bf16` output.
5. `integrations/sana/tests/test_quant_cuda.py` (existing `ci_gpu` suite, unmodified) —
   `uv run pytest integrations/sana/tests/test_quant_cuda.py -v -m ci_gpu`: **3 passed**, including
   `test_fp4_linear_runs_scaled_mm_on_blackwell` (the test name itself asserts this chip is recognized
   as Blackwell-class).

No blockers found. Both FP8 and FP4 are viable on this hardware today, via the exact SANA-WM code path
this doc proposes reusing — Phase 1/2 below can proceed without further hardware validation. (Scope note:
this spike used ad hoc square matmuls and one bare `nn.Linear`, not a full CMD block/network — Phase 2's
own `ci_gpu` numerics test, run against real CMD shapes, is still the gating check before calling
quantization "done.")

## Phase 0.5 — per-layer speedup microbenchmark (**done**, 2026-09-01)

Ran `TorchScaledMMFP8Linear`/`TorchScaledMMFP4Linear` head-to-head against plain `bf16` `nn.Linear`
at real CMD/Cosmos shapes (`model_channels=2048`, `num_heads=16`, `mlp_ratio=4.0` → `d_ff=8192`,
`SEQ=4096` tokens), one layer type at a time, `cuda.Event`-timed, 50 iters after 10–15 warmup:

**Eager mode (no `torch.compile`) — every quantized layer type is SLOWER than `bf16`:**

| layer                 | shape       | bf16 (ms) | fp8 (ms) | fp8× | fp4 (ms) | fp4× |
|------------------------|-------------|-----------|----------|------|----------|------|
| `self_attn.q/k/v_proj` | 2048×2048   | 0.39–0.41 | 1.57–1.61| 0.24–0.25× | 5.25–5.29 | 0.07–0.08× |
| `self_attn.output_proj`| 2048×2048   | 0.40      | 1.61     | 0.25× | 5.26 | 0.08× |
| `cross_attn.q_proj`/`output_proj` | 2048×2048 | 0.40–0.41 | 1.59–1.61 | 0.25× | 5.25–5.26 | 0.08× |
| `mlp.layer1`           | 2048×8192   | 1.83      | 2.26     | 0.81× | 6.11 | 0.30× |
| `mlp.layer2`           | 8192×2048   | 1.92      | 6.27     | 0.31× | 20.73 | 0.09× |

Root cause: per-forward-call activation quantization (amax reduction, clamp, cast, and for FP4 the
RHT16 Hadamard transform + Triton packing kernel) is a handful of extra eager PyTorch/Triton kernel
launches, and their overhead exceeds any GEMM-side saving at these shapes/this chip in eager mode.

**Under `torch.compile` (`mode="max-autotune-no-cudagraphs"`, matching CMD's real
`config.compile_network=True` path) — the picture inverts for FP8, not for FP4:**

| shape              | compiled bf16 | compiled fp8 | fp8 speedup | compiled fp4 | fp4 speedup |
|---------------------|---------------|--------------|-------------|---------------|-------------|
| 2048×2048 (q_proj-like) | 0.385 ms | 0.331 ms | **1.17×** | 1.320 ms | 0.29× (slower) |
| 2048×8192 (mlp1-like)   | 1.796 ms | 0.965 ms | **1.84×** | 2.118 ms | 0.85× (slower) |

**Conclusion — narrows scope for Phase 1/2: pursue FP8 only, not FP4, at least initially.** Compiled
FP8 is a real, meaningful win (1.17–1.84×, bigger on the larger `mlp.layer1`-shaped GEMM) and only
under `torch.compile` — an eager-mode-only caller would see a regression, so `weight_quantization`
must be documented/enforced as effectively requiring `compile_network=True` to pay off (not a hard
code dependency, but worth a docstring/test note). Compiled FP4 is *still* slower than `bf16` at these
shapes on this chip — the RHT16 + Triton packing overhead isn't amortized here, possibly because these
GEMMs are too small relative to FP4's per-call preprocessing cost, or because GB10 doesn't give FP4 a
large-enough tensor-core throughput edge over FP8 to cover it. Don't invest Phase 2 effort in the FP4
recipe until/unless a real profiling pass explains and fixes the gap; `weight_quantization`'s `"fp4"`
literal can stay defined for forward-compat but shouldn't be the initial target.

(Caveat: single isolated `nn.Linear` calls, not a full compiled `CosmosDiTNetwork` forward — inter-layer
fusion opportunities inside a real compiled graph could shift these numbers further in either direction.
The `ci_gpu` test in Phase 2 should re-measure at the full-block level before calling this "validated.")

## Phase 1 — port the generic quant module

Move `TorchScaledMMFP8Linear`/`TorchScaledMMFP4Linear`/`replace_linear_with_quant` (and their
`_require_*` hardware guards) out of `sana_wm/quant.py` into shared `flashdreams` code so both SANA-WM
and CMD (and any future integration) import the same implementation instead of forking it. This is a
mechanical move (the module has zero SANA-specific logic today), but touches SANA-WM's imports too —
coordinate so SANA-WM's existing tests keep passing against the moved location rather than duplicating
the module for CMD.

## Phase 2 — plumbing into CMD

1. `CosmosTransformerConfig.weight_quantization: Literal["none","fp8","fp4"] = "none"`.
2. `CosmosTransformer.__init__`, between `update_parameters_after_loading_checkpoint()` and the
   `compile_network` branch: if `weight_quantization != "none"`, call `replace_linear_with_quant` with a
   skip-pattern list covering embedders/AdaLN-modulation/final-layer/`cam_encoder` and no
   include-pattern restriction (unlike SANA-WM's narrow include list, Cosmos's `Block` doesn't have
   SANA's pointwise-conv-as-Linear wrinkle, so an include-pattern probably isn't needed — verify during
   implementation, don't assume).
3. One new preset for A/B comparison, following the existing `derive_config` idiom (same as
   `sage_attention_plan.md`'s `cmd-chunk4-short-i2v-sage`).

## Dependencies

Nothing new beyond what SANA-WM already declares (`torch._scaled_mm` is stdlib-torch; Triton is already
a dependency via the attention stack). No new `pyproject.toml` extras needed if the quant module moves
to a location already in CMD's dependency closure.

## Validation

Per `skills/validate-performance-quality/SKILL.md`: opt-in only (default `"none"`), numerics check
(cosine-sim/PSNR against `bf16`, not bit-exactness — FP8/FP4 are lossy) before any perf claim. Reuse
`integrations/sana/tests/test_quant_cuda.py`'s tolerance approach rather than inventing new thresholds.

## Tests

- `ci_cpu`: construction-only — a tiny CMD network builds with `weight_quantization="fp8"` without CUDA
  (the quantize call should be skippable/no-op-safe off-GPU the same way `TorchScaledMMFP8Linear`
  guards on `input.is_cuda`); assert released presets default to `"none"`.
- `ci_gpu` (`pytest.importorskip`-gated on `torch._scaled_mm` availability): numerics parity test
  against `bf16`, modeled on `sana/tests/test_quant_cuda.py`.
- `manual`: full benchmark comparison against real CMD weights, alongside the existing
  `cmd-chunk4-short-i2v` reference run.
