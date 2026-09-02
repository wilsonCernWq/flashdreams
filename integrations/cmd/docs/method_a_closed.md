# Method A (`torch._scaled_mm` weight quantization) — closed

Closes the work in `quantization_plan.md`. **Dropped, not on hold.** Superseded
by the native FP8 DiT (`omnidreams_singleview`), which measures 1.50x on the
diffusion step against Method A's 1.17x and cannot be combined with it.

## What this branch built

FP8/FP4 replacement of `nn.Linear` in the Cosmos transformer via
`torch._scaled_mm`, opt-in through `CosmosTransformerConfig.weight_quantization`,
plus a `cmd-chunk4-short-i2v-fp8` preset for A/B.

## What it measured

CMD `chunk4_short`, 60 blocks, n=3, RTX PRO 6000:

| | bf16 | fp8 | |
|---|---|---|---|
| diffuse/step | 704 ms | 622 ms | 1.13x median, 1.24x mean |
| total/step | 988 ms | 843 ms | **1.17x** |
| peak mem | 38.5 GiB | 36.8 GiB | −1.7 GiB |

The kernel is not the problem: `_scaled_mm` runs the `mlp.layer2` shape in
0.3277 ms against `mm`'s 0.5448 ms, a 1.66x that beat all 105 Triton candidates.
The ceiling is Amdahl. Attention, the VAE and the scheduler are untouched, so a
1.66x kernel becomes roughly 1.2x on the transformer and 1.17x on the step.

Also worth recording: compiled FP4 reached 1.51x per-Linear here.
`quantization_plan.md` states FP4 is slower than bf16, which was measured on
GB10 — that claim needs per-hardware scoping if anyone revisits FP4.

## Why it cannot stack on the native path

Two independent reasons, both checked 2026-09-01.

**Mutually exclusive by construction.** The native gate on
`dev/cmd-fp8-native-cutlass` (`flashdreams_cmd/transformer/native_fp8.py:151`)
explicitly refuses `weight_quantization != "none"`. Method A rewrites the module
tree, swapping `nn.Linear` for `TorchScaledMMFP8Linear`, while the native weight
prep resolves `.weight` out of the state dict by literal name — after the swap
those keys no longer exist.

**Lifting that restriction would buy no speed.** Comparing every `nn.Linear` in
the CMD network against the nine names the bridge consumes
(`integrations/omnidreams/omnidreams_singleview/src/dit_streaming/pyext/streaming_dit_bridge.cu`),
Method A covers exactly three things the native path does not:

| | count |
|---|---|
| `blocks.N.cross_attn.k_proj` | 28 |
| `blocks.N.cross_attn.v_proj` | 28 |
| `crossattn_proj.0` | 1 |

All three are on the once-per-rollout path, not the per-step one.
`flashdreams/recipes/cosmos/transformer/impl/modules.py:576` builds the
cross-attention KV cache with `self.cross_attn.initialize_cache(context)` from
the text embedding and keeps it for the whole rollout, so none of them appear in
`diffuse`.

Memory is all that is left, and it is noise at this scale: those 220.2 M
parameters are 440 MB in bf16 against the native path's 1644.2 M / 3288 MB, and
native FP8 already costs 8.3 GiB *more* than eager for its workspace and FP8 KV
cache.

## What outlives the decision

- `flashdreams/core/quant.py` — moved here from `integrations/sana/sana_wm/`, so
  SANA-WM and Cosmos share one implementation instead of two copies. Unaffected
  by this closure.
- The `keep_source_weight` flag. Keeping both the bf16 and FP8 weights cost
  1.8 GiB; the bf16 copy is dead in inference *except* that SANA-WM's
  `stage1_model.py:474,478` reads `.weight` off quantized layers, so it is
  opt-in rather than deleted.
- `integrations/cmd/tools/bench_fp8_linear.py` — per-Linear FP8/FP4 vs bf16,
  eager and compiled. Still the quickest way to sanity-check a shape.

## If anyone revisits

`flashdreams.accelerated` (`441e842`, on `main`) is the principled version of
this idea: `QuantizedNonPersistentLinear` offers per-out-channel granularity
where this branch is per-tensor, and `OptimizedMultiHeadAttention` quantizes
attention as well — precisely what Method A was missing. It is also
model-agnostic, where `omnidreams_singleview` is written against the Cosmos DiT
specifically. Nobody has measured it end to end. Reaching it requires rebasing
onto the v2 layout, which was ruled out for this workstream on 2026-09-01.
