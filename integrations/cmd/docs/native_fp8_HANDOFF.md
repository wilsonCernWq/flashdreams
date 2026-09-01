<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HANDOFF: native FP8 DiT port — for an agent on RTX 5090 / RTX PRO 6000 hardware

You are picking up a port that was developed on a **DGX Spark GB10 (`sm_121a`)**, on which the shipped
`12.0a` build **could not be made to run**. Everything blocked on running it was left undone on purpose
and is listed here.

> **Correction, 2026-09-01 — the reason matters and it is not what the older docs say.** The kernels
> themselves are fine on cc-12.1: `cutlass::arch::Sm120` templates were measured producing correct
> numerics there under `sm_120f`/`sm_121a` builds. The `12.0a` failure is that an `a`-suffixed cubin has
> no forward compatibility, so the runtime refuses to load it (CUDA 209) — and
> `cosmos_gemm_bf16.cu:210` discards that code and reports "unknown error". The arch fix is a fat-binary
> default (`"12.0a;12.1a"`). See `integrations/omnidreams/docs/multi_arch_support.md` (in the `main/`
> checkout) for the full diagnosis, including two non-arch blockers — a repo-local cudnn-frontend API
> bug that likely breaks the bf16 backend **on your RTX hardware too**, and hard-coded `minor == 0` /
> device-name allowlists in host dispatch. **Read that doc before trusting §1 or §5 below.**

Read in this order:

1. `native_fp8_port_plan.md` — the phased plan (§4 defines validation tiers T0/T1/T2 referenced below).
2. `quantization_native_port_scoping.md` — why GB10 can't run this; all arch measurements.
3. This file — exact state, exact next commands.

**Branch:** `dev/integrate-cmd-fp8-native`, in the worktree
`main.native-cutlass/`. Uncommitted (nothing was committed or pushed).
A sibling branch `dev/integrate-cmd-fp8` holds the *unrelated* `torch._scaled_mm` FP8 route.

---

## 1. First thing to do: confirm the premise this port rests on

Everything downstream assumes the kernels run on your GPU. **Verify that before writing any code** —
it takes ~4 minutes and it is the one thing GB10 could never check.

```bash
cd <repo>/main.native-cutlass
python3 integrations/omnidreams/omnidreams_singleview/tools/sync_thirdparty.py sync   # ~1 min, needs network
OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST=1 \
  uv run pytest integrations/omnidreams/tests/test_omnidreams_singleview_native.py::test_cuda_native_extension_builds -v -s
```

Expected: passes in ~165 s (it passed on GB10 too — building was never the problem).
Note the default arch is `12.0a`, which is correct for your GPU; **do not** set
`OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST`.

Then the real premise check — the DiT block kernels actually executing:

```bash
uv run python3 <repo>/main.native-cutlass/integrations/cmd/docs/handoff_smoke.py
```

That script does not exist yet — write it from `diag_native_matrix.py`, described in §3. On GB10 every
cell of that matrix failed; **on your hardware the `bf16` row must pass before anything else matters.**

If it does *not* pass on an allowlisted SM120a GPU, stop and report — that would mean the native path is
broken for reasons unrelated to architecture, and the whole port premise needs re-examining.

---

## 2. What is already implemented (all of it CPU-verified, none of it GPU-verified)

| File | State |
|---|---|
| `integrations/cmd/flashdreams_cmd/transformer/native_fp8.py` | **New.** The eligibility gate: `resolve_native_fp8()`, the `sm_120a` capability+allowlist probe, and the two hard refusals (wrong arch, camera conditioning). No native call yet — this is the gate in front of it. |
| `integrations/cmd/flashdreams_cmd/transformer/__init__.py` | **Modified.** Added `native_dit_acceleration: Literal["disabled","auto","required"] = "disabled"` to `CMDTransformerConfig` + re-exports. |
| `integrations/cmd/tests/test_native_fp8.py` | **New.** 19 `ci_cpu` tests, all passing. |

Verified locally at the time: `pytest integrations/cmd/tests/ -m ci_cpu` → **85 passed**;
`pre-commit run --files <the three>` → ruff + ruff-format + ty all **Passed**. §7 below adds Phase 2/3
work on top of this — the current suite total is **152**.

Two of those tests are load-bearing and worth understanding before you change anything:

- `test_camera_presets_are_model_ineligible_for_the_native_path` — CMD's shipped camera presets
  (`camera_dim=1536`) are refused. This is **correct until plan Phase 5**: the C++ bridge looks weights up
  by literal name and *silently ignores unknown keys*, so a camera-conditioned model would run natively
  and emit **camera-blind video with no error**. Do not relax this to make a demo work.
- `test_sm120_kernel_sites_match_expected_inventory` — pins the arch-conditional kernel surface at
  exactly 48 sites (`ops.cu` ×40, `cosmos_fp8_tc_probe.cu` ×8). If a CUTLASS/vendored bump changes this,
  the hardware assumptions need re-verifying, and you'll find out here instead of at kernel launch.

**Not implemented** (plan Phase 1 items deliberately skipped): the
`flashdreams_native_cosmos` package extraction, the optional-extra dependency wiring, and
`max_jobs` plumbing. The gate currently has **no** import of `omnidreams.*` at all — it is pure
capability/config logic, so it has no packaging consequences yet. Adding the actual native call is what
forces that decision; see plan §6 for extraction-vs-depend (both are architecturally legal).

---

## 3. Reproducing the diagnostic matrix

The GB10 diagnostic lived at `$CLAUDE_JOB_DIR/tmp/diag_native_matrix.py` (a scratch dir, not in the
repo — it is gone). Rewrite it; it is ~90 lines and the shape matters more than the code:

- Build omnidreams' **own** transformer (`omnidreams.transformer.CosmosTransformerConfig` +
  `CosmosDiTNetworkConfig`), `checkpoint_path=None` (random init is fine), `compile_network=False`,
  `use_cuda_graph=False`, `native_dit_acceleration="required"`.
- Sweep `native_dit_backend` over `("bf16", "fp8_kvcache_cudnn")` × three sizes:
  `C=512 h=4 b=2 16×16 t=4`, `C=2048 h=16 b=2 16×16 t=4`, `C=2048 h=16 b=2 64×64 t=8`.
- Drive `transformer.predict_flow(...)` inside `cache.start(i)` / `cache.finalize(i)`.
- **Gotcha that cost an hour:** `predict_flow(..., input=None)` makes `hdmap_patched` resolve to Python
  `None`, but the pybind signature demands a real tensor even when `additional_concat_ch=0`. Pass an
  explicit zero-width tensor: `torch.empty((*latent.shape[:-1], 0), dtype=..., device=...)`.
- `OMNIDREAMS_DIT_PROFILE=1` enables the kernel's built-in per-stage timers
  (`cosmos_block.cu:1404`) — per-block and per-substage (`EV_AFTER_SA_QKV`, `EV_AFTER_FFN1`, …). Use it
  for the FP8-vs-BF16 breakdown; it is far better than wrapping the whole forward.

---

## 4. The number nobody has: FP8 vs BF16 speedup

**No native speedup measurement exists.** It could not be taken on GB10. The only measured FP8 number in
this repo is from the *other* route (`quantization_plan.md` §Phase 0.5): `torch._scaled_mm`, isolated
CMD-shaped GEMMs, **1.17–1.84× over BF16, and only under `torch.compile`** (eager was 0.24–0.81×, i.e.
slower). Treat that as the bar the native path must clear to justify its complexity — and as a useful
oracle, since both quantize the same linears.

Measure native FP8 with `native_dit_backend` toggled `bf16` ↔ `fp8_kvcache_cudnn` on the **same**
random-initialized network, steady-state after warmup (the first call pays one-time weight quantization),
`torch.cuda.Event` timing. Report ms/step for both plus the per-stage profile.

---

## 5. Known blockers waiting for you, in priority order

Full detail in `native_fp8_port_plan.md` §2 (risk register) and §14 (open questions). The ones that are
*specifically* blocked on your hardware:

1. **R4 — prefill FP8 self-KV seeding.** CMD's `prefill` seeds a conditioning prefix into the self-attn
   KV cache. The FP8 runtime allocates its own `k_self_fp8_caches`/`v_self_fp8_caches` as **zeros**
   (`optimized_dit.py:1248-1249`); nothing populates them from CMD's prefix. Expected symptom: the first
   chunk attends to zeros. Confirmed by reading, never observed — needs a real forward.
2. **R5 — rank/`batch_shape` mismatch.** The bridge wants 5D `[B,V,T,HW,D]` with `V==1`; CMD's latents are
   rank-2. Mechanical, but only verifiable once kernels run.
3. **R6 — FP8 numerics at CMD shapes.** Genuinely unknown. Compare against BF16 with cosine-sim/PSNR, not
   bit-exactness. If it fails, note that static activation scales are *not* a quick fix: plan §3 verified
   that 9 of the 10 declared scale sites are unconsumed by any kernel, so wiring them is a new kernel
   project, not a calibration script.
4. **R7 — per-token timesteps** (`conditional_frame_timestep`) appear architecturally unsupported by the
   native AdaLN path. Verify against a real CMD preset early; it may constrain which presets can ever use
   this.

**Good news you can rely on** (verified during planning, no GPU needed): the state-dict keys of
omnidreams' transformer and the shared flashdreams one CMD uses are **byte-identical**, and
`prepare_cosmos_quantized_streaming_weights` was run successfully against a real CMD checkpoint
(600 keys). Nothing needs renaming. And FP8 activation calibration is **not** required — quantization is
dynamic per-tensor amax on-device.

---

## 6. Two environment papercuts that will bite you

- **`3rdparty/` cannot be shared between checkouts.** `sync_thirdparty.py`'s `_source_hash()` hashes the
  *absolute* patch path, so a tree synced elsewhere fails validation with
  `cutlass stamp does not match manifest`. Each worktree needs its own ~348 MB copy + its own `sync` run.
  (A symlink was tried; that is exactly how this was discovered.)
- **A wrong-arch build can fail silently.** On GB10, an `sm_120a` kernel launch returned "no error" while
  never executing (output stayed 0). If you ever see suspiciously perfect zeros, check the arch before
  debugging the math. This is why `native_fp8.py` refuses rather than attempts.

---

## 7. Open items from the overnight Phase 2/3 pass

An unattended pass implemented most of plan Phase 2 and the mechanical half of Phase 3. What landed:

| File | What it is |
|---|---|
| `flashdreams_cmd/transformer/native_weights.py` | The weights-dict contract: `build_native_weights` (refuses camera models up front), `validate_native_weights` (a pure-Python re-implementation of every bridge `TORCH_CHECK`), `move_native_weights_to_device` (alias-preserving). |
| `flashdreams_cmd/transformer/native_adapter.py` | Shape/config translation: `to_bridge_latent`/`from_bridge_latent` (rank-2 CMD ↔ 5-D bridge), `resolve_patch_dims`, `latent_grid`, `empty_hdmap_like`. |
| `flashdreams_cmd/transformer/native_fp8.py` | Extended with the **R7 per-token-timestep refusal**. |
| `tests/test_native_fp8_weights.py` (44) / `test_native_fp8_adapter.py` (23) | `ci_cpu`. Suite total 152, all passing; ruff + ruff-format + ty clean. |

**Plan claims re-verified against the code — all held.** CMD and omnidreams state-dict keys are identical
(diffed on a real network); `prepare_cosmos_quantized_streaming_weights` runs on a CMD state dict with no
native extension and no GPU; `_fp8_prepared` aliases really do share storage (`data_ptr` equality), so the
double-GPU-copy problem is real; fused QKV really does drop the split q/k/v.

Two things worth knowing that the plan does not say:

- **All six released presets have `conditional_frame_timestep=None`**, so the R7 cap blocks nothing today.
  It is a guard against a future preset, not a present-day limitation.
- **`cross_attn.k_proj` / `v_proj` stay bf16** — they feed the cross-attention KV cache, not a block
  linear, so they are absent from the FP8 linear set. Do not "fix" this by quantizing them.

### Still to do — needs your GPU, or is simply unfinished

1. **The `_ensure_fp8_runtime` fake-extension drive** (Phase 3 exit criterion 1) — NOT done. This is the
   largest remaining blind piece: stand up the executor against a `SimpleNamespace` fake with the
   workspace builder stubbed (idiom at `integrations/omnidreams/tests/test_omnidreams_singleview_native.py:839-914`)
   and assert the produced config dict key by key against the bridge's `workspace_tensor` rules
   (`streaming_dit_bridge.cu:2043-2069`). The translation primitives it needs now exist; the drive does not.
2. **The `CMDTransformer` lifecycle hooks** — `initialize_autoregressive_cache` / `predict_flow` /
   `finalize_kv_cache`. Note the plan's ordering requirement: the post-cache-init hook must fire *after*
   CMD's two `self.network.prefill(...)` calls (`transformer/__init__.py:200-215`), or the FP8 shadow
   caches are built from an unseeded BF16 cache. Phase 4 (R4) depends on getting this right.
3. **The T2a `ci_gpu` skeleton** (collected-and-skipped on non-SM120a) — not created. `_require_sm120a()`
   is specified in plan §4; nothing imports it yet.
4. **`move_native_weights_to_device` is implemented but unwired** — there is no call path to attach it to
   until item 1 exists. It is tested in isolation (via `device="meta"`, which exercises the dedup logic
   without a GPU).
5. **`prepare_tensor_for_native` / `NativeTensorSpec` layout checks** (Phase 2 item 3) — skipped
   deliberately. Those helpers live in the compiled extension module, so using them would have made the
   Phase 2 tests depend on a native build. The weights contract is validated directly instead, which is
   the property that actually matters.
