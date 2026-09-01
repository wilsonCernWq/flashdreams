<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CMD: scoping the native CUTLASS FP8 port (research only, not a plan)

> **⚠️ SUPERSEDED IN PART — read `integrations/omnidreams/docs/multi_arch_support.md` first.**
> This doc's central conclusion — that omnidreams' native kernels cannot run on GB10 and that "no
> build flag can fix this" — is **wrong**, and was disproved by measurement on 2026-09-01. The
> `cutlass::arch::Sm120` kernels *do* execute correctly on this cc-12.1 device: a standalone repro of
> the exact production template stack produces correct numerics under both `sm_120f` and `sm_121a`
> builds, with identical `QMMA.16832.F32.E4M3.E4M3` SASS counts. Nothing was ever "compiled out to a
> stub".
> The actual cause of the `12.0a` failure is that an `a`-suffixed cubin has no forward compatibility
> at all, so the cc-12.1 runtime refuses to load it (CUDA error 209) — and omnidreams' own
> `cosmos_gemm_bf16.cu:210` then *discards* that error code and reports "unknown error", which is what
> sent this investigation down the wrong path. The arch-layer fix is a one-line fat-binary default
> (`"12.0a;12.1a"`), needing no CUTLASS or PyTorch patches.
> The measurements below are still accurate as *observations*; the *interpretation* is not. Two
> further blockers found later (a repo-local cudnn-frontend API bug that is not arch-specific, and
> hard-coded `minor == 0` / device-name allowlists) are why the `12.0f` whole-extension build still
> failed even though the kernels themselves were fine.

Status: **scoping note, no commitment implied**. This is not a phased implementation plan like
`sage_attention_plan.md` or `quantization_plan.md` — it exists to replace that doc's hand-wavy
"materially larger, separate undertaking" verdict on porting `omnidreams`'s native FP8 CUTLASS path with
actual numbers, so a human can decide whether to pursue this at all. Nothing here should be read as a
recommendation to build it. Written as the CUTLASS-route counterpart to the `torch._scaled_mm` route
being prototyped in `quantization_plan.md` (out of scope here, not touched).

Note on repo state while researching this: this worktree, like the sibling one prototyping
`quantization_plan.md`, was created from `main` (`289da6f`), which lacks `integrations/cmd/` entirely and
has moved `omnidreams` to `integrations_v2/`. Overlaid `origin/dev/integrate-cmd`'s tree
(`git checkout origin/dev/integrate-cmd -- .`) to research against the actual target code. All line
numbers/counts below are from that branch's `integrations/omnidreams/omnidreams_singleview/` tree.

## What "the native FP8 path" actually consists of

Total native extension: `find ... -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' | wc -l`
= **38,107 lines** — confirms `sage_attention_plan.md`'s "~38k-line" figure is the literal, accurate size
of the *whole* `omnidreams_singleview` extension (VAE streaming + DiT streaming + attention + everything),
not the marginal cost of reusing just FP8 quantization. Breaking down the FP8/quant-relevant subset:

| File | Lines | What it is |
|---|---|---|
| `dit_streaming/kernels/block_quant.cu` + `.cuh` | 806 + 286 = 1,092 | Hand-written CUDA quantize/dequantize/swizzle kernels (`quantize_per_block_128_kernel`, `quantize_weights_swizzled_128_kernel`, etc.). Only `#include <cutlass/numeric_types.h>` for FP8 type defs — **no GEMM**, this is data prep only. |
| `dit_streaming/kernels/cosmos_fp8_two_gemm.cu` | 274 | An actual FP8 GEMM: `cutlass::gemm::device::GemmBatched<..., cutlass::arch::Sm89, ...>` (:16-30) — **targets Ada (SM89 / RTX 4090-class), not Blackwell.** |
| `dit_streaming/kernels/cosmos_fp8_tc_probe.cu` | 387 | A *runtime* tensor-core capability probe using `cutlass::arch::Sm120` GEMM templates (:63-105) with `KernelTmaWarpSpecializedBlockwisePingpongSm120` schedules (:231-279) — the codebase itself doesn't assume SM120 always works; it probes first. |
| `dit_streaming/kernels/ops.cu` | 5,880 | General DiT compute kernel library — 57 top-level functions. Only a subset is FP8-specific (`Sm120Fp8RcrColscale*StageConfig` structs, `compute_activation_scale`, `convert_half_to_int8`/`int8_to_half`, `cutlass_linear_layer_rcr_int8*`); the rest (RoPE, timestep/text embedding, SiLU, patchify, `NCTHW`↔`NDHWC` layout conversion) is shared DiT infra any backend needs, not isolable as "FP8 code." Both `cutlass::arch::Sm89` (:1791, :2462, :2690, :4099...) and `cutlass::arch::Sm120` (:697-2414, dozens of instantiations) code paths coexist here. |
| `dit_streaming/kernels/cosmos_gemm_bf16.cu` | 525 | BF16 baseline GEMM, needed by the dispatch/fallback machinery even though not FP8 itself. |
| `dit_streaming/pyext/streaming_dit_bridge.cu` | 3,112 | Python↔C++ bridge; `grep -c fp8` = **434 matches** — FP8 dispatch is threaded deeply through the bridge, not cleanly separable from the rest of the streaming runtime. |
| `dit_streaming/pyext/sage3_fp4_quant_shim.cu` | 14 | Thin wrapper `#include`ing an **out-of-tree** file: `sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu` (:9). See open risk below — this source isn't in the tracked third-party manifest. |
| `python/optimized_dit.py` | 1,857 | Python glue: `_ensure_fp8_runtime`, KV-cache quantization bookkeeping, backend-string dispatch (`fp8_kvcache_cudnn`, `sage3_fp8`) — see prior research in this conversation, large fraction is FP8-specific but interleaved with general streaming-cache logic. |
| `python/cosmos_fp8_utils.py` | 826 | Entirely FP8-specific: scale/quantize tensor math (portable) + streaming-weight-preparation glue (not portable, assumes `omnidreams`'s own weight layout). |

**Marginal estimate**: even excluding everything not FP8-specific in `ops.cu`, this is realistically not a
clean lift-and-shift. A GEMM kernel plus its capability probe plus its share of the bridge and Python glue
is on the order of **2,000–4,000 lines of code that would need adapting** (not counting the build
toolchain below) to get *one* FP8 linear layer type executing through a native kernel inside a real
`flashdreams` `CosmosDiTNetwork` forward pass — before any perf tuning to actually beat a
`torch._scaled_mm` baseline. This is a materially smaller number than "38k lines," but still an order of
magnitude more code, and more failure-prone code (custom CUTLASS templates, JIT native build), than the
`quantization_plan.md` route's estimated diff.

## Build toolchain

No `CMakeLists.txt`/`setup.py` in the extension tree. Build orchestration lives entirely in
`omnidreams/native/omnidreams_singleview.py`, JIT-compiling via `torch.utils.cpp_extension.load()`
(:551+). Third-party sources are cloned separately via `omnidreams_singleview/tools/sync_thirdparty.py
sync` (invoked with `--perf` per `README.md:146` and `omnidreams/prepare.py:157`) into
`omnidreams_singleview/3rdparty/`, driven by `thirdparty_sources.json`:

- `cutlass` — `github.com/NVIDIA/cutlass.git` @ pinned commit `f3fde58372d3...`, **with a repo-local patch**
  (`patches/cutlass/sm120-tma-pool.patch`) and header overlays (`patches/cutlass/include` →
  `include`). Stock upstream CUTLASS at that commit was apparently insufficient for SM120 TMA-pool
  support — omnidreams had to patch it themselves. This is a real risk signal: whatever SM121a needs may
  not exist upstream either, and may need its own patch nobody has written.
- `SageAttention` (`thu-ml/SageAttention.git`), `SpargeAttn` (`thu-ml/SpargeAttn.git`), `cudnn-frontend`
  (`NVIDIA/cudnn-frontend.git`) — also pinned commits.
- **Open discrepancy, unresolved**: `sage3_fp4_quant_shim.cu` includes from a
  `sageattention3_blackwell/sageattn3/...` path, but `thirdparty_sources.json` only lists a directory
  named `SageAttention` (from `thu-ml/SageAttention.git`), not `sageattention3_blackwell`. Either the
  manifest is stale, a differently-named/differently-sourced tree is expected at that path by some other
  mechanism not found in this pass, or the FP4 shim is currently dead/unbuildable from a clean
  `sync_thirdparty.py sync`. Not resolved here — would block confidently scoping the FP4 native path
  without further digging.
- No CI build-time record found in-tree (no log files, timing comments, or CI config referencing this
  extension's build duration was found in this pass) — build time from clean is unverified.

## Compute-capability risk (concrete, not speculative)

`sage_attention_plan.md` called sm_121a "unverified territory" in general terms. Now concrete:

- The dev box (per the `torch._scaled_mm` spike run in this same conversation) is `NVIDIA GB10`, compute
  capability `(12, 1)` — **sm_121a**.
- `omnidreams_singleview.py:58`'s default JIT arch flag is `_DEFAULT_CUDA_ARCH_LIST = "12.0a"` —
  **sm_120a**, a different `a`-suffixed (architecture-specific) target. `a`-suffixed CUTLASS kernels use
  arch-specific instructions (TMA, warpgroup MMA) that are generally *not* guaranteed forward-compatible
  across different `a` variants within the same major generation — unlike the `torch._scaled_mm` path
  already empirically confirmed working on this exact chip in `quantization_plan.md`'s Phase 0, nothing
  here has been run on sm_121a. Whether `12.0a`-targeted kernels even load/execute correctly on sm_121a
  hardware, or need a rebuild with a `12.1a` target string CUTLASS may or may not support at the pinned
  commit, is unverified and should be the first thing checked before writing any more of this port.
- `cosmos_fp8_two_gemm.cu`'s GEMM is hardcoded to `cutlass::arch::Sm89` (Ada) — this specific kernel
  would need re-templating for Blackwell entirely, it's not a recompile.
- `ops.cu`/`cosmos_fp8_tc_probe.cu` do target `cutlass::arch::Sm120`, and the codebase's own runtime probe
  pattern (`cosmos_fp8_tc_probe.cu`) suggests the omnidreams team already treats SM120 tensor-core
  availability as something to check, not assume — a signal worth taking seriously for sm_121a too.

## CUTLASS/Blackwell support, independent of omnidreams' fork

Not independently verified in this pass (would need a dedicated check, not attempted here to stay in
scope): whether current upstream CUTLASS (newer than the pinned `f3fde58...` commit) has since gained
native SM120/SM121 FP8 GEMM support that would make omnidreams' patched fork unnecessary, and whether a
from-scratch kernel against stock upstream CUTLASS could be a smaller lift than adapting the vendored,
patched, Sm89/Sm120-mixed kernel set above. Flagging as a real possibility worth checking before
committing to either route, not as a finding.

## Empirical validation on the dev box (2026-09-01) — supersedes several guesses above

The spike this doc called for was actually run. Results, in order:

**1. The build is NOT the blocker.** `sync_thirdparty.py sync` completed cleanly (CUTLASS @ `f3fde58` +
the local `sm120-tma-pool.patch`, SageAttention, SpargeAttn, cudnn-frontend). The full native extension
then JIT-built in **164s** with `OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST=12.1a` (overriding the shipped
`12.0a` default), all 28 units, `.so` loaded, `is_available()` → `True`,
`build_info()["cuda_arch_list"] == "12.1a"`. omnidreams' own `ci_gpu` test
`test_omnidreams_singleview_native.py::test_cuda_native_extension_builds` **passes** on this box
(165s), exercising real GPU ops (`zero_workspace_`, `prepare_contiguous`, descriptors) through the
compiled kernels.

**2. The FP4 third-party discrepancy flagged above is RESOLVED — not a real issue.**
`sageattention3_blackwell/` is a *subdirectory of the `SageAttention` repo itself*, present after a
clean sync; `3rdparty/SageAttention/sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu`
exists. The manifest was never stale.

**3. `sage3` is excluded from this GPU by omnidreams' own design.**
`sage3_is_runtime_supported()` (`dit_streaming/kernels/sage3_attention.cu:618-635`) requires
`prop.major==12 && prop.minor==0` **exactly**, then further name-allowlists
(`"GeForce RTX 5090"`, `"RTX PRO 6000"`, `"RTX 6000"`). GB10 reports `(12,1)` → returns `False`. The
in-source comment says this allowlist is deliberate, limited to "SM120a devices we have explicitly
built and validated." So `sage3`/`sage3_fp8` is out on this hardware regardless of anything else.

**4. The real blocker: the DiT block kernel does not run on `sm_121a`.** Driving the genuine path
(`CosmosTransformer.predict_flow()` → `OptimizedDiTExecutor` → `native_extension.optimized_dit_forward()`,
unmocked, tiny synthetic random-weight network) fails in `cosmos_run_transformer_block_streaming`
at block 0 (`streaming_dit_bridge.cu:3011`):

| config (all `num_blocks=2`) | `bf16` | `fp8_kvcache_cudnn` |
|---|---|---|
| `C=512 h=4 16x16 t=4` | `unknown error` | `operation not supported` |
| `C=2048 h=16 16x16 t=4` (prod width) | `unknown error` | `operation not supported` |
| `C=2048 h=16 64x64 t=8` (prod width + bigger grid) | `unknown error` | `operation not supported` |

Identical across all three sizes → **not** a synthetic-shape/tile-size problem. `compute-sanitizer
--tool memcheck`: **0 errors** → not a memory-safety bug. Note `bf16` fails too, so this is not
FP8-specific.

**5. Root cause.** omnidreams' kernels instantiate `cutlass::arch::Sm120` templates — 48 occurrences
confined to exactly **two files** (`dit_streaming/kernels/ops.cu` = 40,
`dit_streaming/kernels/cosmos_fp8_tc_probe.cu` = 8; named configs:
`Sm120Fp8RcrColscaleBf16StageConfig` ×7, `Sm120Fp8RcrColscaleGeluFp8StageConfig` ×4,
`…GeluFp8NoSrcStageConfig` ×2, `…NoSrcAutoConfig` ×2, `Sm120Fp8ProbeGemm` ×2,
`Sm120PerColScaleEltActNoSrcCallbacks` ×1). Per CUTLASS `include/cutlass/arch/config.h:152-172`,
`CUTLASS_ARCH_MMA_SM120_ENABLED` requires `__CUDA_ARCH__ == 1200`. Building for `12.1a`
(`__CUDA_ARCH__ == 1210`) therefore compiles every one of those kernels out to a stub → the runtime
errors above. And `grep -rn "Sm121" src/` over omnidreams' sources returns **zero hits** — there is no
SM121 path to fall back to.

**6. CUTLASS at the pinned commit cannot supply the missing piece.** `grep -rl "Sm121"` over
`3rdparty/cutlass/include/` returns **nothing**: `arch/arch.h:108` defines `struct Sm120` (and Sm90/
Sm100/Sm101/Sm103) but **no `struct Sm121`**, and there are no SM121 kernel/collective headers. Only
`config.h`'s SM121 *macros* exist (they arrived with CUDA-version gating, not with kernels). So
"just add `Sm121` instantiations alongside the `Sm120` ones" is **not** available at `f3fde58`.

This turns the earlier vague "12.0a vs 12.1a" note into a specific, load-bearing catch-22 for GB10:

- build `12.1a` → SASS matches the GPU, but every `arch::Sm120` kernel is compiled out → stubs → the
  failures in the table above (**measured**)
- build `12.0a` → kernels are enabled, but the SASS/PTX is arch-conditional `sm_120a`, which is not
  forward-compatible to `sm_121a` hardware (**measured — confirmed**)

**7. Both sides of the catch-22 are now measured.** Rebuilding the whole extension with
`OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST=12.0a` (omnidreams' shipped default) and rerunning the same
matrix:

| config (all `num_blocks=2`) | `bf16` @ 12.0a | `fp8_kvcache_cudnn` @ 12.0a |
|---|---|---|
| `C=512 h=4 16x16 t=4` | `AdaLN-LoRA global down GEMM failed: unknown error` | `CUDA error: no kernel image is available for execution on the device` |
| `C=2048 h=16 16x16 t=4` | same | same |
| `C=2048 h=16 64x64 t=8` | same | same |

`no kernel image is available for execution on the device` is the textbook signature of SASS built for
an incompatible architecture — confirming `sm_120a` binaries do not load on this `sm_121a` GPU. Note
the `bf16` failure also *moved* (from `cosmos_run_transformer_block_streaming` to the earlier
`AdaLN-LoRA global down GEMM`), consistent with different kernels being enabled/disabled per arch
rather than a single stable bug.

**Conclusion: omnidreams' native DiT path does not run on GB10 / `sm_121a` in either build
configuration.** This is not a configuration or harness problem — it is an architecture-support gap in
the vendored kernel set. No FP8-vs-BF16 speedup number could be obtained on this hardware, because the
native path does not execute here at all. (This says nothing about omnidreams' speedup on the
SM120a hardware it *was* built and validated for — RTX 5090 / RTX PRO 6000 per the sage3 allowlist.)

**8. The `f` (family) arch target was tried and is not sufficient either.** CUTLASS's `config.h`
exposes `SM120F`/`SM121F` *family* macros gated on `CUDA_ARCH_FAMILY(...)`, suggesting a family target
as the cross-SM120/121 mechanism. Verified at the compiler and hardware level with a standalone CUDA
program on this box:

| `nvcc -gencode` target | compiles | runs on this sm_121 GPU |
|---|---|---|
| `arch=compute_120f,code=sm_120f` | yes | **yes** (kernel ran, correct result) |
| `arch=compute_120a,code=sm_120a` | yes | **no** — launch returns "no error" but the kernel *silently does not execute* (output stayed 0) |
| `arch=compute_121a,code=sm_121a` | yes | **yes** |

The `sm_120a` silent-no-op is worth flagging on its own: it means a wrong-arch build can fail without
raising, which is consistent with the confusing mix of `unknown error` / stale-looking behaviour seen
while diagnosing this.

PyTorch's `TORCH_CUDA_ARCH_LIST` parser rejects the `f` suffix ("Unknown CUDA arch (12.0f)"), so a
family target can't be requested through the normal env var. Worked around in the
`dev/integrate-cmd-fp8-native` worktree with ~25 lines in `omnidreams/native/omnidreams_singleview.py`
(`_is_family_arch`/`_family_gencode_flags`): emit `-gencode=arch=compute_120f,code=sm_120f` directly
into `extra_cuda_cflags`, which also makes `cpp_extension` skip its own arch-flag generation (it bails
as soon as any cuda cflag contains `arch`). The extension then builds and *loads* — the `no kernel
image` error is gone — but the DiT kernels still fail exactly as in the `12.1a` build
(`unknown error` / `operation not supported`, all three config sizes). So loading was never the only
gate.

**Why no build flag can fix this at the pinned CUTLASS.** `CUTLASS_ARCH_MMA_SM121A_ENABLED` /
`SM120A_ENABLED` appear *only* in `cutlass/arch/config.h`, `cute/arch/config.hpp`, `cutlass/float8.h`,
and `cutlass/float_subbyte.h` — config plumbing and FP8/FP4 datatype enablement. There are **no SM121
GEMM kernel or collective headers** at `f3fde58`, and no `struct Sm121` arch tag (still absent in
upstream **v4.7.1**). omnidreams selects kernels through the `cutlass::arch::Sm120` tag
(`kMinComputeCapability = 120`), and that selection is what has no SM121-capable counterpart to switch
to.

**Bottom line: all three reachable arch targets fail on GB10, for two different reasons, and none is
fixable by build configuration.** Making the native path run on `sm_121a` requires source-level work —
a CUTLASS version that genuinely ships SM121 kernels (then reworking the local `sm120-tma-pool.patch`
and re-templating the 48 `Sm120` sites across those 2 files), or hand-written SM121 paths. That is a
kernel-engineering project, not an integration task.

## What would need to happen before writing any implementation plan here

This section deliberately does not use "Phase N" numbering — that would imply a commitment this doc isn't
making.

- Resolve the `sageattention3_blackwell` source discrepancy — is the FP4 shim buildable today at all.
- A hardware spike, mirroring `quantization_plan.md`'s Phase 0 but for this route: JIT-build
  `cosmos_fp8_tc_probe.cu`'s SM120 probe (or the smallest buildable subset) against `sm_121a` on the dev
  box and see whether it compiles/runs, before assuming any of the rest is worth adapting.
- If that spike fails, decide whether a from-scratch CUTLASS kernel against current upstream (skipping
  omnidreams' pinned/patched fork) is a better starting point than adapting the vendored one.
- Only then would a real phased plan with LOC/timeline estimates make sense.
