<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# omnidreams: multi-architecture GPU support (SM120 / SM121 / beyond)

Status: **design recommendation, ready for review**. Scope is `integrations/omnidreams/` only (the
`omnidreams_singleview` native extension); CMD is out of scope. This doc is a synthesis of four
parallel investigations run on a DGX Spark GB10 (compute capability 12.1) against the repo at
`/home/qiwu/Work/flashdreams-cmd/main`. Everything labelled **[measured]** was executed on that box;
everything labelled **[assumed]** or **[unverified]** was not. No code changes are in the tree as a
result of this investigation — both checkouts were left clean.

Note: `integrations/omnidreams/docs/` did not exist before this document; it is created by adding
this file. Existing sibling design docs live under `integrations/cmd/docs/` and set the house style
(see `integrations/cmd/docs/quantization_native_port_scoping.md`).

Environment for all measurements: NVIDIA GB10, CC (12,1), driver 580.173.02, CUDA 13.0.88 (nvcc),
`torch 2.12.1+cu130` (aarch64), cuDNN 9.20.0 (`torch.backends.cudnn.version() == 92000`), 20 CPU
cores, 48 SMs, 100 KB smem/SM.

---

## TL;DR

The recommendation is a **fat binary of arch-conditional targets plus device-derived defaults**:

```python
# integrations/omnidreams/omnidreams/native/omnidreams_singleview.py:58
_DEFAULT_CUDA_ARCH_LIST = "12.0a;12.1a"   # release default; dev builds derive from the live device
```

That is the whole of the *architecture* change, it needs **zero** patches to PyTorch or CUTLASS, and
it is measured working. But arch targeting is only one of three independent layers that keep
omnidreams off GB10, and it is not the most expensive one. The other two are a **repo-local
cudnn-frontend API bug that is not arch-specific at all** (and almost certainly breaks the bf16
backend on RTX 5090 too), and **hard-coded `minor == 0` / device-name allowlists** in host dispatch.
One genuine external blocker remains — cuDNN 9.20 ships no FP8 fused-MHA engine for sm_121 — and the
professional answer there is an explicit capability gate, a clear error message, a fallback to the
CUTLASS FP8 attention path omnidreams already owns, and a tracked upstream request.

---

## 1. Why it fails today

Three independent mechanisms, in the order a request hits them. None of them is "CUTLASS does not
support sm_121".

### 1.1 The shipped default emits an arch-*conditional* cubin that a cc-12.1 device refuses to load

`omnidreams/native/omnidreams_singleview.py:58` sets `_DEFAULT_CUDA_ARCH_LIST = "12.0a"`, threaded
into `torch.utils.cpp_extension.load` via `TORCH_CUDA_ARCH_LIST` (resolution at
`omnidreams_singleview.py:467-522`). That produces `-gencode=arch=compute_120a,code=sm_120a` and
nothing else.

An `a`-suffixed cubin has **no forward compatibility whatsoever** — not to another minor version, not
via PTX JIT (see §2). On this GB10 the CUDA runtime rejects it at module/function resolution, before
any launch:

**[measured]** (`/home/qiwu/.claude/jobs/0b3caf82/tmp/repro/funcattr.cu`, a one-line kernel, no CUTLASS):

```
--- 120a ---  cudaFuncGetAttributes -> 209 (no kernel image is available for execution on the device)
              cudaFuncSetAttribute  -> 209
              launch -> 209 ; cudaDeviceSynchronize -> "no error" ; result = 0 (expected 42)
--- 120f ---  all 0 ; result = 42
--- 121a ---  all 0 ; result = 42
```

Note the hazard on line 3: **the launch reports 209 but the subsequent sync reports success**, so
code that only error-checks after a sync sees a silent no-op with garbage (zero) output. This is what
made an earlier standalone experiment look like a mysterious "silent failure", and it is the reason
"do nothing and document it" is not a safe end state (§5, option E).

This mechanism explains both 12.0a symptoms exactly:

- **fp8 path → `"no kernel image is available for execution on the device"`**: raw 209 surfaced
  through torch.
- **bf16 path → `"AdaLN-LoRA global down GEMM failed: unknown error"`**: `cutlass_linear_layer_rrr_bf16`
  in `omnidreams_singleview/src/dit_streaming/kernels/cosmos_gemm_bf16.cu` is a CUTLASS **2.x
  `arch::Sm80`** GEMM whose 128×128×32 tile fits under 48 KB smem, so `initialize()` never calls
  `cudaFuncSetAttribute` and succeeds; the launch then fails with 209 and the function's terminal
  `return (status == kSuccess) ? cudaSuccess : cudaErrorUnknown;` (`cosmos_gemm_bf16.cu:210`)
  **discards the real error code**. "unknown error" is omnidreams' own lossy mapping, not CUDA 999.
  The CUTLASS 3.x equivalent fails one step earlier: `GemmUniversalAdapter::initialize()` calls
  `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 80896)`, gets 209, returns
  `Status::kErrorInternal` — which is precisely the `kErrorInternal (7)` the standalone reproducer
  printed under 120a.

### 1.2 The CUTLASS SM120 kernels are **not** the blocker — they run correctly on GB10

This is the single most important correction to the prior mental model.

**[measured]** `/home/qiwu/.claude/jobs/0b3caf82/tmp/repro/probe_repro.cu` lifts the exact
`cutlass::arch::Sm120` blockwise-scaled FP8 GEMM template stack out of
`omnidreams_singleview/src/dit_streaming/kernels/cosmos_fp8_tc_probe.cu`
(`CollectiveBuilder<Sm120, OpClassTensorOp, e4m3, ...>` + `GemmUniversalAdapter`) and compiles it
against the **vendored, patched** CUTLASS in `omnidreams_singleview/3rdparty/cutlass/include`:

| build | `can_implement` | `initialize` | `run` | numerics |
|---|---|---|---|---|
| `compute_120a/sm_120a` | Success | **kErrorInternal (7)** | — | — |
| `compute_120f/sm_120f` | Success | Success | Success | **correct** (bf16 256.0, 65536/65536 written) |
| `compute_121a/sm_121a` | Success | Success | Success | **correct** |

Independently, `/home/qiwu/.claude/jobs/0b3caf82/tmp/sm12x_gemm.cu` extracted the production
`sm120_fp8_rcr_colscale_bf16` config verbatim from `kernels/ops.cu:774` and reproduced the same
result, plus `sm_121f` correct and a `sm_120a + sm_121a` fat binary correct.

Supporting evidence that nothing is being stubbed out:

- `cuobjdump -sass` on all three standalone binaries shows an **identical 384 `QMMA.16832.F32.E4M3.E4M3`**
  instructions and identical `UTMALDG`/`UTMASTG`/`UTMACMDFLUSH` counts. The real extension built for
  121a (44 MB, 23 cubins, all `arch = sm_121a`) contains **4608 QMMA** — exactly the same count as the
  120a-built `.so`, which is only tagged differently.
- An in-kernel macro probe **[measured]**: under 120f, `__CUDA_ARCH__ == 1200`,
  `CUTE_ARCH_MMA_SM120_ENABLED = 1`, `CUTE_ARCH_TMA_SM120_ENABLED = 1`; under 121a,
  `__CUDA_ARCH__ == 1210`, `CUTLASS_ARCH_MMA_SM121{,A,F}_ENABLED = 1`, same two CuTe atoms enabled.
- `cutlass::arch::Sm120::kMinComputeCapability = 120` (`3rdparty/cutlass/include/cutlass/arch/arch.h:108-110`)
  is used **only** in `if constexpr` compile-time branches inside
  `include/cutlass/gemm/device/gemm_universal_adapter.h` — it is never compared against a queried
  device CC. `can_implement()` returned `kSuccess` on this cc-12.1 device in every build.
- omnidreams' own `#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)`
  guards (`cosmos_fp8_tc_probe.cu:23`, `:322`, `:368`) are **compiler-version** gates
  (`__CUDACC_VER >= 12.8/12.9`), always true under nvcc 13.0.88, so their `cudaErrorNotSupported`
  arms never fire. Likewise `cosmos_fp8_flash_tc.cu:282/362/449/535` are shape gates (`Mq/Mk % 128`),
  not arch gates.

**Conclusion: the 26 real `cutlass::arch::Sm120` arch-tag uses in the tree are already correct for
GB10.** They need no change. (The "48 sites" figure over-counts: it mixes the arch tag with
identifiers merely *named* `Sm120…`. The real count is 24 in `kernels/ops.cu` + 2 in
`kernels/cosmos_fp8_tc_probe.cu`, which are 13 kernel configs × 2 builder tags each.)

### 1.3 With a correct arch flag, the remaining failures are cuDNN, in two unrelated places

Instrumenting every non-success early return in the DiT kernels and rebuilding for 12.1a
(instrumentation reverted; both checkouts clean) gives this trace **[measured]**, identical across
three network sizes:

```
bf16 : attention.cu:1222 -> 999 (unknown error)          -> cosmos_block.cu:2638 -> bridge:3011
fp8  : attention.cu:1346 -> 801 (operation not supported) -> cosmos_block.cu:1369 -> :2638 -> bridge:3011
```

**(a) bf16 — a repo bug, arch-independent.** `CUDNN_FRONTEND_LOG_INFO=1` gives the real cause:

```
[cudnn_frontend] ERROR: generate_stats attribute not set. ["ATTRIBUTE_NOT_SET"]
  at cudnn-frontend/include/cudnn_frontend/node/scaled_dot_product_flash_attention.h:194
```

`attention.cu:1161-1164` builds the packed bf16 SDPA graph with `.set_name` / `.set_attn_scale` /
`.set_causal_mask` and **no `.set_generate_stats(...)`**. cudnn-frontend made that mandatory in
**v1.13.0**; `thirdparty_sources.json` pins **v1.14 (`deda80e`)**. `graph->build()` therefore fails
in `validate()` — before any heuristics or device query — for all heuristic modes, and
`run_cudnn_fmha_packed_qkv` falls through to `return cudaErrorUnknown;` at `attention.cu:1222`. The
FP8 sibling at `attention.cu:1313` *does* call `.set_generate_stats(false)`, which is why only bf16
hit this.

**[measured] proof this is the entire bf16 story:** adding `.set_generate_stats(false)` to the four
bf16 SDPA sites (`attention.cu:1161`, `:1527`, `:1767`, `:2057`) and rebuilding made the real
`optimized_dit_forward` bf16 path pass end-to-end on GB10 at three network sizes:

```
=== arch=12.1a  cap=(12, 1) ===
  [OK] bf16 C=512  h=4  b=2 16x16 t=4  out=(1,1,256,64)
  [OK] bf16 C=2048 h=16 b=2 16x16 t=4  out=(1,1,256,64)
  [OK] bf16 C=2048 h=16 b=2 64x64 t=8  out=(1,1,8192,64)
```

**[assumed, needs SM120a hardware to confirm]** Because the failure is in graph *validation*, this
bug should reproduce identically on RTX 5090 / RTX PRO 6000 with the current cudnn-frontend pin. If
so, "omnidreams' bf16 backend works on SM120a" does not currently hold either, and this is a
release-blocking bug independent of GB10.

**(b) fp8 — a genuine external cuDNN gap on sm_121.** `CUDNN_LOGLEVEL_DBG=3`
(`/home/qiwu/.claude/jobs/0b3caf82/tmp/cudnn_full.log:12610-12613`):

```
engineConfig: {"engineId":1,"smVersion":1210,"knobChoices":{"CUDNN_KNOB_TYPE_KERNEL_CFG":3}}
Warning: CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH; Reason: MHA only supports FP16/BF16 I/O for
  selected engine config at: ARCH_8X == engine_identity.arch && (aType == CUDNN_DTYPE_FP8_E4M3 || ...)
[cudnn_frontend] ERROR: No valid engine configs returned from heuristics.
```

cuDNN 9.20 **recognises** `smVersion: 1210` but offers only an `ARCH_8X`-identity engine, whose
support check explicitly rejects FP8 E4M3/E5M2 I/O. i.e. **there is no Blackwell-class fused FP8 MHA
engine for sm_121 in this cuDNN**. That is closed-source and outside the repo — no arch flag or
CUTLASS change fixes it. `attention.cu:1346` correctly maps it to `cudaErrorNotSupported (801)`;
what is *missing* is a fallback: omnidreams already owns a working CUTLASS FP8 attention path
(`kernels/cosmos_fp8_flash_tc.cu`, `kernels/cosmos_fp8_tc_probe.cu` — proven to run on this GPU in
§1.2), but the `fp8_kvcache_cudnn` backend never falls back to it.

### 1.4 Two host-side allowlists that would silently exclude GB10 anyway

Even with correct SASS and working cuDNN, these hard-code CC 12.**0**:

- `omnidreams_singleview/src/dit_streaming/streaming_dit_bindings.cpp:61-62` (inside the
  `sparge_is_runtime_supported` lambda defined at `:45`):
  `return (prop.major == 8 && prop.minor == 9) || (prop.major == 12 && prop.minor == 0);`
- `omnidreams_singleview/src/dit_streaming/kernels/sage3_attention.cu:626`:
  `if (prop.major != 12 || prop.minor != 0) return false;` — followed at `:631-637` by a **device-name
  string allowlist** (`"GeForce RTX 5090"`, `"RTX PRO 6000"`, `"RTX 6000"`).

These did not fire in the traced runs (those backends were not selected), so they are not part of the
current failure, but they are the exact antipattern that will exclude every future SKU. The
`sage3_attention.cu` comment is honest about why it exists (arch-conditional FP4 MMA compiled only
for `sm_120a`) — which makes it a *consequence* of the single-arch build policy, and it becomes
relaxable once the build emits SASS for the device in question (§4, Phase 3).

---

## 2. What the correct targeting model is

### 2.1 nvcc's three target flavors

| target | feature set | cubin runs on | PTX JITs to |
|---|---|---|---|
| `compute_120`/`sm_120` (base) | baseline | CC 12.x, minor ≥ 0 | any later CC |
| `compute_120f`/`sm_120f` (**family**-conditional, CUDA ≥ 12.9) | baseline ⊂ family | **all of the 12.x family: CC 12.0 and 12.1** | same family, minor ≥ |
| `compute_120a`/`sm_120a` (**arch**-conditional) | family ⊂ arch | **CC 12.0 only** | **nothing — no forward compat at all** |

From the CUDA Programming Guide, [Compute Capabilities appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html):
*"The family-specific feature set is a superset of the baseline feature set. The architecture-specific
feature set is a superset of the family-specific feature set."* From the NVIDIA blog
[Blackwell and CUDA 12.9 Introduce Family-Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/):
with `a`, *"the code only runs on GPUs of that specific CC and no others… no forward-compatibility for
either PTX or a cubin."* Programming Guide Table 28 gives the families: `compute_120f` → {12.0, 12.1};
`compute_121f` → {12.1}; `compute_100f` → {10.0, 10.3}.

`f` exists precisely to solve this problem — it was added in CUDA 12.9 because `a` targets became
brittle once one architecture generation forked into multiple minor CCs.

The Blackwell lineup, for orientation: `sm_100` B100/B200, `sm_103` B300, `sm_110` DRIVE/Jetson Thor
(renamed from `sm_101` in CUDA 13.0), `sm_120` GeForce RTX 50-series + RTX PRO 6000, **`sm_121` GB10 /
DGX Spark**, `sm_107` added in CUDA 13.4. The 10.x datacenter family (tcgen05/TMEM) and the 12.x
consumer/SoC family (warp-level `mma.sync.kind::f8f6f4`, no tcgen05) are **disjoint** for `f`
purposes; sm_120 and sm_121 are in the *same* family.

### 2.2 The FP8 instructions omnidreams needs are family-conditional, not arch-conditional

**[measured]** ptxas acceptance of the exact FP8 MMA CUTLASS's SM120 mainloop emits
(`mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`, `cute/arch/mma_sm120.hpp:68`):

```
sm_120a OK   sm_120f OK   sm_121a OK   sm_121f OK
sm_120  ERROR: Feature '.kind::f8f6f4' not supported on .target 'sm_120'
sm_121  ERROR: Feature '.kind::f8f6f4' not supported on .target 'sm_121'
```

Same for `stmatrix…b8` and NVFP4 `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X`. So: base targets
are unusable (FP8 tensor cores unreachable), and both `a` and `f` suffice.

**[measured]** PTX-JIT behaviour, which settles the "ship PTX as a safety net" question:

```
compute_120a PTX-only -> "no kernel image is available"   <-- 'a' PTX does NOT JIT across the family
compute_120f PTX-only -> correct
compute_121a PTX-only -> correct
```

**Arch-conditional PTX has no forward compatibility.** Shipping `compute_120a+PTX` buys exactly
nothing on GB10.

### 2.3 CUTLASS's model: `arch::Sm120` **is** the SM121 tag, by design

The absence of `struct Sm121` is intentional, not a gap:

- Vendored `3rdparty/cutlass/CHANGELOG.md:48-49` (CUTLASS 4.2.0), verbatim:
  *"Support for Blackwell SM121 kernels for DGX Spark GPUs. — Share the major codes with Blackwell
  SM120 kernels."*
- Upstream `main` (checked 2026-08-28, commit `dc45f979`) still has **zero** occurrences of `Sm121` in
  `include/`, while shipping SM121 support. This is stable upstream policy, not a lag.
- NVIDIA's canonical example `examples/79_blackwell_geforce_gemm/79a_*.cu` uses
  `using ArchTag = cutlass::arch::Sm120;` for *both* 12.0 and 12.1, and guards on the host with a CC
  **range** check `props.major == 12 && (props.minor == 0 || props.minor == 1)`.
- Differentiation happens in the preprocessor, not the type system. Vendored
  `3rdparty/cutlass/include/cute/arch/config.hpp:153-156`:
  ```c
  #if (defined(CUTLASS_ARCH_MMA_SM120_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) ||\
       defined(CUTLASS_ARCH_MMA_SM121_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))
  #  define CUTE_ARCH_MMA_SM120_ENABLED
  #  define CUTE_ARCH_TMA_SM120_ENABLED
  #endif
  ```
  with `CUTE_ARCH_F8F6F4_MMA_ENABLED` supplied at `:159-172` from either the SM120 or the SM121 arm.
- `3rdparty/cutlass/CMakeLists.txt:178` lists `100 100a 120 120a 121 121a` as supported arches and
  `:187` adds `100f 120f 121f 103a 103f` on CUDA ≥ 12.9. `121a` is a first-class CUTLASS build target.

**Adding a `struct Sm121` would be actively wrong**: it would fall off the `arch::Sm120` builder
specializations in `include/cutlass/gemm/collective/builders/sm120_*.inl` and instantiate nothing.

**The pinned CUTLASS 4.2.1 (`f3fde58`) already fully supports GB10.** No CUTLASS bump, no new tag, no
source change to the 26 arch-tag sites is required. The only thing missing is the gencode flag.

### 2.4 A correction worth recording, because it is an easy misread

A static reading of `cute/arch/config.hpp:153` suggests `120f`/`121f` builds get *empty* SM120
MMA/TMA bodies, since `SM120F_ENABLED`/`SM121F_ENABLED` are absent from that `#if`. **That reading is
wrong, and it was checked directly.** `include/cutlass/arch/config.h:169-170` defines
`CUTLASS_ARCH_MMA_SM120F_ENABLED` *nested inside* the `#if ... __CUDA_ARCH__ == 1200` block opened at
`:157` — and a family-conditional target still reports `__CUDA_ARCH__ == 1200`. So a `120f` build
defines `CUTLASS_ARCH_MMA_SM120_ENABLED` too, satisfies `config.hpp:153`, and lights up the identical
CuTe atoms. This is confirmed by both the in-kernel macro probe and two independent
numerically-correct `120f` GEMM runs (§1.2). `f` targets are **not** broken for this code.

---

## 3. What the ecosystem does

**[measured] PyTorch's own cu13 wheels are the closest structural analogue.**
`cuobjdump -lelf` on the installed `torch/lib/libtorch_cuda.so` (423 MB, aarch64) → 2423 cubins:

```
448 sm_80   448 sm_90   448 sm_100   448 sm_110   448 sm_120
 59 sm_100a  59 sm_103a  59 sm_110a   4 sm_90a
  1 sm_120a   1 sm_121a
```

`cuobjdump -lptx` → **zero PTX entries**. The single `sm_120a` and single `sm_121a` cubin come from
one translation unit — symbol dump identifies it as `aten/src/ATen/native/cuda/RowwiseScaledMM.cu`,
PyTorch's FP8 rowwise-scaled GEMM built on `MainloopSm120TmaWarpSpecialized`. **Structurally the exact
problem omnidreams has.** PyTorch's answer is not `f` and not a `Sm121` tag — it is *compile that one
file twice*, via `cmake/Codegen.cmake`'s `_BUILD_FOR_ADDITIONAL_ARCHS`:

```cmake
_BUILD_FOR_ADDITIONAL_ARCHS(".../RowwiseScaledMM.cu" "89;90a;100a;103f;110a;120a;121a")
```

Three properties worth copying: **per-source** escalation (not whole-project), **conditional on the
base arch already being requested** (no build-time explosion), and **CUDA-version gated** (`121a` only
on CUDA ≥ 12.9). Note they mix `a` and `f` in one list — they use whichever target the underlying
kernel's gating understands.

| project | policy |
|---|---|
| **PyTorch** | per-source extra `-gencode`; `120a` **and** `121a` for the CUTLASS FP8 GEMM; no PTX at all |
| **vLLM** | per-kernel-group arch sets; on CUDA ≥ 13.0 uses `12.0f`, falls back to `"12.0a;12.1a"` on 12.8. `cuda_archs_loose_intersection()` documents *"SRC='12.0f' matches TGT='12.1a' since SM121 is in the SM12x family."* PR #38126 "[NVIDIA] Fix DGX Spark logic" added `12.1f`/`12.1a` to scaled_mm / NVFP4 / MLA / MoE lists, *"fixing previously silent compilation skips"* |
| **NVIDIA Transformer Engine** | cleanest split: `NVTE_GENERIC_ARCHS` (`120`) for ordinary sources, `NVTE_SPECIFIC_ARCHS` (`120f` on CUDA ≥ 12.9, else `120a`) for arch-specific ones. Never enumerates `121` — relies on `120f` covering the family |
| **TensorRT-LLM** | `-real` always, PTX explicitly rejected; family (`-f`) targets for SM 100+ on CUDA ≥ 12.9 |
| **FlashInfer** | JIT, arch derived from the live device. `compilation_context.py` names DGX Spark: *"SM 12.x → 'f' suffix with minor version preserved (e.g. compute_120f for SM120, **compute_121a for SM121**). Each SM 12.x variant gets its own cubin to avoid running SM120 code on SM121 (DGX Spark) which can cause cudaErrorIllegalInstruction."* It even greps the vendored CUTLASS `arch.h` at runtime (`cutlass_supports_sm107()`) to decide what its dependency can express |
| **FlashAttention** | `arch=compute_120f,code=sm_120` — family *features*, cubin emitted under the plain `sm_120` name so the loader matches any SM12x; plus a PTX-only tail for the newest arch |
| **SGLang** | plain enumeration of both: `compute_120a,code=sm_120a` **and** `compute_121a,code=sm_121a` |
| **DeepGEMM** | 100 % runtime JIT; no shipped cubins at all |

The most useful ecosystem document on this exact problem is
[flashinfer#3170 "DGX Spark (SM121) Current Support Audit"](https://github.com/flashinfer-ai/flashinfer/issues/3170).
Its headline finding matches ours: the arch flags were mostly fine; the real breakage was
**hard-coded minor-version checks in dispatch** (`is_sm120 = major == 12 and minor == 0`, and two
more of the same shape) — i.e. omnidreams' `streaming_dit_bindings.cpp:61-62` and
`sage3_attention.cu:626` are a known, named, industry-wide bug class. Also
[flashinfer#3294](https://github.com/flashinfer-ai/flashinfer/issues/3294): `flashinfer-cubin 0.6.11`
ships 12 681 cubins, **none** for sm_120/sm_121 — the failure mode of per-arch prebuilt wheels.

Two consistent lessons: (1) nobody solves this by forking the kernel templates per CC; (2) everybody
who got burned got burned by **host-side dispatch**, not by the device code.

---

## 4. Recommendation

**Adopt the PyTorch/SGLang pattern: one arch list of arch-conditional (`a`) targets, device-derived
for dev builds and explicitly enumerated for releases, with the `cutlass::arch::Sm120` templates left
untouched — and treat the two non-arch layers (cuDNN, capability gates) as first-class parts of the
same workstream.**

Why `a`-pair over `f`, given both are measured working:

1. **Zero out-of-tree patches.** `torch/utils/cpp_extension.py:2572-2575` already lists
   `'12.0', '12.0a', '12.1', '12.1a'`. **[measured]** `TORCH_CUDA_ARCH_LIST="12.0a;12.1a"` yields
   `-gencode=arch=compute_120a,code=sm_120a -gencode=arch=compute_121a,code=sm_121a`, while
   `"12.0f"` raises `ValueError: Unknown CUDA arch (12.0f)`. Using `f` means carrying a ~25-line
   monkeypatch against a PyTorch internal (`_get_cuda_arch_flags` returns `[]` as soon as any cuda
   cflag contains the substring `arch`) forever. That is a maintenance liability in a library whose
   whole build path is `torch.utils.cpp_extension.load`.
2. **Arch-conditional is a strict superset of family-conditional.** `a` cannot lose a feature that
   `f` has; the reverse is possible. omnidreams reaches for FP4/NVFP4 MMA in `sage3_attention.cu`,
   and the env-gated block-scaled probe (`cosmos_fp8_tc_probe.cu`) is untested under `f`.
3. **Build-time failure, not silent degradation.** A missing `a` target for a new SKU shows up as a
   loader error the CI SASS check (Phase 0) catches. A misjudged `f` assumption shows up as a
   ptxas error at build — also fine — but a *mis-scoped* one shows up as a slow path at runtime.
4. It is what PyTorch itself does for the structurally identical `RowwiseScaledMM.cu`, and what
   FlashInfer does specifically for DGX Spark.

The honest cost of `a` is that **each new CC needs a rebuild** — which is exactly the thing `f` fixes.
That is the fallback, and it is a good one: **when PyTorch's `_get_cuda_arch_flags` accepts the `f`
suffix upstream, switch the SM12x entry to a single `12.0f`** and the arch list stops needing
maintenance for future 12.x parts. File that upstream issue now (Phase 5); it is a ~10-line change to
`supported_arches` plus a suffix-aware normalizer, and vLLM/TE/TRT-LLM all want it.

Note that omnidreams builds JIT-at-first-use on the target machine, so the fat binary is *not*
mandatory for correctness on a single box — a device-derived single target is strictly better there
(half the build time). The `"12.0a;12.1a"` list matters for release/AOT and for multi-GPU-SKU hosts.

### Phased plan

**Phase 0 — make wrong-arch builds impossible to ship silently. (GB10-validatable.)**
The 120a silent-no-op in §1.1 is the real hazard: zeros with `cudaDeviceSynchronize()` reporting
success. Add a post-build assertion in `omnidreams/native/omnidreams_singleview.py` that every arch in
the resolved list has SASS in the produced `.so` (`cuobjdump -lelf | grep -c sm_121a`), and a
first-use check that the *live device's* CC has a matching image, failing loudly with the arch list in
the message. This is ~30 lines and it is what separates "wrong gencode" from "kernel logic broken" for
every future investigation. Do this first regardless of everything else.

**Phase 1 — arch list. (GB10-validatable; ~4 LOC.)**
- `omnidreams/native/omnidreams_singleview.py:58` → `_DEFAULT_CUDA_ARCH_LIST = "12.0a;12.1a"`.
- Prefer a **device-derived** default for local/JIT builds: normalize each visible
  `torch.cuda.get_device_capability()` — `(12,0)→"12.0a"`, `(12,1)→"12.1a"`, `(9,0)→"9.0a"`,
  `(10,x)→"10.xa"`, else `"{maj}.{min}"` — and fall back to the full release list when no device is
  visible (CI/container builds). Gate `12.1a` on CUDA ≥ 12.9, mirroring PyTorch's comment.
- Keep `OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST` as an override that respects an explicit suffix verbatim
  (this is FlashInfer's contract, and it is what lets someone try `12.0f` without a code change).
- Add a one-line comment near `kernels/ops.cu:669` recording *why* `cutlass::arch::Sm120` is the
  correct tag for sm_121, citing `3rdparty/cutlass/CHANGELOG.md:48`. This is the exact trap that
  produced this investigation.

**Phase 2 — fix the cudnn-frontend `generate_stats` bug. (GB10-validatable; needs SM120a regression.)**
Add `.set_generate_stats(false)` at `kernels/attention.cu:1161`, `:1527`, `:1767`, `:2057`. **[measured]**
this alone makes the full bf16 DiT forward pass on GB10. Treat it as a standalone bug fix with its own
regression test, not part of the arch work — it is very likely broken on SM120a today too (§1.3a).
While in there, stop discarding error codes: `cosmos_gemm_bf16.cu:210` and the `cudaErrorUnknown`
returns at `attention.cu:1222` should propagate the real `cudaError_t` (or at minimum log it). The
whole "unknown error at block 0" mystery was self-inflicted by that one line.

**Phase 3 — replace CC-equality gates with family + capability probes. (Needs both boxes.)**
- `streaming_dit_bindings.cpp:61-62`: `(12,0)` → `major == 12`.
- `sage3_attention.cu:626`: `minor != 0` → `major != 12`, and **delete the device-name allowlist** at
  `:631-637`. Its stated justification ("CUDA does not expose the architecture suffix… limited to
  SM120a devices we have explicitly built and validated") is dissolved by Phase 0: the build now knows
  which arches it emitted, so gate on *that*, not on a marketing string.
- Every relaxed gate must be a **logged downgrade** on failure, never a silent `false`.
- These change *which kernels run*, not whether they run, so they need numerical validation on both
  a 5090/PRO 6000 and a GB10 before being trusted.

**Phase 4 — make the cuDNN FP8 gap a graceful, loud degradation. (GB10-validatable.)**
cuDNN 9.20 has no FP8 fused-MHA engine for sm_121 (§1.3b), and that is not fixable in this repo. The
professional handling is three things, none of which is "support sm_120a only":
1. **Fall back** from `fp8_kvcache_cudnn` to the CUTLASS FP8 attention path
   (`kernels/cosmos_fp8_flash_tc.cu`) omnidreams already ships and which **[measured]** runs correctly
   on this GPU — instead of surfacing `801` from `attention.cu:1346` as a hard failure.
2. **Explicit capability report + actionable error** if no fallback is available: name the device CC,
   the cuDNN version, and the specific missing engine, not "operation not supported".
3. **Track it upstream** — file an NVBug / cuDNN request for Blackwell-class FP8 fused MHA on sm_121
   and link the ticket from the fallback's comment, so the fallback can be removed when it lands.

**Phase 5 — forward-looking hygiene. (Ongoing.)**
File the PyTorch issue for `f`-suffix support in `_get_cuda_arch_flags`. Revisit `120f` when it lands
— at that point the whole SM12x arch list collapses to one entry that covers unreleased 12.x parts.
Separately, retune tile/stage selection: the `ops.cu` heuristics (`prefer_sm120_fused_fp8_epilogue`,
`sm120_fp8_stage2`, the m128n32/n64/n128 ladder) were tuned for ~170-SM GB202 parts and will
mis-select on GB10's 48 SMs. That is a performance task, not correctness, and it should not gate
Phases 0-4.

### What can be validated where

| | GB10 dev box (this machine) | Needs SM120a hardware (RTX 5090 / RTX PRO 6000) |
|---|---|---|
| Phase 0 SASS assertion | **yes** — already exercised via `cuobjdump` | confirm the assertion does not false-positive on a 12.0a-only build |
| Phase 1 arch list | **yes** — fat 120a+121a binary built and run correctly | **required**: confirm 5090 still picks the `sm_120a` image and is bit-identical to today |
| Phase 2 `generate_stats` | **yes** — full bf16 DiT forward passes | **required**: confirm this is a fix, not a behaviour change, on SM120a (and whether bf16 is broken there today) |
| Phase 3 gate relaxation | can confirm the paths are *selected* | **required**: Sage3/SpargeAttn numerics on both CCs; these are approximate-attention kernels, "it ran" is not "it is correct" |
| Phase 4 fp8 fallback | **yes** — CUTLASS FP8 GEMM/attention verified on GB10 | **required**: confirm no perf regression where the cuDNN engine *does* exist |
| Phase 5 retune | **yes** — GB10 is the target being tuned for | **required**: no regression on the 170-SM part |

Estimated cost: Phases 0-2 are on the order of 100 LOC and a day, mostly test-writing. Phase 3-4 are
the real work and are dominated by cross-hardware validation, not by code. Build time roughly doubles
on the CUTLASS-heavy TUs for a 2-arch release build (**[measured]** 39 s for a 2-arch standalone;
expect ~164 s → ~300 s full extension), which is why dev builds should stay device-derived and
single-target.

---

## 5. Options considered and rejected

**A. Add `struct Sm121` to CUTLASS and parameterise the 26 arch-tag sites.** Rejected — it is the
wrong design, not merely expensive. `arch::Sm120` *is* the SM121 tag by explicit upstream policy
(`3rdparty/cutlass/CHANGELOG.md:48-49`; zero `Sm121` in upstream `main`'s `include/`). A new tag would
fall off the `sm120_*_mma_builder.inl` specializations and instantiate nothing, requiring 300-800
lines of new CUTLASS specializations carried across every bump — to reach a state the pinned CUTLASS
already reaches with a flag. **[measured]**: `arch::Sm120` + `-arch=sm_121a` produces correct
numerics on this GB10.

**B. Bump vendored CUTLASS (4.2.1 → 4.7.x/4.8dev).** Rejected *for this purpose*; keep on the normal
dependency-hygiene cadence. It buys **no** SM121 enablement — 4.2.1 already has it — and costs a patch
rebase (`omnidreams_singleview/patches/cutlass/sm120-tma-pool.patch` touches ~6 files across ~20
hunks, including invasive `Copy_Traits<SM90_TMA_*>` by-value→pointer surgery that will conflict if
upstream touched `copy_traits_sm90_tma.hpp`) plus revalidation of all 13 kernel configs. Worth doing
eventually for kernel coverage and to test whether that patch — which is an **nvcc-13 codegen
workaround, not an SM120 architecture dependency**, and whose `__CUDA_ARCH__ >= 1200` guard is already
SM121-correct — can be dropped. **[measured, weak]** a minimal repro ran correctly against *unpatched*
CUTLASS at 121a and 120f, but at M=N=K=256 with no graph capture; a descriptor-in-local-memory bug is
exactly what only appears under register pressure. Do not drop the patch without a full-model A/B.

**C. Family targets (`sm_120f`) now.** Rejected *for now*, retained as the stated fallback (§4). It is
technically sound — **[measured]** `120f` compiles and runs these exact kernels correctly on GB10,
`kind::f8f6f4` / `stmatrix.b8` / `kind::mxf4nvf4.block_scale` all assemble, and the static objection
that `cute/arch/config.hpp:153` omits `SM120F_ENABLED` is a misread (§2.4). It is also the genuinely
future-proof answer: one cubin covers 12.0, 12.1, and unreleased 12.x. But it requires carrying a
monkeypatch against `torch.utils.cpp_extension` internals indefinitely, and `f` is a strict subset of
`a`'s feature set, so a future kernel could quietly lose a path. Revisit when PyTorch accepts the
suffix.

**D. PTX-only forward-compat tail.** Rejected. **[measured]** `compute_120a` PTX will not JIT to
sm_121 — arch-conditional PTX has zero forward compatibility. Base `compute_120` PTX would JIT but
cannot express `kind::f8f6f4` at all, so it would silently give you a non-FP8 fallback or nothing.
The ecosystem agrees: torch cu13 ships **zero** PTX (**[measured]** `cuobjdump -lptx` → empty),
TRT-LLM explicitly rejects `-virtual`, vLLM warns against it.

**E. Separate per-arch wheels.** Rejected. N× CI and storage, user must pick correctly, and it is
precisely the mechanism that left SM12x unserved in `flashinfer-cubin`
([flashinfer#3294](https://github.com/flashinfer-ai/flashinfer/issues/3294): 12 681 cubins, none for
sm_120/sm_121).

**F. Document the limitation and stop.** Rejected, and not merely because the user ruled it out. It
is **unsafe**: a `sm_120a`-only binary on GB10 does not reliably error. **[measured]** the launch
returns 209 but `cudaDeviceSynchronize()` returns "no error" and the output buffer stays zero. Any
"document it" outcome must still ship Phase 0's build-time SASS assertion and runtime image check, or
the failure mode is garbage pixels rather than an exception.

---

## 6. Open questions / what would change this recommendation

1. **Is the bf16 `generate_stats` bug live on SM120a today?** [unverified — needs a 5090 / RTX PRO
   6000.] The failure is in cudnn-frontend graph *validation*, before any device query, so it should
   be hardware-independent. If confirmed, Phase 2 is a release-blocking bug fix for existing supported
   hardware and should jump the queue ahead of all arch work. If it somehow does *not* reproduce
   there, our model of that failure is incomplete and Phase 2 needs re-derivation.
2. **Will cuDNN ship an FP8 fused-MHA engine for sm_121?** If a future cuDNN adds it, Phase 4's
   fallback becomes redundant (keep it as a fallback, drop the error path). If not, GB10 permanently
   runs FP8 attention through the CUTLASS path, and that path's *performance* on 48 SMs becomes the
   thing to measure. Either way it does not change the arch recommendation.
3. **Is the `sm120-tma-pool.patch` still needed?** [weak evidence it is not.] If a full-model A/B on
   unpatched CUTLASS at 121a passes, dropping it removes the main obstacle to option B and makes
   CUTLASS bumps routine. Worth an explicit experiment, not a guess.
4. **Does the env-gated block-scaled probe (`cosmos_fp8_tc_probe.cu:63`, `Sm120Fp8ProbeGemm`, behind
   `OMNIDREAMS_DIT_FP8_TC_*`) work at 121a?** [untested end-to-end in the extension, though the
   standalone lift of it did run correctly.] It is the only path using
   `KernelTmaWarpSpecializedBlockwisePingpongSm120` and block-scale layouts, and it is the most likely
   place for an `a`-only assumption to hide. Relevant mainly if `f` is ever adopted (option C).
5. **Sage3 / SpargeAttn numerics on CC 12.1.** [unverified.] These are approximate-attention kernels;
   relaxing their gates (Phase 3) without numerical validation on both CCs would trade a silent
   fallback for a silent quality regression, which is worse. If they turn out to be genuinely
   CC-12.0-specific, the right outcome is a *narrower, build-informed* gate — not the current device-name
   allowlist.
6. **Tile/stage heuristics.** [expected mis-tuned, unmeasured.] 48 SMs / unified LPDDR5X / 24 MB L2 vs
   ~170 SMs / GDDR7. Correctness is unaffected (`StageCountAutoCarveout` computes identically — GB10
   reports the same 100 KB smem/SM, 99 KB optin), but "works on GB10" and "is fast on GB10" are
   different claims and this doc only supports the first.
7. **Would PyTorch accepting `f` change the recommendation?** Yes, and that is the intended
   evolution: at that point `12.0f` alone replaces the SM12x arch list and future 12.x parts need no
   rebuild. Nothing in Phases 0-4 becomes wasted work — Phase 0's SASS check and Phase 3's family
   gating are prerequisites for `f` too.

---

## Appendix: operational notes

- **`sync_thirdparty.py` binds a 3rdparty tree to its checkout path.** `_source_hash()` hashes the
  **absolute** patch path, so copying a synced `3rdparty/` between checkouts yields
  `"cutlass stamp does not match manifest"`. Each checkout needs its own copy and its own `sync` run.
  Worth fixing (hash the path relative to the repo root) if anyone is in there anyway.
- **Diagnostic recipe** for any future "does it run on this GPU?" question, in the order that
  disambiguates fastest:
  1. `cuobjdump -lelf <ext>.so | sort | uniq -c` — which arches are actually in the binary.
  2. `cudaFuncGetAttributes` on a trivial kernel from the same build — 209 here means arch mismatch,
     full stop, and it is far clearer than any launch-site error.
  3. `CUDNN_FRONTEND_LOG_INFO=1` and `CUDNN_LOGLEVEL_DBG=3` — cuDNN failures are opaque without them
     and were the entire remaining mystery in this investigation.
  4. Only then read omnidreams' own error strings, which currently lose the underlying code
     (`cosmos_gemm_bf16.cu:210`).
- **Artifacts from this investigation** (scratch, not repo state):
  `/home/qiwu/.claude/jobs/0b3caf82/tmp/repro/{probe_repro.cu,funcattr.cu,repro_120a,repro_120f,repro_121a}`,
  `/home/qiwu/.claude/jobs/0b3caf82/tmp/sm12x_gemm.cu`,
  `/home/qiwu/.claude/jobs/0b3caf82/tmp/arch/{t.cu,r.cu,f.cu}`,
  `/home/qiwu/.claude/jobs/0b3caf82/tmp/cudnn_full.log` (FP8 arch mismatch at `:12610-12613`),
  `/home/qiwu/.claude/jobs/0b3caf82/tmp/build_120f.py` (torch arch-parser bypass prototype, **not**
  in the tree),
  `/home/qiwu/.claude/jobs/0b3caf82/tmp/torch_cuda_lelf.txt`.

## Sources

- [CUDA Programming Guide — Compute Capabilities (Table 28: arch- and family-specific targets)](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- [NVIDIA Blackwell and CUDA 12.9 Introduce Family-Specific Architecture Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)
- [CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) · [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
- [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md) (4.2.0: SM121 shares SM120 code) · [CUTLASS example 79](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm) · [CUTLASS #3100](https://github.com/NVIDIA/cutlass/issues/3100), [#3227](https://github.com/NVIDIA/cutlass/issues/3227) (both CuTe-DSL only; do not affect the header-only C++ path)
- [PyTorch cmake/Codegen.cmake](https://github.com/pytorch/pytorch/blob/main/cmake/Codegen.cmake) · [torch/utils/cpp_extension.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/cpp_extension.py) · [pytorch#156176](https://github.com/pytorch/pytorch/pull/156176) (sm_121/sm_110 rationale)
- [vLLM CMakeLists.txt](https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt) · [vLLM cmake/utils.cmake](https://github.com/vllm-project/vllm/blob/main/cmake/utils.cmake) · [vLLM PR #38126](https://github.com/vllm-project/vllm/pull/38126) · [PR #40082](https://github.com/vllm-project/vllm/pull/40082)
- [TransformerEngine common/CMakeLists.txt](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/CMakeLists.txt) · [TensorRT-LLM cuda_configuration.cmake](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/cmake/modules/cuda_configuration.cmake), [#11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799)
- [FlashInfer compilation_context.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/compilation_context.py) · [flashinfer#3170 (DGX Spark SM121 audit)](https://github.com/flashinfer-ai/flashinfer/issues/3170) · [#3294](https://github.com/flashinfer-ai/flashinfer/issues/3294)
- [FlashAttention setup.py](https://github.com/Dao-AILab/flash-attention/blob/main/setup.py) · [SGLang aot/CMakeLists.txt](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/CMakeLists.txt) · [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- Prior repo write-up: `integrations/cmd/docs/quantization_native_port_scoping.md`