<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CMD: porting omnidreams' native CUTLASS FP8 DiT path (progressive implementation plan)

Status: **plan, nothing implemented**. This is the implementation plan that
`quantization_native_port_scoping.md` deliberately refused to write ("this section deliberately does not
use 'Phase N' numbering"). It is written now because five targeted investigations closed most of the
questions that doc left open. It supersedes that doc's *scoping* verdict, not its *hardware* findings —
all `sm_121a` / GB10 results are cited from it and **not repeated here**.

Sibling docs, read them rather than expecting this one to restate them:

- `integrations/cmd/docs/quantization_native_port_scoping.md` — why the native path cannot execute on the
  local DGX Spark GB10 (`sm_121a`), all three arch targets measured (`:103-227`). Load-bearing conclusion
  reused throughout this plan: **the local dev box can compile this code but cannot run its kernels.**
- `integrations/cmd/docs/quantization_plan.md` — the *other*, much simpler `torch._scaled_mm` FP8 route.
  Already implemented in the working tree. It is this port's fallback, its numerical oracle, and its
  competitor: compiled FP8 there measured **1.17-1.84x** on isolated CMD-shaped GEMMs (`:141-158`).
- `integrations/cmd/docs/sage_attention_plan.md` — attention-backend selection, deferred.

Verification legend used below: **[V]** = verified by running code or reading the exact cited lines during
investigation; **[A]** = assumed / inferred and explicitly not verified; **[?]** = open question, tracked in
the last section.

---

## 1. What the port actually is

The native path is a Python executor (`OptimizedDiTExecutor`,
`integrations/omnidreams/omnidreams_singleview/python/optimized_dit.py:691-815`) that snapshots a
`CosmosDiTNetwork`'s `state_dict()`, quantizes the block linears to E4M3, and thereafter calls one C++
entry point per denoising step — `optimized_dit_forward` in
`integrations/omnidreams/omnidreams_singleview/src/dit_streaming/pyext/streaming_dit_bridge.cu`. The port
is therefore **not** a kernel-authoring project. It is an adapter project with one small kernel change.

Three facts fix the shape of the work:

**[V] The weight contract is already satisfied.** The two transformer implementations — omnidreams'
(`integrations/omnidreams/omnidreams/transformer/impl/network.py:63`) and the shared flashdreams one CMD
uses (`flashdreams/flashdreams/recipes/cosmos/transformer/impl/network.py:148`) — emit **byte-identical
state-dict keys**. Verified two independent ways: by instantiating all three networks and diffing
`state_dict()` (omnidreams-only keys: none; shared-only: none; shape mismatches: none), and against the
real released checkpoint on this box,
`~/.cache/huggingface/hub/models--nvidia--cmd/snapshots/0fc41b56.../chunk4_camera_control_t29_l24.safetensors`
(600 keys = 12 top-level + 21 x 28 blocks). Every key the bridge reads
(`streaming_dit_bridge.cu:1655-1714, 2698-2726, 3039-3075`) and every key the FP8 prep requires
(`_COSMOS_BLOCK_FP8_LINEAR_KEYS`,
`integrations/omnidreams/omnidreams_singleview/python/cosmos_fp8_utils.py:35-44`) is present with the
right shape. `prepare_cosmos_quantized_streaming_weights` was run on a CMD-shaped state dict and completed
(51 keys in, 95 out). **Nothing needs renaming and nothing is missing.**

**[V] The one architectural delta is `cam_encoder`.** `CMDTransformerBlock`
(`integrations/cmd/flashdreams_cmd/transformer/modules.py:28`) adds exactly one `nn.Linear` per block,
`self_attn.cam_encoder` `[2048, 1536]` (`:36-40`), applied at `:121-127` (and identically in `prefill`,
`:147-150`) between AdaLN modulation and the self-attention QKV projection. The bridge does not know the
key exists; unknown keys in the `weights` dict are silently ignored (lookup is by literal name,
`streaming_dit_bridge.cu:269-273`). **The default failure mode is therefore silently camera-blind video,
not a crash** — the worst possible failure mode, and the reason Phase 2's contract test exists.

**[V] Everything else is plumbing, and all of it is CPU-testable.** Rank/`batch_shape` normalisation,
config-field translation, the runtime/config dict contract, the FP8 self-KV prefix seeding, and the arch
gate are all pure Python against a `SimpleNamespace` fake extension — the technique omnidreams' own suite
already uses for 30 of its 32 tests
(`integrations/omnidreams/tests/test_omnidreams_singleview_native.py:733-770, 786-795, 839-914`).

### 1.1 The premise correction that reshapes the whole plan

`quantization_native_port_scoping.md` establishes that the port cannot be validated on the dev box. That
remains true. But **[V] the repo's standard GPU CI runner is the target hardware**:
`.github/workflows/ci.yml:88` runs the per-PR `gpu` job on `linux-amd64-gpu-rtxpro6000-latest-2`, and
"RTX PRO 6000" is one of exactly three names on omnidreams' own SM120a allowlist
(`integrations/omnidreams/omnidreams_singleview/src/dit_streaming/kernels/sage3_attention.cu:618-637`:
`prop.major==12 && prop.minor==0`, then a device-name match). `CONTRIBUTING.md:272` documents the tier as
"GPU runner (RTX Pro 6000)".

So target hardware is reachable on every PR with **no new runner and no infra request**. The dev-box
constraint is a *developer inner-loop* problem, not a *validation* problem. What genuinely does not exist:

- **[V] No CI job ever builds the native extension.** `test_cuda_native_extension_builds`
  (`integrations/omnidreams/tests/test_omnidreams_singleview_native.py:928-933`) is `ci_gpu` but gated on
  `OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST`, which `grep` shows is set **nowhere** under `.github/`.
- **[V] No CI job ever executes the native kernels.** `native_dit_acceleration` defaults to `"disabled"`
  (`integrations/omnidreams/omnidreams/transformer/__init__.py:208`) and
  `grep -rn "native_dit" .github/ configs/ apps/` returns zero hits.
- **[V] `3rdparty/` is gitignored** (`integrations/omnidreams/omnidreams_singleview/.gitignore:2`) and
  there is no CI cache for it or for the extension build root (`ci.yml:196-241` trims only the uv cache).
- **[V] CMD has zero `ci_gpu` tests.** `pytest integrations/cmd/tests/ -m ci_gpu` → "no tests collected
  (66 deselected)"; all five files are module-level `ci_cpu`.

---

## 2. Risk register, ranked by uncertainty retired per unit of effort

This ordering drives the phase order. Highest first.

| # | Risk | Status | Retired in |
|---|---|---|---|
| R1 | Extension does not build in the CI container (`nvidia/cuda:13.2.1`, `ci.yml:91`) — the 164 s clean build was proven only against local `nvcc 13.0.88`, and CUTLASS is pinned at `f3fde58` **plus a repo-local `sm120-tma-pool.patch`** | **[?] unknown, unretirable locally** | Phase 0 |
| R2 | Per-PR CI cost of building the extension makes the whole approach unaffordable (no `3rdparty/` cache exists; absolute paths poison the extension cache key) | **[V] problem confirmed, fix identified** | Phase 0 |
| R3 | `cam_encoder` silently dropped → camera-blind video that a golden-clip gate localises poorly | **[V] confirmed real** | Phase 2 (contract), Phase 5 (fix) |
| R4 | CMD's `prefill` prefix is never seeded into the FP8 self-KV caches → native attention reads zeros for the conditioning frame | **[V] confirmed real; fix now believed cheap** | Phase 4 |
| R5 | Rank / `batch_shape=()` mismatch: bridge wants 5D `[B,V,T,HW,D]` with `V==1`, CMD's latents are rank-2 | **[V] confirmed real, mechanical** | Phase 3 |
| R6 | FP8 numerics diverge from BF16 beyond an acceptable threshold at CMD shapes | **[?] genuinely unknown** | Phase 4 / Phase 6 |
| R7 | Per-token timesteps (`conditional_frame_timestep`) are architecturally unsupported by the native AdaLN pipeline | **[V] confirmed; currently latent** (CMD presets leave it `None`) | Phase 3 (guard), never (capability cap) |
| R8 | Activation-scale calibration required with no artifact/script | **[V] FALSE ALARM — see §3** | n/a |
| R9 | `_release_network_after_fp8_snapshot` incompatible with CMD (`initialize_cache` signature, patchify rank, `patch_temporal` on config, no `prefill`) | **[V] confirmed; deliberately deferred** | Phase 7 (optional) |
| R10 | Packaging: `cmd -> omnidreams` dependency drags in `<3.13` pin, `ludus-renderer`, cmake/ninja | **[V] confirmed; precedented workaround** | Phase 1 |

---

## 3. The calibration question, answered up front

The scoping brief flagged FP8 activation-scale calibration as a possible early-phase blocker. **[V] It is
not one, and this plan explicitly budgets zero time for a calibration harness.** Evidence, all from
investigation 3:

- Nothing in the repo calls `validate_cosmos_fp8_activation_calibration` (`cosmos_fp8_utils.py:288-327`)
  or `cosmos_fp8_activation_scale_tensor` (`:330-354`) outside their own definitions. A repo-wide grep for
  `cosmos_fp8_activation` hits only those three sites.
- The bridge reads `config["cosmos_fp8_activation_scales"]` under an `if (config.contains(...))` guard
  (`streaming_dit_bridge.cu:2509-2519`), and `_ensure_fp8_runtime` (`optimized_dit.py:1201-1221`) never
  writes that key. Same for the reverse-direction amax tensor (`:2522-2529`) — no Python producer exists.
- When absent, `cosmos_fp8_activation_scale_or_one`
  (`integrations/omnidreams/omnidreams_singleview/src/dit_streaming/kernels/cosmos_block.cu:503-510`)
  returns `1.0f`.
- **Only 1 of the 10 declared sites is wired into a kernel at all.** `cosmos_fp8_activation_scale_or_one`
  has exactly one call site in the whole tree: `cosmos_block.cu:2855`, for
  `kCosmosFp8ActivationScaleFfn1Gelu` (index 9, `cosmos_block.cuh:46`). Sites 0-8 are declared in Python
  and reserved in the ABI but consumed by nothing. And `scale == 1.0f` selects the *fused* GELU→FP8 fast
  path (`cosmos_block.cu:794-807`) — the uncalibrated default is also the fastest one.
- There is no calibration script anywhere:
  `integrations/omnidreams/omnidreams_singleview/tools/` contains only `native_build.py` and
  `sync_thirdparty.py`.

Activation quantization on this path is **dynamic per-tensor amax computed on-device**
(`ops.cuh:286-289, 385-388`, driven from `linear_utils.cuh:126-154`). CMD ships with no calibration
artifact, matching omnidreams' own shipped behaviour.

**The residual risk this leaves is R6, not R8.** If Phase 4 numerics come out bad, static scales are one
of the candidate fixes — but wiring them is a *new kernel project* (9 of 10 sites are unconsumed), not a
"write the missing calibration script" task. That off-ramp is priced in §10, not in a phase.

---

## 4. Validation tiers (defined once, referenced by every phase)

| Tier | Where | Mechanism | What it can prove |
|---|---|---|---|
| **T0** | CPU, every PR | `ci_cpu` in `.github/workflows/ci.yml:83` (`linux-amd64-cpu8`) | Config plumbing, backend selection, weight-key/layout contracts, runtime-dict contract, **arch-dispatch decisions** via `monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _d: (12, 0))` (idiom from `integrations/sana/tests/test_smoke.py:2152`) |
| **T1** | Local GB10, `sm_121a` | `manual` marker, developer inner loop only | **Compilation** (syntax, template instantiation, symbol export — ~164 s clean, `quantization_native_port_scoping.md:105-113`), non-arch-conditional ops through the real `.so`, loader/cache behaviour, and the `torch._scaled_mm` oracle. **Never kernel numerics.** |
| **T2a** | RTX PRO 6000 CI runner, collected-and-skipped | `ci_gpu` + `_require_sm120a()` | Costs milliseconds while the native path is off; keeps the tests from going stale. Use from day one. |
| **T2b** | RTX PRO 6000 CI runner, executing | `ci_gpu` + env opt-in + a `sync_thirdparty` step + build cache | Single-block and full-network numerics parity vs BF16 |
| **T2c** | Nightly / `workflow_dispatch` | New workflow cloned from `.github/workflows/determinism.yml` (already the template for "a `ci_gpu` test gated behind an env var a scheduled job flips", `:5-8`, `:30`) | Golden-clip quality gate, end-to-end perf |

The one shared gate helper, mirroring `sage3_is_runtime_supported` (`sage3_attention.cu:618-637`) so Python
and CUDA agree, and skipping *cleanly* on GB10 as `AGENTS.md:57` requires:

```python
def _require_sm120a() -> None:
    if not torch.cuda.is_available():
        pytest.skip("native FP8 DiT requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        pytest.skip(f"native FP8 DiT kernels are sm_120a-only, got sm_{major}{minor}")
    name = torch.cuda.get_device_name()
    if not any(k in name for k in ("GeForce RTX 5090", "RTX PRO 6000", "RTX 6000")):
        pytest.skip(f"sm_120a allowlist does not cover {name!r}")
```

Do **not** gate any of this on `torch.cuda.is_available()` alone. On GB10 that attempts to run and fails
confusingly, and `quantization_native_port_scoping.md:196-202` measured a *silent no-op* for `sm_120a`
launches — a wrong-arch build can produce zeros without raising.

---

## 5. Phase 0 — CI build bring-up (**gated on SM120a CI; do this first**)

**Why first.** It is the only risk (R1/R2) that no amount of local or CPU work can retire, it blocks every
later phase's validation, and it touches zero CMD code — so a failure here costs nothing already invested.
It is also the phase most likely to surprise: the 164 s build was proven against local `nvcc 13.0.88`,
CI runs `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` (`ci.yml:22`, `:91`), against pinned CUTLASS
`f3fde58` **plus** the repo-local `patches/cutlass/sm120-tma-pool.patch`.

**Work**

1. Add a `sync_thirdparty.py sync` step + `OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST: "1"` to the `gpu`
   job env block (`ci.yml:93-108`), initially on a scratch branch, not on the per-PR path.
2. Confirm `test_cuda_native_extension_builds`
   (`test_omnidreams_singleview_native.py:928-960`) passes on the runner. It already exercises
   `zero_workspace_`, `prepare_contiguous`, `native_tensor_descriptor` through the compiled `.so`.
3. Add the missing **build cache** — the highest-leverage CI change in this plan. Key it on the
   `thirdparty_manifest` hash (`sync_thirdparty.py` already computes `_source_hash:235` /
   `_source_tree_hash:262` and stamps via `_write_stamp:292`) plus the `sm${{ ... }}` arch value the
   workflow already derives at `ci.yml:112-121`.
4. Fix the two portability papercuts that make caching impossible today (**[V] both confirmed**):
   - **Absolute paths in `_source_hash`** — `sync_thirdparty.py:244` (`"path": str(patch.path)`) and
     `:252` (`"source": str(overlay.source)`), baked into `_stamp_metadata:266-275` and strictly compared
     in `verify_source:445-452`. A `3rdparty/` tree copied or bind-mounted from another checkout is
     rejected. Fix: hash `patch.path.relative_to(ROOT).as_posix()`. Content is already covered by the
     sibling `sha256` fields, so nothing is lost.
   - **Absolute paths in the compiled-extension cache key** — `_extension_name()` folds
     `json.dumps(thirdparty_info)`
     (`integrations/omnidreams/omnidreams/native/omnidreams_singleview.py:440`) and
     `SourceInfo.as_dict()` emits `"path"` / `"stamp_path"` (`native_build.py:58-67`). Two checkouts
     produce different extension names and cannot share the ~95 MB / ~164 s build. Fix: drop
     `path`/`stamp_path` from the dict fed to `_extension_name`, keep `commit`/`source_sha256`/
     `tree_sha256`. This also unblocks the three worktrees currently on this box each needing their own
     348 MB.
5. Optionally address the full-tree SHA256 on every cold load (`omnidreams_singleview.py:553` →
   `_hash_tree`, `sync_thirdparty.py:104-120`) — ~348 MB of I/O per process start even when the `.so` is
   cached. Painful for a pytest session that spawns workers. Not blocking; note it.

**Exit criteria (independently verifiable)**

- A green CI run on `linux-amd64-gpu-rtxpro6000-latest-2` that syncs third-party sources, JIT-builds the
  extension, and passes `test_cuda_native_extension_builds`. Wall-clock recorded.
- A second run demonstrating cache hit, with the delta in wall-clock recorded.
- T1 corroboration: the same two path fixes verified locally by building from two different worktrees and
  observing a shared extension name / shared build dir.

**Tier:** T2b (build only) + T1. **Effort: 2-4 days**, dominated by CI iteration latency, with a real
chance of one round lost to the CUDA 13.2 / CUTLASS `f3fde58` delta. **Cannot be done blind.**

**Off-ramp.** If the extension does not build under CUDA 13.2 and the fix is not obvious within ~3 days,
stop the whole port and stay on `quantization_plan.md`'s route. Do not attempt to bump CUTLASS — the
pinned commit carries a local patch (`patches/cutlass/sm120-tma-pool.patch`) whose rebase cost is unknown
**[?]**.

---

## 6. Phase 1 — packaging and the gating skeleton (**blind**)

**Goal:** CMD can *ask for* the native path and *cleanly not get it*, on every machine, before any of it
works. Nothing in this phase changes behaviour for existing CMD users.

**Legality: no architecture exception is needed. [V]** The dependency rule as written
(`AGENTS.md:65`, `CLAUDE.md` "Architecture notes") is one-directional: `core` and `infra` must never import
from `integrations/`. Integration→integration is explicitly blessed
(`skills/flashdreams-integrations/SKILL.md:16`) and precedented: `integrations/hy_worldplay/pyproject.toml:39`
depends on `flashdreams-wan22`, imported at `hy_worldplay/_checkpoint.py:23`. Native code as its own nested
workspace member is also precedented: `integrations/omnidreams/ludus-renderer/` (root `pyproject.toml:12`).

**Recommended structure** — extract, don't depend:

```
integrations/flashdreams_native_cosmos/
├── pyproject.toml                  # name = "flashdreams-native-cosmos"
├── flashdreams_native_cosmos/
│   ├── acceleration.py             # moved from omnidreams/native/acceleration.py — zero omnidreams content
│   ├── primitives.py               # moved from omnidreams/native/primitives.py
│   └── loader.py                   # load_extension / select_backend orchestration
├── src/  python/  tools/  patches/  thirdparty_sources.json
└── .gitignore                      # /3rdparty/
```

`omnidreams/native/__init__.py` becomes a thin re-export shim so its 20+ call sites and its public docs
(`docs/source/models/omnidreams.rst:315-343`) keep working.

**Why not just add `"flashdreams-omnidreams"` to CMD's deps** (the `hy_worldplay` shortcut, which *is*
legal): it narrows CMD from `>=3.10` to omnidreams' `>=3.10,<3.13` (`integrations/omnidreams/pyproject.toml:26`,
a PyNvVideoCodec constraint), pulls in `ludus-renderer` → `cmake>=4.0`/`ninja`/`nvidia-nvcomp-cu12`, breaks
the narrow-env contract in `CONTRIBUTING.md:321-337`, and transitively registers omnidreams' runners
(entry-point discovery is group-wide). Acceptable as an explicitly-temporary interim; the extraction gets
harder the more CMD code imports `omnidreams.*`.

**Work**

1. Extract the package; add its path to root `pyproject.toml` `[tool.pyright] extraPaths` (**line 45**)
   *and* `[tool.ty.environment] extra-paths` (**line 71**) — hand-maintained lists, not globs. Add the row
   to `CONTRIBUTING.md:339-351`. Update `THIRD-PARTY-NOTICES:91-111`, which documents CUTLASS /
   SageAttention / SpargeAttn / cudnn-frontend by their *current path* (see `skills/maintaining-oss-state`).
2. Declare it as an **optional extra**, not a hard dependency:
   `[project.optional-dependencies] native = ["flashdreams-native-cosmos"]` (precedent: omnidreams'
   `interactive-drive` / `rtx-postprocess` extras, `omnidreams/pyproject.toml:79-87`). Import lazily,
   function-local, inside the `!= "disabled"` branch (precedent: `omnidreams/transformer/__init__.py:318, 350`).
3. Add config slots. Per `quantization_plan.md`'s already-landed precedent
   (`weight_quantization: Literal[...]` on the shared config,
   `flashdreams/flashdreams/recipes/cosmos/transformer/__init__.py:156`), put
   `native_dit_acceleration` / `native_dit_backend` / `native_dit_attention_backend` on the **shared**
   `CosmosTransformerConfig` (`:101`) typed as plain `Literal`/`str` — **never** as omnidreams'
   `NativeAccelerationMode`, which would put an `integrations/` import in `recipes/`. Default
   `"disabled"`. Reuse the three-state `auto`/`disabled`/`required` vocabulary verbatim
   (`omnidreams/native/acceleration.py:59-77`, `select_native_extension:128-175`,
   `_unavailable_or_raise:178-190`); do not invent a second one.
4. **Add the compute-capability guard.** This is the single most important gating addition the port makes.
   `_default_availability_check` (`acceleration.py:193-201`) only probes an optional `is_available()`
   symbol, so on GB10 the extension compiles for `12.0a`, *loads*, and then hard-errors at kernel launch.
   CMD's availability check must return `(False, "native FP8 requires SM 12.0; found sm_121")` for
   anything but `(12, 0)`.
5. Parameterise the remediation string. `NATIVE_EXTENSION_SYNC_COMMAND` (`acceleration.py:35-40`) is
   omnidreams-branded and is emitted into every `auto`-mode warning and `required`-mode exception
   (`:120-125`); a CMD user would be told to run a command that doesn't work for them.
6. Plumb `max_jobs` through CMD's config — `_DEFAULT_MAX_JOBS_CAP = 8` (`omnidreams_singleview.py:54`)
   caps Ninja at 8 on a 20-core box.

**Exit criteria**

- **T0:** new `integrations/cmd/tests/test_native_fp8.py`, `pytestmark = pytest.mark.ci_cpu`, asserting:
  every released CMD preset defaults to `"disabled"`; `mode="disabled"` never touches the compiler
  (`acceleration.py:138`); `mode="auto"` with a fake-unavailable loader falls back with a reason;
  `mode="required"` raises `NativeAccelerationUnavailable`; and the arch decision under
  `get_device_capability` monkeypatched to `(12,0)` / `(12,1)` / `(9,0)`. Mirror
  `test_omnidreams_singleview_native.py:499-640`.
- **T0:** a static kernel-coverage test that greps the vendored `src/` for `arch::Sm12x` tags and asserts
  the set equals an explicit expected set (currently: 48 sites, `ops.cu` x40 +
  `cosmos_fp8_tc_probe.cu` x8). Cheap, and it catches exactly the class of surprise that burned the
  scoping investigation.
- **T1:** on GB10, `mode="auto"` logs a graceful "sm_121 not supported" fallback instead of a CUDA error.
- `uv sync --package flashdreams-cmd --extra dev` still resolves without cmake/ninja.

**Effort: 3-5 days.** **Fully blind.**

---

## 7. Phase 2 — weight-contract conformance, on CPU (**blind**)

**Goal:** make R3 (silently-dropped `cam_encoder`) impossible to ship, before any kernel work exists.

**Work**

1. Build the CMD-side weights-dict producer against the contract enumerated in investigation 1. The
   required key list is long but fully known; the hard-failure semantics are `TORCH_CHECK(d.contains(key))`
   in `get_w` (`streaming_dit_bridge.cu:269-273`). Notable sub-contracts:
   - `linear_backend="fp8"` ⇒ all 8 block linears must be `uint8` (`:2561-2573`); every uint8 weight needs
     a 1-D `[out_features]` `<key>_scale` cast to fp16 (`get_block_linear_scale:2580-2611`).
   - `cosmos_quantized_prepared_strict=True` (which `_ensure_fp8_runtime` always sets,
     `optimized_dit.py:1217-1218`) ⇒ every `<key>_fp8_prepared` + `<key>_fp8_prepared_scale` must exist,
     the latter strictly `torch.float16` (`:2540-2559, 2592-2602`).
   - **[V]** The `_fp8_prepared` aliases are **not** transposed — the docstring says so
     (`cosmos_fp8_utils.py:486-489`) and `data_ptr()` equality was verified. They are a pure contract
     assertion; CMD could pass `add_fp8_prepared=False` + `cosmos_quantized_prepared=False` and lose only
     the strict validation.
   - Fused QKV activates **only when `self_attn.qkv_proj.weight` exists AND is `uint8`**
     (`:1892-1894, 2645-2647`), and `drop_split_self_attn_qkv` defaults to `True`
     (`cosmos_fp8_utils.py:589-590`) — **[V]** the output dict then contains no `q_proj` at all. Fusion
     stays valid for CMD because `cam_encoder` acts on `normed_x` *before* QKV, not on Q/K/V.
2. **The `cam_encoder` contract test.** Assert that the packing step either (a) emits a tensor for
   `blocks.{i}.self_attn.cam_encoder.weight` with the right shape/dtype/contiguity **and** the config dict
   carries the corresponding camera key, or (b) refuses to enable the native path when
   `config.network.camera_dim is not None`. Until Phase 5 lands, (b) is the correct behaviour and this test
   is what enforces it.
3. Weight-prep / layout checks over CMD's real key set with `prepare_tensor_for_native` /
   `NativeTensorSpec` (mirroring `test_omnidreams_singleview_native.py:663-707`), run against **every** CMD
   runner preset (`cmd-chunk1/4-{short,long,camera}-i2v`, `integrations/cmd/pyproject.toml` entry points),
   so a preset whose dims violate a kernel tile constraint fails on CPU.
4. Guard: the native path requires `weight_quantization="none"`. The shared `CosmosTransformer.__init__`
   rewrites the module tree via `replace_linear_with_quant` when it isn't
   (`recipes/cosmos/transformer/__init__.py:245-256`), and the snapshot would then contain quantized-Linear
   keys `prepare_cosmos_quantized_streaming_weights` cannot parse. Note the precedent that
   `_COSMOS_WEIGHT_QUANTIZATION_SKIP_PATTERNS` already lists `cam_encoder` (`:192`).
5. Cheap memory win while here: **[V]** because the `_fp8_prepared` aliases share CPU storage, the
   per-entry `.to(device=...)` loop at `optimized_dit.py:1074-1079` materialises **two independent GPU
   copies** of every FP8 weight — ~1.64 GiB becoming ~3.29 GiB for CMD's 28x2048 geometry, i.e. no better
   than the BF16 it replaced. Deduplicate by `data_ptr` during the device move.

**Exit criteria**

- **T0:** all of the above as `ci_cpu` tests. A weights dict built from a real CMD `state_dict()` passes a
  pure-Python re-implementation of every `TORCH_CHECK` the bridge performs (shape, dtype, rank,
  contiguity), with no GPU and no compiled extension. This is the phase that makes later T2b failures
  interpretable: if the contract test is green, a T2b failure is a *numerics* problem, not a *plumbing*
  problem.
- **T1 (optional):** run the real `prepare_cosmos_quantized_streaming_weights` against the real CMD
  checkpoint on GB10 and diff key sets against the expected list. No kernel involved.

**Effort: 2-3 days.** **Fully blind.**

---

## 8. Phase 3 — the adapter (**blind**)

**Goal:** a `CMDTransformer`-shaped host object that satisfies every attribute the executor reaches for,
with all rank/config translation done and unit-tested — still with a fake extension.

**What the executor requires from the host** (investigation 1, section (c)) is large but entirely
enumerable: `transformer.{config, network (writable), _maybe_inject_image, _select_mask, _use_cuda_graph,
_cuda_graph_dispatch, _network_call, _network_call_uncond, _cuda_graph_capture_ar_idx, _output_height,
_output_width}`. **[V] Every one of these exists identically on the shared `CosmosTransformer`**
(`recipes/cosmos/transformer/__init__.py:226-227, 268-285, 447, 458`), and
`CUDAGraphDispatch.disable(fn=...)` exists at
`flashdreams/flashdreams/infra/acceleration/cuda_graph_dispatch.py:92`.

**The real gaps, all mechanical:**

| Gap | Fix |
|---|---|
| `config.num_views` (`optimized_dit.py:711`) | default 1 |
| `config.network.patch_temporal` / `.patch_spatial` (read at `:548-550, 585-587, 1758-1763`) | **[V]** shared config exposes `patch_size: tuple[int,int,int]` (`impl/network.py:109`); the *module* has the derived attrs (`:165-169`), the *config* does not. Translate in the adapter — do **not** add omnidreams' fields to the shared config. |
| `config.native_dit_*` | added in Phase 1 |
| `additional_concat_ch` / `additional_patch_embedding` (HDMap) | absent in CMD; pass `hdmap_patched` empty, omit `cosmos_hdmap_embed`. Guarded at `streaming_dit_bridge.cu:1660`. |
| Multi-view fields (`enable_cross_view_attn`, `n_cameras_emb`, `adaln_view_embedder`) | not needed; `V==1` is the only supported mode (`:1622`) |
| `set_context_parallel_group` arity (omnidreams: `self_attn_group=`/`cross_view_attn_group=`; shared: `cp_group=`, `impl/network.py:229`) | cosmetic; single-rank on this path |
| **Rank normalisation (R5)** | bridge wants 5D `[B,V,T,HW,D_in]`, `V==1` (`:1556-1624`), `B==1` (`:1747-1750`); executor's un-flatten assumes `ndim == 4` (`:1749-1774`); shared/CMD patchify to `[..., L, D]` (`impl/network.py:344-350`) and **every CMD preset sets `batch_shape=()`** (`integrations/cmd/flashdreams_cmd/config.py:83`) → rank-2. Adapter inserts the `B`/`V` axes and splits `L → (T, HW)`. |
| **Mask concat** | CMD concats in the transformer (`integrations/cmd/flashdreams_cmd/transformer/__init__.py:114`); the bridge concats internally (`:1650`). Pass `noisy_latent` and `mask` **separately**. |
| **Per-token timesteps (R7)** | CMD calls `_build_per_token_timesteps` (`transformer/__init__.py:112` → shared `:471-495`); bridge validates `t_emb` as exactly `[B, K]` (`:1684-1695`) and broadcasts one shift/scale row to all `M` rows (`cosmos_modulate.cu:280-305`). **Hard-guard: refuse the native path when `conditional_frame_timestep is not None`.** Per-token AdaLN is a capability cap, not a bug to fix. |
| `flatten_thw` (`omnidreams/transformer/__init__.py:345`) | shared always flattens |

**Also in scope:** `initialize_autoregressive_cache` / `predict_flow` / `finalize_kv_cache` hooks on
`CMDTransformer`, mirroring `omnidreams/transformer/__init__.py:615-616, 685-690, 729-730`. **Ordering
requirement, [V]:** the `after_initialize_autoregressive_cache` hook must fire *after* CMD's two
`self.network.prefill(...)` calls (`integrations/cmd/flashdreams_cmd/transformer/__init__.py:200-215`), so
that the BF16 self-KV caches already contain the prefix when the FP8 shadow caches are built. Phase 4
depends on this.

**Deliberately out of scope in this phase:** `_release_network_after_fp8_snapshot`
(`optimized_dit.py:1115-1146`) and its `_CosmosNetworkShapeOps` stub (`:419-587`). **[V]** The stub is
incompatible with CMD in four independent ways — `initialize_cache(chunk_size, window_size, sink_size,
context)` positional vs shared's `text_embeddings=` keyword (`impl/network.py:406-414`, called that way at
`recipes/.../__init__.py:329-333`); `patchify_and_maybe_split_cp` asserting `ndim == 6` (`:531-556`) vs
shared's `>= 4` (`impl/network.py:341-352`); `self.config.patch_temporal` (`:547-549`); and **no `prefill`
at all**. Keep the BF16 network resident (~3.3 GiB). See Phase 7.

**Exit criteria**

- **T0:** the adapter drives a full `_ensure_fp8_runtime` against a `SimpleNamespace` fake extension with
  CPU bf16 tensors (workspace builder stubbed, exactly as
  `test_omnidreams_singleview_native.py:839-914` does) and the produced config dict is asserted key by key:
  `num_blocks`, `num_heads`, `model_channels`, `cosmos_linear_backend`, `cosmos_kv_cache_backend`,
  `cosmos_write_bf16_kv_cache`, cache-list lengths, and every workspace tensor's exact shape/dtype against
  the bridge's `workspace_tensor` rules (`:2043-2069`).
- **T0:** rank-normalisation round-trip test: CMD `batch_shape=()` latent → 5D bridge input → back, for
  every preset's `(len_t, pH, pW)`.
- **T0:** the per-token-timestep guard raises for `conditional_frame_timestep != None`.
- **T2a:** the `ci_gpu` skeleton lands here (collected, skipped, milliseconds).

**Effort: 5-8 days.** **Fully blind.** This is the largest pure-Python phase.

---

## 9. Phase 4 — first real native forward, camera **off** (**gated on SM120a**)

**Goal:** one CMD preset without camera conditioning produces numerically-close output through
`optimized_dit_forward` on the RTX PRO 6000 runner. This is the phase where the port either works or
doesn't.

Camera is deliberately excluded here (Phase 5) so that a numerics failure has one fewer candidate cause.
Prefill is *not* excludable — every CMD preset is I2V with a clean first-frame prefix
(`cache.image = None`, `integrations/cmd/flashdreams_cmd/transformer/__init__.py:154`; all conditioning
flows through `prefill`), so R4 must be solved in this phase.

**R4, and why it is now believed cheap. [V]** The FP8 shadow self-KV caches are allocated **zero-filled**
(`optimized_dit.py:1248-1249`: `torch.zeros_like(t, dtype=torch.uint8)`) and are only ever written by the
kernel at `self_attn_write_start`. CMD's prefix K/V, written by `BlockKVCache.prefill`
(`flashdreams/flashdreams/core/attention/kvcache.py:132-174`) into the BF16 `_k`/`_v`, would never reach
them → the native attention reads zeros for the conditioning frame. **The fix is five lines**, because the
FP8 KV quantization on this path is an **unscaled identity-scale E4M3 cast done in plain PyTorch** — the
cross caches are built exactly that way at `optimized_dit.py:1235-1241`
(`t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()`). Seeding self is the same cast applied to the
already-prefilled BF16 buffers instead of `zeros_like`. **[A]** This assumes the kernel's self-KV write
path uses the same identity scaling as its cross-KV read path — verify by reading
`cosmos_block.cu:2522-2523, 2619-2620` before implementing, and confirm empirically at T2b.

Two related pieces of good news from investigation 2: **[V]** `compute_self_attn_write_start`
(`optimized_dit.py:656-680`) still returns the correct cursor after a prefill (the `_curr == _prev + 1`
branch returns `_n_cached == prefix_size`), and `_roll_fp8_cache_left_like_block_cache` (`:636-653`)
matches `_roll_local_window_left` (`kvcache.py:205-228`) even with the prefix offset. **The cursor
arithmetic survives; only the contents seeding does not.**

**Work**

1. Seed the FP8 self caches from the prefilled BF16 caches (above).
2. Wire the T2b CI job: extend Phase 0's scratch-branch job into a real gated `ci_gpu` execution path,
   env-opt-in, using the Phase 0 cache.
3. Numerics ladder, cheapest first — do not skip rungs:
   - **Rung A:** one `CMDTransformerBlock` forward, native FP8 vs BF16 reference, relative-MAE bound in the
     style of `integrations/sana/tests/test_quant_cuda.py:79-113` (which uses `fp8_rel_mae <= 0.06`).
   - **Rung B:** full 28-block network, one denoising step, AR index 0 (prefix just seeded).
   - **Rung C:** AR index >= 1, exercising the rolling window and `_roll_fp8_cache_left`.
   - **Rung D:** a short rollout (a few AR steps) against the BF16 path, cosine-sim / PSNR per step, to
     catch drift that a single step hides.
4. Use `cosmos_trace_tensor` (`[num_blocks, 4, B, M, K]`, debug taps for sa/ca/ffn/block-out,
   `streaming_dit_bridge.cu:2440-2457, 2875-2885`) to localise the first diverging block when a rung
   fails. This is the single most useful debugging affordance the bridge exposes; budget time to wire it.

**Exit criteria**

- **T2b:** rungs A-D pass on the RTX PRO 6000 runner with recorded thresholds, on `cmd-chunk4-short-i2v`
  (or the equivalent camera-free preset).
- **T2b:** an explicit negative test — with the prefix-seeding fix reverted, rung B *fails*. This is what
  proves R4 was real and is fixed, rather than masked.
- **T0:** every threshold and every config-dict value used above is also asserted on CPU, so a later
  regression is caught before it reaches the GPU job.

**Effort: 5-8 days** if numerics come out clean; **open-ended if they do not** (see §11, R6 off-ramp).
**Cannot be done blind** past the point of writing the tests. Write all the tests during Phase 3 so this
phase is purely "run and debug on the runner".

---

## 10. Phase 5 — camera injection (**kernel change; partly blind**)

**Goal:** close R3 for real, so the two camera presets (`cmd-chunk1-camera-i2v`, `cmd-chunk4-camera-i2v`,
`integrations/cmd/flashdreams_cmd/config.py:147-189`) work on the native path.

**Why it cannot be pre-applied outside the kernel. [V]** HDMap is a *single, network-level* add — one
shared `additional_patch_embedding` applied once after `x_embedder`, before the block loop, which is why
the executor can precompute it in Python (`_make_cosmos_hdmap_cache`, `optimized_dit.py:243-269`) and the
bridge does one `cur = cur + hdmap_embed` (`:1659-1660`). CMD's camera conditioning is **28 independent,
per-block adds, each landing after that block's `layer_norm_self_attn` + AdaLN modulation**. Folding them
into one pre-block add is mathematically wrong: LayerNorm is not additive and each block's `x` differs.

**But the value is `x`-independent**, so it is precomputable — the delivery, not the computation, is what
needs a kernel. And **[V] the seam is in non-CUTLASS code**: the FP8 path dispatches to
`cosmos_layernorm_modulate_to_fp8_only<bf16>` (`cosmos_block.cu:2398-2409`), which writes the modulated
activation straight to FP8 in `linear_fp8_scratch`, consumed directly by the fused QKV GEMM (`:2434-2437`).
There is no BF16 `normed` buffer in the fast path and no host-visible seam.

**The change:**

- Three kernels in `src/dit_streaming/kernels/cosmos_modulate.cu` — `cosmos_layernorm_modulate_kernel`
  (`:94`), `..._to_fp8_kernel` (`:157`), `..._to_fp8_only_kernel` (`:219`) — gain an optional
  `const ElementT* __restrict__ cam = nullptr`, and the epilogue at `:271-273` becomes
  `v = v * (1.f + sc) + sh + (cam ? to_float(cam[idx]) : 0.f);`
- Their three host launchers (`:280`, `:311`, `:340`), a `cam_sa` pointer on the block-params struct, and a
  `cosmos_cam_embed` config key in the bridge mirroring the existing `cosmos_hdmap_embed` validation
  (`streaming_dit_bridge.cu:1631-1647`).
- A Python precompute cloned from `_make_cosmos_hdmap_cache`.
- **Only the SA call site needs it.** The two fused `..._residual_layernorm_modulate_to_fp8_only` variants
  (`cosmos_block.cu:2659`, `:2807`) serve cross-attn and MLP, which take no camera term — the harder fused
  epilogues (`ops.cuh:185`) stay untouched.

**[V] `cosmos_modulate.cu` contains zero `cutlass::arch::Sm120` references** (all 48 sites are in `ops.cu`
and `cosmos_fp8_tc_probe.cu`). These are plain hand-written elementwise CUDA kernels, so **this change is
arch-neutral and its kernel is unit-testable on the local GB10** even though the end-to-end FP8 DiT is not.
That makes Phase 5 the one phase with real T1 signal, and it should be exploited: add a small pybind entry
point exposing `cosmos_layernorm_modulate*` directly and test it against a PyTorch reference on GB10.

**Memory note:** materialising `cam_embed` as `[num_blocks, B, L, 2048]` bf16 costs ~716 MB at the chunk-4
camera preset (`config.py:169-180`: 480x832 → latent 60x104, `patch_size=(1,2,2)` → `pH*pW=1560`, `T=4`,
`L=6240`) and ~179 MB at chunk-1. Affordable, but the chunk-4 figure argues for computing it per block
inside the C++ loop from the raw `[L, 1536]` tokens plus the 28 weights — one extra `[L,1536]x[1536,2048]`
bf16 GEMM per block, ~19.6 GFLOP/forward at `L=6240`, noticeable but not dominant next to the FFN.
**[?] Which of the two is better is unmeasured**; implement the precompute first (simpler, no new GEMM
dispatch), measure at Phase 6, switch if it matters.

**Exit criteria**

- **T1:** the modified elementwise kernels pass a numerics test against a PyTorch reference **on GB10**
  (arch-neutral, so this is genuine signal).
- **T0:** `cam_encoder` now appears in the weights/config dict for camera presets; the Phase 2 refusal
  guard flips from "refuse" to "supply", asserted by the same test.
- **T2b:** a parity test where **two different camera inputs produce measurably different outputs** — this
  is the specific assertion that guards against the silent-drop failure, which a golden-clip gate localises
  badly.
- **T2b:** rungs A-D from Phase 4 re-run on a camera preset.

**Effort: 5-8 days.** **Roughly half blind** (kernel authoring + T1 kernel test locally; end-to-end on CI).

---

## 11. Phase 6 — golden clip and perf (**gated on SM120a; nightly**)

**Goal:** answer the only two questions the SM120a tier uniquely has to answer — *did it change the video*
and *is it faster*.

**[V] The comparison machinery already exists in core** —
`flashdreams/flashdreams/quality/clip_compare.py` (`ClipComparisonThresholds:33`,
`assert_clip_within_thresholds:203`, `bottom_half:138`, `read_video_rgb:78`) — so it is importable from
`integrations/cmd/` with no dependency-direction violation. A CMD gate is a copy of omnidreams'
231-line `test_quality_regression.py` with `_ENV_PREFIX = "FLASHDREAMS_CMD_QUALITY_"` and a different
default runner, **not new infrastructure**.

**Work**

1. New workflow `.github/workflows/cmd-native-fp8.yml`, cloned from `.github/workflows/determinism.yml`
   (cron nightly + `workflow_dispatch`, same `runs-on: linux-amd64-gpu-rtxpro6000-latest-2`).
2. Produce the CMD reference clip **from the BF16 path on the same RTX PRO 6000**, publish to the HF
   dataset the way `ci.yml:171-186` does, promote per `tests/README.md`. **CMD has no reference clip
   today — this is net-new work.**
3. Run the native FP8 candidate against that BF16 reference. Start looser than omnidreams'
   (`min_psnr_db=30.0`, `max_mean_abs=4.0`, `max_rmse=8.0`, `max_mean_flip=0.070`,
   `integrations/omnidreams/tests/test_quality_regression.py:68-77`) — FP8-vs-BF16 is a genuine numerical
   change — and calibrate from the first artifact upload.
4. Perf: FP8 vs BF16 end-to-end latency, per `skills/validate-performance-quality/SKILL.md`. This number
   is the entire point of the port and **cannot be obtained anywhere else**
   (`quantization_native_port_scoping.md:183-187`). Compare against `quantization_plan.md`'s
   `torch._scaled_mm` route measured on the same runner, not against its GB10 numbers.

**Exit criteria:** nightly green; recorded PSNR/RMSE table; recorded latency table with the
`torch._scaled_mm` route as the comparison baseline. **T2c. Effort: 3-5 days** (plus reference-asset
turnaround). **Cannot be done blind.**

---

## 12. Phase 7 — deferred optimisations (**explicitly out of the critical path**)

Do not start any of these before Phase 6 has a number.

- **Network release / `_CosmosNetworkShapeOps`.** ~3.3 GiB of BF16 weights. **[V]** Blocked on four
  incompatibilities (§8) *and* on the fact that `prefill` is a full 28-block forward through every self-attn
  KV cache — it needs the whole network, so the stub cannot have a cheap version. Making this work means
  routing `prefill` through the native kernel too, which reopens the camera gap at a *different token
  count* (`pH*pW` vs `len_t*pH*pW`) while `_make_cosmos_streaming_workspace` sizes every scratch buffer for
  one fixed `tokens` (`optimized_dit.py:272-416`) and the call is CUDA-graph-captured. Note also that
  release force-disables CUDA graphs (`:1136`) while CMD's presets set `compile_network=True` and
  `use_cuda_graph=True` (`integrations/cmd/flashdreams_cmd/config.py:88-89`, asserted in
  `integrations/cmd/tests/test_config.py:104-105`) — the preset defaults would have to change.
- **`sage3` / `sparge` attention backends.** `fp8_kvcache_cudnn` does not need sage3; if pursued, note the
  name allowlist (`sage3_attention.cu:618-637`) and the `OMNIDREAMS_SINGLEVIEW_DISABLE_SAGE3=1` escape
  hatch, which is baked into the extension name and the in-process cache key
  (`omnidreams_singleview.py:437, 442, 544`) so both variants coexist.
- **`OMNIDREAMS_DIT_FP8_SDPA_LAYOUT=bhmd`** cuDNN/TC-layout shadow caches (`optimized_dit.py:1268-1311`).
- **Per-block camera GEMM** instead of the `[num_blocks, B, L, 2048]` precompute, if Phase 6 shows the
  716 MB matters.

**Effort: 5-10 days each, none scheduled.**

---

## 13. Effort summary and hardware gating

| Phase | What it retires | Blind? | Effort |
|---|---|---|---|
| 0 — CI build bring-up | R1, R2 | **No — SM120a CI required** | 2-4 d |
| 1 — packaging + gating skeleton | R10, arch-gate correctness | **Yes** (T0 + T1) | 3-5 d |
| 2 — weight-contract conformance | R3 (as a guard) | **Yes** (T0) | 2-3 d |
| 3 — adapter | R5, R7 | **Yes** (T0) | 5-8 d |
| 4 — first native forward, camera off | R4, R6 (first read) | **No — SM120a CI required** | 5-8 d, open-ended if numerics fail |
| 5 — camera injection | R3 (for real) | **Half** (kernel + T1 local; e2e on CI) | 5-8 d |
| 6 — golden clip + perf | the actual product question | **No — SM120a CI required** | 3-5 d |
| 7 — deferred optimisations | R9 and perf tail | mixed | not scheduled |

**Total for phases 0-6: roughly 25-40 engineer-days**, i.e. **6-9 calendar weeks** at realistic
CI-iteration latency, assuming no off-ramp is taken. Phases 1-3 (10-16 days, ~40% of the work) are fully
blind and can proceed in parallel with, or entirely before, securing anything beyond Phase 0's CI access.

**Practical sequencing consequence:** push as much as possible into T0. The `cam_encoder` contract, the
runtime dict, the weight-prep layout, and the arch-dispatch decision are all CPU-testable. If the port is
structured so those surfaces carry the CMD-vs-omnidreams delta, the SM120a tier only has to answer two
questions — "same video?" and "faster?" — and both fit in one nightly job.

---

## 14. Known blockers and open questions

Nothing in this section is resolved. Each is flagged where it bites.

1. **[?] CUDA 13.2 vs 13.0 build.** The 164 s build was proven only on local `nvcc 13.0.88`. CI runs
   `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` (`ci.yml:22`, `:91`) against pinned CUTLASS `f3fde58` plus
   a local patch. **No local work can retire this.** First CI build attempt is the test; budget one lost
   round. This is Phase 0's entire purpose and the plan's hardest off-ramp.
2. **[?] FP8 numerics at CMD shapes.** Unknown. omnidreams validated its own model, not CMD's, and CMD's
   camera-conditioned presets are a different data distribution. If Phase 4 rungs fail, the candidate
   fixes are, in increasing cost: `cosmos_linear_backend="mixed"` (keep specific layers BF16,
   `streaming_dit_bridge.cu:2561-2573`); BF16 KV cache instead of FP8
   (`cosmos_kv_cache_backend="bf16"`); and — expensive — static activation scales, which as §3 establishes
   is a **new kernel project** (9 of 10 scale sites are unconsumed), not a scripting task.
3. **[?] Self-KV FP8 scaling convention.** The prefix-seeding fix (Phase 4) assumes the kernel's self-KV
   write uses the same identity-scale E4M3 convention as the cross-KV path built in Python at
   `optimized_dit.py:1235-1241`. Read `cosmos_block.cu:2522-2523, 2619-2620` before implementing. If the
   convention differs, the fix becomes a C++ entry point instead of five lines of Python.
4. **[?] The 3-line `cosmos_modulate.cu` change is a code estimate, not a verified patch.** Nobody has
   written or compiled it. The `[V]` part is that the three kernels and their epilogue line exist as cited
   and contain no CUTLASS arch tags; the `[A]` part is that no other consumer of those launchers breaks
   and that the params struct threads through cleanly.
5. **[V, and permanent] Per-token timesteps are architecturally unsupported.** The bridge produces one
   modulation vector per batch element (`:1675-1683, 2477-2485`), not per token. CMD presets currently set
   `conditional_frame_timestep=None`, so this is latent — but it is a hard capability cap on the native
   path, not a to-do. Phase 3 makes it an explicit refusal.
6. **[V] Camera-blind output is the default failure mode**, not a crash: `cam_encoder.weight` passes
   through weight prep untouched (not in `_COSMOS_BLOCK_FP8_LINEAR_KEYS`, `cosmos_fp8_utils.py:35-44`) and
   the bridge never looks it up. Until Phase 5, the native path **must refuse** camera presets. Do not
   allow "it produced video, ship it" between Phases 4 and 5.
7. **[V] `V==1` and `B==1` only.** `streaming_dit_bridge.cu:1622, 1747-1750`. Any CFG batching must run as
   two separate calls, matching the executor's existing cond/uncond split (`optimized_dit.py:1041-1042`).
8. **[V] Native sources do not ship in the wheel.** No `__init__.py` under `omnidreams_singleview/`, no
   matching `package-data` (`integrations/omnidreams/pyproject.toml:123-141`), `/3rdparty/` gitignored, and
   the loader resolves sources via `_ROOT = Path(__file__).resolve().parents[2]`
   (`omnidreams_singleview.py:40`). `docs/source/models/omnidreams.rst:331-332` says so outright. **The
   native path works only from a source checkout / editable workspace install**; a future
   `pip install flashdreams-cmd[native]` from a wheel silently degrades to PyTorch. Must be stated in
   CMD's docs.
9. **[?] cuDNN discovery is install-shape-dependent.** `_python_package_dir("nvidia.cudnn")`
   (`omnidreams_singleview.py:479-486, 564`) selects `-l:libcudnn.so.9` vs plain `-lcudnn`
   (`:309-318`). omnidreams declares `nvidia-cudnn-cu13` only for win32 and relies on torch's transitive
   cuDNN on Linux. CMD's env must resolve the same cuDNN or link flags silently change.
10. **[?] Whether this port is worth it at all versus `quantization_plan.md`.** That route is already
    implemented, needs no CUTLASS, no JIT build, no `3rdparty/` sync, and no arch allowlist — and it
    measured 1.17-1.84x on compiled CMD-shaped GEMMs (`quantization_plan.md:141-158`, isolated layers, not
    a full network). The native route's advantage is FP8 KV caching and fused attention, which
    `torch._scaled_mm` cannot express — but **the size of that advantage on CMD is unmeasured and only
    Phase 6 can measure it.** If Phase 0 or Phase 4 stalls, the honest answer is to stop and keep the
    simpler route. This plan is worth executing precisely because Phases 0-3 make that decision cheaply and
    early rather than after the kernel work.

---

## 15. Repo-rule compliance checklist (applies to every phase)

- **[V] No architecture exception is needed** for `cmd -> flashdreams-native-cosmos` or
  `cmd -> omnidreams`. Precedent: `hy_worldplay -> wan22`
  (`integrations/hy_worldplay/pyproject.toml:39`, `hy_worldplay/_checkpoint.py:23`).
- **Would need an exception, therefore forbidden:** any `flashdreams/core` or `flashdreams/infra` module
  importing the native package, or the shared `CosmosTransformerConfig` importing omnidreams'
  `NativeAccelerationMode`. Use a plain `Literal` in `recipes/`. `grep -rn cpp_extension flashdreams/`
  returns nothing today — keep it that way.
- **Exactly one marker per test** (`ci_cpu` / `ci_gpu` / `manual`), enforced at collection by
  `flashdreams/_pytest_plugins/marker_enforcement.py:33-56`. Policy/contract tests → `ci_cpu`. Anything
  that launches a kernel → `ci_gpu` **plus** `_require_sm120a()`. Anything that triggers a cold ~164 s
  build outside the designated CI job → `manual`.
- **SPDX headers** on every new file; REUSE lint runs in `.github/workflows/reuse-lint.yml`.
- **`THIRD-PARTY-NOTICES:91-111`** documents the vendored third-party trees by their current path — any
  relocation in Phase 1 must update it (`skills/maintaining-oss-state`).
- **New workspace package checklist:** root `pyproject.toml` `[tool.pyright] extraPaths` (**:45**) and
  `[tool.ty.environment] extra-paths` (**:71**) are hand-maintained; `[tool.uv.workspace] members`
  `integrations/*` (**:9**) covers a top-level `integrations/<new>/`, a nested path needs an explicit entry
  (**:12**). `.github/scripts/sync_version.py:68-70` rglobs all `pyproject.toml`, so version sync is
  automatic. Add the row to `CONTRIBUTING.md:339-351`.
- `ruff` + `ty` via `uv run --group lint pre-commit run -a`; DCO sign-off on commits.