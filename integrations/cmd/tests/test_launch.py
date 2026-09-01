# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-safe checks for CMD's WebRTC launch capability and runtime plumbing.

These stop at construction and pure logic -- no CUDA, no `sageattention`/
`aiortc` network activity, no checkpoint downloads. GPU/manual coverage for
an actual live session lives outside this suite (see
`integrations/cmd/docs/sage_attention_plan.md`'s sibling WebRTC test-plan
notes in the implementation plan, not in this file).
"""

from __future__ import annotations

import sys

import pytest
import torch
from flashdreams_cmd.config import RUNNER_CONFIGS
from flashdreams_cmd.inputs import load_cmd_camera, resolve_total_latent_frames
from flashdreams_cmd.launch import LAUNCH_CAPABILITY
from flashdreams_cmd.runner import CMDRunnerConfig

from flashdreams.serving.launch import LaunchOptions, available_launch_modes

pytestmark = pytest.mark.ci_cpu

_RUNNER_CONFIG = RUNNER_CONFIGS["cmd-chunk4-short-i2v"]


def test_default_runner_config_points_at_cmd_launch_capability() -> None:
    assert isinstance(_RUNNER_CONFIG, CMDRunnerConfig)
    assert (
        _RUNNER_CONFIG.launch_capability == "flashdreams_cmd.launch:LAUNCH_CAPABILITY"
    )


def test_supported_modes_add_webrtc_alongside_the_builtin_run_mode() -> None:
    options = LaunchOptions()
    assert LAUNCH_CAPABILITY.supported_modes(_RUNNER_CONFIG, options) == ("webrtc",)
    assert available_launch_modes(_RUNNER_CONFIG, options) == ("run", "webrtc")


def test_resolve_rejects_unsupported_mode() -> None:
    assert (
        LAUNCH_CAPABILITY.resolve(_RUNNER_CONFIG, mode="mp4", options=LaunchOptions())
        is None
    )


def test_resolve_rejects_unknown_scenario_and_output_fields() -> None:
    with pytest.raises(ValueError, match="scenario fields"):
        LAUNCH_CAPABILITY.resolve(
            _RUNNER_CONFIG,
            mode="webrtc",
            options=LaunchOptions(scenario={"prompt": "hello"}),
        )
    with pytest.raises(ValueError, match="output fields"):
        LAUNCH_CAPABILITY.resolve(
            _RUNNER_CONFIG,
            mode="webrtc",
            options=LaunchOptions(output={"bogus_field": 1}),
        )


def test_resolve_accepts_recognized_webrtc_output_fields() -> None:
    resolved = LAUNCH_CAPABILITY.resolve(
        _RUNNER_CONFIG,
        mode="webrtc",
        options=LaunchOptions(host="0.0.0.0", port=9000, output={"fps": 16}),
    )
    assert resolved is not None
    assert resolved.mode == "webrtc"
    assert resolved.summary["runner"] == _RUNNER_CONFIG.runner_name
    assert resolved.summary["mode"] == "webrtc"


def test_resolve_accepts_live_camera_intrinsics_output_field() -> None:
    resolved = LAUNCH_CAPABILITY.resolve(
        _RUNNER_CONFIG,
        mode="webrtc",
        options=LaunchOptions(
            output={"live_camera_intrinsics": [416.0, 416.0, 208.0, 120.0]}
        ),
    )
    assert resolved is not None


def test_serve_helpers_parse_bool_output_values() -> None:
    from flashdreams_cmd.webrtc.serve import _bool_value

    assert _bool_value(True, name="x") is True
    assert _bool_value("true", name="x") is True
    assert _bool_value("false", name="x") is False
    with pytest.raises(TypeError):
        _bool_value("not-a-bool", name="x")


def test_serve_helpers_parse_camera_intrinsics_output_values() -> None:
    from flashdreams_cmd.webrtc.serve import _camera_intrinsics_value

    assert _camera_intrinsics_value([416.0, 416.0, 208.0, 120.0]) == (
        416.0,
        416.0,
        208.0,
        120.0,
    )
    with pytest.raises(TypeError):
        _camera_intrinsics_value([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        _camera_intrinsics_value("not-a-sequence")


def test_runtime_config_default_warmup_chunks_is_nonzero() -> None:
    """Regression test: warmup_chunks=0 doesn't run a smaller warmup, it
    skips warmup entirely (flashdreams/serving/webrtc/warmup.py returns
    immediately on num_chunks==0), so the first real client used to pay the
    full torch.compile/autotune stall live. Default now matches lingbot's
    own default (10) instead of 0.

    Only checks the dataclass default, not that warmup actually runs -- the
    real mechanism (flashdreams.serving.webrtc.warmup.run_loopback_warmup_session)
    has its own coverage at flashdreams/tests/test_webrtc_warmup.py."""
    from flashdreams_cmd.webrtc.session import CMDRuntimeConfig

    assert CMDRuntimeConfig().warmup_chunks == 10


def test_warmup_chunks_kwargs_regression_omits_key_when_unset() -> None:
    """Regression test: serve_cmd_webrtc previously always passed
    warmup_chunks=0 to CMDRuntimeConfig(...) regardless of its own default,
    clobbering it on every real CLI launch that didn't set
    --output.warmup-chunks -- the same bug class as the earlier
    live_camera_intrinsics clobbering fix."""
    from flashdreams_cmd.webrtc.serve import _warmup_chunks_kwargs
    from flashdreams_cmd.webrtc.session import CMDRuntimeConfig

    assert _warmup_chunks_kwargs({}) == {}
    config = CMDRuntimeConfig(**_warmup_chunks_kwargs({}))
    assert config.warmup_chunks == CMDRuntimeConfig().warmup_chunks

    assert _warmup_chunks_kwargs({"warmup_chunks": 3}) == {"warmup_chunks": 3}


def test_seed_kwargs_omits_key_when_unset() -> None:
    """Regression test: serve_cmd_webrtc previously never forwarded
    config.pipeline.diffusion_model.seed into CMDRuntimeConfig at all --
    CMDRuntimeConfig.seed always stayed at its own hardcoded default (22)
    regardless of --pipeline.diffusion-model.seed, because
    _resolve_camera_pipeline_config's diffusion_model override reads from
    CMDRuntimeConfig.seed, not from the already-CLI-resolved
    pipeline_config_base. Same bug class as warmup_chunks/live_camera_intrinsics."""
    from flashdreams_cmd.webrtc.serve import _seed_kwargs
    from flashdreams_cmd.webrtc.session import CMDRuntimeConfig

    assert _seed_kwargs(None) == {}
    config = CMDRuntimeConfig(**_seed_kwargs(None))
    assert config.seed == CMDRuntimeConfig().seed

    assert _seed_kwargs(1234) == {"seed": 1234}
    assert CMDRuntimeConfig(**_seed_kwargs(1234)).seed == 1234


def test_serve_cmd_webrtc_regression_seed_and_compile_network_are_cli_reachable() -> (
    None
):
    """Regression test for the actual attribute paths serve_cmd_webrtc reads
    (not just the pure _seed_kwargs helper): every shipped CMD preset's
    resolved pipeline config must have a CMDTransformerConfig transformer
    (so the isinstance narrowing serve.py relies on doesn't crash) with a
    real compile_network value, and a non-None diffusion_model.seed by
    default (so _seed_kwargs's None branch is only a defensive fallback, not
    the common case)."""
    from flashdreams_cmd.transformer import CMDTransformerConfig

    for name, runner_config in RUNNER_CONFIGS.items():
        transformer_config = runner_config.pipeline.diffusion_model.transformer
        assert isinstance(transformer_config, CMDTransformerConfig), name
        assert isinstance(transformer_config.compile_network, bool)
        assert runner_config.pipeline.diffusion_model.seed is not None, name


def test_webrtc_video_kwargs_falls_back_to_runner_defaults_when_unset() -> None:
    from flashdreams_cmd.webrtc.serve import _webrtc_video_kwargs

    assert _webrtc_video_kwargs(
        {}, default_height=480, default_width=832, default_fps=16
    ) == {"video_height": 480, "video_width": 832, "fps": 16}


def test_webrtc_video_kwargs_regression_output_overrides_actually_apply() -> None:
    """Regression test: --output.fps/video-height/video-width previously
    validated as recognized fields but were silently dropped -- serve_cmd_webrtc
    always sourced these from the offline runner's CLI config instead of
    options.output, so the flags had zero effect."""
    from flashdreams_cmd.webrtc.serve import _webrtc_video_kwargs

    assert _webrtc_video_kwargs(
        {"fps": 24, "video_height": 720, "video_width": 1280},
        default_height=480,
        default_width=832,
        default_fps=16,
    ) == {"video_height": 720, "video_width": 1280, "fps": 24}


def test_encoder_backend_kwargs_regression_prefer_sw_encoder_actually_applies() -> None:
    """Regression test: --prefer-sw-encoder / --output.prefer-sw-encoder
    validated but was never read, so encoder_backend always stayed 'auto'.
    Mapping matches lingbot's own (lingbot/runtime.py): 'default' when software
    is preferred, 'auto' otherwise."""
    from flashdreams_cmd.webrtc.serve import _encoder_backend_kwargs

    assert _encoder_backend_kwargs({}, prefer_sw_encoder=False) == {
        "encoder_backend": "auto"
    }
    assert _encoder_backend_kwargs({}, prefer_sw_encoder=True) == {
        "encoder_backend": "default"
    }
    # --output.prefer-sw-encoder (manifest-level) takes precedence over the
    # top-level --prefer-sw-encoder flag, matching lingbot's own precedence.
    assert _encoder_backend_kwargs(
        {"prefer_sw_encoder": True}, prefer_sw_encoder=False
    ) == {"encoder_backend": "default"}


def test_live_camera_intrinsics_kwargs_omits_key_when_unset() -> None:
    """Regression test: an absent --output.live-camera-intrinsics must NOT
    override CMDRuntimeConfig's own default with an explicit None.

    Caught live: an earlier version always passed `live_camera_intrinsics=`
    to `CMDRuntimeConfig(...)`, which silently clobbered the dataclass
    default with `None` on every real CLI launch that didn't set the flag,
    even though the class-level default looked correct in isolation.
    """
    from flashdreams_cmd.webrtc.serve import _live_camera_intrinsics_kwargs
    from flashdreams_cmd.webrtc.session import (
        _DEFAULT_LIVE_CAMERA_INTRINSICS,
        CMDRuntimeConfig,
    )

    assert _live_camera_intrinsics_kwargs({}) == {}
    config = CMDRuntimeConfig(**_live_camera_intrinsics_kwargs({}))
    assert config.live_camera_intrinsics == _DEFAULT_LIVE_CAMERA_INTRINSICS

    overridden = _live_camera_intrinsics_kwargs(
        {"live_camera_intrinsics": [1.0, 2.0, 3.0, 4.0]}
    )
    assert overridden == {"live_camera_intrinsics": (1.0, 2.0, 3.0, 4.0)}


def test_rollout_seed_passes_base_seed_through_on_a_single_rank() -> None:
    from flashdreams_cmd.webrtc.session import _rollout_seed

    assert _rollout_seed(22, rank=0, world_size=1) == 22


def test_rollout_seed_offsets_by_rank_under_context_parallelism() -> None:
    from flashdreams_cmd.webrtc.session import _rollout_seed

    assert _rollout_seed(22, rank=0, world_size=4) == 22
    assert _rollout_seed(22, rank=3, world_size=4) == 25


def test_importing_launch_module_does_not_pull_in_webrtc_runtime() -> None:
    """`import flashdreams_cmd.launch` must stay aiortc-free.

    `_launch` only imports `.webrtc.serve` (and thus aiortc-dependent shared
    infra) lazily, inside the function body, so a plain CLI `--help`/
    `--no-instantiate` invocation never needs the `sage`/`serving` extras.
    """
    for module_name in (
        "flashdreams_cmd.webrtc.session",
        "flashdreams_cmd.webrtc.serve",
        "aiortc",
    ):
        sys.modules.pop(module_name, None)

    sys.modules.pop("flashdreams_cmd.launch", None)
    import flashdreams_cmd.launch  # noqa: F401

    assert "flashdreams_cmd.webrtc.session" not in sys.modules
    assert "flashdreams_cmd.webrtc.serve" not in sys.modules


def test_resolve_total_latent_frames_matches_released_preset_geometry() -> None:
    # cmd-chunk4-short-i2v: prefix 1 + 5 chunks * 4 latent frames = 21.
    assert resolve_total_latent_frames(prefix_len_t=1, len_t=4, num_chunks=5) == 21


def test_load_cmd_camera_returns_none_for_non_camera_variant_without_a_path() -> None:
    assert (
        load_cmd_camera(
            None,
            camera_conditioned=False,
            total_latent_frames=21,
            camera_frame_stride=4,
        )
        is None
    )


def test_load_cmd_camera_requires_a_path_for_camera_conditioned_variants() -> None:
    with pytest.raises(ValueError, match="requires --camera-path"):
        load_cmd_camera(
            None,
            camera_conditioned=True,
            total_latent_frames=21,
            camera_frame_stride=4,
        )


def test_load_cmd_camera_rejects_a_path_for_non_camera_variants() -> None:
    with pytest.raises(ValueError, match="only valid for camera CMD variants"):
        load_cmd_camera(
            "camera.npz",
            camera_conditioned=False,
            total_latent_frames=21,
            camera_frame_stride=4,
        )


class _StubModelSession:
    """Fake `CMDModelSessionCore` seam: only what `_next_step_request_sync`
    and `_runtime_step_index` need, without a real pipeline."""

    def __init__(self, *, step_index: int, num_frames: int = 4) -> None:
        self.step_index = step_index
        self._num_frames = num_frames

    def next_num_frames(self) -> int:
        return self._num_frames


def _cpu_runtime():
    from flashdreams_cmd.webrtc.session import CMDInferenceRuntime, CMDRuntimeConfig

    return CMDInferenceRuntime(config=CMDRuntimeConfig(device="cpu"))


def test_runtime_constructs_on_cpu_without_cuda() -> None:
    runtime = _cpu_runtime()
    assert runtime._is_runtime_initialized() is False


def test_initialize_sync_rolls_back_partial_state_on_failure() -> None:
    """Regression test: a failure partway through _initialize_sync used to
    leave self._pipeline/self._model_session (and the input mapping/
    canonicalizer built alongside them) partially set. Depending on exactly
    where the failure happened, a retry (the shared manager calls
    runtime.initialize() again on the next connection attempt whenever
    _is_runtime_initialized() is still False) would either silently no-op
    -- this class's own top-of-function guard (self._pipeline is not None)
    already short-circuiting while the base class's _is_runtime_initialized()
    (both fields non-None) still reported "not ready" -- or, worse, look
    fully initialized to both checks and let the caller proceed straight
    into real work (under multi-GPU, a distributed collective against a
    process group whose other ranks had already crashed and exited,
    hanging until the NCCL/Gloo watchdog timeout -- exactly the class of
    hang _run_master_only_with_broadcast_failure exists to prevent, just
    reintroduced one step later). Every partial side effect must be rolled
    back so a retry genuinely re-attempts from scratch."""
    runtime = _cpu_runtime()

    def _boom() -> None:
        runtime._pipeline = object()  # type: ignore[assignment]
        runtime._model_session = object()  # type: ignore[assignment]
        runtime._input_canonicalizer = object()  # type: ignore[assignment]
        runtime._input_mapping = object()  # type: ignore[assignment]
        raise RuntimeError("simulated failure partway through initialization")

    runtime._initialize_sync_body = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="simulated failure"):
        runtime._initialize_sync()

    assert runtime._pipeline is None
    assert runtime._model_session is None
    assert runtime._input_canonicalizer is None
    assert runtime._input_mapping is None
    assert runtime._is_runtime_initialized() is False

    # A subsequent retry (e.g. the next connection attempt) must actually
    # re-attempt the work, not silently no-op because of leftover state.
    calls: list[None] = []

    def _record() -> None:
        calls.append(None)

    runtime._initialize_sync_body = _record  # type: ignore[method-assign]
    runtime._initialize_sync()
    assert calls == [None]


def test_input_mapping_properties_raise_before_initialization() -> None:
    runtime = _cpu_runtime()
    with pytest.raises(RuntimeError, match="input mapping is not initialized"):
        _ = runtime.input_mapping
    with pytest.raises(RuntimeError, match="canonicalizer is not initialized"):
        _ = runtime.input_canonicalizer
    with pytest.raises(RuntimeError, match="input source schema is not initialized"):
        _ = runtime.input_source_schema


def test_runtime_uses_fixed_conditioning_once_initialized_with_no_camera_mapping() -> (
    None
):
    from flashdreams.runtime.canonical import InputCanonicalizer
    from flashdreams.runtime.inputs import UserInputSchema
    from flashdreams.runtime.mapping import IdentityInputMapping

    runtime = _cpu_runtime()
    runtime._pipeline = object()  # type: ignore[assignment]  # Simulate a resolved non-camera preset.
    assert isinstance(runtime.input_mapping, IdentityInputMapping)
    assert isinstance(runtime.input_canonicalizer, InputCanonicalizer)
    assert isinstance(runtime.input_source_schema, UserInputSchema)
    assert runtime._next_input_frame_count() == 0


def test_next_step_request_sync_always_reports_the_next_chunk() -> None:
    """WebRTC sessions run until the client disconnects -- no chunk-count
    cutoff, matching lingbot (which never checks a chunk count at all)."""
    runtime = _cpu_runtime()
    runtime._model_session = _StubModelSession(step_index=0)  # type: ignore[assignment]

    request = runtime._next_step_request_sync()
    assert request.step_index == 0
    assert request.metadata["input_frame_count"] == 4

    runtime._model_session.step_index = 1000  # type: ignore[union-attr]
    far_request = runtime._next_step_request_sync()
    assert far_request.step_index == 1000


def test_generate_one_chunk_sync_is_unreachable_for_cmd() -> None:
    """CMD always drives through `start_inference_session`; the raw
    segment-stepping path (built for live-input models) must never run."""
    runtime = _cpu_runtime()
    with pytest.raises(RuntimeError, match="no live input"):
        runtime._generate_one_chunk_sync(segments=[], frame_times=[])


def test_create_session_manager_hooks_are_wired() -> None:
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeConfig,
        create_cmd_webrtc_session_manager,
    )

    manager = create_cmd_webrtc_session_manager(
        runtime_config=CMDRuntimeConfig(device="cpu", fps=12)
    )
    assert manager.busy_message == "A CMD session is already active."
    assert manager.warmup_label == "CMD WebRTC"
    assert manager.fps == 12


def test_runtime_config_has_live_camera_intrinsics_default() -> None:
    from flashdreams_cmd.webrtc.session import CMDRuntimeConfig

    config = CMDRuntimeConfig()
    # Verified against CMD's own shipped example camera.npz (target_intrinsics[0]).
    assert config.live_camera_intrinsics == (529.0876, 529.1866, 416.03174, 239.36508)


def _live_cpu_runtime(*, intrinsics: tuple[float, float, float, float] | None = None):
    from flashdreams_cmd.webrtc.session import CMDInferenceRuntime, CMDRuntimeConfig

    return CMDInferenceRuntime(
        config=CMDRuntimeConfig(
            device="cpu",
            # A camera-conditioned preset: live camera control is automatic
            # for these, not a separate opt-in flag.
            preset_id="cmd-chunk4-camera-i2v",
            live_camera_intrinsics=intrinsics,
        )
    )


def test_initialize_sync_requires_live_camera_intrinsics_for_a_camera_conditioned_preset() -> (
    None
):
    runtime = _live_cpu_runtime(intrinsics=None)
    with pytest.raises(RuntimeError, match="requires live_camera_intrinsics"):
        runtime._initialize_sync()


def test_resolve_camera_pipeline_config_wires_live_encoder_for_camera_conditioned_preset() -> (
    None
):
    """Regression test: _initialize_sync's camera-conditioning auto-detection
    and encoder-overlay construction previously had zero test coverage --
    the only test calling _initialize_sync hit an early guard before any of
    this wiring ran. This exercises the pure-Python construction directly,
    without needing CUDA/.setup()."""
    from flashdreams_cmd.encoder import CMDLiveCameraEncoderConfig
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeConfig,
        _resolve_camera_pipeline_config,
    )

    config = CMDRuntimeConfig(
        device="cpu",
        preset_id="cmd-chunk4-camera-i2v",
        live_camera_intrinsics=(416.0, 416.0, 208.0, 120.0),
        seed=22,
    )
    pipeline_config, camera_conditioned = _resolve_camera_pipeline_config(
        config, rank=0, world_size=1
    )

    assert camera_conditioned is True
    encoder_config = pipeline_config.encoder
    assert isinstance(encoder_config, CMDLiveCameraEncoderConfig)
    assert encoder_config.len_t == 4  # cmd-chunk4-camera-i2v's transformer len_t
    assert encoder_config.frame_stride == pipeline_config.camera_frame_stride
    assert encoder_config.patch_size == pipeline_config.camera_patch_size
    assert encoder_config.image_height == config.video_height
    assert encoder_config.image_width == config.video_width
    assert encoder_config.base_intrinsics == config.live_camera_intrinsics
    assert pipeline_config.diffusion_model.seed == 22
    assert pipeline_config.diffusion_model.transformer.compile_network is True


def test_rescale_live_camera_intrinsics_is_identity_at_the_reference_resolution() -> (
    None
):
    from flashdreams_cmd.webrtc.session import _rescale_live_camera_intrinsics

    intrinsics = (529.0876, 529.1866, 416.03174, 239.36508)
    assert _rescale_live_camera_intrinsics(
        intrinsics, video_height=480, video_width=832
    ) == pytest.approx(intrinsics)


def test_rescale_live_camera_intrinsics_regression_scales_with_resolution() -> None:
    """Regression test: live_camera_intrinsics used to be applied raw,
    unscaled, regardless of video_height/video_width -- unlike lingbot's
    equivalent, which always rescales via _transform_intrinsics. Overriding
    the output resolution away from the 480x832 reference used to silently
    feed the model wrong focal length/principal point with no error."""
    from flashdreams_cmd.webrtc.session import _rescale_live_camera_intrinsics

    fx, fy, cx, cy = _rescale_live_camera_intrinsics(
        (529.0876, 529.1866, 416.03174, 239.36508),
        video_height=960,  # 2x the 480 reference
        video_width=416,  # 0.5x the 832 reference
    )
    assert fx == pytest.approx(529.0876 * 0.5)
    assert cx == pytest.approx(416.03174 * 0.5)
    assert fy == pytest.approx(529.1866 * 2.0)
    assert cy == pytest.approx(239.36508 * 2.0)


def test_resolve_camera_pipeline_config_rescales_intrinsics_for_non_reference_resolution() -> (
    None
):
    from flashdreams_cmd.encoder import CMDLiveCameraEncoderConfig
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeConfig,
        _resolve_camera_pipeline_config,
    )

    config = CMDRuntimeConfig(
        device="cpu",
        preset_id="cmd-chunk4-camera-i2v",
        live_camera_intrinsics=(529.0876, 529.1866, 416.03174, 239.36508),
        video_height=960,
        video_width=416,
    )
    pipeline_config, _ = _resolve_camera_pipeline_config(config, rank=0, world_size=1)
    encoder_config = pipeline_config.encoder
    assert isinstance(encoder_config, CMDLiveCameraEncoderConfig)
    assert encoder_config.base_intrinsics == pytest.approx(
        (529.0876 * 0.5, 529.1866 * 2.0, 416.03174 * 0.5, 239.36508 * 2.0)
    )


def test_build_live_input_mapping_regression_rescales_intrinsics() -> None:
    """Regression test: unlike _resolve_camera_pipeline_config's rescale call
    (covered above), the second _rescale_live_camera_intrinsics call site --
    building CMDInputMapping's per-step base_intrinsics inside
    _initialize_sync_body -- previously had zero test coverage proving the
    rescaled (not raw) value reached it: every test constructing
    CMDInputMapping directly hand-wrote raw, unscaled reference-resolution
    matrices. Extracted into _build_live_input_mapping so this is directly
    testable on CPU (len_t/frame_stride are already known pre-pipeline.setup(),
    identical to what _resolve_camera_pipeline_config computes)."""
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeConfig,
        _build_live_input_mapping,
    )

    config = CMDRuntimeConfig(
        device="cpu",
        live_camera_intrinsics=(529.0876, 529.1866, 416.03174, 239.36508),
        video_height=960,
        video_width=416,
    )
    mapping = _build_live_input_mapping(config, len_t=4, frame_stride=4)

    expected = torch.tensor(
        [
            [529.0876 * 0.5, 0.0, 416.03174 * 0.5],
            [0.0, 529.1866 * 2.0, 239.36508 * 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(mapping._base_intrinsics, expected)


def test_resolve_camera_pipeline_config_offsets_seed_by_rank_under_cp() -> None:
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeConfig,
        _resolve_camera_pipeline_config,
    )

    config = CMDRuntimeConfig(
        device="cpu",
        preset_id="cmd-chunk4-camera-i2v",
        live_camera_intrinsics=(416.0, 416.0, 208.0, 120.0),
        seed=22,
    )
    pipeline_config, _ = _resolve_camera_pipeline_config(config, rank=3, world_size=4)
    assert pipeline_config.diffusion_model.seed == 25


def test_resolve_camera_pipeline_config_leaves_non_camera_preset_without_live_encoder() -> (
    None
):
    from flashdreams_cmd.encoder import CMDLiveCameraEncoderConfig
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeConfig,
        _resolve_camera_pipeline_config,
    )

    config = CMDRuntimeConfig(device="cpu", preset_id="cmd-chunk4-short-i2v")
    pipeline_config, camera_conditioned = _resolve_camera_pipeline_config(
        config, rank=0, world_size=1
    )

    assert camera_conditioned is False
    assert not isinstance(pipeline_config.encoder, CMDLiveCameraEncoderConfig)


def test_run_master_only_with_broadcast_failure_raises_on_every_rank() -> None:
    """Regression test: a plain dist.barrier() after an unguarded master-only
    asset load used to leave non-master ranks hung indefinitely (until the
    NCCL watchdog timeout) if rank 0's load raised, since nothing told them
    rank 0 already failed and would never reach the barrier. Every rank must
    instead raise the same clear error. PayloadBus.broadcast_object falls
    back to identity when no process group is initialized, so this is
    directly exercisable outside a real distributed run: is_master=True
    covers rank 0's own path, is_master=False covers a rank that never even
    attempts the load.

    TODO: this does NOT prove a non-master rank actually reads the
    broadcast error message rather than its own local (always-None) state
    -- doing that would need mocking PayloadBus.broadcast_object itself.
    Skipped for now: this is real, working distributed-coordination
    infrastructure with no evidence of being flaky, so chasing this
    specific coverage gap isn't worth it unless we actually see a
    regression here."""
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeError,
        _run_master_only_with_broadcast_failure,
    )

    def _boom() -> None:
        raise RuntimeError("simulated rank-0 fetch failure")

    with pytest.raises(CMDRuntimeError, match="simulated rank-0 fetch failure"):
        _run_master_only_with_broadcast_failure(is_master=True, load_fn=_boom)

    calls: list[None] = []

    def _record() -> None:
        calls.append(None)

    _run_master_only_with_broadcast_failure(is_master=True, load_fn=_record)
    assert calls == [None]

    # A non-master rank never calls load_fn at all.
    _run_master_only_with_broadcast_failure(is_master=False, load_fn=_boom)


def test_resolve_camera_pipeline_config_rejects_camera_preset_without_intrinsics() -> (
    None
):
    from flashdreams_cmd.webrtc.session import (
        CMDRuntimeConfig,
        CMDRuntimeError,
        _resolve_camera_pipeline_config,
    )

    config = CMDRuntimeConfig(
        device="cpu", preset_id="cmd-chunk4-camera-i2v", live_camera_intrinsics=None
    )
    with pytest.raises(CMDRuntimeError, match="requires live_camera_intrinsics"):
        _resolve_camera_pipeline_config(config, rank=0, world_size=1)


def test_input_mapping_properties_return_live_variants_once_configured() -> None:
    from flashdreams.runtime.canonical import InputCanonicalizer
    from flashdreams_cmd.input_mapping import CMDInputMapping, KeyboardToCameraCommand
    from flashdreams_cmd.webrtc.session import CMD_LIVE_WEBRTC_SOURCE_SCHEMA

    runtime = _live_cpu_runtime(intrinsics=(416.0, 416.0, 208.0, 120.0))
    # Simulate what `_initialize_sync` builds, without a real pipeline/CUDA.
    runtime._pipeline = object()  # type: ignore[assignment]
    runtime._input_canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    runtime._input_mapping = CMDInputMapping(
        fps=16,
        base_intrinsics=torch.tensor(
            [[416.0, 0.0, 208.0], [0.0, 416.0, 120.0], [0.0, 0.0, 1.0]]
        ),
        len_t=4,
        frame_stride=4,
    )

    assert runtime.input_mapping is runtime._input_mapping
    assert runtime.input_canonicalizer is runtime._input_canonicalizer
    assert runtime.input_source_schema is CMD_LIVE_WEBRTC_SOURCE_SCHEMA


class _StubModelSessionRecordingStep(_StubModelSession):
    """Extends the base stub to record what `.step()` was called with."""

    def __init__(self, *, step_index: int = 0, num_frames: int = 16) -> None:
        super().__init__(step_index=step_index, num_frames=num_frames)
        self.last_camera_input: object = None

    def reset(self, **kwargs: object) -> None:
        del kwargs

    def step(self, camera_input=None, *, metadata=None):
        self.last_camera_input = camera_input
        from flashdreams.runtime import StepResult

        return StepResult(step_index=self.step_index, frame_count=self._num_frames)


def test_reset_rollout_sync_also_clears_canonicalizer_held_key_state() -> None:
    """Regression test: an earlier version's ``_reset_rollout_sync`` only
    reset ``self._input_mapping`` (the pose integrator), never
    ``self._input_canonicalizer`` (``KeyboardToCameraCommand``'s held-key
    state). ``CMDInferenceRuntime`` is long-lived across many sequential
    WebRTC sessions; a client that disconnects mid-keypress (dropped tab,
    network loss -- no guaranteed ``key_up``) would leave that key held, and
    the *next* client's session would silently start moving on frame 1
    despite receiving no input of its own.

    Caught live: confirmed by numerically reproducing this exact scenario
    against the real reset path -- a second "session" with zero real events
    still moved, until the canonicalizer was also reset.
    """
    from flashdreams.runtime import (
        InferenceInput,
        InputCanonicalizer,
        StepRequest,
        TimeWindow,
        UserInputEvent,
        UserInputs,
    )
    from flashdreams_cmd.input_mapping import (
        FIELD_CAMERA_POSES,
        CMDInputMapping,
        KeyboardToCameraCommand,
    )
    from flashdreams_cmd.webrtc.session import CMD_LIVE_WEBRTC_SOURCE_SCHEMA

    runtime = _live_cpu_runtime(intrinsics=(416.0, 416.0, 208.0, 120.0))
    runtime._pipeline = object()  # type: ignore[assignment]
    runtime._model_session = _StubModelSessionRecordingStep()  # type: ignore[assignment]
    runtime._load_rollout_inputs_sync = lambda: torch.zeros(1, 3, 4, 4)  # type: ignore[method-assign]

    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    runtime._input_canonicalizer = canonicalizer
    runtime._input_mapping = CMDInputMapping(
        fps=16,
        base_intrinsics=torch.tensor(
            [[416.0, 0.0, 208.0], [0.0, 416.0, 120.0], [0.0, 0.0, 1.0]]
        ),
        len_t=4,
        frame_stride=4,
    )

    # Simulate a client holding 'w' with no matching key_up (a dropped tab).
    dt = 1.0 / 16
    window = TimeWindow(start_s=0.0, end_s=16 * dt)
    canonicalizer.canonicalize(
        UserInputs(
            events=(
                UserInputEvent(
                    timestamp_s=0.0, event_type="key_down", payload={"key": "w"}
                ),
            )
        ),
        window=window,
        source_schema=CMD_LIVE_WEBRTC_SOURCE_SCHEMA,
    )

    runtime._reset_rollout_sync()

    # A brand-new session with zero real events must show zero motion.
    step_inputs = runtime._input_mapping.map_step_inputs(
        canonical_inputs=runtime._input_canonicalizer.canonicalize(
            UserInputs(), window=window, source_schema=CMD_LIVE_WEBRTC_SOURCE_SCHEMA
        ),
        inference_input=InferenceInput(),
        request=StepRequest(step_index=0, user_input_window=window),
    )
    poses = step_inputs.step[FIELD_CAMERA_POSES]
    torch.testing.assert_close(poses[0], torch.eye(4))
    torch.testing.assert_close(poses[-1], torch.eye(4))


def test_step_sync_live_mode_requires_camera_fields() -> None:
    from flashdreams.runtime import InferenceInput
    from flashdreams_cmd.webrtc.session import CMDRuntimeError

    runtime = _live_cpu_runtime(intrinsics=(416.0, 416.0, 208.0, 120.0))
    runtime._model_session = _StubModelSessionRecordingStep()  # type: ignore[assignment]
    runtime._input_mapping = object()  # type: ignore[assignment]  # Simulate a built live mapping.
    with pytest.raises(CMDRuntimeError, match="camera_poses"):
        runtime._step_sync(InferenceInput())


def test_step_sync_live_mode_builds_cam_ctrl_input_from_step_fields() -> None:
    from flashdreams.runtime import InferenceInput
    from flashdreams_cmd.encoder import CMDCamCtrlInput

    runtime = _live_cpu_runtime(intrinsics=(416.0, 416.0, 208.0, 120.0))
    session = _StubModelSessionRecordingStep()
    runtime._model_session = session  # type: ignore[assignment]
    runtime._input_mapping = object()  # type: ignore[assignment]  # Simulate a built live mapping.
    poses = torch.eye(4).repeat(16, 1, 1)
    intrinsics = torch.eye(3).repeat(16, 1, 1)
    inputs = InferenceInput(
        step={"camera_poses": poses, "camera_intrinsics": intrinsics}
    )

    runtime._step_sync(inputs)

    assert isinstance(session.last_camera_input, CMDCamCtrlInput)
    torch.testing.assert_close(session.last_camera_input.poses, poses)
    torch.testing.assert_close(session.last_camera_input.intrinsics, intrinsics)


def test_step_sync_fixed_mode_ignores_inputs_and_matches_prior_behavior() -> None:
    from flashdreams.runtime import InferenceInput

    runtime = _cpu_runtime()
    session = _StubModelSessionRecordingStep()
    runtime._model_session = session  # type: ignore[assignment]

    runtime._step_sync(InferenceInput())

    assert session.last_camera_input is None
