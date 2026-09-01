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

"""CMD WebRTC runtime and session management.

CMD's WebRTC mode is always interactive and runs until the client
disconnects, matching lingbot: there is no fixed-trajectory camera replay
and no chunk-count cutoff here (that remains exclusive to the offline
``run`` mode -- see ``flashdreams_cmd.runner.CMDRunner``).

A resolved preset's own weights decide whether a session gets live
keyboard-driven (WASD) camera control: camera-conditioned presets
(``camera_dim is not None``) get it automatically; other presets stream a
fixed prompt/image rollout with no camera input at all. Either way the
prompt/image are the runner config's own CLI-level defaults, resolved once
per session.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from flashdreams.core.distributed.rank_orchestration import (
    PayloadBus,
    distributed_op,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.runner_io import resolve_prompt_value
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.inputs import (
    InferenceInput,
    UserInputCapability,
    UserInputSchema,
)
from flashdreams.runtime.mapping import IdentityInputMapping
from flashdreams.runtime.types import StepRequest, StepResult
from flashdreams.serving.webrtc.encoders import EncoderBackend
from flashdreams.serving.webrtc.manager import (
    DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
    BaseWebRTCSessionManager,
    WebRTCControlSignal,
)
from flashdreams.serving.webrtc.runtime import ThreadAffineDistributedWebRTCRuntime

from ..encoder import CMDCamCtrlInput, CMDLiveCameraEncoderConfig
from ..input_mapping import (
    FIELD_CAMERA_INTRINSICS,
    FIELD_CAMERA_POSES,
    CMDInputMapping,
    KeyboardToCameraCommand,
)
from ..inputs import load_cmd_image
from ..model_session import CMDModelSessionCore
from ..runner import DEFAULT_IMAGE_URL, DEFAULT_PROMPT

CMD_LIVE_WEBRTC_SOURCE_SCHEMA = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
    ),
    description="CMD live WASD camera control over WebRTC.",
)

_DEFAULT_LIVE_CAMERA_INTRINSICS: tuple[float, float, float, float] = (
    529.0876,
    529.1866,
    416.03174,
    239.36508,
)
"""``(fx, fy, cx, cy)`` read directly from CMD's shipped example
``camera.npz``'s ``target_intrinsics[0]`` (https://raw.githubusercontent.com/
nv-tlabs/cmd/main/examples/camera.npz), calibrated for the default 480x832
video geometry."""

_LIVE_CAMERA_INTRINSICS_REFERENCE_HEIGHT = 480
_LIVE_CAMERA_INTRINSICS_REFERENCE_WIDTH = 832
"""Resolution ``live_camera_intrinsics`` (default or operator-supplied) is
always defined against; rescaled to the runtime's actual
``video_height``/``video_width`` by :func:`_rescale_live_camera_intrinsics`
before use, matching lingbot's own ``_transform_intrinsics``."""


class CMDRuntimeError(RuntimeError):
    """Raised when the CMD WebRTC runtime is used incorrectly."""


@dataclass(slots=True)
class CMDRuntimeConfig:
    """Configuration for one CMD WebRTC runtime instance."""

    preset_id: str = "cmd-chunk4-short-i2v"
    pipeline_config: Any | None = None
    """Optional pre-resolved pipeline config; overrides ``preset_id`` lookup."""
    compile_network: bool = True
    seed: int = 22
    device: str = "cuda:0"
    video_height: int = 480
    video_width: int = 832
    fps: int = 16
    default_prompt: str | Path = DEFAULT_PROMPT
    default_image_path: str | Path = DEFAULT_IMAGE_URL
    warmup_chunks: int = 10
    """Loopback chunks to run at startup before accepting real clients, so
    torch.compile/autotune stalls happen once at server start instead of on
    the first real client's first chunks. 0 skips warmup entirely (not just
    a smaller one) -- matches lingbot's own default."""
    warmup_timeout_s: float = 600.0
    encoder_backend: EncoderBackend = "auto"
    encoder_bitrate_bps: int = 6_000_000
    encoder_gop: int = 30
    live_camera_intrinsics: tuple[float, float, float, float] | None = (
        _DEFAULT_LIVE_CAMERA_INTRINSICS
    )
    """``(fx, fy, cx, cy)`` for live camera control, required when the
    resolved preset is camera-conditioned. Defaults to the shipped example
    ``camera.npz``'s own ``target_intrinsics[0]`` (verified: fx=529.0876,
    fy=529.1866, cx=416.03174, cy=239.36508), calibrated for the
    ``_LIVE_CAMERA_INTRINSICS_REFERENCE_HEIGHT``/``_WIDTH`` (480x832)
    reference geometry -- automatically rescaled to the runtime's actual
    ``video_height``/``video_width`` (see
    :func:`_rescale_live_camera_intrinsics`), so overriding either an
    operator-supplied intrinsics value or the video resolution doesn't
    require manually recalibrating the other."""


def _cmd_pipeline_config(preset_id: str) -> Any:
    from flashdreams_cmd.config import CMD_CONFIGS

    if preset_id not in CMD_CONFIGS:
        supported = ", ".join(sorted(CMD_CONFIGS))
        raise ValueError(f"Unknown preset_id={preset_id!r}. Supported: {supported}")
    return CMD_CONFIGS[preset_id]


def _rescale_live_camera_intrinsics(
    intrinsics: tuple[float, float, float, float],
    *,
    video_height: int,
    video_width: int,
) -> tuple[float, float, float, float]:
    """Rescale ``(fx, fy, cx, cy)`` from the reference resolution
    ``live_camera_intrinsics`` was calibrated against to the runtime's
    actual output resolution.

    Mirrors lingbot's own ``_transform_intrinsics`` (with no crop term,
    since CMD has no separate resize-vs-final-size step -- lingbot's own
    call site for a plain default-intrinsics rescale always has
    ``height_resize == height_final`` and ``width_resize == width_final``
    too, which cancels its crop term the same way). Without this, overriding
    ``video_height``/``video_width`` away from the reference resolution
    used to silently feed the model geometrically wrong focal
    length/principal point, with no error.
    """
    scale_x = video_width / _LIVE_CAMERA_INTRINSICS_REFERENCE_WIDTH
    scale_y = video_height / _LIVE_CAMERA_INTRINSICS_REFERENCE_HEIGHT
    fx, fy, cx, cy = intrinsics
    return (fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y)


def _resolve_camera_pipeline_config(
    config: CMDRuntimeConfig, *, rank: int, world_size: int
) -> tuple[Any, bool]:
    """Resolve the (possibly camera-overlaid) pipeline config plus whether
    the preset is camera-conditioned, without constructing the real
    pipeline -- no CUDA or checkpoint access, safe to call from ``ci_cpu``
    tests. ``_initialize_sync`` calls ``pipeline_config.setup()`` on the
    returned config separately.
    """
    pipeline_config_base = config.pipeline_config
    if pipeline_config_base is None:
        pipeline_config_base = _cmd_pipeline_config(config.preset_id)

    transformer_config = pipeline_config_base.diffusion_model.transformer
    camera_conditioned = transformer_config.network.camera_dim is not None
    live_camera_intrinsics = config.live_camera_intrinsics
    if camera_conditioned and live_camera_intrinsics is None:
        raise CMDRuntimeError(
            f"{config.preset_id!r} is camera-conditioned; WebRTC sessions "
            "drive it with live camera control, which requires "
            "live_camera_intrinsics (fx, fy, cx, cy) -- a live session has "
            "no .npz calibration to fall back on."
        )

    diffusion_model_overrides: dict[str, Any] = {
        "seed": _rollout_seed(config.seed, rank=rank, world_size=world_size),
        "transformer": {"compile_network": config.compile_network},
    }
    if not camera_conditioned:
        return (
            derive_config(
                base_config=pipeline_config_base,
                diffusion_model=diffusion_model_overrides,
            ),
            False,
        )

    if transformer_config.prefix_len_t != 1:
        raise CMDRuntimeError(
            "live camera control assumes prefix_len_t == 1; got "
            f"{transformer_config.prefix_len_t}."
        )
    assert live_camera_intrinsics is not None  # validated above
    pipeline_config = derive_config(
        base_config=pipeline_config_base,
        encoder=CMDLiveCameraEncoderConfig(
            len_t=transformer_config.len_t,
            frame_stride=pipeline_config_base.camera_frame_stride,
            patch_size=pipeline_config_base.camera_patch_size,
            image_height=config.video_height,
            image_width=config.video_width,
            dtype=transformer_config.dtype,
            base_intrinsics=_rescale_live_camera_intrinsics(
                live_camera_intrinsics,
                video_height=config.video_height,
                video_width=config.video_width,
            ),
        ),
        diffusion_model=diffusion_model_overrides,
    )
    return pipeline_config, True


def _build_live_input_mapping(
    config: CMDRuntimeConfig, *, len_t: int, frame_stride: int
) -> CMDInputMapping:
    """Build the live-session ``CMDInputMapping``, with intrinsics correctly
    rescaled to the runtime's actual output resolution.

    ``len_t``/``frame_stride`` are already known before ``pipeline.setup()``
    -- identical to what :func:`_resolve_camera_pipeline_config` computes
    for the encoder config -- so this is CPU-testable without a real
    pipeline, unlike the ``decoder_ratio == camera_frame_stride`` sanity
    check next to its only call site, which does need one.
    """
    live_camera_intrinsics = config.live_camera_intrinsics
    assert (
        live_camera_intrinsics is not None
    )  # validated in _resolve_camera_pipeline_config
    fx, fy, cx, cy = _rescale_live_camera_intrinsics(
        live_camera_intrinsics,
        video_height=config.video_height,
        video_width=config.video_width,
    )
    return CMDInputMapping(
        fps=config.fps,
        base_intrinsics=torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        len_t=len_t,
        frame_stride=frame_stride,
    )


def _run_master_only_with_broadcast_failure(
    *, is_master: bool, load_fn: Callable[[], object]
) -> None:
    """Run ``load_fn`` only on the master rank, then broadcast its outcome to
    every rank, raising consistently everywhere it failed.

    A plain ``dist.barrier()`` after an unguarded master-only call has a real
    hang risk: if ``load_fn`` raises on rank 0 before reaching the barrier,
    every other rank -- which has nothing gating it beforehand -- is already
    waiting at the barrier and has no way to learn that rank 0 already
    failed and will never arrive, stalling the whole distributed server
    until the NCCL watchdog times out (~30 minutes by default). Broadcasting
    the outcome instead means rank 0 always reaches a collective call
    (whether ``load_fn`` succeeded or not), so the other ranks are always
    unblocked, and everyone raises the same clear error together instead of
    hanging.

    ``PayloadBus.broadcast_object`` itself falls back to returning its
    argument unchanged when no process group is initialized, so this is
    directly callable (and testable) outside a real distributed run too.

    OPEN QUESTION (not yet verified, single-GPU only so far): a live test
    with a failing ``--image-path`` showed the failure happening during
    ``on_startup``'s eager ``preload_runtime()`` call crashes the *entire*
    process before ``web.run_app()`` finishes starting -- there is no
    surviving server left for a client to retry against, single-GPU. The
    multi-GPU hang scenario this function was written for assumes rank 0's
    process specifically *survives* a failed ``load_fn()`` (via
    ``server.py``'s ``offer()`` handler catching ``Exception`` broadly) so a
    later reconnect's lazy ``preload_runtime()`` call can retry and hang
    against already-dead worker ranks. But ``offer()`` can only run once the
    server has finished starting, i.e. after ``on_startup`` has already
    returned successfully -- so if the very first ``_load_rollout_inputs_sync()``
    call (which happens unconditionally at the end of ``_initialize_sync_body``,
    inside ``_reset_rollout_sync``) is what's failing, rank 0 should crash at
    startup exactly like the single-GPU case, not survive to reach
    ``offer()`` at all. Haven't traced whether something differs under real
    ``torchrun`` (e.g. how non-zero ranks without an aiohttp app of their own
    behave, or whether ``preload_runtime()`` can ever fail *after* a
    successful first startup) that would make the originally-described hang
    reachable. Revisit with an actual multi-GPU repro before trusting that
    write-up at face value.
    """
    error_message: str | None = None
    if is_master:
        try:
            load_fn()
        except Exception as exc:
            error_message = str(exc)
    error_message = PayloadBus().broadcast_object(error_message)
    if error_message is not None:
        raise CMDRuntimeError(
            f"Rank 0 failed to warm the rollout-input cache: {error_message}"
        )


def _rollout_seed(base_seed: int, *, rank: int, world_size: int) -> int:
    """Offset ``base_seed`` by ``rank`` under CP so each rank draws a
    distinct RNG stream, matching ``Runner.offset_seed_by_global_rank``
    (``flashdreams.infra.runner``) and lingbot's WebRTC runtime -- neither
    of which this thread-affine runtime goes through directly."""
    return base_seed + rank if world_size > 1 else base_seed


class CMDInferenceRuntime(ThreadAffineDistributedWebRTCRuntime[CMDRuntimeConfig, None]):
    """Single-session CMD runtime: a fixed rollout streamed over WebRTC."""

    _STEADY_STATE_AR_PROBE_INDEX: int = 1000
    """AR index past CMD's transient AR-0 frame count (see the prefix-frame
    handling in ``CMDInferencePipeline.get_num_output_frames``)."""

    def __init__(self, config: CMDRuntimeConfig | None = None) -> None:
        super().__init__(
            config=config or CMDRuntimeConfig(),
            runtime_error_type=CMDRuntimeError,
            thread_name="cmd-webrtc-runtime",
        )
        self._pipeline: Any | None = None
        self._model_session: CMDModelSessionCore | None = None
        self._sync_step_lock = threading.Lock()
        self._input_mapping: CMDInputMapping | None = None
        self._input_canonicalizer: InputCanonicalizer | None = None

    async def start_inference_session(self) -> CMDWebRTCInferenceSession:
        """Return an ``InferenceSession`` view of the current rollout."""
        if self._closed:
            raise CMDRuntimeError("Runtime is closed.")
        if not self._is_runtime_initialized():
            raise CMDRuntimeError("Runtime is not initialized.")
        return CMDWebRTCInferenceSession(runtime=self)

    @property
    def input_mapping(self) -> IdentityInputMapping | CMDInputMapping:
        """Live camera intent for a camera-conditioned preset; fixed
        (no-op) conditioning otherwise."""
        if self._pipeline is None:
            raise CMDRuntimeError("Runtime input mapping is not initialized.")
        if self._input_mapping is None:
            return IdentityInputMapping()
        return self._input_mapping

    @property
    def input_canonicalizer(self) -> InputCanonicalizer:
        if self._pipeline is None:
            raise CMDRuntimeError("Runtime canonicalizer is not initialized.")
        if self._input_canonicalizer is None:
            return InputCanonicalizer()
        return self._input_canonicalizer

    @property
    def input_source_schema(self) -> UserInputSchema:
        if self._pipeline is None:
            raise CMDRuntimeError("Runtime input source schema is not initialized.")
        if self._input_mapping is None:
            return UserInputSchema()
        return CMD_LIVE_WEBRTC_SOURCE_SCHEMA

    def _next_step_request_sync(self) -> StepRequest:
        """Describe the next chunk. Sessions run until the client disconnects
        -- there is no chunk-count cutoff, matching lingbot."""
        if self._model_session is None:
            raise CMDRuntimeError("Runtime is not initialized.")
        return StepRequest(
            step_index=self._model_session.step_index,
            metadata={"input_frame_count": self._model_session.next_num_frames()},
        )

    def _step_blocking(self, inputs: InferenceInput) -> StepResult:
        """Run one step from synchronous ``InferenceSession`` code."""
        if self._closed:
            raise CMDRuntimeError("Session is closed.")
        with self._sync_step_lock:
            if self._closed:
                raise CMDRuntimeError("Session is closed.")
            return self._worker.call_blocking(self._step_sync_all_ranks, inputs)

    def _is_runtime_initialized(self) -> bool:
        return self._pipeline is not None and self._model_session is not None

    def _runtime_step_index(self) -> int:
        if self._model_session is None:
            raise CMDRuntimeError("Runtime is not initialized.")
        return self._model_session.step_index

    def _next_input_frame_count(self) -> int:
        # CMD never consumes live input frames.
        return 0

    def _steady_output_frame_count(self) -> int:
        """Return the steady-state per-chunk frame count.

        Master-only read with no distributed broadcast, matching the base
        class's ``peek_steady_output_num_frames`` contract.
        """
        if self._pipeline is None:
            raise CMDRuntimeError("Runtime is not initialized.")
        return int(
            self._pipeline.get_num_output_frames(self._STEADY_STATE_AR_PROBE_INDEX)
        )

    @distributed_op(WebRTCControlSignal.SESSION_STEP)
    def _step_sync_all_ranks(self, inputs: InferenceInput) -> StepResult:
        return self._step_sync(inputs)

    def _step_sync(self, inputs: InferenceInput) -> StepResult:
        if self._model_session is None:
            raise CMDRuntimeError("Runtime is not initialized.")
        step_index = self._model_session.step_index
        if self._input_mapping is None:
            del inputs  # Non-camera preset: no per-step model input.
            return self._model_session.step(metadata={"step": step_index})
        poses = _require_camera_tensor(inputs, FIELD_CAMERA_POSES)
        intrinsics = _require_camera_tensor(inputs, FIELD_CAMERA_INTRINSICS)
        camera_input = CMDCamCtrlInput(
            poses=poses.to(device=self._device, dtype=torch.float32),
            intrinsics=intrinsics.to(device=self._device, dtype=torch.float32),
        )
        return self._model_session.step(camera_input, metadata={"step": step_index})

    def _load_rollout_inputs_sync(self) -> torch.Tensor:
        """Resolve CMD's fixed first-frame image.

        WebRTC never replays a fixed camera trajectory (that's the offline
        ``run`` mode's job); camera conditioning for a camera-conditioned
        preset comes entirely from per-step keyboard input (see
        :meth:`_step_sync`).
        """
        if self._pipeline is None:
            raise CMDRuntimeError("Runtime is not initialized.")
        return load_cmd_image(
            self.config.default_image_path,
            pixel_height=self.config.video_height,
            pixel_width=self.config.video_width,
            device=self._device,
        )

    def _initialize_sync(self) -> None:
        if self._pipeline is not None:
            return
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the CMD WebRTC runtime.")

        try:
            self._initialize_sync_body()
        except BaseException:
            # Roll every partial side effect back to "never attempted" so a
            # retry (the shared manager's preload_runtime() calls
            # runtime.initialize() again on the next connection whenever
            # _is_runtime_initialized() is still False) actually re-runs the
            # work instead of silently no-op'ing. Without this, a failure
            # partway through used to leave self._pipeline/self._model_session
            # in a state the base class's _is_runtime_initialized() check
            # (both non-None) and this method's own top-of-function guard
            # (self._pipeline is not None) disagreed about -- depending on
            # exactly where the failure happened, retry would either not
            # redo the failed step (leaving the runtime permanently broken)
            # or, worse under multi-GPU, look fully initialized to both
            # checks and let preload_runtime() proceed straight to a real
            # dist.broadcast/dist.broadcast_object_list against a process
            # group whose other ranks already crashed -- reintroducing the
            # exact class of hang _run_master_only_with_broadcast_failure
            # exists to prevent, just one step later.
            self._pipeline = None
            self._model_session = None
            self._input_canonicalizer = None
            self._input_mapping = None
            raise

    def _initialize_sync_body(self) -> None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        pipeline_config, camera_conditioned = _resolve_camera_pipeline_config(
            self.config, rank=self.rank, world_size=world_size
        )
        self._pipeline = pipeline_config.setup().to(device=self._device)

        if camera_conditioned:
            decoder_ratio = self._pipeline.decoder.temporal_compression_ratio
            if decoder_ratio != self._pipeline.config.camera_frame_stride:
                raise CMDRuntimeError(
                    "live camera control assumes camera_frame_stride == "
                    "decoder.temporal_compression_ratio; got "
                    f"{self._pipeline.config.camera_frame_stride} != {decoder_ratio}."
                )
            self._input_canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
            self._input_mapping = _build_live_input_mapping(
                self.config,
                len_t=self._pipeline._cmd_transformer_config.len_t,
                frame_stride=self._pipeline.config.camera_frame_stride,
            )

        self._model_session = CMDModelSessionCore(
            pipeline=self._pipeline,
            output_stream_factory=lambda: VideoOutputStream(
                postprocess_stream=None,
                output_layout="tchw",
            ),
        )

        if dist.is_initialized():
            # `_load_rollout_inputs_sync` fetches remote defaults through
            # `resolve_input_path`, which has no cross-process file lock, and
            # every rank calls it with identical paths. Warm the cache from
            # rank 0 before every rank loads for real below.
            _run_master_only_with_broadcast_failure(
                is_master=self.is_master, load_fn=self._load_rollout_inputs_sync
            )

        self._reset_rollout_sync()
        self._initialize_video_encoder_sync()

    def _reset_rollout_sync(self, session_input: None = None) -> None:
        del session_input
        if self._pipeline is None or self._model_session is None:
            raise CMDRuntimeError("Runtime pipeline is not initialized.")
        # TODO: unlike _initialize_sync's call to the same method, this one
        # has no _run_master_only_with_broadcast_failure protection. Left
        # unguarded on purpose: by the time any reset runs, _initialize_sync
        # already warmed the local cache, so single-node multi-GPU deploys
        # (our only deployment model today) never re-hit the network here.
        # Revisit if we ever deploy across multiple nodes (each with its own
        # local cache) -- only `is_master` (global rank 0) is protected, not
        # a per-node leader, so a genuinely multi-node deploy could still
        # race unprotected downloads on every non-rank-0 node.
        image = self._load_rollout_inputs_sync()
        self._model_session.reset(
            prompt=resolve_prompt_value(self.config.default_prompt),
            image=image,
            camera_to_world=None,
            intrinsics=None,
            expected_latent_frames=None,
        )
        if self._input_mapping is not None:
            self._input_mapping.reset()
        if self._input_canonicalizer is not None:
            # Otherwise a client that disconnects mid-keypress (dropped tab,
            # network loss -- no guaranteed key_up) leaves KeyboardToCameraCommand
            # holding that key, and the *next* client's session silently starts
            # already moving on frame 1 despite no input of its own.
            self._input_canonicalizer.reset()

    def _close_sync(self) -> None:
        model_session = self._model_session
        pipeline = self._pipeline
        self._model_session = None
        self._pipeline = None
        if model_session is not None:
            model_session.close()
        if pipeline is not None:
            del pipeline
        if self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _generate_one_chunk_sync(
        self,
        *,
        segments: list[Any],
        frame_times: list[float],
    ) -> StepResult:
        # CMD defines `start_inference_session` above, so the shared manager
        # always drives it through `_step_sync` and this raw segment path
        # (built for live-input models such as LingBot) is unreachable. Raise
        # loudly rather than silently stepping with no meaningful segments.
        del segments, frame_times
        raise CMDRuntimeError(
            "CMD has no live input; sessions must drive through "
            "start_inference_session(), not the raw segment-stepping path."
        )


def _require_camera_tensor(inputs: InferenceInput, name: str) -> torch.Tensor:
    """Return one required per-step camera tensor from a mapped ``InferenceInput``."""
    if name not in inputs.step:
        raise CMDRuntimeError(
            f"CMD live step inputs are missing {name!r}; the selected input "
            "mapping must produce it for every step."
        )
    value = inputs.step[name]
    if not isinstance(value, torch.Tensor):
        raise CMDRuntimeError(f"CMD live step input {name!r} must be a tensor.")
    return value


class CMDWebRTCInferenceSession:
    """``InferenceSession`` view of a live CMD WebRTC rollout.

    The rollout is owned by :class:`CMDInferenceRuntime`; this only adapts it
    to the runtime-API stepping surface so the shared manager can drive it.
    """

    def __init__(self, *, runtime: CMDInferenceRuntime) -> None:
        self._runtime = runtime

    def next_step_request(self) -> StepRequest | None:
        return self._runtime._next_step_request_sync()

    def step(self, inputs: InferenceInput) -> StepResult:
        return self._runtime._step_blocking(inputs)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        raise CMDRuntimeError(
            "Reset a CMD WebRTC rollout through the runtime's session "
            "lifecycle, not through the inference session."
        )

    def close(self) -> None:
        # The runtime outlives the session and is closed by the serve loop.
        return None


def create_cmd_webrtc_session_manager(
    *,
    runtime: CMDInferenceRuntime | None = None,
    runtime_config: CMDRuntimeConfig | None = None,
    fps: int | None = None,
    client_liveness_timeout_s: float = DEFAULT_CLIENT_LIVENESS_TIMEOUT_S,
) -> BaseWebRTCSessionManager[CMDInferenceRuntime, CMDRuntimeConfig]:
    """Configure the shared WebRTC manager for the CMD runtime."""
    runtime_config = runtime_config or getattr(runtime, "config", None)
    if not isinstance(runtime_config, CMDRuntimeConfig):
        runtime_config = CMDRuntimeConfig()
    fps = runtime_config.fps if fps is None else fps
    if fps <= 0:
        raise ValueError("fps must be > 0")
    runtime = runtime or CMDInferenceRuntime(config=runtime_config)
    return BaseWebRTCSessionManager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=fps,
        identity=runtime_config.preset_id,
        busy_message="A CMD session is already active.",
        warmup_label="CMD WebRTC",
        client_liveness_timeout_s=client_liveness_timeout_s,
    )


__all__ = [
    "CMDInferenceRuntime",
    "CMDRuntimeConfig",
    "CMDRuntimeError",
    "CMDWebRTCInferenceSession",
    "create_cmd_webrtc_session_manager",
]
