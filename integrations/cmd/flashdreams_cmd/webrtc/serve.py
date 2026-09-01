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

"""Typed ``flashdreams-run <cmd-preset> webrtc`` launch implementation."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from flashdreams.runtime.demo import WebRTCAppResources, WebRTCOutputSpec
from flashdreams.runtime.demo.bootstrap import (
    configure_logging,
    initialize_cuda_distributed,
)
from flashdreams.serving.launch import LaunchOptions
from flashdreams.serving.webrtc.demo import serve_webrtc_demo

from ..runner import CMDRunnerConfig
from ..transformer import CMDTransformerConfig
from .session import (
    CMDInferenceRuntime,
    CMDRuntimeConfig,
    create_cmd_webrtc_session_manager,
)


def serve_cmd_webrtc(*, config: CMDRunnerConfig, options: LaunchOptions) -> object:
    """Serve one CMD runner preset as an interactive WebRTC session.

    Sessions run until the client disconnects, matching lingbot: there is no
    fixed camera-trajectory replay and no chunk-count cutoff here (that
    remains exclusive to the offline ``run`` mode). The prompt and
    first-frame image are the runner config's own CLI-level defaults, read
    once at server start and reused for every connecting session. A
    camera-conditioned preset gets live WASD camera control automatically;
    other presets stream with no camera input.
    """
    configure_logging()
    context = initialize_cuda_distributed(default_device=str(config.device))

    transformer_config = config.pipeline.diffusion_model.transformer
    assert isinstance(transformer_config, CMDTransformerConfig)

    runtime_config = CMDRuntimeConfig(
        preset_id=config.pipeline.name,
        pipeline_config=config.pipeline,
        device=str(context.device),
        # config.pipeline already carries the operator's --pipeline.diffusion-model.seed/
        # --pipeline.diffusion-model.transformer.compile-network CLI overrides;
        # CMDRuntimeConfig has its own separate seed/compile_network fields
        # that _resolve_camera_pipeline_config's diffusion_model override
        # reads from instead of pipeline_config_base, so without this they
        # silently stayed at CMDRuntimeConfig's hardcoded defaults (22/True)
        # regardless of what the operator passed on the CLI.
        compile_network=transformer_config.compile_network,
        default_prompt=config.prompt,
        default_image_path=config.image_path,
        **_seed_kwargs(config.pipeline.diffusion_model.seed),
        **_warmup_chunks_kwargs(options.output),
        warmup_timeout_s=_float_value(
            options.output.get("warmup_timeout_s", 600.0), name="warmup_timeout_s"
        ),
        **_webrtc_video_kwargs(
            options.output,
            default_height=config.pixel_height,
            default_width=config.pixel_width,
            default_fps=config.fps,
        ),
        **_encoder_backend_kwargs(
            options.output, prefer_sw_encoder=options.prefer_sw_encoder
        ),
        **_live_camera_intrinsics_kwargs(options.output),
    )
    runtime = CMDInferenceRuntime(config=runtime_config)
    manager = create_cmd_webrtc_session_manager(
        runtime=runtime,
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        client_liveness_timeout_s=_float_value(
            options.output.get("client_liveness_timeout_s", 30.0),
            name="client_liveness_timeout_s",
        ),
    )
    output = WebRTCOutputSpec(
        host=str(options.host or options.output.get("host", "0.0.0.0")),
        port=_int_value(
            options.port
            if options.port is not None
            else options.output.get("port", 8080),
            name="port",
        ),
        fps=runtime_config.fps,
        video_width=runtime_config.video_width,
        video_height=runtime_config.video_height,
        warmup_chunks=runtime_config.warmup_chunks,
        warmup_timeout_s=runtime_config.warmup_timeout_s,
        preload_name="CMD",
    )
    return serve_webrtc_demo(
        output=output,
        model_id="flashdreams-cmd",
        session_manager=manager,
        app_resources=WebRTCAppResources(
            model_web_resource=files("flashdreams_cmd.webrtc").joinpath("web"),
            preload_name="CMD",
        ),
        world_rank=context.world_rank,
    )


def _int_value(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"{name} must be convertible to int, got {type(value).__name__}.")


def _float_value(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"{name} must be convertible to float, got {type(value).__name__}.")


def _bool_value(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    raise TypeError(f"{name} must be convertible to bool, got {value!r}.")


def _camera_intrinsics_value(
    value: object,
) -> tuple[float, float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        fx, fy, cx, cy = (
            _float_value(component, name="live_camera_intrinsics component")
            for component in value
        )
        return (fx, fy, cx, cy)
    raise TypeError(
        "live_camera_intrinsics must be a 4-element [fx, fy, cx, cy] sequence, "
        f"got {value!r}."
    )


def _seed_kwargs(seed: int | None) -> dict[str, Any]:
    """Build the ``seed`` kwarg only when the resolved pipeline config
    actually set one.

    The base ``DiffusionModelConfig.seed`` field defaults to ``None``
    (meaning "no fixed seed"); CMD's own presets always set ``seed=22``
    explicitly, so ``None`` here is a defensive fallback rather than an
    expected runtime case. Omitting the kwarg when unset lets
    ``CMDRuntimeConfig``'s own default apply instead of clobbering it,
    mirroring ``_live_camera_intrinsics_kwargs``/``_warmup_chunks_kwargs``.
    """
    if seed is None:
        return {}
    return {"seed": seed}


def _warmup_chunks_kwargs(output: Mapping[str, object]) -> dict[str, Any]:
    """Build the ``warmup_chunks`` kwarg only when the caller set it.

    ``CMDRuntimeConfig.warmup_chunks`` already defaults to a real warmup (not
    0, which skips it entirely); hardcoding a separate ``0`` fallback here
    would silently clobber that default on every real CLI launch that didn't
    set ``--output.warmup-chunks``, exactly like the earlier
    ``live_camera_intrinsics`` bug this mirrors.
    """
    if "warmup_chunks" not in output:
        return {}
    return {"warmup_chunks": _int_value(output["warmup_chunks"], name="warmup_chunks")}


def _webrtc_video_kwargs(
    output: Mapping[str, object],
    *,
    default_height: int,
    default_width: int,
    default_fps: int,
) -> dict[str, Any]:
    """Resolve fps/video_height/video_width from ``--output.*`` overrides.

    Falls back to the runner config's own CLI-level values when unset.
    Caught live: these previously validated as recognized output fields
    (``CMDLaunchCapability._WEBRTC_OUTPUT_FIELDS``) but were never actually
    read, so setting them silently did nothing.
    """
    return {
        "video_height": _int_value(
            output.get("video_height", default_height), name="video_height"
        ),
        "video_width": _int_value(
            output.get("video_width", default_width), name="video_width"
        ),
        "fps": _int_value(output.get("fps", default_fps), name="fps"),
    }


def _encoder_backend_kwargs(
    output: Mapping[str, object], *, prefer_sw_encoder: bool
) -> dict[str, Any]:
    """Resolve the encoder backend from ``--output.prefer-sw-encoder`` (or the
    top-level ``--prefer-sw-encoder`` flag), matching lingbot's own mapping
    (``lingbot/runtime.py``: ``"default" if prefer_sw_encoder else "auto"``).
    """
    prefer_sw = _bool_value(
        output.get("prefer_sw_encoder", prefer_sw_encoder), name="prefer_sw_encoder"
    )
    return {"encoder_backend": "default" if prefer_sw else "auto"}


def _live_camera_intrinsics_kwargs(output: Mapping[str, object]) -> dict[str, Any]:
    """Build the ``live_camera_intrinsics`` kwarg only when the caller set it.

    ``CMDRuntimeConfig.live_camera_intrinsics`` already defaults to CMD's
    verified shipped calibration; explicitly passing ``None`` through here
    for an absent ``--output.live-camera-intrinsics`` would silently clobber
    that default with every ``CMDRuntimeConfig(...)`` construction.
    """
    if "live_camera_intrinsics" not in output:
        return {}
    return {
        "live_camera_intrinsics": _camera_intrinsics_value(
            output["live_camera_intrinsics"]
        )
    }


__all__ = ["serve_cmd_webrtc"]
