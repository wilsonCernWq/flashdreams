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

"""CMD launch capability for ``flashdreams-run``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.launch import LaunchMode, LaunchOptions, ResolvedLaunch

# V1 accepts no per-connection scenario overrides: session inputs are the
# runner config's own CLI-level prompt/image defaults. WebRTC sessions never
# replay a fixed camera trajectory (that's the offline `run` mode's job).
_WEBRTC_SCENARIO_FIELDS: frozenset[str] = frozenset()
_WEBRTC_OUTPUT_FIELDS = frozenset(
    {
        "host",
        "port",
        "fps",
        "video_height",
        "video_width",
        "warmup_chunks",
        "warmup_timeout_s",
        "client_liveness_timeout_s",
        "prefer_sw_encoder",
        "live_camera_intrinsics",
    }
)


class CMDLaunchCapability:
    """Expose an interactive WebRTC mode for CMD runner presets.

    CMD's existing offline ``run`` mode already covers batch/mp4 generation
    and is always available regardless of ``launch_capability``, so this only
    adds ``"webrtc"`` rather than replumbing ``run`` through the shared demo
    adapter machinery.
    """

    def supported_modes(
        self,
        config: RunnerConfig,
        options: LaunchOptions,
    ) -> tuple[LaunchMode, ...]:
        del config, options
        return ("webrtc",)

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: LaunchMode,
        options: LaunchOptions,
    ) -> ResolvedLaunch | None:
        if mode != "webrtc":
            return None
        _validate_fields("scenario", options.scenario, _WEBRTC_SCENARIO_FIELDS)
        _validate_fields("output", options.output, _WEBRTC_OUTPUT_FIELDS)
        return ResolvedLaunch(
            mode="webrtc",
            label="CMD WebRTC server",
            summary={
                "runner": config.runner_name,
                "mode": "webrtc",
                "device": config.device,
            },
            launch=partial(_launch, config=config, options=options),
        )


def _launch(*, config: RunnerConfig, options: LaunchOptions) -> object:
    from .runner import CMDRunnerConfig
    from .webrtc.serve import serve_cmd_webrtc

    if not isinstance(config, CMDRunnerConfig):
        raise TypeError(
            f"CMD launch capability requires CMDRunnerConfig, got {type(config).__name__}."
        )
    return serve_cmd_webrtc(config=config, options=options)


def _validate_fields(
    section: str,
    values: Mapping[str, object],
    allowed: frozenset[str],
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported CMD {section} fields: {', '.join(unknown)}.")


LAUNCH_CAPABILITY = CMDLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "CMDLaunchCapability"]
