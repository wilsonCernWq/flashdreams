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

"""Runner for released CMD causal I2V and camera-control checkpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import (
    load_first_frame_tensor,
    read_image_rgb,
    resolve_input_path,
    resolve_prompt_value,
    runner_artifact_path,
    write_runner_stats,
)
from flashdreams.runtime.video_output import Mp4VideoOutputTarget

from .pipeline import CMDInferencePipeline, CMDInferencePipelineCache

DEFAULT_PROMPT = (
    "A grounded first-person game camera surfs smoothly along a slow, glassy "
    "water current through an ancient desert caravanserai just before sunrise. "
    "Near-field stonework creates clear parallax while the view glides toward "
    "a sunlit central courtyard with fluid, continuous motion."
)
"""Compact version of CMD's released non-camera example prompt."""

DEFAULT_CAMERA_PROMPT = (
    "A first-person view of a desolate arid settlement under a gloomy sky. "
    "A modern golden sports car is parked among rusted shacks and crumbling "
    "buildings while raindrops speckle the foreground."
)
"""Compact version of CMD's released camera-control example prompt."""

_RAW_EXAMPLE_ROOT = "https://raw.githubusercontent.com/nv-tlabs/cmd/main/examples"
DEFAULT_IMAGE_URL = f"{_RAW_EXAMPLE_ROOT}/image.png"
DEFAULT_CAMERA_IMAGE_URL = f"{_RAW_EXAMPLE_ROOT}/camera_image.png"
DEFAULT_CAMERA_URL = f"{_RAW_EXAMPLE_ROOT}/camera.npz"

CMD_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "cmd"
)
"""User-writable cache for CMD example inputs."""


@dataclass(kw_only=True)
class CMDRunnerConfig(RunnerConfig):
    """CLI configuration shared by all released CMD variants."""

    _target: type["CMDRunner"] = field(default_factory=lambda: CMDRunner)

    prompt: str | Path = DEFAULT_PROMPT
    """Inline prompt or path to a text file."""

    image_path: str | Path = DEFAULT_IMAGE_URL
    """First-frame RGB image as a local path or HTTP(S) URL."""

    camera_path: str | Path | None = None
    """CMD camera ``.npz`` containing ``target_w2c`` and intrinsics."""

    pixel_height: int = 480
    """Output video pixel height."""

    pixel_width: int = 832
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate used by released CMD checkpoints."""

    num_chunks: int = 20
    """Number of generated causal chunks, excluding the image prefix."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""


class CMDRunner(Runner[CMDRunnerConfig, CMDInferencePipeline]):
    """Run a complete CMD I2V rollout and write one MP4 artifact."""

    config: CMDRunnerConfig

    @property
    def total_latent_frames(self) -> int:
        """Total latent frames including CMD's independent image prefix."""
        transformer = self.pipeline._cmd_transformer_config
        return transformer.prefix_len_t + self.config.num_chunks * transformer.len_t

    def _load_image(self) -> torch.Tensor:
        """Resolve, resize, and normalize the first-frame image."""
        path = resolve_input_path(
            self.config.image_path,
            cache_dir=CMD_CACHE_DIR,
            validator=read_image_rgb,
        )
        return load_first_frame_tensor(
            path,
            pixel_height=self.config.pixel_height,
            pixel_width=self.config.pixel_width,
            device=self.pipeline.device,
            dtype=torch.bfloat16,
        )

    def _load_camera(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load and trim CMD's pixel-rate camera trajectory."""
        camera_path = self.config.camera_path
        if camera_path is None:
            if self.pipeline.camera_conditioned:
                raise ValueError(
                    "This CMD variant requires --camera-path with a camera .npz"
                )
            return None
        if not self.pipeline.camera_conditioned:
            raise ValueError("--camera-path is only valid for camera CMD variants")

        path = resolve_input_path(camera_path, cache_dir=CMD_CACHE_DIR)
        with np.load(path, allow_pickle=False) as payload:
            try:
                world_to_camera = np.asarray(
                    payload["target_w2c"],
                    dtype=np.float32,
                )
                intrinsics = np.asarray(
                    payload["target_intrinsics"],
                    dtype=np.float32,
                )
            except KeyError as error:
                raise ValueError(
                    "camera .npz must contain target_w2c and target_intrinsics"
                ) from error

        pixel_frames = 1 + (self.total_latent_frames - 1) * (
            self.pipeline.config.camera_frame_stride
        )
        if (
            world_to_camera.shape[0] < pixel_frames
            or intrinsics.shape[0] < pixel_frames
        ):
            raise ValueError(
                f"camera input needs {pixel_frames} pixel frames for "
                f"{self.total_latent_frames} latent frames"
            )
        camera_to_world = np.linalg.inv(world_to_camera[:pixel_frames]).astype(
            np.float32
        )
        return (
            torch.from_numpy(camera_to_world),
            torch.from_numpy(intrinsics[:pixel_frames]),
        )

    def _initialize_cache(self) -> CMDInferencePipelineCache:
        """Load rollout inputs and initialize all component caches."""
        if self.config.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        camera = self._load_camera()
        camera_to_world, intrinsics = camera if camera is not None else (None, None)
        return self.pipeline.initialize_cache(
            text=[resolve_prompt_value(self.config.prompt)],
            image=self._load_image(),
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            expected_latent_frames=(
                self.total_latent_frames if camera is not None else None
            ),
        )

    def run(self) -> None:
        """Generate all configured chunks and stream them into an MP4."""
        cache = self._initialize_cache()
        output_stream = self.create_video_output_stream(fps=self.config.fps)
        video_path = runner_artifact_path(
            self.config.output_dir,
            self.config.runner_name,
            "mp4",
        )
        output_target = Mp4VideoOutputTarget(
            output_path=video_path,
            fps=self.config.fps,
            output_layout=output_stream.output_layout,
            enabled=self.is_rank_zero,
        )
        output_target.open()
        for autoregressive_index in range(self.config.num_chunks):
            generated = self.pipeline.generate(autoregressive_index, cache)
            stats = self.pipeline.finalize(autoregressive_index, cache)
            output_target.write(
                output_stream.process(
                    generated,
                    autoregressive_index=autoregressive_index,
                    metrics=stats,
                )
            )
        tail = output_stream.finish()
        if tail is not None:
            output_target.write(tail)
        artifacts = output_target.close()
        if not artifacts:
            return

        artifact = artifacts[0]
        logger.info(
            "[{}] wrote {} latent frames to {}",
            self.config.runner_name,
            self.total_latent_frames,
            Path(artifact.uri).resolve(),
        )
        stats_history = artifact.metadata["stats_history"]
        if stats_history:
            stats_path = write_runner_stats(
                self.config.output_dir,
                self.config.runner_name,
                list(stats_history),
            )
            logger.info("[{}] wrote stats to {}", self.config.runner_name, stats_path)


__all__ = [
    "CMDRunner",
    "CMDRunnerConfig",
    "DEFAULT_CAMERA_IMAGE_URL",
    "DEFAULT_CAMERA_PROMPT",
    "DEFAULT_CAMERA_URL",
    "DEFAULT_IMAGE_URL",
    "DEFAULT_PROMPT",
]
