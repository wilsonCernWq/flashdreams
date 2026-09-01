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

"""Shared rollout-input loading for CMD's offline and WebRTC runners."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from flashdreams.infra.runner_io import (
    load_first_frame_tensor,
    read_image_rgb,
    resolve_input_path,
)

CMD_CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "cmd"
)
"""User-writable cache for CMD example inputs."""


def resolve_total_latent_frames(
    *, prefix_len_t: int, len_t: int, num_chunks: int
) -> int:
    """Total latent frames including CMD's independent image prefix."""
    return prefix_len_t + num_chunks * len_t


def load_cmd_image(
    image_path: str | Path,
    *,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Resolve, resize, and normalize a CMD first-frame image."""
    path = resolve_input_path(
        image_path,
        cache_dir=CMD_CACHE_DIR,
        validator=read_image_rgb,
    )
    return load_first_frame_tensor(
        path,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        device=device,
        dtype=torch.bfloat16,
    )


def load_cmd_camera(
    camera_path: str | Path | None,
    *,
    camera_conditioned: bool,
    total_latent_frames: int,
    camera_frame_stride: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load and trim CMD's pixel-rate camera trajectory."""
    if camera_path is None:
        if camera_conditioned:
            raise ValueError(
                "This CMD variant requires --camera-path with a camera .npz"
            )
        return None
    if not camera_conditioned:
        raise ValueError("--camera-path is only valid for camera CMD variants")

    path = resolve_input_path(camera_path, cache_dir=CMD_CACHE_DIR)
    with np.load(path, allow_pickle=False) as payload:
        try:
            world_to_camera = np.asarray(payload["target_w2c"], dtype=np.float32)
            intrinsics = np.asarray(payload["target_intrinsics"], dtype=np.float32)
        except KeyError as error:
            raise ValueError(
                "camera .npz must contain target_w2c and target_intrinsics"
            ) from error

    pixel_frames = 1 + (total_latent_frames - 1) * camera_frame_stride
    if world_to_camera.shape[0] < pixel_frames or intrinsics.shape[0] < pixel_frames:
        raise ValueError(
            f"camera input needs {pixel_frames} pixel frames for "
            f"{total_latent_frames} latent frames"
        )
    camera_to_world = np.linalg.inv(world_to_camera[:pixel_frames]).astype(np.float32)
    return (
        torch.from_numpy(camera_to_world),
        torch.from_numpy(intrinsics[:pixel_frames]),
    )


__all__ = [
    "CMD_CACHE_DIR",
    "load_cmd_camera",
    "load_cmd_image",
    "resolve_total_latent_frames",
]
