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

"""Per-step camera-token slicing and live camera-token computation for CMD."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)

from .camera import _rays_to_camera_tokens, camera_rays, incremental_camera_conditioning


@dataclass(kw_only=True)
class CMDCameraEncoderConfig(EncoderConfig):
    """Config for selecting one generated camera-control chunk."""

    _target: type["CMDCameraEncoder"] = field(default_factory=lambda: CMDCameraEncoder)

    len_t: int = 1
    """Generated latent frames per AR chunk."""

    prefix_len_t: int = 1
    """Independent camera-prefix frames preceding generated chunks."""


class CMDCameraEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Select camera tokens for the current generated block."""

    config: CMDCameraEncoderConfig

    def __init__(self, config: CMDCameraEncoderConfig) -> None:
        super().__init__(config)
        self.config = config

    def initialize_autoregressive_cache(self) -> StreamingEncoderCache:
        """Build the stateless per-rollout encoder cache."""
        return StreamingEncoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> Tensor:
        """Return the camera-token slice for one AR step.

        Args:
            input: Full camera sequence shaped ``[..., T, C, H, W]``.
            autoregressive_index: Generated chunk index.
            cache: Stateless cache required by the streaming-encoder contract.

        Returns:
            Current chunk shaped ``[..., len_t, C, H, W]``.
        """
        assert cache is not None, "CMDCameraEncoder requires an encoder cache"
        start = self.config.prefix_len_t + autoregressive_index * self.config.len_t
        end = start + self.config.len_t
        if input.shape[-4] < end:
            raise ValueError(
                "Camera conditioning is too short for AR step "
                f"{autoregressive_index}: need {end}, got {input.shape[-4]} latent frames"
            )
        return input[..., start:end, :, :, :]


@dataclass(kw_only=True)
class CMDCamCtrlInput:
    """Per-AR-step live camera payload for :class:`CMDLiveCameraEncoder`."""

    poses: Tensor
    """This chunk's dense pixel-rate poses, shape ``[..., len_t * frame_stride,
    4, 4]`` -- no anchor/prefix frame included."""

    intrinsics: Tensor
    """``[..., 3, 3]`` or ``[..., len_t * frame_stride, 3, 3]``, aligned with
    ``poses``."""


@dataclass(kw_only=True)
class CMDLiveCameraEncoderConfig(EncoderConfig):
    """Config for computing camera-control tokens per AR step from live poses."""

    _target: type["CMDLiveCameraEncoder"] = field(
        default_factory=lambda: CMDLiveCameraEncoder
    )

    len_t: int = 1
    """Generated latent frames per AR chunk."""

    frame_stride: int = 4
    """Pixel-frames per latent-frame interval."""

    patch_size: int = 16
    """Spatial unshuffle factor aligned with the DiT token grid."""

    image_height: int = 480
    """Pixel-frame height."""

    image_width: int = 832
    """Pixel-frame width."""

    dtype: torch.dtype = torch.bfloat16
    """Output camera-token dtype."""

    base_intrinsics: tuple[float, float, float, float] | None = None
    """``(fx, fy, cx, cy)`` used for both per-step tokens (as a passthrough
    default) and the one-time identity-pose prefix token CMD's independent
    first-frame prefill needs -- see :meth:`CMDLiveCameraEncoder.prefix_camera_tokens`.
    Required; there is no live-session ``.npz`` calibration to fall back on."""


@dataclass(kw_only=True)
class CMDLiveCameraEncoderCache(StreamingEncoderCache):
    """Per-rollout cache carrying the block-relative anchor pose."""

    anchor_pose: Tensor
    """Raw absolute pose of the frame preceding the next chunk, ``[1, 4, 4]``."""


class CMDLiveCameraEncoder(StreamingEncoder[CMDLiveCameraEncoderCache]):
    """Compute camera-control tokens per AR step from live poses+intrinsics.

    Unlike :class:`CMDCameraEncoder`, which slices a precomputed whole-
    trajectory tensor, this encoder *computes* tokens fresh each step from a
    small per-step payload, carrying the block-relative anchor pose forward
    in its cache -- see ``incremental_camera_conditioning``.
    """

    config: CMDLiveCameraEncoderConfig

    def __init__(self, config: CMDLiveCameraEncoderConfig) -> None:
        super().__init__(config)
        if config.base_intrinsics is None:
            raise ValueError("CMDLiveCameraEncoderConfig requires base_intrinsics")
        self.config = config

    def initialize_autoregressive_cache(
        self, *, device: torch.device | None = None
    ) -> CMDLiveCameraEncoderCache:
        """Seed the anchor at identity: live sessions have no real prefix pose."""
        return CMDLiveCameraEncoderCache(
            anchor_pose=torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        )

    def forward(
        self,
        input: CMDCamCtrlInput,
        autoregressive_index: int = 0,
        cache: CMDLiveCameraEncoderCache | None = None,
    ) -> Tensor:
        """Return this AR step's camera tokens, advancing the anchor cache."""
        assert cache is not None, "CMDLiveCameraEncoder requires an encoder cache"
        del autoregressive_index
        tokens, new_anchor = incremental_camera_conditioning(
            input.poses,
            input.intrinsics,
            anchor_pose=cache.anchor_pose,
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            len_t=self.config.len_t,
            frame_stride=self.config.frame_stride,
            patch_size=self.config.patch_size,
            output_dtype=self.config.dtype,
        )
        cache.anchor_pose = new_anchor
        return tokens

    def prefix_camera_tokens(self, *, device: torch.device) -> Tensor:
        """Camera token for CMD's independent first-frame prefix.

        The prefix frame is always self-anchored (see
        ``camera.block_relative_poses``'s anchor rule, where frame 0 solves
        against its own pose), so its relative pose is always identity --
        this reproduces exactly what
        ``build_camera_conditioning(...)[..., :1, :, :, :]`` would give a
        fixed-trajectory rollout, without needing a real prefix pose to
        exist for a live session.

        Returns:
            ``[1, C, H/patch_size, W/patch_size]``, matching the shape
            :meth:`CMDCameraEncoder.forward` slices out of a precomputed
            ``camera_condition`` tensor for AR step 0's prefix.
        """
        assert self.config.base_intrinsics is not None  # validated in __init__
        fx, fy, cx, cy = self.config.base_intrinsics
        identity_pose = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        intrinsics = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        rays = camera_rays(
            identity_pose,
            intrinsics,
            image_height=self.config.image_height,
            image_width=self.config.image_width,
        )
        return _rays_to_camera_tokens(
            rays, patch_size=self.config.patch_size, output_dtype=self.config.dtype
        )
