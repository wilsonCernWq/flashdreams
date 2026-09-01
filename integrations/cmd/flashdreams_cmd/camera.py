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

"""Camera-ray conditioning for CMD's Cosmos latent grid."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor

CAMERA_FEATURE_DIM = 6
"""Ray-origin plus unit-direction channels per pixel."""


def camera_frame_indices(
    num_pixel_frames: int,
    frame_stride: int,
    *,
    device: torch.device,
) -> Tensor:
    """Select pixel-frame cameras aligned with compressed latent frames."""
    if num_pixel_frames <= 0:
        raise ValueError("num_pixel_frames must be positive")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if (num_pixel_frames - 1) % frame_stride:
        raise ValueError(
            "Camera sequence length must equal 1 + k * frame_stride; got "
            f"{num_pixel_frames} frames with stride {frame_stride}"
        )
    return torch.arange(
        0,
        num_pixel_frames,
        frame_stride,
        device=device,
        dtype=torch.long,
    )


def block_relative_poses(camera_to_world: Tensor, block_size: int) -> Tensor:
    """Express each latent camera relative to its preceding block boundary."""
    if camera_to_world.ndim < 3 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError(
            "camera_to_world must have shape [..., T, 4, 4]; got "
            f"{tuple(camera_to_world.shape)}"
        )
    if camera_to_world.shape[-3] == 0:
        raise ValueError("camera_to_world must contain at least one frame")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    frame_indices = torch.arange(
        camera_to_world.shape[-3],
        device=camera_to_world.device,
        dtype=torch.long,
    )
    anchor_indices = (
        torch.div(
            torch.clamp(frame_indices - 1, min=0),
            block_size,
            rounding_mode="floor",
        )
        * block_size
    )
    anchors = camera_to_world.index_select(-3, anchor_indices)
    return torch.linalg.solve(anchors.float(), camera_to_world.float())


def _sample_intrinsics(
    intrinsics: Tensor,
    frame_indices: Tensor,
    num_pixel_frames: int,
    batch_shape: tuple[int, ...],
) -> Tensor:
    matrix_shape = batch_shape + (3, 3)
    sequence_shape = batch_shape + (num_pixel_frames, 3, 3)
    singleton_shape = batch_shape + (1, 3, 3)
    if intrinsics.shape == matrix_shape:
        return intrinsics.unsqueeze(-3).expand(
            *batch_shape, frame_indices.numel(), 3, 3
        )
    if intrinsics.shape == singleton_shape:
        return intrinsics.expand(*batch_shape, frame_indices.numel(), 3, 3)
    if intrinsics.shape != sequence_shape:
        raise ValueError(
            "intrinsics must have shape [..., 3, 3] or [..., T, 3, 3]; got "
            f"{tuple(intrinsics.shape)}"
        )
    return intrinsics.index_select(-3, frame_indices)


def camera_rays(
    camera_to_world: Tensor,
    intrinsics: Tensor,
    *,
    image_height: int,
    image_width: int,
) -> Tensor:
    """Build world-space ray origins and directions as ``[..., T, H, W, 6]``."""
    if camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must end in [T, 4, 4]")
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end in [T, 3, 3]")
    if camera_to_world.shape[:-2] != intrinsics.shape[:-2]:
        raise ValueError("camera poses and intrinsics must share batch and time dims")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")

    poses = camera_to_world.float()
    calibration = intrinsics.to(device=poses.device, dtype=torch.float32)
    focal_x = calibration[..., 0, 0]
    focal_y = calibration[..., 1, 1]
    if torch.any(focal_x <= 0) or torch.any(focal_y <= 0):
        raise ValueError("camera focal lengths must be positive")

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(image_height, device=poses.device, dtype=torch.float32) + 0.5,
        torch.arange(image_width, device=poses.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    direction_x = (pixel_x - calibration[..., 0, 2, None, None]) / focal_x[
        ..., None, None
    ]
    direction_y = (pixel_y - calibration[..., 1, 2, None, None]) / focal_y[
        ..., None, None
    ]
    camera_direction = torch.stack(
        (direction_x, direction_y, torch.ones_like(direction_x)), dim=-1
    )
    camera_direction = torch.nn.functional.normalize(camera_direction, dim=-1)

    ray_direction = torch.einsum(
        "...tij,...thwj->...thwi",
        poses[..., :3, :3],
        camera_direction,
    )
    ray_origin = poses[..., :3, 3][..., None, None, :].expand_as(ray_direction)
    return torch.cat((ray_origin, ray_direction), dim=-1)


def _rays_to_camera_tokens(
    rays: Tensor,
    *,
    patch_size: int,
    output_dtype: torch.dtype,
) -> Tensor:
    """Patchify per-pixel ray channels into DiT-token-grid camera tokens."""
    return rearrange(
        rays,
        "... t (h ph) (w pw) c -> ... t (c ph pw) h w",
        ph=patch_size,
        pw=patch_size,
    ).to(dtype=output_dtype)


def build_camera_conditioning(
    camera_to_world: Tensor,
    intrinsics: Tensor,
    *,
    image_height: int,
    image_width: int,
    frame_stride: int = 4,
    patch_size: int = 16,
    block_size: int = 1,
    expected_latent_frames: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Build patch-flattened block-relative camera tokens.

    Args:
        camera_to_world: Pixel-frame poses shaped ``[..., T, 4, 4]``.
        intrinsics: One or per-frame calibration matrix shaped
            ``[..., 3, 3]`` or ``[..., T, 3, 3]``.
        image_height: Pixel-frame height.
        image_width: Pixel-frame width.
        frame_stride: Pixel frames per latent-frame interval.
        patch_size: Spatial unshuffle factor aligned with the DiT token grid.
        block_size: Generated latent frames per causal block.
        expected_latent_frames: Optional exact latent-camera count.
        output_dtype: Output camera-token dtype.

    Returns:
        Camera tokens shaped ``[..., T_latent, 6 * patch_size**2,
        H/patch_size, W/patch_size]``.
    """
    if camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must end in [T, 4, 4]")
    if image_height % patch_size or image_width % patch_size:
        raise ValueError(
            f"image size {(image_height, image_width)} must divide patch_size={patch_size}"
        )

    num_pixel_frames = camera_to_world.shape[-3]
    frame_indices = camera_frame_indices(
        num_pixel_frames,
        frame_stride,
        device=camera_to_world.device,
    )
    if (
        expected_latent_frames is not None
        and frame_indices.numel() != expected_latent_frames
    ):
        raise ValueError(
            f"Camera sequence produces {frame_indices.numel()} latent frames; "
            f"expected {expected_latent_frames}"
        )
    sampled_poses = camera_to_world.index_select(-3, frame_indices)
    sampled_intrinsics = _sample_intrinsics(
        intrinsics.to(device=camera_to_world.device),
        frame_indices,
        num_pixel_frames,
        camera_to_world.shape[:-3],
    )
    rays = camera_rays(
        block_relative_poses(sampled_poses, block_size),
        sampled_intrinsics,
        image_height=image_height,
        image_width=image_width,
    )
    return _rays_to_camera_tokens(
        rays, patch_size=patch_size, output_dtype=output_dtype
    )


def incremental_camera_conditioning(
    camera_to_world: Tensor,
    intrinsics: Tensor,
    *,
    anchor_pose: Tensor,
    image_height: int,
    image_width: int,
    len_t: int,
    frame_stride: int = 4,
    patch_size: int = 16,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor]:
    """Build one AR chunk's camera tokens incrementally, carrying the block anchor.

    Reproduces exactly what a full :func:`build_camera_conditioning` call
    would produce for this chunk's latent frames, without needing the whole
    trajectory materialized upfront. Assumes ``prefix_len_t == 1`` (true for
    all six released CMD presets); callers must assert this explicitly.

    Args:
        camera_to_world: This chunk's newly-arrived, dense pixel-rate poses,
            shaped ``[..., len_t * frame_stride, 4, 4]`` -- exactly the
            frames generated in this AR step, no anchor/prefix frame
            included.
        intrinsics: ``[..., 3, 3]`` or ``[..., len_t * frame_stride, 3, 3]``,
            aligned with ``camera_to_world``.
        anchor_pose: ``[..., 1, 4, 4]`` raw absolute pose of the frame
            immediately preceding this chunk -- either the previous chunk's
            returned ``new_anchor_pose``, or, for AR step 0, the true prefix
            frame's raw pose (reproduces the fixed-trajectory/batch path
            exactly) or a caller-chosen virtual origin such as
            ``torch.eye(4)`` (live control, where no real prefix camera pose
            exists).
        image_height: Pixel-frame height.
        image_width: Pixel-frame width.
        len_t: Generated latent frames per AR/causal block -- must match the
            transformer's own ``len_t``. Call boundaries must align with
            block boundaries: unlike :func:`build_camera_conditioning`
            (which derives block boundaries from the whole trajectory this
            call's caller doesn't have), this function has no way to detect
            a mismatch on its own, so it's enforced explicitly here instead
            of silently producing wrong tokens for every frame from the next
            AR step onward.
        frame_stride: Pixel frames per latent-frame interval; must match
            ``build_camera_conditioning``'s and equal the decoder's
            temporal compression ratio.
        patch_size: Spatial unshuffle factor aligned with the DiT token grid.
        output_dtype: Output camera-token dtype.

    Returns:
        ``(tokens, new_anchor_pose)``: tokens shaped ``[..., len_t, 6 *
        patch_size**2, H/patch_size, W/patch_size]``; ``new_anchor_pose``
        shaped ``[..., 1, 4, 4]``, to pass as ``anchor_pose`` for the next
        chunk.
    """
    if camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must end in [T, 4, 4]")
    if anchor_pose.shape[-3:] != (1, 4, 4):
        raise ValueError("anchor_pose must have shape [..., 1, 4, 4]")
    expected_new_frames = len_t * frame_stride
    if camera_to_world.shape[-3] != expected_new_frames:
        raise ValueError(
            "camera_to_world must contain exactly len_t * frame_stride = "
            f"{expected_new_frames} new pixel frames (one AR block); got "
            f"{camera_to_world.shape[-3]}."
        )

    windowed = torch.cat(
        [
            anchor_pose.to(device=camera_to_world.device, dtype=camera_to_world.dtype),
            camera_to_world,
        ],
        dim=-3,
    )
    frame_indices = camera_frame_indices(
        windowed.shape[-3],
        frame_stride,
        device=windowed.device,
    )
    if frame_indices.numel() < 2:
        raise ValueError(
            "incremental_camera_conditioning requires at least one new latent frame"
        )
    sampled_poses = windowed.index_select(-3, frame_indices)
    new_frame_indices = frame_indices[1:] - 1
    sampled_intrinsics = _sample_intrinsics(
        intrinsics.to(device=camera_to_world.device),
        new_frame_indices,
        camera_to_world.shape[-3],
        camera_to_world.shape[:-3],
    )
    anchors = anchor_pose.expand_as(sampled_poses)
    relative = torch.linalg.solve(anchors.float(), sampled_poses.float())[..., 1:, :, :]
    rays = camera_rays(
        relative,
        sampled_intrinsics,
        image_height=image_height,
        image_width=image_width,
    )
    tokens = _rays_to_camera_tokens(
        rays, patch_size=patch_size, output_dtype=output_dtype
    )
    new_anchor_pose = sampled_poses[..., -1:, :, :]
    return tokens, new_anchor_pose
