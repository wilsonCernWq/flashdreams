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

"""Prove chunk-by-chunk incremental camera conditioning matches the batch path.

This is the crux correctness test for live camera control: it establishes
that ``incremental_camera_conditioning`` (which a live WebRTC session calls
once per AR step, knowing only that step's raw poses plus a carried anchor)
reproduces exactly what ``build_camera_conditioning`` (the existing,
shipped, whole-trajectory-upfront path) would have produced for the same
poses. Any future change to either function that breaks this equivalence
must fail this test.
"""

from __future__ import annotations

import torch
import pytest
from flashdreams_cmd.camera import (
    build_camera_conditioning,
    incremental_camera_conditioning,
)
from flashdreams_cmd.encoder import CMDCamCtrlInput, CMDLiveCameraEncoderConfig

pytestmark = pytest.mark.ci_cpu

_IMAGE_HEIGHT = 16
_IMAGE_WIDTH = 16
_INTRINSICS = torch.tensor([[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]])


def _random_rigid_poses(
    *batch_and_time: int, generator: torch.Generator
) -> torch.Tensor:
    """Build random valid rigid transforms shaped ``[*batch_and_time, 4, 4]``."""
    count = 1
    for dim in batch_and_time:
        count *= dim
    random_matrices = torch.randn(count, 3, 3, generator=generator)
    rotations, _ = torch.linalg.qr(random_matrices)
    # QR can hand back a reflection (det=-1); flip one column to force a
    # proper rotation (det=+1) so this looks like a real camera pose.
    det_sign = torch.det(rotations).sign()
    rotations = rotations.clone()
    rotations[:, :, 0] *= det_sign.unsqueeze(-1)
    translations = torch.randn(count, 3, generator=generator) * 2.0
    poses = torch.eye(4).unsqueeze(0).repeat(count, 1, 1)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    return poses.reshape(*batch_and_time, 4, 4)


def _run_incremental(
    poses: torch.Tensor,
    *,
    anchor_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    len_t: int,
    frame_stride: int,
    num_chunks: int,
) -> torch.Tensor:
    """Drive incremental_camera_conditioning chunk-by-chunk over ``poses``."""
    pixel_frames_per_chunk = len_t * frame_stride
    anchor = anchor_pose
    chunks = []
    for chunk_index in range(num_chunks):
        start = 1 + chunk_index * pixel_frames_per_chunk
        chunk_poses = poses[..., start : start + pixel_frames_per_chunk, :, :]
        tokens, anchor = incremental_camera_conditioning(
            chunk_poses,
            intrinsics,
            anchor_pose=anchor,
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
            len_t=len_t,
            frame_stride=frame_stride,
            output_dtype=torch.float32,
        )
        chunks.append(tokens)
    return torch.cat(chunks, dim=-4)


def test_incremental_matches_batch_with_prefix_seeded_anchor() -> None:
    """Seeding the anchor from the real prefix pose reproduces the batch path."""
    generator = torch.Generator().manual_seed(0)
    len_t, frame_stride, num_chunks = 4, 4, 3
    total_pixel_frames = 1 + num_chunks * len_t * frame_stride
    poses = _random_rigid_poses(total_pixel_frames, generator=generator)

    batch = build_camera_conditioning(
        poses,
        _INTRINSICS,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        frame_stride=frame_stride,
        block_size=len_t,
        output_dtype=torch.float32,
    )
    incremental = _run_incremental(
        poses,
        anchor_pose=poses[:1],
        intrinsics=_INTRINSICS,
        len_t=len_t,
        frame_stride=frame_stride,
        num_chunks=num_chunks,
    )

    torch.testing.assert_close(incremental, batch[1:], atol=0.0, rtol=0.0)


def test_incremental_matches_batch_with_identity_seeded_anchor() -> None:
    """The live-mode convention (identity anchor) matches a trajectory whose
    prefix pose really is the origin."""
    generator = torch.Generator().manual_seed(1)
    len_t, frame_stride, num_chunks = 4, 4, 2
    total_pixel_frames = 1 + num_chunks * len_t * frame_stride
    poses = _random_rigid_poses(total_pixel_frames, generator=generator)
    poses[0] = torch.eye(4)

    batch = build_camera_conditioning(
        poses,
        _INTRINSICS,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        frame_stride=frame_stride,
        block_size=len_t,
        output_dtype=torch.float32,
    )
    incremental = _run_incremental(
        poses,
        anchor_pose=torch.eye(4).unsqueeze(0),
        intrinsics=_INTRINSICS,
        len_t=len_t,
        frame_stride=frame_stride,
        num_chunks=num_chunks,
    )

    torch.testing.assert_close(incremental, batch[1:], atol=0.0, rtol=0.0)


def test_incremental_matches_batch_with_leading_batch_dimension() -> None:
    """The incremental path preserves an arbitrary leading batch dimension."""
    generator = torch.Generator().manual_seed(2)
    len_t, frame_stride, num_chunks = 4, 4, 2
    total_pixel_frames = 1 + num_chunks * len_t * frame_stride
    poses = _random_rigid_poses(2, total_pixel_frames, generator=generator)
    intrinsics = _INTRINSICS.expand(2, 3, 3)

    batch = build_camera_conditioning(
        poses,
        intrinsics,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        frame_stride=frame_stride,
        block_size=len_t,
        output_dtype=torch.float32,
    )
    incremental = _run_incremental(
        poses,
        anchor_pose=poses[:, :1],
        intrinsics=intrinsics,
        len_t=len_t,
        frame_stride=frame_stride,
        num_chunks=num_chunks,
    )

    torch.testing.assert_close(incremental, batch[:, 1:], atol=0.0, rtol=0.0)


def test_live_camera_encoder_carries_anchor_and_matches_batch() -> None:
    """CMDLiveCameraEncoder advances its cached anchor and matches the batch path."""
    generator = torch.Generator().manual_seed(3)
    len_t, frame_stride, num_chunks = 4, 4, 3
    total_pixel_frames = 1 + num_chunks * len_t * frame_stride
    poses = _random_rigid_poses(total_pixel_frames, generator=generator)
    poses[0] = torch.eye(4)

    batch = build_camera_conditioning(
        poses,
        _INTRINSICS,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        frame_stride=frame_stride,
        block_size=len_t,
        output_dtype=torch.float32,
    )

    encoder = CMDLiveCameraEncoderConfig(
        len_t=len_t,
        frame_stride=frame_stride,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        dtype=torch.float32,
        base_intrinsics=(8.0, 8.0, 8.0, 8.0),
    ).setup()
    cache = encoder.initialize_autoregressive_cache()
    torch.testing.assert_close(cache.anchor_pose, torch.eye(4).unsqueeze(0))

    pixel_frames_per_chunk = len_t * frame_stride
    chunks = []
    for chunk_index in range(num_chunks):
        start = 1 + chunk_index * pixel_frames_per_chunk
        chunk_poses = poses[start : start + pixel_frames_per_chunk]
        tokens = encoder(
            CMDCamCtrlInput(poses=chunk_poses, intrinsics=_INTRINSICS),
            chunk_index,
            cache,
        )
        chunks.append(tokens)
        torch.testing.assert_close(cache.anchor_pose, chunk_poses[-1:])

    torch.testing.assert_close(torch.cat(chunks, dim=0), batch[1:], atol=0.0, rtol=0.0)


def test_prefix_camera_tokens_matches_batch_prefix_slice() -> None:
    """CMD's independent first-frame prefill needs its own camera token too
    (caught live: this was missing entirely on the first real end-to-end
    smoke test -- ``CMDTransformer.initialize_autoregressive_cache`` raised
    ``camera-conditioned CMD requires camera_condition`` because
    ``pipeline.py`` still left ``camera_condition=None`` in live mode).
    ``prefix_camera_tokens()`` must reproduce exactly what
    ``build_camera_conditioning(...)[..., :1, :, :, :]`` gives a
    fixed-trajectory rollout whose prefix pose is the identity (the live
    convention -- see the identity-seeded-anchor test above)."""
    generator = torch.Generator().manual_seed(4)
    len_t, frame_stride = 4, 4
    total_pixel_frames = 1 + len_t * frame_stride
    poses = _random_rigid_poses(total_pixel_frames, generator=generator)
    poses[0] = torch.eye(4)

    batch = build_camera_conditioning(
        poses,
        _INTRINSICS,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        frame_stride=frame_stride,
        block_size=len_t,
        output_dtype=torch.float32,
    )

    encoder = CMDLiveCameraEncoderConfig(
        len_t=len_t,
        frame_stride=frame_stride,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        dtype=torch.float32,
        base_intrinsics=(8.0, 8.0, 8.0, 8.0),
    ).setup()

    prefix_tokens = encoder.prefix_camera_tokens(device=torch.device("cpu"))

    torch.testing.assert_close(prefix_tokens, batch[:1], atol=0.0, rtol=0.0)


def test_incremental_camera_conditioning_rejects_a_frame_count_not_matching_len_t() -> (
    None
):
    """Regression test: earlier there was no enforced invariant between a
    call's frame count and len_t*frame_stride (one AR block). Two half-size
    calls covering what should be one block silently produced wrong tokens
    with no error (verified live: 50% of elements mismatched, diffs up to
    ~6.0) instead of raising -- this is the one guard that used to be
    missing, kept in sync only by hand-duplicated len_t literals elsewhere.
    """
    generator = torch.Generator().manual_seed(5)
    len_t, frame_stride = 4, 4
    wrong_frame_count = 2 * frame_stride  # half of len_t * frame_stride = 16
    poses = _random_rigid_poses(wrong_frame_count, generator=generator)

    with pytest.raises(ValueError, match="len_t \\* frame_stride"):
        incremental_camera_conditioning(
            poses,
            _INTRINSICS,
            anchor_pose=torch.eye(4).unsqueeze(0),
            image_height=_IMAGE_HEIGHT,
            image_width=_IMAGE_WIDTH,
            len_t=len_t,
            frame_stride=frame_stride,
            output_dtype=torch.float32,
        )


def test_incremental_camera_conditioning_normalizes_anchor_device_and_dtype() -> None:
    """Regression test: the anchor/camera_to_world concat only normalized
    anchor_pose's dtype, not its device, asymmetric with the explicit device
    handling intrinsics gets three lines later. Not reachable through the
    current webrtc wiring (which keeps everything on one device already),
    but a real gap in the function's own contract -- a caller-supplied
    anchor on a different device than camera_to_world used to crash with a
    device-mismatch RuntimeError instead of being handled transparently."""
    generator = torch.Generator().manual_seed(6)
    len_t, frame_stride = 4, 4
    poses = _random_rigid_poses(len_t * frame_stride, generator=generator).to(
        torch.float64
    )
    anchor_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # different dtype

    tokens, new_anchor = incremental_camera_conditioning(
        poses,
        _INTRINSICS,
        anchor_pose=anchor_pose,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
        len_t=len_t,
        frame_stride=frame_stride,
        output_dtype=torch.float32,
    )
    assert tokens.shape[0] == len_t
    assert new_anchor.device == poses.device
