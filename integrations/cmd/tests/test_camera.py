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

"""CPU tests for CMD camera geometry and per-step slicing."""

from __future__ import annotations

import pytest
import torch
from flashdreams_cmd.camera import (
    block_relative_poses,
    build_camera_conditioning,
    camera_frame_indices,
)
from flashdreams_cmd.encoder import CMDCameraEncoderConfig

pytestmark = pytest.mark.ci_cpu


def test_camera_frame_indices_follow_vae_stride() -> None:
    """Select the first frame and every fourth pixel frame thereafter."""
    indices = camera_frame_indices(9, 4, device=torch.device("cpu"))
    assert indices.tolist() == [0, 4, 8]
    with pytest.raises(ValueError, match=r"1 \+ k"):
        camera_frame_indices(8, 4, device=torch.device("cpu"))


def test_block_relative_poses_anchor_generated_chunks() -> None:
    """Use frame zero for the first generated block and its end thereafter."""
    poses = torch.eye(4).repeat(6, 1, 1)
    poses[:, 0, 3] = torch.arange(6)
    relative = block_relative_poses(poses, block_size=4)
    torch.testing.assert_close(
        relative[:, 0, 3],
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 1.0]),
    )


def test_camera_conditioning_matches_cmd_token_layout() -> None:
    """Flatten six ray channels over one 16-by-16 spatial token."""
    poses = torch.eye(4).repeat(9, 1, 1)
    intrinsics = torch.tensor([[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]])
    condition = build_camera_conditioning(
        poses,
        intrinsics,
        image_height=16,
        image_width=16,
        expected_latent_frames=3,
        output_dtype=torch.float32,
    )
    assert condition.shape == (3, 1536, 1, 1)
    origins = condition[:, : 3 * 16 * 16]
    assert torch.count_nonzero(origins) == 0


def test_camera_encoder_slices_after_independent_prefix() -> None:
    """AR zero begins after the prefix and each chunk advances by len_t."""
    encoder = CMDCameraEncoderConfig(len_t=4).setup()
    cache = encoder.initialize_autoregressive_cache()
    camera = torch.arange(9).reshape(9, 1, 1, 1)
    assert encoder(camera, 0, cache).flatten().tolist() == [1, 2, 3, 4]
    assert encoder(camera, 1, cache).flatten().tolist() == [5, 6, 7, 8]
    with pytest.raises(ValueError, match="too short"):
        encoder(camera, 2, cache)
