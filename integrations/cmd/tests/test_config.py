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

"""CPU-safe structural checks for CMD checkpoints and presets."""

from __future__ import annotations

import pytest
import torch
from flashdreams_cmd.config import CMD_CONFIGS, RUNNER_CONFIGS
from flashdreams_cmd.pipeline import CMDInferencePipelineConfig
from flashdreams_cmd.runner import (
    DEFAULT_CAMERA_IMAGE_URL,
    DEFAULT_CAMERA_URL,
    DEFAULT_IMAGE_URL,
    CMDRunnerConfig,
)
from flashdreams_cmd.transformer import (
    CMDDiTNetworkConfig,
    CMDTransformerConfig,
    state_dict_transform,
)

from flashdreams.recipes.wan import WanVAEDecoderConfig, WanVAEEncoderConfig

pytestmark = pytest.mark.ci_cpu


def test_default_example_assets_use_raw_git_urls() -> None:
    """Fetch ordinary Git blobs from GitHub's raw-content host."""
    root = "https://raw.githubusercontent.com/nv-tlabs/cmd/main/examples/"
    assert DEFAULT_IMAGE_URL == f"{root}image.png"
    assert DEFAULT_CAMERA_IMAGE_URL == f"{root}camera_image.png"
    assert DEFAULT_CAMERA_URL == f"{root}camera.npz"


def test_state_dict_transform_keeps_cmd_camera_weights() -> None:
    """Drop runtime RoPE buffers without discarding camera projections."""
    camera_weight = torch.randn(4, 6)
    transformed = state_dict_transform(
        {
            "net.blocks.0.self_attn.cam_encoder.weight": camera_weight,
            "net.pos_embedder.seq": torch.ones(1),
        }
    )
    assert transformed == {
        "blocks.0.self_attn.cam_encoder.weight": camera_weight,
    }


def test_small_cmd_network_exposes_checkpoint_camera_key() -> None:
    """Attach camera projections at the exact released checkpoint path."""
    network = CMDDiTNetworkConfig(
        in_channels=2,
        out_channels=1,
        model_channels=32,
        num_blocks=1,
        num_heads=4,
        mlp_ratio=2.0,
        use_adaln_lora=False,
        use_crossattn_projection=False,
        crossattn_emb_channels=16,
        camera_dim=6,
    ).setup()
    state = network.state_dict()
    assert state["blocks.0.self_attn.cam_encoder.weight"].shape == (32, 6)


def test_released_presets_match_cmd_temporal_geometry() -> None:
    """Expose all six checkpoints with their trained frame counts and windows."""
    expected = {
        "cmd-chunk1-short-i2v": (1, 23, 24, 21, False),
        "cmd-chunk1-long-i2v": (1, 125, 126, 21, False),
        "cmd-chunk4-short-i2v": (4, 5, 21, 16, False),
        "cmd-chunk4-long-i2v": (4, 30, 121, 16, False),
        "cmd-chunk1-camera-i2v": (1, 31, 32, 21, True),
        "cmd-chunk4-camera-i2v": (4, 7, 29, 24, True),
    }
    assert set(CMD_CONFIGS) == set(RUNNER_CONFIGS) == set(expected)
    for name, (chunk, chunks, total, window, camera) in expected.items():
        runner = RUNNER_CONFIGS[name]
        assert isinstance(runner, CMDRunnerConfig)
        pipeline = runner.pipeline
        assert isinstance(pipeline, CMDInferencePipelineConfig)
        transformer = pipeline.diffusion_model.transformer
        assert isinstance(transformer, CMDTransformerConfig)
        assert transformer.len_t == chunk
        assert runner.num_chunks == chunks
        assert transformer.prefix_len_t + chunks * chunk == total
        assert transformer.window_size_t == window
        assert (transformer.network.camera_dim is not None) is camera
        assert transformer.compile_network is True
        assert transformer.use_cuda_graph is True
        assert isinstance(pipeline.image_encoder, WanVAEEncoderConfig)
        assert pipeline.image_encoder.use_compile is True
        assert pipeline.image_encoder.use_cuda_graph is True
        assert isinstance(pipeline.decoder, WanVAEDecoderConfig)
        assert pipeline.decoder.use_compile is True
        assert pipeline.decoder.use_cuda_graph is True
        assert runner.runner_name == runner.pipeline.name
