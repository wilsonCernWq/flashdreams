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

"""Configuration literals for the six released CMD checkpoints."""

from __future__ import annotations

from flashdreams.infra.config import derive_config
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.runner import RunnerConfig
from flashdreams.recipes.wan import WanVAEDecoderConfig, WanVAEEncoderConfig

from .camera import CAMERA_FEATURE_DIM
from .encoder import CMDCameraEncoderConfig
from .pipeline import CMDInferencePipelineConfig
from .runner import (
    DEFAULT_CAMERA_IMAGE_URL,
    DEFAULT_CAMERA_PROMPT,
    DEFAULT_CAMERA_URL,
    CMDRunnerConfig,
)
from .transformer import (
    CMDDiTNetworkConfig,
    CMDTransformerConfig,
    state_dict_transform,
)

CHECKPOINT_ROOT = "https://huggingface.co/nvidia/cmd/blob/main"
"""Hugging Face repository containing released CMD safetensors."""

CHECKPOINT_CHUNK1_SHORT = f"{CHECKPOINT_ROOT}/chunk1_short_t24_l21.safetensors"
CHECKPOINT_CHUNK1_LONG = f"{CHECKPOINT_ROOT}/chunk1_long_t126_l21.safetensors"
CHECKPOINT_CHUNK4_SHORT = f"{CHECKPOINT_ROOT}/chunk4_short_t21_l16.safetensors"
CHECKPOINT_CHUNK4_LONG = f"{CHECKPOINT_ROOT}/chunk4_long_t121_l16.safetensors"
CHECKPOINT_CHUNK1_CAMERA = (
    f"{CHECKPOINT_ROOT}/chunk1_camera_control_t32_l21.safetensors"
)
CHECKPOINT_CHUNK4_CAMERA = (
    f"{CHECKPOINT_ROOT}/chunk4_camera_control_t29_l24.safetensors"
)

CAMERA_TOKEN_DIM = CAMERA_FEATURE_DIM * 16 * 16
"""Six ray channels flattened over each 16-by-16 pixel patch."""

CMD_SCHEDULER = FlowMatchSchedulerConfig(
    num_inference_steps=4,
    denoising_timesteps=[1000, 750, 500, 250],
    warp_denoising_step=True,
    shift=5.0,
    sigma_min=0.0,
    extra_one_step=True,
    num_train_timesteps=1000,
)
"""Released CMD four-step self-forcing schedule."""

PIPELINE_CMD_CHUNK1_SHORT = CMDInferencePipelineConfig(
    name="cmd-chunk1-short-i2v",
    enable_sync_and_profile=True,
    encoder=None,
    image_encoder=WanVAEEncoderConfig(use_compile=True, use_cuda_graph=True),
    decoder=WanVAEDecoderConfig(use_compile=True, use_cuda_graph=True),
    diffusion_model=DiffusionModelConfig(
        seed=22,
        context_noise=128,
        noise_in_unpatchified_shape=True,
        transformer=CMDTransformerConfig(
            network=CMDDiTNetworkConfig(cp_method="ring"),
            checkpoint_path=CHECKPOINT_CHUNK1_SHORT,
            state_dict_transform=state_dict_transform,
            batch_shape=(),
            len_t=1,
            window_size_t=21,
            sink_size_t=0,
            guidance_scale=1.0,
            compile_network=True,
            use_cuda_graph=True,
        ),
        scheduler=CMD_SCHEDULER,
    ),
)
RUNNER_CMD_CHUNK1_SHORT = CMDRunnerConfig(
    runner_name=PIPELINE_CMD_CHUNK1_SHORT.name,
    description="CMD chunk-1 short I2V: 24 latent frames, four denoising steps.",
    pipeline=PIPELINE_CMD_CHUNK1_SHORT,
    num_chunks=23,
)

PIPELINE_CMD_CHUNK1_LONG = derive_config(
    PIPELINE_CMD_CHUNK1_SHORT,
    name="cmd-chunk1-long-i2v",
    diffusion_model=dict(
        transformer=dict(checkpoint_path=CHECKPOINT_CHUNK1_LONG),
    ),
)
RUNNER_CMD_CHUNK1_LONG = CMDRunnerConfig(
    runner_name=PIPELINE_CMD_CHUNK1_LONG.name,
    description="CMD chunk-1 long I2V: 126 latent frames, four steps per chunk.",
    pipeline=PIPELINE_CMD_CHUNK1_LONG,
    num_chunks=125,
)

PIPELINE_CMD_CHUNK4_SHORT = derive_config(
    PIPELINE_CMD_CHUNK1_SHORT,
    name="cmd-chunk4-short-i2v",
    diffusion_model=dict(
        transformer=dict(
            checkpoint_path=CHECKPOINT_CHUNK4_SHORT,
            len_t=4,
            window_size_t=16,
        ),
    ),
)
RUNNER_CMD_CHUNK4_SHORT = CMDRunnerConfig(
    runner_name=PIPELINE_CMD_CHUNK4_SHORT.name,
    description="CMD chunk-4 short I2V: 21 latent frames, four steps per chunk.",
    pipeline=PIPELINE_CMD_CHUNK4_SHORT,
    num_chunks=5,
)

PIPELINE_CMD_CHUNK4_SHORT_FP8 = derive_config(
    PIPELINE_CMD_CHUNK4_SHORT,
    name="cmd-chunk4-short-i2v-fp8",
    diffusion_model=dict(
        transformer=dict(weight_quantization="fp8"),
    ),
)
"""A/B twin of ``cmd-chunk4-short-i2v`` with FP8 block linears.

Differs from its base in exactly one field, so a side-by-side run isolates the
effect of quantization. FP8 only pays off under ``torch.compile`` (which this
preset inherits as ``compile_network=True``); in eager mode the per-call
activation quantization makes it slower than bf16.
"""

RUNNER_CMD_CHUNK4_SHORT_FP8 = CMDRunnerConfig(
    runner_name=PIPELINE_CMD_CHUNK4_SHORT_FP8.name,
    description="CMD chunk-4 short I2V with FP8 block linears (A/B against the bf16 twin).",
    pipeline=PIPELINE_CMD_CHUNK4_SHORT_FP8,
    num_chunks=5,
)

PIPELINE_CMD_CHUNK4_LONG = derive_config(
    PIPELINE_CMD_CHUNK4_SHORT,
    name="cmd-chunk4-long-i2v",
    diffusion_model=dict(
        transformer=dict(checkpoint_path=CHECKPOINT_CHUNK4_LONG),
    ),
)
RUNNER_CMD_CHUNK4_LONG = CMDRunnerConfig(
    runner_name=PIPELINE_CMD_CHUNK4_LONG.name,
    description="CMD chunk-4 long I2V: 121 latent frames, four steps per chunk.",
    pipeline=PIPELINE_CMD_CHUNK4_LONG,
    num_chunks=30,
)

PIPELINE_CMD_CHUNK1_CAMERA = derive_config(
    PIPELINE_CMD_CHUNK1_SHORT,
    name="cmd-chunk1-camera-i2v",
    encoder=CMDCameraEncoderConfig(len_t=1),
    diffusion_model=dict(
        transformer=dict(
            checkpoint_path=CHECKPOINT_CHUNK1_CAMERA,
            network=dict(camera_dim=CAMERA_TOKEN_DIM),
        ),
    ),
)
RUNNER_CMD_CHUNK1_CAMERA = CMDRunnerConfig(
    runner_name=PIPELINE_CMD_CHUNK1_CAMERA.name,
    description="CMD chunk-1 camera-controlled I2V: 32 latent frames.",
    pipeline=PIPELINE_CMD_CHUNK1_CAMERA,
    prompt=DEFAULT_CAMERA_PROMPT,
    image_path=DEFAULT_CAMERA_IMAGE_URL,
    camera_path=DEFAULT_CAMERA_URL,
    num_chunks=31,
)

PIPELINE_CMD_CHUNK4_CAMERA = derive_config(
    PIPELINE_CMD_CHUNK4_SHORT,
    name="cmd-chunk4-camera-i2v",
    encoder=CMDCameraEncoderConfig(len_t=4),
    diffusion_model=dict(
        transformer=dict(
            checkpoint_path=CHECKPOINT_CHUNK4_CAMERA,
            window_size_t=24,
            network=dict(camera_dim=CAMERA_TOKEN_DIM),
        ),
    ),
)
RUNNER_CMD_CHUNK4_CAMERA = CMDRunnerConfig(
    runner_name=PIPELINE_CMD_CHUNK4_CAMERA.name,
    description="CMD chunk-4 camera-controlled I2V: 29 latent frames.",
    pipeline=PIPELINE_CMD_CHUNK4_CAMERA,
    prompt=DEFAULT_CAMERA_PROMPT,
    image_path=DEFAULT_CAMERA_IMAGE_URL,
    camera_path=DEFAULT_CAMERA_URL,
    num_chunks=7,
)

CMD_CONFIGS: dict[str, CMDInferencePipelineConfig] = {
    config.name: config
    for config in (
        PIPELINE_CMD_CHUNK1_SHORT,
        PIPELINE_CMD_CHUNK1_LONG,
        PIPELINE_CMD_CHUNK4_SHORT,
        PIPELINE_CMD_CHUNK4_SHORT_FP8,
        PIPELINE_CMD_CHUNK4_LONG,
        PIPELINE_CMD_CHUNK1_CAMERA,
        PIPELINE_CMD_CHUNK4_CAMERA,
    )
}
"""All CMD pipeline presets keyed by stable name."""

RUNNER_CONFIGS: dict[str, RunnerConfig] = {
    config.runner_name: config
    for config in (
        RUNNER_CMD_CHUNK1_SHORT,
        RUNNER_CMD_CHUNK1_LONG,
        RUNNER_CMD_CHUNK4_SHORT,
        RUNNER_CMD_CHUNK4_SHORT_FP8,
        RUNNER_CMD_CHUNK4_LONG,
        RUNNER_CMD_CHUNK1_CAMERA,
        RUNNER_CMD_CHUNK4_CAMERA,
    )
}
"""All CMD runner presets discovered by ``flashdreams-run``."""


__all__ = [
    "CMD_CONFIGS",
    "RUNNER_CONFIGS",
    "RUNNER_CMD_CHUNK1_CAMERA",
    "RUNNER_CMD_CHUNK1_LONG",
    "RUNNER_CMD_CHUNK1_SHORT",
    "RUNNER_CMD_CHUNK4_CAMERA",
    "RUNNER_CMD_CHUNK4_LONG",
    "RUNNER_CMD_CHUNK4_SHORT",
]
