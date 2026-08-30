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

"""CMD Cosmos DiT with per-block camera-ray projections."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.recipes.cosmos.transformer.impl.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkCache,
    CosmosDiTNetworkConfig,
)

from .modules import CMDTransformerBlock


@dataclass
class CMDDiTNetworkConfig(CosmosDiTNetworkConfig):
    """Configuration for CMD's checkpoint-compatible Cosmos DiT."""

    _target: type["CMDDiTNetwork"] = field(default_factory=lambda: CMDDiTNetwork)

    camera_dim: int | None = None
    """Patch-flattened ray channels; ``None`` disables camera conditioning."""


class CMDDiTNetwork(CosmosDiTNetwork):
    """Run CMD's causal Cosmos DiT with optional camera control."""

    config: CMDDiTNetworkConfig

    def _build_block(self) -> CMDTransformerBlock:
        """Build one CMD transformer block."""
        return CMDTransformerBlock(
            x_dim=self.config.model_channels,
            context_dim=self.config.crossattn_emb_channels,
            num_heads=self.config.num_heads,
            mlp_ratio=self.config.mlp_ratio,
            use_adaln_lora=self.config.use_adaln_lora,
            adaln_lora_dim=self.config.adaln_lora_dim,
            cp_method=self.config.cp_method,
            camera_dim=self.config.camera_dim,
        )

    def _prepare_hidden(
        self,
        x: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        timesteps = timesteps * self.config.timestep_scale
        batch_shape = x.shape[:-2]
        sequence_length = x.shape[-2]
        x = self.x_embedder(x)

        if timesteps.ndim == 0:
            timestep_slots = 1
        else:
            assert timesteps.shape == batch_shape + (sequence_length,), (
                f"per-token timesteps shape {tuple(timesteps.shape)} must equal "
                f"{tuple(batch_shape + (sequence_length,))}"
            )
            timestep_slots = sequence_length
        time_embedding, adaln_lora = self.t_embedder(timesteps)
        time_embedding = self.t_embedding_norm(time_embedding)
        time_embedding = torch.broadcast_to(
            time_embedding, batch_shape + (timestep_slots, time_embedding.shape[-1])
        )
        if adaln_lora is not None:
            adaln_lora = torch.broadcast_to(
                adaln_lora,
                batch_shape + (timestep_slots, adaln_lora.shape[-1]),
            )
        return x, time_embedding, adaln_lora

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: CosmosDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        camera: Tensor | None = None,
    ) -> Tensor:
        """Run one CMD denoising forward pass."""
        assert self._parameters_updated_after_loading_checkpoint, (
            "update_parameters_after_loading_checkpoint() must run before forward()"
        )
        x, time_embedding, adaln_lora = self._prepare_hidden(x, timesteps)

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, CMDTransformerBlock)
            x = block(
                x=x,
                emb=time_embedding,
                rope_freqs=rope_freqs,
                adaln_lora=adaln_lora,
                cache=cache[block_idx],
                camera=camera,
            )
        if eager_mode:
            cache.after_update(current_chunk_idx)
        return self.final_layer(x, time_embedding, adaln_lora)

    def prefill(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: CosmosDiTNetworkCache,
        rope_freqs: Tensor,
        camera: Tensor | None = None,
    ) -> None:
        """Seed every self-attention cache with an independent prefix."""
        assert self._parameters_updated_after_loading_checkpoint, (
            "update_parameters_after_loading_checkpoint() must run before prefill()"
        )
        x, time_embedding, adaln_lora = self._prepare_hidden(x, timesteps)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, CMDTransformerBlock)
            x = block.prefill(
                x=x,
                emb=time_embedding,
                rope_freqs=rope_freqs,
                adaln_lora=adaln_lora,
                cache=cache[block_idx],
                camera=camera,
            )
