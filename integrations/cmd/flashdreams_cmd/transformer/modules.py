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

"""CMD camera-conditioned Cosmos transformer block."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from flashdreams.recipes.cosmos.transformer.impl.modules import Block, BlockCache


class CMDTransformerBlock(Block):
    """Inject projected camera rays into Cosmos self-attention inputs."""

    def __init__(
        self, *args: Any, camera_dim: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.camera_dim = camera_dim
        if camera_dim is not None:
            self.self_attn.add_module(
                "cam_encoder",
                nn.Linear(camera_dim, self.x_dim, bias=False),
            )

    def _modulations(
        self,
        emb: Tensor,
        adaln_lora: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if self.use_adaln_lora:
            assert adaln_lora is not None, (
                "adaln_lora is required when use_adaln_lora is True"
            )
            self_values = self.adaln_modulation_self_attn(emb) + adaln_lora
            cross_values = self.adaln_modulation_cross_attn(emb) + adaln_lora
            mlp_values = self.adaln_modulation_mlp(emb) + adaln_lora
        else:
            self_values = self.adaln_modulation_self_attn(emb)
            cross_values = self.adaln_modulation_cross_attn(emb)
            mlp_values = self.adaln_modulation_mlp(emb)
        return (
            *self_values.chunk(3, dim=-1),
            *cross_values.chunk(3, dim=-1),
            *mlp_values.chunk(3, dim=-1),
        )

    def _inject_camera(self, normed_x: Tensor, camera: Tensor | None) -> Tensor:
        if self.camera_dim is None:
            if camera is not None:
                raise ValueError(
                    "camera tokens were provided to an unconditional CMD block"
                )
            return normed_x
        if camera is None:
            raise ValueError("camera-conditioned CMD block requires camera tokens")
        if camera.shape[:-1] != normed_x.shape[:-1]:
            raise ValueError(
                "camera and video token grids must match; got "
                f"{tuple(camera.shape)} and {tuple(normed_x.shape)}"
            )
        encoder = getattr(self.self_attn, "cam_encoder")
        return normed_x + encoder(camera.to(dtype=normed_x.dtype))

    def _finish_branches(
        self,
        x: Tensor,
        emb: Tensor,
        cache: BlockCache,
        adaln_lora: Tensor | None,
        *,
        self_attn_output: Tensor,
    ) -> Tensor:
        (
            _shift_self,
            _scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._modulations(emb, adaln_lora)
        x = x + gate_self * self_attn_output

        normed_x = self.layer_norm_cross_attn(x) * (1 + scale_cross) + shift_cross
        x = x + gate_cross * self.cross_attn(normed_x, kv_cache=cache.cross_attn)

        normed_x = self.layer_norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(normed_x)

    def forward(
        self,
        x: Tensor,
        emb: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        adaln_lora: Tensor | None = None,
        camera: Tensor | None = None,
    ) -> Tensor:
        """Run one camera-conditioned transformer block update."""
        assert emb.ndim == x.ndim, "emb and x must have the same number of dimensions"
        shift_self, scale_self, *_ = self._modulations(emb, adaln_lora)
        normed_x = self.layer_norm_self_attn(x) * (1 + scale_self) + shift_self
        normed_x = self._inject_camera(normed_x, camera)
        self_attn_output = self.self_attn(
            normed_x,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        return self._finish_branches(
            x,
            emb,
            cache,
            adaln_lora,
            self_attn_output=self_attn_output,
        )

    def prefill(
        self,
        x: Tensor,
        emb: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        adaln_lora: Tensor | None = None,
        camera: Tensor | None = None,
    ) -> Tensor:
        """Run and cache a variable-length conditioning prefix."""
        shift_self, scale_self, *_ = self._modulations(emb, adaln_lora)
        normed_x = self.layer_norm_self_attn(x) * (1 + scale_self) + shift_self
        normed_x = self._inject_camera(normed_x, camera)

        prefix_cache = self.self_attn.compute_kv(normed_x, rope_freqs)
        cache.self_attn.prefill(prefix_cache.cached_k(), prefix_cache.cached_v())
        self_attn_output = self.self_attn.apply_kv(
            normed_x,
            prefix_cache,
            rope_freqs,
        )
        return self._finish_branches(
            x,
            emb,
            cache,
            adaln_lora,
            self_attn_output=self_attn_output,
        )
