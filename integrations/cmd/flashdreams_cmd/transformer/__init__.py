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

"""CMD transformer with independent I2V-prefix cache seeding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from flashdreams.core.attention.rope import RotaryPositionEmbedding3D
from flashdreams.core.distributed.context_parallel import split_inputs_cp
from flashdreams.recipes.cosmos.transformer import (
    CosmosTransformer,
    CosmosTransformerCache,
    CosmosTransformerConfig,
)
from flashdreams.recipes.cosmos.transformer.impl.network import (
    state_dict_transform,
)

from .native_fp8 import (
    NativeAccelerationMode,
    NativeFP8Unavailable,
    resolve_native_fp8,
)
from .network import CMDDiTNetwork, CMDDiTNetworkConfig

__all__ = [
    "CMDDiTNetworkConfig",
    "CMDTransformer",
    "CMDTransformerCache",
    "CMDTransformerConfig",
    "NativeAccelerationMode",
    "NativeFP8Unavailable",
    "resolve_native_fp8",
    "state_dict_transform",
]


@dataclass(kw_only=True)
class CMDTransformerCache(CosmosTransformerCache):
    """CMD AR cache with one independently cached prefix frame."""

    prefix_len_t: int = 1
    """Temporal positions cached before generated AR step 0."""

    def start(self, autoregressive_index: int) -> None:
        self.rope_freqs = self.rope_adapter.shift_t(
            autoregressive_index,
            temporal_offset=self.prefix_len_t,
        )
        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)


@dataclass(kw_only=True)
class CMDTransformerConfig(CosmosTransformerConfig):
    """Configuration for CMD's causal Cosmos transformer."""

    _target: type["CMDTransformer"] = field(default_factory=lambda: CMDTransformer)

    network: CMDDiTNetworkConfig = field(default_factory=CMDDiTNetworkConfig)
    """CMD DiT network configuration."""

    prefix_len_t: int = 1
    """Independent I2V prefix length. Released CMD checkpoints use ``1``."""

    native_dit_acceleration: NativeAccelerationMode = "disabled"
    """Native CUTLASS FP8 DiT path (``integrations/cmd/docs/native_fp8_port_plan.md``).

    ``disabled`` (default) never probes for the extension. ``auto`` uses it when
    the model and GPU both qualify and falls back with a logged reason otherwise.
    ``required`` raises :class:`NativeFP8Unavailable` instead of falling back.
    The kernels are ``sm_120a``-only; see :mod:`.native_fp8` for why the
    preconditions are hard refusals rather than best-effort attempts."""


class CMDTransformer(CosmosTransformer):
    """Run CMD generation after seeding the clean first-frame prefix."""

    config: CMDTransformerConfig
    network: CMDDiTNetwork

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        """Patchify latents or flatten pre-patchified camera tokens."""
        camera_dim = self.config.network.camera_dim
        if camera_dim is None or x.shape[-3] != camera_dim:
            return super().patchify_and_maybe_split_cp(x)

        camera = rearrange(x, "... t c h w -> ... (t h w) c")
        if self._cp_group is not None:
            camera = split_inputs_cp(camera, seq_dim=-2, cp_group=self._cp_group)
        return camera

    def _predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CMDTransformerCache,
        network_cache: Any,
        input: Tensor | None,
        *,
        uncond: bool,
    ) -> Tensor:
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0 and cache.rope_freqs is not None, (
            "Cache.start() must run before predict_flow()"
        )
        mask = self._select_mask(cache)
        timesteps = self._build_per_token_timesteps(timestep, cache)
        return self._select_network(ar_idx, uncond=uncond)(
            x=torch.cat((noisy_latent, mask), dim=-1),
            timesteps=timesteps,
            rope_freqs=cache.rope_freqs,
            cache=network_cache,
            current_chunk_idx=ar_idx,
            eager_mode=False,
            camera=input,
        )

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        negative_text_embeddings: Tensor | None = None,
        camera_condition: Tensor | None = None,
        **unused: Any,
    ) -> CMDTransformerCache:
        """Build caches and prefill the clean first-frame conditioning prefix."""
        if image_embeddings is None:
            raise ValueError("CMD inference requires first-frame image embeddings")
        if self.config.prefix_len_t != 1:
            raise ValueError("Released CMD checkpoints require prefix_len_t=1")

        parent = super().initialize_autoregressive_cache(
            height=height,
            width=width,
            text_embeddings=text_embeddings,
            image_embeddings=None,
            negative_text_embeddings=negative_text_embeddings,
            **unused,
        )
        cache = CMDTransformerCache(
            network_cache=parent.network_cache,
            network_cache_uncond=parent.network_cache_uncond,
            rope_adapter=parent.rope_adapter,
            image=None,
            mask_first_block=parent.mask_first_block,
            mask_other_blocks=parent.mask_other_blocks,
            prefix_len_t=self.config.prefix_len_t,
        )

        image_embeddings = image_embeddings.to(device=self.device, dtype=self.dtype)
        prefix_latent = self.patchify_and_maybe_split_cp(image_embeddings)
        prefix_mask = torch.ones(
            *self.config.batch_shape,
            1,
            1,
            height,
            width,
            device=self.device,
            dtype=self.dtype,
        )
        prefix_mask = super().patchify_and_maybe_split_cp(prefix_mask)
        prefix_input = torch.cat((prefix_latent, prefix_mask), dim=-1)

        camera_prefix: Tensor | None = None
        if self.config.network.camera_dim is not None:
            if camera_condition is None:
                raise ValueError("camera-conditioned CMD requires camera_condition")
            camera_prefix = self.patchify_and_maybe_split_cp(
                camera_condition[..., :1, :, :, :]
            )
        elif camera_condition is not None:
            raise ValueError(
                "camera_condition was provided to a non-camera CMD variant"
            )

        _, kh, kw = self.config.network.patch_size
        prefix_rope = RotaryPositionEmbedding3D(
            len_t=1,
            len_h=height // kh,
            len_w=width // kw,
            head_dim=self.config.network.model_channels
            // self.config.network.num_heads,
            h_extrapolation_ratio=self.config.h_extrapolation_ratio,
            w_extrapolation_ratio=self.config.w_extrapolation_ratio,
            device=self.device,
        )
        prefix_rope.set_context_parallel_group(self._cp_group)
        prefix_freqs = prefix_rope.shift_t(0)
        timestep = torch.zeros((), device=self.device, dtype=self.dtype)

        self.network.prefill(
            x=prefix_input,
            timesteps=timestep,
            cache=cache.network_cache,
            rope_freqs=prefix_freqs,
            camera=camera_prefix,
        )
        if cache.network_cache_uncond is not None:
            self.network.prefill(
                x=prefix_input,
                timesteps=timestep,
                cache=cache.network_cache_uncond,
                rope_freqs=prefix_freqs,
                camera=camera_prefix,
            )
        return cache
