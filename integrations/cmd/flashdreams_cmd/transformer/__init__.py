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

from . import native_adapter
from .network import CMDDiTNetwork, CMDDiTNetworkConfig

__all__ = [
    "CMDDiTNetworkConfig",
    "CMDTransformer",
    "CMDTransformerCache",
    "CMDTransformerConfig",
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

    ## Native DiT acceleration
    #
    # CMD borrows omnidreams' `omnidreams_singleview` extension rather than
    # carrying its own kernels: the two models emit byte-identical per-block
    # state-dict keys, so the same C++ bridge consumes both. These mirror the
    # fields omnidreams exposes, because they are forwarded verbatim to the
    # shared OptimizedDiTExecutor.

    native_dit_acceleration: str = "disabled"
    """``disabled``, ``auto`` (fall back silently), or ``required`` (raise)."""

    native_dit_backend: str = "fp8_kvcache_cudnn"
    """``fp8_kvcache_cudnn`` or ``bf16``. Only meaningful when enabled."""

    native_dit_attention_backend: str = "auto"
    """``auto``, ``cudnn``, ``sparge``, ``sage3`` or ``sage3_fp8``."""

    native_dit_build_root: str | None = None
    """Where to cache the JIT-built extension; ``None`` uses the default."""

    native_dit_max_jobs: int | str | None = None
    """nvcc fan-out. Keep small (2) on a shared host: the build is RAM-bound."""

    native_dit_verbose_build: bool = False
    """Echo nvcc lines, for diagnosing a build that fails to load."""


class CMDTransformer(CosmosTransformer):
    """Run CMD generation after seeding the clean first-frame prefix."""

    config: CMDTransformerConfig
    network: CMDDiTNetwork

    def __init__(self, config: CMDTransformerConfig) -> None:
        native_enabled = getattr(config, "native_dit_acceleration", "disabled") != "disabled"
        if native_enabled and config.compile_network:
            # The base __init__ wraps self.network in compile_module before we
            # get a chance to build the executor, and the executor snapshots the
            # module's weights. Rather than reorder the base or mutate config
            # behind its back, refuse the combination -- it is already the
            # documented requirement for the native path, and running both costs
            # a torch.compile graph that is then thrown away.
            raise ValueError(
                "native_dit_acceleration requires compile_network=False; "
                "pass --pipeline.diffusion-model.transformer.compile-network False"
            )
        super().__init__(config)

        self._optimized_dit_executor: Any | None = None
        self._optimized_dit_selection: Any | None = None
        if native_enabled:
            self._configure_optimized_dit_from_config()

    def _configure_optimized_dit_from_config(self) -> None:
        """Build the shared omnidreams executor for CMD's network.

        Imported here rather than at module scope so ``import flashdreams_cmd``
        never requires the omnidreams package; only actually enabling the native
        path does. The kernels, the C++ bridge and the executor all live in
        omnidreams_singleview -- CMD supplies tensors in the bridge's layout and
        nothing else.
        """
        try:
            from omnidreams.native import omnidreams_singleview
            from omnidreams.native.acceleration import (
                NativeAccelerationConfig,
                require_extension_symbols,
            )
        except ImportError as exc:
            raise ImportError(
                "native_dit_acceleration requires the omnidreams package "
                "(flashdreams-omnidreams) to be installed"
            ) from exc

        helper = omnidreams_singleview.load_python_module("optimized_dit")
        selection = omnidreams_singleview.select_backend(
            "optimized_dit",
            NativeAccelerationConfig(
                mode=self.config.native_dit_acceleration,
                build_root=self.config.native_dit_build_root,
                max_jobs=self.config.native_dit_max_jobs,
                verbose_build=self.config.native_dit_verbose_build,
            ),
            availability_check=require_extension_symbols("optimized_dit_forward"),
        )
        self._optimized_dit_selection = selection
        if not selection.enabled:
            # "auto" degrades silently by design; "required" already raised.
            return
        self._optimized_dit_executor = helper.OptimizedDiTExecutor(
            self,
            selection.require_extension(),
            dit_backend=self.config.native_dit_backend,
            attention_backend=self.config.native_dit_attention_backend,
        )

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        """Patchify latents or flatten pre-patchified camera tokens."""
        camera_dim = self.config.network.camera_dim
        if camera_dim is None or x.shape[-3] != camera_dim:
            return super().patchify_and_maybe_split_cp(x)

        camera = rearrange(x, "... t c h w -> ... (t h w) c")
        if self._cp_group is not None:
            camera = split_inputs_cp(camera, seq_dim=-2, cp_group=self._cp_group)
        return camera

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Route to the native executor when one is configured.

        **``input`` means different things in the two models.** omnidreams
        passes an HDMap here and the executor forwards it as ``hdmap_patched``;
        CMD passes *camera* tokens (see ``_predict_flow``, ``camera=input``).
        Handing CMD's camera straight to the executor would feed it to the HDMap
        branch -- accepted without complaint, and wrong. So camera is
        intercepted here and an explicitly empty HDMap goes down instead.
        """
        executor = self._optimized_dit_executor
        if executor is None:
            return super().predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                input=input,
            )
        if self.config.network.camera_dim is not None:
            raise NotImplementedError(
                "camera-conditioned CMD on the native path needs the camera "
                "producer, which is not wired up yet; the kernels and transport "
                "are in place (dev/camera-inject-f62a0d) but nothing fills the "
                "buffer. Use a camera-free preset for now."
            )
        if input is not None:
            raise ValueError(
                "a camera tensor was supplied to a CMD model with "
                "camera_dim=None; the native path has nowhere to put it"
            )
        return executor.predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=native_adapter.empty_hdmap_like(noisy_latent),
        )

    def finalize_kv_cache(self, *args: Any, **kwargs: Any) -> None:
        try:
            super().finalize_kv_cache(*args, **kwargs)
        finally:
            if self._optimized_dit_executor is not None:
                # Runs even when skip_finalize_kv_cache short-circuits the base:
                # the executor's per-chunk caches are keyed on the AR index and
                # must be dropped either way.
                self._optimized_dit_executor.after_finalize_kv_cache()

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
        if self._optimized_dit_executor is not None:
            # Must run after prefill: the executor snapshots the cache templates
            # here, and it resets the CUDA graph in the same call, so anything
            # allocated per rollout is released in lockstep with the graph.
            #
            # Note prefill itself stays on the eager network. It runs once per
            # rollout at a different token count than the streaming steps, so
            # routing it natively would be a separate piece of work.
            self._optimized_dit_executor.after_initialize_autoregressive_cache(cache)
        return cache
