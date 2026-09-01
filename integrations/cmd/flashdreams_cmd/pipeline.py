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

"""CMD causal I2V pipeline with optional camera-ray conditioning."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.encoder import StreamingEncoderCache
from flashdreams.infra.encoder.text.cosmos_reason1 import CosmosReason1TextEncoder
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
)
from flashdreams.infra.profiler import EventProfiler
from flashdreams.recipes.cosmos.pipeline import CosmosInferencePipelineConfig
from flashdreams.recipes.cosmos.transformer.constants import NEGATIVE_PROMPT
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAECache,
    WanVAEEncoder,
)

from .camera import build_camera_conditioning
from .encoder import CMDCamCtrlInput, CMDLiveCameraEncoder
from .transformer import CMDTransformerCache, CMDTransformerConfig


@dataclass(kw_only=True)
class CMDInferencePipelineCache(
    StreamInferencePipelineCache[
        StreamingEncoderCache,
        CMDTransformerCache,
        WanVAECache,
    ]
):
    """Per-rollout CMD state in addition to component caches."""

    transformer_cache: CMDTransformerCache
    """CMD transformer cache with the independent prefix already seeded."""

    prefix_image_embeddings: Tensor
    """Clean first-frame latent prepended to the first decoder chunk."""

    camera_condition: Tensor | None = None
    """Full latent-rate camera sequence, including the first-frame prefix."""


@dataclass(kw_only=True)
class CMDInferencePipelineConfig(CosmosInferencePipelineConfig):
    """Configuration for CMD causal I2V inference."""

    _target: type["CMDInferencePipeline"] = field(
        default_factory=lambda: CMDInferencePipeline
    )

    camera_frame_stride: int = 4
    """Pixel-camera frames per temporally compressed latent interval."""

    camera_patch_size: int = 16
    """Spatial ray-patch size aligned with VAE and DiT downsampling."""


class CMDInferencePipeline(
    StreamInferencePipeline[
        StreamingEncoderCache,
        CMDTransformerCache,
        WanVAECache,
    ]
):
    """Stream CMD chunks after independently caching the clean first frame."""

    config: CMDInferencePipelineConfig
    text_encoder: CosmosReason1TextEncoder
    image_encoder: WanVAEEncoder

    def __init__(self, config: CMDInferencePipelineConfig) -> None:
        super().__init__(config)
        self.text_encoder = config.text_encoder.setup()
        if config.image_encoder is None:
            raise ValueError("CMD requires a configured first-frame image encoder")
        self.image_encoder = config.image_encoder.setup()

    @property
    def _cmd_transformer_config(self) -> CMDTransformerConfig:
        config = self.diffusion_model.transformer.config
        assert isinstance(config, CMDTransformerConfig)
        return config

    @property
    def camera_conditioned(self) -> bool:
        """Whether this variant requires a camera trajectory."""
        return self._cmd_transformer_config.network.camera_dim is not None

    @property
    def _live_camera_encoder(self) -> bool:
        """Whether this pipeline is wired for per-step live camera input."""
        return isinstance(self.encoder, CMDLiveCameraEncoder)

    @torch.no_grad()
    def initialize_cache(
        self,
        text: list[str],
        image: Tensor | None = None,
        *,
        camera_to_world: Tensor | None = None,
        intrinsics: Tensor | None = None,
        expected_latent_frames: int | None = None,
    ) -> CMDInferencePipelineCache:
        """Initialize text, first-frame, camera, and streaming component caches.

        Args:
            text: One prompt per configured batch element.
            image: First-frame pixels shaped ``[..., 1, 3, H, W]`` in
                ``[-1, 1]``.
            camera_to_world: Optional pixel-rate poses shaped
                ``[..., T, 4, 4]``. Required by camera variants.
            intrinsics: Optional constant or pixel-rate calibration matrices.
                Required by camera variants.
            expected_latent_frames: Optional exact camera length after temporal
                sampling, including the first-frame prefix.

        Returns:
            Initialized CMD rollout cache.
        """
        if not text:
            raise ValueError("text must be non-empty")
        if image is None:
            raise ValueError("CMD is I2V-only and requires a first-frame image")
        if self.image_encoder is None:
            raise ValueError("CMD requires a configured first-frame image encoder")
        if image.shape[-4] != 1:
            raise ValueError(
                "image must contain exactly one temporal frame; got "
                f"shape {tuple(image.shape)}"
            )

        text_embeddings = self.text_encoder(text)
        if self._cmd_transformer_config.guidance_scale > 1.0:
            negative_text_embeddings = self.text_encoder([NEGATIVE_PROMPT] * len(text))
        else:
            negative_text_embeddings = None
        image_embeddings = self.image_encoder(image)

        if not isinstance(self.decoder, StreamingVideoDecoder):
            raise TypeError("CMD requires a StreamingVideoDecoder")
        spatial_ratio = self.decoder.spatial_compression_ratio
        pixel_height, pixel_width = image.shape[-2:]
        if pixel_height % spatial_ratio or pixel_width % spatial_ratio:
            raise ValueError(
                f"image size {(pixel_height, pixel_width)} must be divisible by "
                f"decoder spatial ratio {spatial_ratio}"
            )
        latent_height = pixel_height // spatial_ratio
        latent_width = pixel_width // spatial_ratio

        camera_condition: Tensor | None = None
        if self.camera_conditioned and not self._live_camera_encoder:
            if camera_to_world is None or intrinsics is None:
                raise ValueError(
                    "camera-conditioned CMD requires camera_to_world and intrinsics"
                )
            camera_condition = build_camera_conditioning(
                camera_to_world.to(device=self.device, dtype=torch.float32),
                intrinsics.to(device=self.device, dtype=torch.float32),
                image_height=pixel_height,
                image_width=pixel_width,
                frame_stride=self.config.camera_frame_stride,
                patch_size=self.config.camera_patch_size,
                block_size=self._cmd_transformer_config.len_t,
                expected_latent_frames=expected_latent_frames,
                output_dtype=self._cmd_transformer_config.dtype,
            )
        elif self._live_camera_encoder:
            if camera_to_world is not None or intrinsics is not None:
                raise ValueError(
                    "live camera-conditioned CMD does not take a fixed camera "
                    "trajectory; camera comes from per-step live input"
                )
            assert isinstance(self.encoder, CMDLiveCameraEncoder)
            # CMD's independent first-frame prefix is prefilled once here,
            # separately from the per-step tokens produced by `generate()`;
            # it needs its own (identity-pose) camera token too.
            camera_condition = self.encoder.prefix_camera_tokens(device=self.device)
        elif camera_to_world is not None or intrinsics is not None:
            raise ValueError("camera inputs were provided to a non-camera CMD variant")

        encoder_context = {"device": self.device} if self._live_camera_encoder else {}
        parent = super().initialize_cache(
            transformer_context={
                "height": latent_height,
                "width": latent_width,
                "text_embeddings": text_embeddings,
                "negative_text_embeddings": negative_text_embeddings,
                "image_embeddings": image_embeddings,
                "camera_condition": camera_condition,
            },
            encoder_context=encoder_context,
        )
        assert isinstance(parent.transformer_cache, CMDTransformerCache)
        return CMDInferencePipelineCache(
            transformer_cache=parent.transformer_cache,
            encoder_cache=parent.encoder_cache,
            decoder_cache=parent.decoder_cache,
            prefix_image_embeddings=image_embeddings,
            camera_condition=camera_condition,
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: CMDInferencePipelineCache,
        input: CMDCamCtrlInput | None = None,
    ) -> Tensor:
        """Generate and decode one CMD autoregressive chunk.

        Args:
            autoregressive_index: This AR step's index.
            cache: The rollout cache from :meth:`initialize_cache`.
            input: Optional live camera payload for this step, consumed only
                when the pipeline is wired with a
                :class:`~.encoder.CMDLiveCameraEncoder`. Ignored (must be
                ``None``) for every other rollout, which conditions on
                ``cache.camera_condition`` instead.
        """
        previous = cache.autoregressive_index
        expected = previous + 1 if previous is not None else 0
        if autoregressive_index != expected:
            raise ValueError(
                f"AR step out of order: expected {expected}, got {autoregressive_index}"
            )
        cache.autoregressive_index = autoregressive_index

        events: EventProfiler | None = None
        if self.config.enable_sync_and_profile:
            events = EventProfiler()
            cache.event_profiler = events

        camera_chunk: Tensor | None = None
        if self._live_camera_encoder:
            if input is None:
                raise ValueError(
                    "this pipeline is wired with a CMDLiveCameraEncoder, which "
                    "requires a per-step CMDCamCtrlInput on every generate() call"
                )
            if self.encoder is None or cache.encoder_cache is None:
                raise RuntimeError(
                    "live camera input was provided but no camera encoder is configured"
                )
            camera_chunk = self.encoder(
                input=input,
                autoregressive_index=autoregressive_index,
                cache=cache.encoder_cache,
            )
        elif input is not None:
            raise ValueError(
                "a live camera input was provided, but this pipeline is not "
                "wired with a CMDLiveCameraEncoder; camera conditioning comes "
                "from cache.camera_condition instead"
            )
        elif cache.camera_condition is not None:
            if self.encoder is None or cache.encoder_cache is None:
                raise RuntimeError(
                    "camera condition exists without a configured encoder cache"
                )
            camera_chunk = self.encoder(
                input=cache.camera_condition,
                autoregressive_index=autoregressive_index,
                cache=cache.encoder_cache,
            )
        if events is not None:
            events.record("encode")

        clean_latent, final_state = self.diffusion_model.generate(
            autoregressive_index=autoregressive_index,
            cache=cache.transformer_cache,
            input=camera_chunk,
        )
        cache.final_state = final_state
        if events is not None:
            events.record("diffuse")

        decoder_input = clean_latent
        if autoregressive_index == 0:
            decoder_input = torch.cat(
                (cache.prefix_image_embeddings, clean_latent),
                dim=-4,
            )
        if self.decoder is None or cache.decoder_cache is None:
            raise RuntimeError("CMD requires an initialized streaming decoder")
        output = self.decoder(
            input=decoder_input,
            autoregressive_index=autoregressive_index,
            cache=cache.decoder_cache,
        )
        if events is not None:
            events.record("decode")
        return output

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        """Return decoded pixel frames emitted for one CMD chunk."""
        if not isinstance(self.decoder, StreamingVideoDecoder):
            raise TypeError("CMD requires a StreamingVideoDecoder")
        latent_frames = self._cmd_transformer_config.len_t
        if autoregressive_index == 0:
            latent_frames += self._cmd_transformer_config.prefix_len_t
        return self.decoder.get_output_temporal_size(
            autoregressive_index,
            latent_frames,
        )


__all__ = [
    "CMDInferencePipeline",
    "CMDInferencePipelineCache",
    "CMDInferencePipelineConfig",
]
