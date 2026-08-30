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

"""Per-step camera-token slicing for CMD rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from flashdreams.infra.encoder import (
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)


@dataclass(kw_only=True)
class CMDCameraEncoderConfig(EncoderConfig):
    """Config for selecting one generated camera-control chunk."""

    _target: type["CMDCameraEncoder"] = field(default_factory=lambda: CMDCameraEncoder)

    len_t: int = 1
    """Generated latent frames per AR chunk."""

    prefix_len_t: int = 1
    """Independent camera-prefix frames preceding generated chunks."""


class CMDCameraEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Select camera tokens for the current generated block."""

    config: CMDCameraEncoderConfig

    def __init__(self, config: CMDCameraEncoderConfig) -> None:
        super().__init__(config)
        self.config = config

    def initialize_autoregressive_cache(self) -> StreamingEncoderCache:
        """Build the stateless per-rollout encoder cache."""
        return StreamingEncoderCache()

    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> Tensor:
        """Return the camera-token slice for one AR step.

        Args:
            input: Full camera sequence shaped ``[..., T, C, H, W]``.
            autoregressive_index: Generated chunk index.
            cache: Stateless cache required by the streaming-encoder contract.

        Returns:
            Current chunk shaped ``[..., len_t, C, H, W]``.
        """
        assert cache is not None, "CMDCameraEncoder requires an encoder cache"
        start = self.config.prefix_len_t + autoregressive_index * self.config.len_t
        end = start + self.config.len_t
        if input.shape[-4] < end:
            raise ValueError(
                "Camera conditioning is too short for AR step "
                f"{autoregressive_index}: need {end}, got {input.shape[-4]} latent frames"
            )
        return input[..., start:end, :, :, :]
