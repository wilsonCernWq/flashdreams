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

"""Shape and config translation between CMD and the native FP8 bridge.

Phase 3 of ``integrations/cmd/docs/native_fp8_port_plan.md``. CMD and
omnidreams describe the same architecture with different conventions, and the
gaps are all mechanical — but each one is a silent-corruption risk if it is
guessed rather than translated:

Rank
    The bridge indexes latents as 5-D ``[B, V, T, HW, D]`` and requires ``V == 1``
    and ``B == 1`` (``streaming_dit_bridge.cu:1556-1624, 1747-1750``). CMD
    patchifies to ``[..., L, D]`` and **every released preset sets
    ``batch_shape=()``**, so the tensor that actually arrives is rank-2. The
    adapter inserts the batch/view axes and splits ``L`` into ``(T, HW)``.

Patch dimensions
    The shared config exposes ``patch_size: tuple[int, int, int]``
    (``impl/network.py:109``); only the *module* derives ``patch_temporal`` /
    ``patch_spatial`` (``:165-169``). omnidreams' executor reads the derived
    names off the *config*. Translate here rather than adding omnidreams' field
    names to the shared config, which would leak an integration's vocabulary
    into ``recipes/``.

None of this needs a GPU, so all of it is asserted on CPU.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

BRIDGE_BATCH_SIZE = 1
"""The bridge validates ``B == 1`` (``streaming_dit_bridge.cu:1747-1750``)."""

BRIDGE_NUM_VIEWS = 1
"""Single-view is the only supported mode (``:1622``); CMD is never multi-view."""


class NativeLayoutError(ValueError):
    """A tensor or config cannot be expressed in the bridge's layout."""


def resolve_patch_dims(network_config: Any) -> tuple[int, int]:
    """Return ``(patch_temporal, patch_spatial)`` from a shared Cosmos config.

    The shared config stores ``patch_size`` as ``(t, h, w)`` and asserts
    ``h == w`` when the module is built (``impl/network.py:165-167``); this
    reproduces that assertion so a bad config fails during translation rather
    than deep inside a kernel.
    """
    patch_size = getattr(network_config, "patch_size", None)
    if patch_size is None:
        raise NativeLayoutError("network config has no patch_size")
    if len(patch_size) != 3:
        raise NativeLayoutError(
            f"patch_size must be (t, h, w); got {tuple(patch_size)}"
        )
    patch_temporal, patch_h, patch_w = patch_size
    if patch_h != patch_w:
        raise NativeLayoutError(
            f"native FP8 requires square spatial patches; got {patch_h}x{patch_w}"
        )
    return int(patch_temporal), int(patch_h)


def latent_grid(
    transformer_config: Any,
    *,
    height: int,
    width: int,
) -> tuple[int, int]:
    """Return ``(temporal_positions, spatial_positions)`` for a rollout.

    Mirrors the shared ``latent_shape`` arithmetic
    (``recipes/cosmos/transformer/__init__.py:245-263``): ``L`` is
    ``(len_t // kt) * (height // ks) * (width // ks)``.
    """
    patch_temporal, patch_spatial = resolve_patch_dims(transformer_config.network)
    len_t = int(transformer_config.len_t)
    if len_t % patch_temporal:
        raise NativeLayoutError(
            f"len_t ({len_t}) must be divisible by patch_temporal ({patch_temporal})"
        )
    if height % patch_spatial or width % patch_spatial:
        raise NativeLayoutError(
            f"height/width ({height}x{width}) must be divisible by "
            f"patch_spatial ({patch_spatial})"
        )
    temporal = len_t // patch_temporal
    spatial = (height // patch_spatial) * (width // patch_spatial)
    return temporal, spatial


def to_bridge_latent(
    latent: Tensor,
    *,
    temporal_positions: int,
    spatial_positions: int,
) -> Tensor:
    """Reshape a CMD latent ``[..., L, D]`` into the bridge's ``[B, V, T, HW, D]``.

    Accepts rank-2 (``batch_shape=()``, what every released preset produces) and
    rank-3 (``batch_shape=(B,)``). Refuses anything the bridge would reject
    rather than silently reinterpreting memory.
    """
    if latent.dim() not in (2, 3):
        raise NativeLayoutError(
            f"expected a CMD latent of rank 2 or 3 [..., L, D]; got rank {latent.dim()}"
        )

    if latent.dim() == 3:
        batch = latent.shape[0]
        if batch != BRIDGE_BATCH_SIZE:
            raise NativeLayoutError(
                f"the native bridge requires batch size {BRIDGE_BATCH_SIZE}; got {batch}"
            )
        sequence, channels = latent.shape[1], latent.shape[2]
    else:
        sequence, channels = latent.shape[0], latent.shape[1]

    expected = temporal_positions * spatial_positions
    if sequence != expected:
        raise NativeLayoutError(
            f"latent sequence length {sequence} does not match "
            f"temporal_positions*spatial_positions ({temporal_positions}*"
            f"{spatial_positions}={expected})"
        )
    return latent.reshape(
        BRIDGE_BATCH_SIZE,
        BRIDGE_NUM_VIEWS,
        temporal_positions,
        spatial_positions,
        channels,
    )


def from_bridge_latent(
    tensor: Tensor,
    *,
    batch_shape: tuple[int, ...],
) -> Tensor:
    """Invert :func:`to_bridge_latent`, restoring CMD's leading batch shape."""
    if tensor.dim() != 5:
        raise NativeLayoutError(
            f"expected a bridge latent of rank 5 [B, V, T, HW, D]; got rank {tensor.dim()}"
        )
    batch, views = tensor.shape[0], tensor.shape[1]
    if batch != BRIDGE_BATCH_SIZE or views != BRIDGE_NUM_VIEWS:
        raise NativeLayoutError(f"expected [B=1, V=1, ...]; got B={batch}, V={views}")
    sequence = tensor.shape[2] * tensor.shape[3]
    channels = tensor.shape[4]
    return tensor.reshape(*batch_shape, sequence, channels)


def empty_hdmap_like(latent: Tensor) -> Tensor:
    """A zero-width HDMap tensor for a model without HDMap conditioning.

    ``predict_flow(..., input=None)`` leaves ``hdmap_patched`` as Python ``None``,
    but the pybind signature demands a real tensor even when
    ``additional_concat_ch == 0`` (the C++ side then guards on the zero width,
    ``streaming_dit_bridge.cu:1660``). Passing this keeps the call well-typed.
    """
    return torch.empty(
        (*latent.shape[:-1], 0), dtype=latent.dtype, device=latent.device
    )


def native_num_views(transformer_config: Any) -> int:
    """omnidreams' executor reads ``config.num_views``; CMD has no such field."""
    return int(getattr(transformer_config, "num_views", BRIDGE_NUM_VIEWS))


__all__ = [
    "BRIDGE_BATCH_SIZE",
    "BRIDGE_NUM_VIEWS",
    "NativeLayoutError",
    "empty_hdmap_like",
    "from_bridge_latent",
    "latent_grid",
    "native_num_views",
    "resolve_patch_dims",
    "to_bridge_latent",
]
