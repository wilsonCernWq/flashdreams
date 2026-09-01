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

"""CPU tests for CMD <-> native-bridge shape and config translation.

Tier T0 of ``integrations/cmd/docs/native_fp8_port_plan.md`` Phase 3. The rank
mismatch (R5) and the per-token-timestep cap (R7) are both silent-corruption
risks: the first would reinterpret memory, the second would quietly drop a
per-token noise schedule. Both are asserted here without a GPU.
"""

from __future__ import annotations

import pytest
import torch
from flashdreams_cmd.config import CMD_CONFIGS
from flashdreams_cmd.transformer import CMDDiTNetworkConfig, CMDTransformerConfig
from flashdreams_cmd.transformer.native_adapter import (
    BRIDGE_BATCH_SIZE,
    BRIDGE_NUM_VIEWS,
    NativeLayoutError,
    empty_hdmap_like,
    from_bridge_latent,
    latent_grid,
    native_num_views,
    resolve_patch_dims,
    to_bridge_latent,
)
from flashdreams_cmd.transformer.native_fp8 import (
    NativeFP8Unavailable,
    check_model_supports_native_fp8,
    resolve_native_fp8,
)

pytestmark = pytest.mark.ci_cpu


def _config(**overrides) -> CMDTransformerConfig:
    network_overrides = overrides.pop("network", {})
    network = CMDDiTNetworkConfig(camera_dim=None, **network_overrides)
    return CMDTransformerConfig(network=network, **overrides)


def _preset(name: str) -> CMDTransformerConfig:
    transformer = CMD_CONFIGS[name].diffusion_model.transformer
    assert isinstance(transformer, CMDTransformerConfig)
    return transformer


# --------------------------------------------------------------------------
# Patch-dimension translation
# --------------------------------------------------------------------------


def test_patch_dims_are_derived_from_the_shared_patch_size_tuple() -> None:
    """The shared config has patch_size; only the module derives kt/ks."""
    config = _config()
    assert resolve_patch_dims(config.network) == (
        config.network.patch_size[0],
        config.network.patch_size[1],
    )


def test_non_square_spatial_patches_are_rejected() -> None:
    config = _config(network={"patch_size": (1, 2, 4)})
    with pytest.raises(NativeLayoutError, match="square spatial patches"):
        resolve_patch_dims(config.network)


def test_malformed_patch_size_is_rejected() -> None:
    config = _config()
    object.__setattr__(config.network, "patch_size", (1, 2))
    with pytest.raises(NativeLayoutError, match=r"\(t, h, w\)"):
        resolve_patch_dims(config.network)


def test_num_views_defaults_to_one_for_cmd() -> None:
    """omnidreams' executor reads config.num_views; CMD has no such field."""
    assert native_num_views(_config()) == BRIDGE_NUM_VIEWS


# --------------------------------------------------------------------------
# Rank normalisation (R5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CMD_CONFIGS))
def test_rank_round_trip_for_every_released_preset(name: str) -> None:
    """CMD rank-2 latent -> 5-D bridge input -> back, bit-identical."""
    config = _preset(name)
    height, width = 64, 96
    temporal, spatial = latent_grid(config, height=height, width=width)

    patch_t, patch_s, _ = config.network.patch_size
    channels = config.network.out_channels * patch_t * patch_s * patch_s
    latent = torch.randn(temporal * spatial, channels)

    bridged = to_bridge_latent(
        latent, temporal_positions=temporal, spatial_positions=spatial
    )
    assert bridged.shape == (
        BRIDGE_BATCH_SIZE,
        BRIDGE_NUM_VIEWS,
        temporal,
        spatial,
        channels,
    )

    restored = from_bridge_latent(bridged, batch_shape=config.batch_shape)
    assert restored.shape == latent.shape
    assert torch.equal(restored, latent)


def test_every_released_preset_uses_the_rank_two_layout() -> None:
    """The rank-2 case is not hypothetical; it is what all presets produce."""
    for name in CMD_CONFIGS:
        assert _preset(name).batch_shape == ()


def test_rank_three_latent_with_batch_one_is_accepted() -> None:
    latent = torch.randn(1, 12, 8)
    bridged = to_bridge_latent(latent, temporal_positions=3, spatial_positions=4)
    assert bridged.shape == (1, 1, 3, 4, 8)
    assert torch.equal(from_bridge_latent(bridged, batch_shape=(1,)), latent)


def test_batch_larger_than_one_is_rejected() -> None:
    latent = torch.randn(2, 12, 8)
    with pytest.raises(NativeLayoutError, match="batch size"):
        to_bridge_latent(latent, temporal_positions=3, spatial_positions=4)


def test_sequence_length_mismatch_is_rejected() -> None:
    """Guards against reinterpreting memory with the wrong T/HW split."""
    latent = torch.randn(13, 8)
    with pytest.raises(NativeLayoutError, match="does not match"):
        to_bridge_latent(latent, temporal_positions=3, spatial_positions=4)


@pytest.mark.parametrize("rank", [1, 4])
def test_unsupported_latent_rank_is_rejected(rank: int) -> None:
    latent = torch.randn(*([2] * rank))
    with pytest.raises(NativeLayoutError, match="rank 2 or 3"):
        to_bridge_latent(latent, temporal_positions=1, spatial_positions=2)


def test_from_bridge_latent_rejects_wrong_rank() -> None:
    with pytest.raises(NativeLayoutError, match="rank 5"):
        from_bridge_latent(torch.randn(2, 3), batch_shape=())


def test_from_bridge_latent_rejects_multi_view() -> None:
    with pytest.raises(NativeLayoutError, match="V=1"):
        from_bridge_latent(torch.randn(1, 2, 3, 4, 8), batch_shape=())


def test_latent_grid_rejects_indivisible_spatial_dims() -> None:
    config = _preset("cmd-chunk4-short-i2v")
    with pytest.raises(NativeLayoutError, match="divisible"):
        latent_grid(config, height=65, width=64)


# --------------------------------------------------------------------------
# HDMap placeholder
# --------------------------------------------------------------------------


def test_empty_hdmap_has_zero_width_and_matching_leading_shape() -> None:
    """`input=None` yields a Python None the pybind signature will not accept."""
    latent = torch.randn(12, 8, dtype=torch.bfloat16)
    hdmap = empty_hdmap_like(latent)
    assert hdmap.shape == (12, 0)
    assert hdmap.dtype == latent.dtype


# --------------------------------------------------------------------------
# Per-token timesteps (R7) — a capability cap, enforced as a refusal
# --------------------------------------------------------------------------


def test_per_token_timesteps_disqualify_the_native_path() -> None:
    config = _config()
    object.__setattr__(config, "conditional_frame_timestep", 0.5)
    outcome = check_model_supports_native_fp8(config)
    assert not outcome
    assert "per-token" in outcome.reason


def test_per_token_timesteps_raise_in_required_mode() -> None:
    config = _config()
    object.__setattr__(config, "conditional_frame_timestep", 0.5)
    with pytest.raises(NativeFP8Unavailable, match="per-token"):
        resolve_native_fp8(config, mode="required")


def test_released_presets_do_not_use_per_token_timesteps() -> None:
    """So the R7 cap blocks no shipped preset today."""
    for name in CMD_CONFIGS:
        assert _preset(name).conditional_frame_timestep is None
