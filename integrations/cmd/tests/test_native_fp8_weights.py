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

"""CPU conformance tests for the native FP8 weights-dict contract.

Tier T0 of ``integrations/cmd/docs/native_fp8_port_plan.md`` Phase 2. Each test
here corresponds to a ``TORCH_CHECK`` the C++ bridge performs at kernel-launch
time on hardware this dev box does not have; the point is that a plumbing
mistake fails here, on CPU, instead of surfacing as an opaque CUDA error later.
"""

from __future__ import annotations

import pytest
import torch
from flashdreams_cmd.config import CMD_CONFIGS
from flashdreams_cmd.transformer import CMDDiTNetworkConfig, CMDTransformerConfig
from flashdreams_cmd.transformer.native_weights import (
    BLOCK_BF16_REL_KEYS,
    FUSED_QKV_REL_KEY,
    SPLIT_QKV_REL_KEYS,
    TOP_LEVEL_KEYS,
    NativeWeightContractError,
    assert_no_camera_weights,
    block_prefix,
    build_native_weights,
    camera_weight_keys,
    find_camera_weights,
    move_native_weights_to_device,
    validate_native_weights,
)

pytestmark = pytest.mark.ci_cpu

_NUM_BLOCKS = 2
_MODEL_CHANNELS = 64


def _network(camera_dim: int | None = None):
    """A small CMD network with the released architecture's key structure."""
    return CMDDiTNetworkConfig(
        in_channels=16,
        out_channels=16,
        model_channels=_MODEL_CHANNELS,
        num_blocks=_NUM_BLOCKS,
        num_heads=4,
        mlp_ratio=4.0,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_crossattn_projection=False,
        crossattn_emb_channels=32,
        camera_dim=camera_dim,
    ).setup()


def _state_dict(camera_dim: int | None = None) -> dict[str, torch.Tensor]:
    return dict(_network(camera_dim).to(torch.bfloat16).state_dict())


@pytest.fixture(scope="module")
def packed_weights() -> dict[str, torch.Tensor]:
    """A real quantized weights dict, built from a real CMD state dict."""
    pytest.importorskip("omnidreams", reason="native weight prep lives in omnidreams")
    return build_native_weights(_state_dict(), num_blocks=_NUM_BLOCKS)


def _validate(
    weights,
    *,
    num_blocks: int = _NUM_BLOCKS,
    model_channels: int = _MODEL_CHANNELS,
    quantized_prepared_strict: bool = True,
) -> None:
    validate_native_weights(
        weights,
        num_blocks=num_blocks,
        model_channels=model_channels,
        quantized_prepared_strict=quantized_prepared_strict,
    )


# --------------------------------------------------------------------------
# The happy path: a genuinely-packed CMD model satisfies the whole contract.
# --------------------------------------------------------------------------


def test_packed_cmd_weights_satisfy_the_bridge_contract(packed_weights) -> None:
    """The end-to-end assertion this phase exists to make."""
    _validate(packed_weights)


def test_packing_produces_fused_qkv_and_drops_the_split_projections(
    packed_weights,
) -> None:
    """``drop_split_self_attn_qkv`` defaults to True, so q/k/v are gone."""
    prefix = block_prefix(0)
    fused = packed_weights[prefix + FUSED_QKV_REL_KEY]
    assert fused.dtype == torch.uint8
    assert tuple(fused.shape) == (3 * _MODEL_CHANNELS, _MODEL_CHANNELS)
    for rel_key in SPLIT_QKV_REL_KEYS:
        assert prefix + rel_key not in packed_weights


def test_prepared_aliases_share_storage_with_their_canonical_tensors(
    packed_weights,
) -> None:
    """They are contract assertions, not copies — the memory dedup relies on it."""
    key = block_prefix(0) + "mlp.layer1.weight"
    assert (
        packed_weights[key].data_ptr()
        == packed_weights[key + "_fp8_prepared"].data_ptr()
    )
    assert (
        packed_weights[key + "_scale"].data_ptr()
        == packed_weights[key + "_fp8_prepared_scale"].data_ptr()
    )


# --------------------------------------------------------------------------
# cam_encoder: the failure this phase is chiefly designed to make impossible.
# --------------------------------------------------------------------------


def test_camera_state_dict_is_refused_rather_than_silently_stripped() -> None:
    """The bridge ignores unknown keys, so packing must refuse, not drop."""
    with pytest.raises(NativeWeightContractError, match="cam_encoder"):
        build_native_weights(_state_dict(camera_dim=6), num_blocks=_NUM_BLOCKS)


def test_camera_weights_are_detectable_in_a_state_dict() -> None:
    state = _state_dict(camera_dim=6)
    assert find_camera_weights(state, num_blocks=_NUM_BLOCKS) == camera_weight_keys(
        _NUM_BLOCKS
    )


def test_non_camera_state_dict_has_no_camera_weights() -> None:
    assert find_camera_weights(_state_dict(), num_blocks=_NUM_BLOCKS) == ()


def test_assert_no_camera_weights_rejects_a_smuggled_camera_tensor(
    packed_weights,
) -> None:
    """Guards the packed dict itself, not just its input."""
    smuggled = dict(packed_weights)
    smuggled[block_prefix(0) + "self_attn.cam_encoder.weight"] = torch.zeros(
        _MODEL_CHANNELS, 6, dtype=torch.bfloat16
    )
    with pytest.raises(NativeWeightContractError, match="camera-blind"):
        assert_no_camera_weights(smuggled, num_blocks=_NUM_BLOCKS)


def test_every_camera_preset_is_refused_and_every_other_preset_is_not() -> None:
    """Preset-level coverage, so a preset gaining camera cannot slip through."""
    camera, plain = [], []
    for name in CMD_CONFIGS:
        transformer = CMD_CONFIGS[name].diffusion_model.transformer
        assert isinstance(transformer, CMDTransformerConfig)
        target = camera if transformer.network.camera_dim is not None else plain
        target.append(name)
    assert camera, "expected at least one camera-conditioned preset"
    assert plain, "expected at least one preset without camera conditioning"


# --------------------------------------------------------------------------
# Each bridge TORCH_CHECK, exercised by violating it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", TOP_LEVEL_KEYS)
def test_missing_top_level_key_is_rejected(packed_weights, missing: str) -> None:
    broken = {k: v for k, v in packed_weights.items() if k != missing}
    with pytest.raises(NativeWeightContractError, match="missing weight key"):
        _validate(broken)


@pytest.mark.parametrize(
    "rel_key",
    ["self_attn.output_proj.weight", "mlp.layer1.weight", "cross_attn.q_proj.weight"],
)
def test_missing_block_linear_is_rejected(packed_weights, rel_key: str) -> None:
    key = block_prefix(1) + rel_key
    broken = {k: v for k, v in packed_weights.items() if k != key}
    with pytest.raises(NativeWeightContractError, match="missing weight key"):
        _validate(broken)


def test_bf16_linear_is_rejected_under_fp8_backend(packed_weights) -> None:
    key = block_prefix(0) + "mlp.layer2.weight"
    broken = dict(packed_weights)
    broken[key] = torch.zeros(64, 256, dtype=torch.bfloat16)
    broken.pop(key + "_fp8_prepared", None)
    with pytest.raises(NativeWeightContractError, match="must be torch.uint8"):
        _validate(broken)


def test_missing_prepared_alias_is_rejected_when_strict(packed_weights) -> None:
    key = block_prefix(0) + "mlp.layer1.weight"
    broken = {k: v for k, v in packed_weights.items() if k != key + "_fp8_prepared"}
    with pytest.raises(NativeWeightContractError, match="missing FP8 prepared alias"):
        _validate(broken)


def test_missing_prepared_alias_is_tolerated_when_not_strict(packed_weights) -> None:
    """Non-strict mode falls back to the canonical tensor (:2555-2559)."""
    key = block_prefix(0) + "mlp.layer1.weight"
    relaxed = {
        k: v
        for k, v in packed_weights.items()
        if k not in (key + "_fp8_prepared", key + "_fp8_prepared_scale")
    }
    _validate(relaxed, quantized_prepared_strict=False)


def test_prepared_alias_with_wrong_shape_is_rejected(packed_weights) -> None:
    key = block_prefix(0) + "mlp.layer1.weight"
    broken = dict(packed_weights)
    broken[key + "_fp8_prepared"] = torch.zeros(8, 8, dtype=torch.uint8)
    with pytest.raises(NativeWeightContractError, match="same shape"):
        _validate(broken)


def test_prepared_alias_with_wrong_dtype_is_rejected(packed_weights) -> None:
    key = block_prefix(0) + "mlp.layer1.weight"
    canonical = packed_weights[key]
    broken = dict(packed_weights)
    broken[key + "_fp8_prepared"] = torch.zeros(
        tuple(canonical.shape), dtype=torch.bfloat16
    )
    with pytest.raises(NativeWeightContractError, match="raw E4M3 bytes"):
        _validate(broken)


def test_non_contiguous_prepared_alias_is_rejected(packed_weights) -> None:
    key = block_prefix(0) + "mlp.layer1.weight"
    canonical = packed_weights[key]
    wide = torch.zeros((canonical.shape[0], canonical.shape[1] * 2), dtype=torch.uint8)
    broken = dict(packed_weights)
    broken[key + "_fp8_prepared"] = wide[:, ::2]
    with pytest.raises(NativeWeightContractError, match="must be contiguous"):
        _validate(broken)


def test_prepared_scale_must_be_float16(packed_weights) -> None:
    """Only the *prepared* scale is dtype-strict; the canonical one is cast."""
    key = block_prefix(0) + "mlp.layer1.weight"
    broken = dict(packed_weights)
    broken[key + "_fp8_prepared_scale"] = packed_weights[
        key + "_fp8_prepared_scale"
    ].to(torch.float32)
    with pytest.raises(NativeWeightContractError, match="must be torch.float16"):
        _validate(broken)


def test_canonical_scale_may_be_any_float_dtype(packed_weights) -> None:
    """The bridge casts it (:2606-2608), so float32 must be accepted."""
    key = block_prefix(0) + "mlp.layer1.weight"
    relaxed = {
        k: v for k, v in packed_weights.items() if k != key + "_fp8_prepared_scale"
    }
    relaxed[key + "_scale"] = packed_weights[key + "_scale"].to(torch.float32)
    _validate(relaxed, quantized_prepared_strict=False)


def test_scale_with_wrong_element_count_is_rejected(packed_weights) -> None:
    key = block_prefix(0) + "mlp.layer1.weight"
    broken = dict(packed_weights)
    broken[key + "_fp8_prepared_scale"] = torch.zeros(3, dtype=torch.float16)
    with pytest.raises(NativeWeightContractError, match="elements"):
        _validate(broken)


def test_two_dimensional_scale_is_rejected(packed_weights) -> None:
    key = block_prefix(0) + "mlp.layer1.weight"
    broken = dict(packed_weights)
    rows = packed_weights[key].shape[0]
    broken[key + "_fp8_prepared_scale"] = torch.zeros(rows, 1, dtype=torch.float16)
    with pytest.raises(NativeWeightContractError, match=r"\[out_features\]"):
        _validate(broken)


def test_missing_scale_is_rejected(packed_weights) -> None:
    key = block_prefix(0) + "self_attn.output_proj.weight"
    broken = {
        k: v
        for k, v in packed_weights.items()
        if k not in (key + "_scale", key + "_fp8_prepared_scale")
    }
    with pytest.raises(NativeWeightContractError, match="missing FP8 prepared scale"):
        _validate(broken)


def test_fused_qkv_with_wrong_shape_is_rejected(packed_weights) -> None:
    key = block_prefix(0) + FUSED_QKV_REL_KEY
    broken = dict(packed_weights)
    broken[key] = torch.zeros(2 * _MODEL_CHANNELS, _MODEL_CHANNELS, dtype=torch.uint8)
    broken[key + "_fp8_prepared"] = broken[key]
    broken[key + "_fp8_prepared_scale"] = torch.zeros(
        2 * _MODEL_CHANNELS, dtype=torch.float16
    )
    broken[key + "_scale"] = broken[key + "_fp8_prepared_scale"]
    with pytest.raises(NativeWeightContractError, match="must have shape"):
        _validate(broken)


def test_bf16_only_block_tensor_must_stay_bf16(packed_weights) -> None:
    key = block_prefix(0) + BLOCK_BF16_REL_KEYS[0]
    broken = dict(packed_weights)
    broken[key] = packed_weights[key].to(torch.float32)
    with pytest.raises(NativeWeightContractError, match="raw bf16 pointer"):
        _validate(broken)


@pytest.mark.parametrize("rel_key", BLOCK_BF16_REL_KEYS)
def test_missing_bf16_block_tensor_is_rejected(packed_weights, rel_key: str) -> None:
    key = block_prefix(1) + rel_key
    broken = {k: v for k, v in packed_weights.items() if k != key}
    with pytest.raises(NativeWeightContractError, match="missing weight key"):
        _validate(broken)


def test_non_tensor_value_is_rejected(packed_weights) -> None:
    broken = dict(packed_weights)
    broken[TOP_LEVEL_KEYS[0]] = "not a tensor"
    with pytest.raises(NativeWeightContractError, match="must be a torch.Tensor"):
        _validate(broken)


def test_zero_blocks_is_rejected(packed_weights) -> None:
    with pytest.raises(NativeWeightContractError, match="num_blocks must be positive"):
        _validate(packed_weights, num_blocks=0)


# --------------------------------------------------------------------------
# Device move: aliased storage must survive, or FP8 costs as much as BF16.
# --------------------------------------------------------------------------


def test_device_move_preserves_alias_sharing(packed_weights) -> None:
    """A naive per-entry ``.to()`` would double the GPU footprint."""
    weights = dict(packed_weights)
    deduplicated = move_native_weights_to_device(weights, device="meta")

    assert deduplicated > 0, "expected prepared aliases to be recognised as aliases"
    key = block_prefix(0) + "mlp.layer1.weight"
    assert weights[key] is weights[key + "_fp8_prepared"]
    assert weights[key + "_scale"] is weights[key + "_fp8_prepared_scale"]
    assert all(
        v.device.type == "meta" for v in weights.values() if isinstance(v, torch.Tensor)
    )


def test_device_move_is_a_noop_when_already_on_target(packed_weights) -> None:
    weights = dict(packed_weights)
    assert move_native_weights_to_device(weights, device="cpu") == 0
