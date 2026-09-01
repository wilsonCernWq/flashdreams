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

"""Weights-dict contract for CMD's native CUTLASS FP8 DiT path.

Phase 2 of ``integrations/cmd/docs/native_fp8_port_plan.md``. The C++ bridge
(``omnidreams_singleview/src/dit_streaming/pyext/streaming_dit_bridge.cu``)
consumes a plain ``dict[str, Tensor]`` and enforces its expectations with
``TORCH_CHECK`` at kernel-launch time, on a GPU this repo's dev box cannot even
run. :func:`validate_native_weights` re-implements every one of those checks in
pure Python so the same failures surface on CPU, in ``ci_cpu``, with a readable
message and no compiled extension.

The point is diagnostic separation: if this validator passes and the real kernel
still fails, the problem is *numerics*, not *plumbing*.

``cam_encoder`` is the reason this module leads with a refusal rather than a
best-effort pack. The bridge resolves weights by literal name and ignores keys
it does not know (``get_w``, ``streaming_dit_bridge.cu:269-273``), so a
camera-conditioned CMD model packed for the native path would lose its camera
projection *silently* and render camera-blind video. :func:`build_native_weights`
therefore refuses outright rather than emitting a dict that looks complete.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any, Literal

import torch
from torch import Tensor

CosmosLinearBackend = Literal["fp8", "bf16", "mixed"]

BLOCK_LINEAR_REL_KEYS: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
)
"""Block linears the bridge quantizes (``streaming_dit_bridge.cu:1878-1887``)."""

FUSED_QKV_REL_KEY = "self_attn.qkv_proj.weight"
"""Optional fused Q/K/V; only honoured when present *and* ``uint8`` (:1892-1894)."""

SPLIT_QKV_REL_KEYS: tuple[str, ...] = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
)

BLOCK_BF16_REL_KEYS: tuple[str, ...] = (
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "cross_attn.q_norm.weight",
    "adaln_modulation_self_attn.1.weight",
    "adaln_modulation_self_attn.2.weight",
    "adaln_modulation_cross_attn.1.weight",
    "adaln_modulation_cross_attn.2.weight",
    "adaln_modulation_mlp.1.weight",
    "adaln_modulation_mlp.2.weight",
)
"""Per-block tensors read as raw bf16 pointers (:2699-2719); never quantized."""

TOP_LEVEL_KEYS: tuple[str, ...] = (
    "x_embedder.proj.1.weight",
    "t_embedder.1.linear_1.weight",
    "t_embedder.1.linear_2.weight",
    "t_embedding_norm.weight",
    "final_layer.adaln_modulation.1.weight",
    "final_layer.adaln_modulation.2.weight",
    "final_layer.linear.weight",
)
"""Non-block tensors the bridge fetches unconditionally (:1655-1714, :3039-3075)."""

CAMERA_REL_KEY = "self_attn.cam_encoder.weight"
"""CMD-only per-block camera projection. The bridge has no slot for this."""

FP8_PREPARED_SUFFIX = "_fp8_prepared"
FP8_PREPARED_SCALE_SUFFIX = "_fp8_prepared_scale"
SCALE_SUFFIX = "_scale"


class NativeWeightContractError(ValueError):
    """A weights dict violates a check the C++ bridge would enforce."""


def block_prefix(block_idx: int) -> str:
    """Mirror ``block_prefix`` (``streaming_dit_bridge.cu:1111-1113``)."""
    return f"blocks.{block_idx}."


def camera_weight_keys(num_blocks: int) -> tuple[str, ...]:
    """Per-block ``cam_encoder`` keys a camera-conditioned CMD network carries."""
    return tuple(block_prefix(i) + CAMERA_REL_KEY for i in range(num_blocks))


def find_camera_weights(
    weights: Mapping[str, Any], *, num_blocks: int
) -> tuple[str, ...]:
    """Return any ``cam_encoder`` keys present, in block order.

    Used to prove the packing path cannot drop camera weights unnoticed: either
    they are absent because the model has no camera conditioning, or packing was
    refused. There is no third case in which they are silently discarded.
    """
    return tuple(key for key in camera_weight_keys(num_blocks) if key in weights)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeWeightContractError(message)


def _get(weights: Mapping[str, Any], key: str) -> Tensor:
    """Mirror ``get_w`` (:269-273): presence is a hard failure, not a default."""
    _require(key in weights, f"cosmos_forward: missing weight key '{key}'")
    value = weights[key]
    _require(
        isinstance(value, Tensor),
        f"{key} must be a torch.Tensor; got {type(value).__name__}",
    )
    return value


def _has_fused_qkv(weights: Mapping[str, Any], prefix: str) -> bool:
    """Fused QKV needs the key *and* ``uint8`` (:1892-1894, :2645-2647)."""
    key = prefix + FUSED_QKV_REL_KEY
    return key in weights and _get(weights, key).dtype == torch.uint8


def _validate_linear(
    weights: Mapping[str, Any],
    *,
    prefix: str,
    rel_key: str,
    linear_backend: CosmosLinearBackend,
    quantized_prepared: bool,
    quantized_prepared_strict: bool,
) -> Tensor:
    """Mirror ``get_block_linear_weight`` (:2536-2574). Returns the used tensor."""
    key = prefix + rel_key
    canonical = _get(weights, key)
    tensor = canonical

    if quantized_prepared and canonical.dtype == torch.uint8:
        prepared_key = key + FP8_PREPARED_SUFFIX
        if prepared_key in weights:
            tensor = _get(weights, prepared_key)
            _require(
                tensor.dtype == torch.uint8,
                f"{prepared_key} must be torch.uint8 raw E4M3 bytes; got {tensor.dtype}",
            )
            _require(
                tensor.dim() == canonical.dim(),
                f"{prepared_key} must have the same rank as {key}",
            )
            _require(
                tuple(tensor.shape) == tuple(canonical.shape),
                f"{prepared_key} must have the same shape as {key}",
            )
            _require(tensor.is_contiguous(), f"{prepared_key} must be contiguous")
        elif quantized_prepared_strict:
            _require(
                False,
                f"missing FP8 prepared alias {prepared_key} "
                "while cosmos_quantized_prepared_strict=True",
            )

    if linear_backend == "fp8":
        _require(
            tensor.dtype == torch.uint8,
            f"{key} must be torch.uint8 raw E4M3 bytes when "
            f"cosmos_linear_backend=fp8; got {tensor.dtype}",
        )
    elif linear_backend == "bf16":
        _require(
            tensor.dtype == torch.bfloat16,
            f"{key} must be torch.bfloat16 when cosmos_linear_backend=bf16; "
            f"got {tensor.dtype}",
        )
    else:
        _require(
            tensor.dtype in (torch.uint8, torch.bfloat16),
            f"{key} must be torch.bfloat16 or torch.uint8 raw E4M3 bytes when "
            f"cosmos_linear_backend=mixed; got {tensor.dtype}",
        )
    return tensor


def _validate_linear_scale(
    weights: Mapping[str, Any],
    *,
    prefix: str,
    rel_key: str,
    weight: Tensor,
    quantized_prepared: bool,
    quantized_prepared_strict: bool,
) -> None:
    """Mirror ``get_block_linear_scale`` (:2580-2611).

    A non-``uint8`` weight needs no scale at all, and a *canonical* scale may be
    any dtype (the bridge casts it). Only the ``_fp8_prepared_scale`` alias is
    required to be ``float16`` already.
    """
    if weight.dtype != torch.uint8:
        return

    key = prefix + rel_key + SCALE_SUFFIX
    using_prepared_scale = False
    if quantized_prepared:
        prepared_key = prefix + rel_key + FP8_PREPARED_SCALE_SUFFIX
        if prepared_key in weights:
            key = prepared_key
            using_prepared_scale = True
        elif quantized_prepared_strict:
            _require(
                False,
                f"missing FP8 prepared scale alias {prepared_key} "
                "while cosmos_quantized_prepared_strict=True",
            )

    scale = _get(weights, key)
    if using_prepared_scale:
        _require(
            scale.dtype == torch.float16,
            f"{key} must be torch.float16; got {scale.dtype}",
        )
    _require(scale.dim() == 1, f"{key} must be [out_features]")
    _require(
        scale.numel() == weight.shape[0],
        f"{key} must have {weight.shape[0]} elements, got {scale.numel()}",
    )
    _require(scale.is_contiguous(), f"{key} must be contiguous")


def validate_native_weights(
    weights: Mapping[str, Any],
    *,
    num_blocks: int,
    model_channels: int,
    linear_backend: CosmosLinearBackend = "fp8",
    quantized_prepared: bool = True,
    quantized_prepared_strict: bool = True,
) -> None:
    """Assert a weights dict satisfies every bridge-side ``TORCH_CHECK``.

    Raises :class:`NativeWeightContractError` on the first violation, with the
    bridge's own wording so a CPU failure and a GPU failure read the same.

    The defaults match what ``_ensure_fp8_runtime`` always sends
    (``optimized_dit.py:1202-1218``): FP8 linears with strict prepared aliases.
    """
    _require(num_blocks > 0, f"num_blocks must be positive, got {num_blocks}")

    for key in TOP_LEVEL_KEYS:
        _get(weights, key)

    for block_idx in range(num_blocks):
        prefix = block_prefix(block_idx)
        fused = _has_fused_qkv(weights, prefix)

        rel_keys: list[str] = []
        if fused:
            rel_keys.append(FUSED_QKV_REL_KEY)
        rel_keys.extend(
            rel_key
            for rel_key in BLOCK_LINEAR_REL_KEYS
            if not (fused and rel_key in SPLIT_QKV_REL_KEYS)
        )

        for rel_key in rel_keys:
            weight = _validate_linear(
                weights,
                prefix=prefix,
                rel_key=rel_key,
                linear_backend=linear_backend,
                quantized_prepared=quantized_prepared,
                quantized_prepared_strict=quantized_prepared_strict,
            )
            _validate_linear_scale(
                weights,
                prefix=prefix,
                rel_key=rel_key,
                weight=weight,
                quantized_prepared=quantized_prepared,
                quantized_prepared_strict=quantized_prepared_strict,
            )

        if fused:
            fused_key = prefix + FUSED_QKV_REL_KEY
            fused_weight = _get(weights, fused_key)
            expected = (3 * model_channels, model_channels)
            _require(
                fused_weight.dim() == 2 and tuple(fused_weight.shape) == expected,
                f"{fused_key} must have shape [{expected[0]}, {expected[1]}], "
                f"got {list(fused_weight.shape)}",
            )

        for rel_key in BLOCK_BF16_REL_KEYS:
            key = prefix + rel_key
            tensor = _get(weights, key)
            _require(
                tensor.dtype == torch.bfloat16,
                f"{key} is read as a raw bf16 pointer; got {tensor.dtype}",
            )


def assert_no_camera_weights(
    weights: Mapping[str, Any],
    *,
    num_blocks: int,
) -> None:
    """Fail if camera weights are present in a dict bound for the bridge.

    The bridge would ignore them silently, so their presence means the caller
    packed a camera-conditioned model that the native block cannot express.
    """
    present = find_camera_weights(weights, num_blocks=num_blocks)
    if present:
        raise NativeWeightContractError(
            f"native FP8 weights contain {len(present)} camera-projection tensor(s) "
            f"(e.g. {present[0]}) that the C++ bridge has no slot for and would "
            "ignore silently, producing camera-blind video. The native path must "
            "be refused for camera-conditioned models until port plan Phase 5."
        )


def build_native_weights(
    state_dict: Mapping[str, Tensor],
    *,
    num_blocks: int,
    linear_policy: str = "all",
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """Quantize a CMD ``state_dict`` into the bridge's weights dict.

    Delegates the quantization itself to omnidreams'
    ``prepare_cosmos_quantized_streaming_weights`` — CMD and omnidreams emit
    byte-identical state-dict keys, so no renaming is needed (verified against a
    real CMD network; see the port plan §1).

    Refuses camera-conditioned inputs up front: packing them would produce a
    dict that looks complete while having dropped every ``cam_encoder``.
    """
    camera_keys = find_camera_weights(state_dict, num_blocks=num_blocks)
    if camera_keys:
        raise NativeWeightContractError(
            f"refusing to pack native FP8 weights: {len(camera_keys)} camera "
            f"projection(s) present (e.g. {camera_keys[0]}). The native block has "
            "no cam_encoder equivalent and the bridge ignores unknown keys, so "
            "packing would silently discard camera conditioning."
        )

    prepare = _load_cosmos_fp8_utils().prepare_cosmos_quantized_streaming_weights
    weights = prepare(
        dict(state_dict),
        num_blocks=num_blocks,
        device=device,
        linear_policy=linear_policy,
    )
    assert_no_camera_weights(weights, num_blocks=num_blocks)
    return weights


def _load_cosmos_fp8_utils() -> Any:
    """Load omnidreams' FP8 helpers lazily.

    Kept function-local so importing this module never requires omnidreams to be
    installed — the validator above is pure Python and stays usable (and
    testable) without it.
    """
    try:
        from omnidreams.native import omnidreams_singleview as native
    except ImportError as exc:  # pragma: no cover - exercised only without omnidreams
        raise NativeWeightContractError(
            "native FP8 weight preparation requires the omnidreams package "
            "(flashdreams-omnidreams) to be installed"
        ) from exc
    return native.load_python_module("cosmos_fp8_utils")


def move_native_weights_to_device(
    weights: MutableMapping[str, Tensor],
    *,
    device: torch.device | str,
    keys: Iterable[str] | None = None,
) -> int:
    """Move a weights dict to ``device``, preserving aliased storage.

    ``prepare_cosmos_quantized_streaming_weights`` emits ``_fp8_prepared`` and
    ``_fp8_prepared_scale`` as *aliases* — they share storage with their
    canonical tensors (verified by ``data_ptr`` equality). A naive per-entry
    ``.to(device=...)`` breaks that sharing and materialises two independent
    copies of every FP8 weight on the GPU, roughly doubling the footprint the
    quantization was meant to shrink (``optimized_dit.py:1074-1079``).

    Copying once per distinct source storage keeps the aliasing intact. Returns
    the number of tensors that were re-pointed at an existing copy rather than
    copied again.
    """
    target = torch.device(device)
    selected = tuple(keys) if keys is not None else tuple(weights.keys())
    moved: dict[tuple[int, int], Tensor] = {}
    deduplicated = 0

    for key in selected:
        value = weights.get(key)
        if not isinstance(value, Tensor):
            continue
        if value.device == target:
            continue
        identity = (int(value.data_ptr()), int(value.storage_offset()))
        cached = moved.get(identity)
        if cached is not None and cached.shape == value.shape:
            weights[key] = cached
            deduplicated += 1
            continue
        relocated = value.to(device=target)
        moved[identity] = relocated
        weights[key] = relocated
    return deduplicated


__all__ = [
    "BLOCK_BF16_REL_KEYS",
    "BLOCK_LINEAR_REL_KEYS",
    "CAMERA_REL_KEY",
    "FUSED_QKV_REL_KEY",
    "SPLIT_QKV_REL_KEYS",
    "TOP_LEVEL_KEYS",
    "CosmosLinearBackend",
    "NativeWeightContractError",
    "assert_no_camera_weights",
    "block_prefix",
    "build_native_weights",
    "camera_weight_keys",
    "find_camera_weights",
    "move_native_weights_to_device",
    "validate_native_weights",
]
