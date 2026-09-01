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

"""Eligibility gating for CMD's native CUTLASS FP8 DiT path.

Phase 1 of ``integrations/cmd/docs/native_fp8_port_plan.md``: decide whether the
native path may be used, and refuse *loudly and early* when it may not. The
native forward itself is not wired up yet — this module is the gate in front of
it, and every refusal it can return is a failure mode the port must not ship.

Two refusals matter enough to spell out:

Wrong GPU architecture
    omnidreams' kernels are compiled from ``cutlass::arch::Sm120`` templates and
    are ``sm_120a``-only. On any other Blackwell part the extension still builds
    and *loads*, then fails at kernel launch — and a wrong-arch launch can be a
    silent no-op that returns zeros without raising (measured on GB10/``sm_121a``;
    see ``integrations/cmd/docs/quantization_native_port_scoping.md``). So the
    capability probe is a hard precondition, not a best-effort hint. The
    allowlist mirrors ``sage3_is_runtime_supported`` in
    ``omnidreams_singleview/src/dit_streaming/kernels/sage3_attention.cu`` so
    Python and CUDA agree on which devices qualify.

Camera conditioning
    ``CMDTransformerBlock`` adds a per-block ``self_attn.cam_encoder`` projection
    that the native block kernel has no concept of, and the C++ bridge looks
    weights up by literal name — unknown keys are ignored silently. Enabling the
    native path on a camera-conditioned model would therefore produce
    camera-blind video rather than an error. Until the kernel learns camera
    injection (plan Phase 5), a configured ``camera_dim`` disqualifies the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

NativeAccelerationMode = Literal["disabled", "auto", "required"]
"""Vocabulary reused verbatim from ``omnidreams/native/acceleration.py``."""

SM120A_COMPUTE_CAPABILITY = (12, 0)
"""The only compute capability omnidreams' FP8 DiT kernels are built for."""

SM120A_DEVICE_ALLOWLIST = (
    "GeForce RTX 5090",
    "RTX PRO 6000",
    "RTX 6000",
)
"""Device names omnidreams itself has built and validated the SM120a path on."""


class NativeFP8Unavailable(RuntimeError):
    """Raised when ``native_dit_acceleration="required"`` cannot be honoured."""


@dataclass(frozen=True)
class NativeFP8Eligibility:
    """Outcome of the native-path precondition checks."""

    available: bool
    reason: str
    """Human-readable justification; non-empty whenever ``available`` is False."""

    def __bool__(self) -> bool:
        return self.available


def _describe_device(device_index: int | None = None) -> tuple[tuple[int, int], str]:
    capability = torch.cuda.get_device_capability(device_index)
    return capability, torch.cuda.get_device_name(device_index)


def check_device_supports_native_fp8(
    device_index: int | None = None,
) -> NativeFP8Eligibility:
    """Whether this GPU can execute omnidreams' ``sm_120a`` FP8 DiT kernels."""

    if not torch.cuda.is_available():
        return NativeFP8Eligibility(False, "native FP8 DiT requires CUDA")

    capability, name = _describe_device(device_index)
    if capability != SM120A_COMPUTE_CAPABILITY:
        major, minor = capability
        return NativeFP8Eligibility(
            False,
            f"native FP8 DiT kernels are sm_120a-only; found sm_{major}{minor} ({name}). "
            "A wrong-architecture launch can silently produce zeros, so the native "
            "path is refused rather than attempted.",
        )
    if not any(allowed in name for allowed in SM120A_DEVICE_ALLOWLIST):
        allowed = ", ".join(SM120A_DEVICE_ALLOWLIST)
        return NativeFP8Eligibility(
            False,
            f"{name!r} is not on the validated sm_120a allowlist ({allowed})",
        )
    return NativeFP8Eligibility(True, "")


def check_model_supports_native_fp8(config: Any) -> NativeFP8Eligibility:
    """Whether this CMD model configuration is expressible in the native block.

    Checks the model, not the machine, so it stays meaningful on CPU and in
    ``ci_cpu`` — the config-level refusals are exactly the ones that would
    otherwise degrade silently at runtime.
    """

    network = getattr(config, "network", None)

    camera_dim = getattr(network, "camera_dim", None)
    if camera_dim is not None:
        return NativeFP8Eligibility(
            False,
            f"camera conditioning (camera_dim={camera_dim}) has no native block "
            "equivalent; the C++ bridge would ignore cam_encoder weights and emit "
            "camera-blind video. Blocked until port plan Phase 5 lands.",
        )

    # The bridge validates the timestep embedding as exactly [B, K]
    # (streaming_dit_bridge.cu:1684-1695) and broadcasts a single shift/scale row
    # across all M token rows (cosmos_modulate.cu:280-305). CMD's per-token
    # timesteps have no native representation, so this is a capability cap rather
    # than a bug to fix later. Every released preset leaves it None.
    conditional_frame_timestep = getattr(config, "conditional_frame_timestep", None)
    if conditional_frame_timestep is not None:
        return NativeFP8Eligibility(
            False,
            f"conditional_frame_timestep={conditional_frame_timestep} requires "
            "per-token AdaLN modulation; the native block broadcasts one "
            "timestep row to every token and would silently ignore the "
            "per-token schedule",
        )

    # Torch-level weight quantization rewrites the module tree, so the state-dict
    # snapshot the native weight-prep consumes would no longer contain plain
    # nn.Linear keys. The two FP8 routes are mutually exclusive by construction.
    weight_quantization = getattr(config, "weight_quantization", "none")
    if weight_quantization != "none":
        return NativeFP8Eligibility(
            False,
            f"weight_quantization={weight_quantization!r} rewrites the module tree; "
            "the native FP8 path needs unquantized nn.Linear weights to snapshot",
        )
    return NativeFP8Eligibility(True, "")


def resolve_native_fp8(
    config: Any,
    *,
    mode: NativeAccelerationMode = "disabled",
    device_index: int | None = None,
) -> NativeFP8Eligibility:
    """Resolve whether CMD should run the native FP8 DiT path.

    ``disabled`` short-circuits without probing CUDA at all (so it stays free on
    CPU-only machines), ``auto`` degrades to a reason string, and ``required``
    raises rather than silently falling back.
    """

    if mode == "disabled":
        return NativeFP8Eligibility(False, "native_dit_acceleration='disabled'")
    if mode not in ("auto", "required"):
        raise ValueError(
            f"native_dit_acceleration must be 'disabled', 'auto', or 'required'; got {mode!r}"
        )

    for eligibility in (
        check_model_supports_native_fp8(config),
        check_device_supports_native_fp8(device_index),
    ):
        if not eligibility.available:
            if mode == "required":
                raise NativeFP8Unavailable(
                    f"native_dit_acceleration='required' but {eligibility.reason}"
                )
            return eligibility
    return NativeFP8Eligibility(True, "")


__all__ = [
    "SM120A_COMPUTE_CAPABILITY",
    "SM120A_DEVICE_ALLOWLIST",
    "NativeAccelerationMode",
    "NativeFP8Eligibility",
    "NativeFP8Unavailable",
    "check_device_supports_native_fp8",
    "check_model_supports_native_fp8",
    "resolve_native_fp8",
]
