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

"""CPU-safe gating checks for CMD's native CUTLASS FP8 DiT path.

Tier T0 of ``integrations/cmd/docs/native_fp8_port_plan.md``: every refusal the
native path can produce is asserted here, on CPU, so the failure modes that
would otherwise surface as a wrong-architecture launch (potentially a silent
no-op) or as camera-blind video are caught before any GPU is involved.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from flashdreams_cmd.config import CMD_CONFIGS
from flashdreams_cmd.transformer import CMDDiTNetworkConfig, CMDTransformerConfig
from flashdreams_cmd.transformer.native_fp8 import (
    SM120A_COMPUTE_CAPABILITY,
    NativeFP8Unavailable,
    check_device_supports_native_fp8,
    check_model_supports_native_fp8,
    resolve_native_fp8,
)

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OMNIDREAMS_KERNEL_DIR = (
    _REPO_ROOT
    / "integrations"
    / "omnidreams"
    / "omnidreams_singleview"
    / "src"
    / "dit_streaming"
    / "kernels"
)

# Every site that pins a kernel to an SM120-only CUTLASS template. Building for
# any other architecture compiles these out to stubs, which is why the port is
# sm_120a-only. Asserted explicitly so a vendored-source update that adds or
# removes arch-conditional kernels shows up as a test failure rather than as a
# runtime "operation not supported" much later.
_EXPECTED_SM120_SITES = {"ops.cu": 40, "cosmos_fp8_tc_probe.cu": 8}


def _uncameraed_config() -> CMDTransformerConfig:
    """A CMD config with camera conditioning off (native-eligible by model)."""
    return CMDTransformerConfig(network=CMDDiTNetworkConfig(camera_dim=None))


def _preset_transformer(name: str) -> CMDTransformerConfig:
    """Return a released preset's transformer config, narrowed to CMD's type."""
    transformer = CMD_CONFIGS[name].diffusion_model.transformer
    assert isinstance(transformer, CMDTransformerConfig)
    return transformer


def _fake_gpu(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capability: tuple[int, int],
    name: str,
) -> None:
    """Present an arbitrary GPU to the capability probe.

    Lets every architecture decision be asserted on CPU, including for GPUs this
    machine does not have.
    """
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device=None: capability,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _device=None: name,
    )


def test_released_presets_do_not_enable_native_acceleration() -> None:
    """Every shipped preset stays on the default PyTorch path."""
    assert CMD_CONFIGS, "expected at least one released CMD preset"
    for name in CMD_CONFIGS:
        transformer = _preset_transformer(name)
        assert transformer.native_dit_acceleration == "disabled", (
            f"preset {name} enables native acceleration"
        )


def test_camera_presets_are_model_ineligible_for_the_native_path() -> None:
    """The shipped camera presets are exactly the ones Phase 5 must unblock.

    Asserted over the real presets rather than a synthetic config so that a
    preset gaining camera conditioning later cannot quietly become native-
    eligible.
    """
    camera_presets = {
        name: transformer
        for name in CMD_CONFIGS
        if (transformer := _preset_transformer(name)).network.camera_dim is not None
    }
    assert camera_presets, "expected at least one camera-conditioned CMD preset"
    for name, transformer in camera_presets.items():
        outcome = check_model_supports_native_fp8(transformer)
        assert not outcome, f"camera preset {name} must not be native-eligible"
        assert "cam_encoder" in outcome.reason


def test_disabled_mode_never_probes_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """``disabled`` short-circuits before touching CUDA at all."""

    def _explode(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("disabled mode must not probe CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", _explode)
    outcome = resolve_native_fp8(_uncameraed_config(), mode="disabled")
    assert not outcome
    assert "disabled" in outcome.reason


def test_camera_conditioned_model_is_refused() -> None:
    """A configured ``camera_dim`` disqualifies the model until Phase 5.

    The C++ bridge resolves weights by literal name and ignores unknown keys, so
    running a camera-conditioned CMD model natively would drop ``cam_encoder``
    and emit camera-blind video instead of failing.
    """
    config = CMDTransformerConfig(network=CMDDiTNetworkConfig(camera_dim=6))
    outcome = check_model_supports_native_fp8(config)
    assert not outcome
    assert "camera" in outcome.reason.lower()
    assert "cam_encoder" in outcome.reason


def test_camera_refusal_raises_in_required_mode() -> None:
    config = CMDTransformerConfig(network=CMDDiTNetworkConfig(camera_dim=6))
    with pytest.raises(NativeFP8Unavailable, match="camera"):
        resolve_native_fp8(config, mode="required")


def test_torch_weight_quantization_conflicts_with_native_path() -> None:
    """The two FP8 routes are mutually exclusive by construction."""
    config = _uncameraed_config()
    if not hasattr(config, "weight_quantization"):
        pytest.skip("weight_quantization lands with the torch._scaled_mm route")
    object.__setattr__(config, "weight_quantization", "fp8")
    outcome = check_model_supports_native_fp8(config)
    assert not outcome
    assert "weight_quantization" in outcome.reason


@pytest.mark.parametrize(
    ("capability", "device_name", "expected"),
    [
        (SM120A_COMPUTE_CAPABILITY, "NVIDIA RTX PRO 6000 Blackwell", True),
        (SM120A_COMPUTE_CAPABILITY, "NVIDIA GeForce RTX 5090", True),
        ((12, 1), "NVIDIA GB10", False),
        ((9, 0), "NVIDIA H100 80GB HBM3", False),
        ((8, 9), "NVIDIA GeForce RTX 4090", False),
        (SM120A_COMPUTE_CAPABILITY, "NVIDIA L40S", False),
    ],
)
def test_device_gate_matches_sm120a_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    device_name: str,
    expected: bool,
) -> None:
    """The arch decision is asserted without needing any of these GPUs."""
    _fake_gpu(monkeypatch, capability=capability, name=device_name)

    outcome = check_device_supports_native_fp8()
    assert bool(outcome) is expected
    if not expected:
        assert outcome.reason


def test_wrong_arch_reason_warns_about_silent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GB10 refusal explains *why* it is a refusal and not an attempt."""
    _fake_gpu(monkeypatch, capability=(12, 1), name="NVIDIA GB10")

    outcome = check_device_supports_native_fp8()
    assert not outcome
    assert "sm_121" in outcome.reason
    assert "silently" in outcome.reason


def test_auto_mode_falls_back_with_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gpu(monkeypatch, capability=(12, 1), name="NVIDIA GB10")

    outcome = resolve_native_fp8(_uncameraed_config(), mode="auto")
    assert not outcome
    assert outcome.reason


def test_required_mode_raises_on_unsupported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_gpu(monkeypatch, capability=(12, 1), name="NVIDIA GB10")

    with pytest.raises(NativeFP8Unavailable, match="sm_121"):
        resolve_native_fp8(_uncameraed_config(), mode="required")


def test_eligible_device_and_model_resolve_to_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_gpu(
        monkeypatch,
        capability=SM120A_COMPUTE_CAPABILITY,
        name="NVIDIA RTX PRO 6000 Blackwell",
    )

    outcome = resolve_native_fp8(_uncameraed_config(), mode="required")
    assert outcome
    assert outcome.reason == ""


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="native_dit_acceleration"):
        resolve_native_fp8(
            _uncameraed_config(),
            mode="enabled",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.skipif(
    not _OMNIDREAMS_KERNEL_DIR.is_dir(),
    reason="omnidreams native sources are not checked out",
)
def test_sm120_kernel_sites_match_expected_inventory() -> None:
    """Pin the arch-conditional kernel surface the port depends on.

    These 48 sites are why the native path is sm_120a-only: CUTLASS emits device
    code for ``arch::Sm120`` templates only when ``__CUDA_ARCH__ == 1200``. If a
    vendored-source bump changes this inventory, the port's hardware assumptions
    need re-verifying — so drift fails here rather than at kernel launch.
    """
    pattern = re.compile(r"Sm120[A-Za-z0-9_]*")
    actual = {
        path.name: len(pattern.findall(path.read_text(encoding="utf-8")))
        for path in sorted(_OMNIDREAMS_KERNEL_DIR.glob("*.cu"))
        if pattern.search(path.read_text(encoding="utf-8"))
    }
    assert actual == _EXPECTED_SM120_SITES


@pytest.mark.skipif(
    not _OMNIDREAMS_KERNEL_DIR.is_dir(),
    reason="omnidreams native sources are not checked out",
)
def test_no_sm121_kernel_support_is_vendored() -> None:
    """Document the constraint that makes GB10 unusable for this path.

    If a CUTLASS bump ever introduces SM121 kernels, this test fails and the
    dev-box story in ``quantization_native_port_scoping.md`` should be revisited.
    """
    sm121_hits = [
        path.name
        for path in sorted(_OMNIDREAMS_KERNEL_DIR.glob("*.cu"))
        if "Sm121" in path.read_text(encoding="utf-8")
    ]
    assert sm121_hits == []
