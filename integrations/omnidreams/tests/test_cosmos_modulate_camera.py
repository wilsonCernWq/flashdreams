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

"""Numerics for the LayerNorm + AdaLN-modulate kernels, with and without camera.

These kernels are the seam where CMD's per-block ``cam_encoder`` projection is
added to the native DiT (``CMDTransformerBlock.forward``,
``flashdreams_cmd/transformer/modules.py:119-120``). They are arch-neutral --
``cosmos_modulate.cu`` contains no ``cutlass::arch::Sm120`` -- so unlike the FP8
GEMM path these run anywhere, including sm_121 where cuDNN has no FP8 fused MHA.

The assertion that matters most is not "the numbers are close": it is that a
camera term which is silently dropped is *caught*. A dropped camera produces
plausible video with no error, which is exactly the failure this whole feature
exists to prevent. See ``test_camera_actually_changes_the_output`` and
``test_camera_is_indexed_per_token_not_per_batch``.
"""

from __future__ import annotations

import os

import pytest
import torch

from omnidreams.native import omnidreams_singleview as native

pytestmark = pytest.mark.ci_gpu

_RUN = os.environ.get("OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST") == "1"
_EPS = 1e-6

# bf16 carries ~8 mantissa bits, so a single rounding of an fp32 result lands
# within ~2^-8. E4M3 carries 3, hence the much looser FP8 bound.
_BF16_TOL = 0.02
_FP8_TOL = 0.15


def _extension():
    if not _RUN:
        pytest.skip(
            "Set OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST=1 to build the native extension."
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    extension = native.load_extension()
    assert extension is not None, native.extension_load_error()
    if not hasattr(extension, "cosmos_test_layernorm_modulate"):
        pytest.skip("extension predates the cosmos_test_layernorm_modulate probe")
    return extension


def _reference(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    cam: torch.Tensor | None,
    *,
    batch: int,
) -> torch.Tensor:
    """Float32 reference for ``norm(x) * (1 + scale) + shift + cam``.

    ``torch.nn.functional.layer_norm`` uses the same biased variance the kernel
    computes as ``E[x^2] - mean^2``, and the DiT's LayerNorms are
    ``elementwise_affine=False, eps=1e-6``
    (``flashdreams/recipes/cosmos/transformer/impl/modules.py:488``), so this is
    faithful rather than merely close.

    ``shift``/``scale`` are ``[B, K]`` broadcast across ``M / B`` consecutive
    rows; ``cam`` is ``[M, K]``, one row per token.
    """
    rows, width = x.shape
    per_batch = rows // batch
    normed = torch.nn.functional.layer_norm(x.float(), (width,), eps=_EPS)
    shift_rows = shift.float().repeat_interleave(per_batch, dim=0)
    scale_rows = scale.float().repeat_interleave(per_batch, dim=0)
    out = normed * (1.0 + scale_rows) + shift_rows
    if cam is not None:
        out = out + cam.float()
    return out


def _inputs(batch: int, per_batch: int, width: int, *, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs = {"device": "cuda", "dtype": torch.bfloat16, "generator": generator}
    rows = batch * per_batch
    return (
        torch.randn(rows, width, **kwargs),
        torch.randn(batch, width, **kwargs) * 0.1,
        torch.randn(batch, width, **kwargs) * 0.1,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.abs().max().clamp_min(1e-6)
    return ((actual.float() - expected).abs().max() / scale).item()


def _run(extension, x, shift, scale, cam, *, batch, variant):
    return extension.cosmos_test_layernorm_modulate(
        x=x, shift=shift, scale=scale, cam=cam, B=batch, variant=variant
    )


@pytest.mark.parametrize("variant", ["plain", "to_fp8", "to_fp8_only"])
@pytest.mark.parametrize("batch,per_batch", [(1, 5), (2, 3)])
def test_matches_pytorch_reference_with_camera(
    variant: str, batch: int, per_batch: int
) -> None:
    """All three variants, with and without a camera term, against fp32 torch.

    ``batch=1`` is what production runs; ``batch=2`` with ``per_batch>1`` is the
    shape that distinguishes per-token from per-batch camera indexing.
    """
    extension = _extension()
    width = 256
    x, shift, scale = _inputs(batch, per_batch, width)
    cam = torch.randn(
        batch * per_batch, width, device="cuda", dtype=torch.bfloat16
    )

    for camera in (None, cam):
        result = _run(
            extension, x, shift, scale, camera, batch=batch, variant=variant
        )
        torch.cuda.synchronize()
        expected = _reference(x, shift, scale, camera, batch=batch)

        if "y" in result:
            assert _relative_error(result["y"], expected) < _BF16_TOL
        if "y_fp8" in result:
            actual = result["y_fp8"].view(torch.float8_e4m3fn)
            assert _relative_error(actual, expected) < _FP8_TOL


@pytest.mark.parametrize("variant", ["plain", "to_fp8", "to_fp8_only"])
def test_null_camera_is_bit_identical_to_zero_camera(variant: str) -> None:
    """``cam=None`` and an explicit zero camera must agree exactly.

    The null branch is mathematically the identity, so any difference here means
    the pointer is being read when it should not be, or vice versa. Exact
    equality, not a tolerance: there is no floating-point reason to differ.
    """
    extension = _extension()
    batch, per_batch, width = 2, 3, 256
    x, shift, scale = _inputs(batch, per_batch, width)
    zeros = torch.zeros(batch * per_batch, width, device="cuda", dtype=torch.bfloat16)

    without = _run(extension, x, shift, scale, None, batch=batch, variant=variant)
    with_zeros = _run(extension, x, shift, scale, zeros, batch=batch, variant=variant)
    torch.cuda.synchronize()

    for key in without:
        assert torch.equal(without[key], with_zeros[key]), (
            f"{variant}/{key}: cam=None and cam=zeros diverged"
        )


@pytest.mark.parametrize("variant", ["plain", "to_fp8", "to_fp8_only"])
def test_camera_actually_changes_the_output(variant: str) -> None:
    """Two different cameras must produce measurably different outputs.

    This is the assertion the feature exists for. A kernel that accepts the
    pointer and ignores it passes every accuracy check above -- the reference
    would simply be recomputed without the term -- but fails here.
    """
    extension = _extension()
    batch, per_batch, width = 1, 4, 256
    x, shift, scale = _inputs(batch, per_batch, width)
    rows = batch * per_batch
    cam_a = torch.full((rows, width), 0.5, device="cuda", dtype=torch.bfloat16)
    cam_b = torch.full((rows, width), -0.5, device="cuda", dtype=torch.bfloat16)

    out_a = _run(extension, x, shift, scale, cam_a, batch=batch, variant=variant)
    out_b = _run(extension, x, shift, scale, cam_b, batch=batch, variant=variant)
    none_out = _run(extension, x, shift, scale, None, batch=batch, variant=variant)
    torch.cuda.synchronize()

    for key in out_a:
        assert not torch.equal(out_a[key], out_b[key]), (
            f"{variant}/{key}: the camera term is being ignored"
        )
        assert not torch.equal(out_a[key], none_out[key]), (
            f"{variant}/{key}: a non-zero camera matched the no-camera result"
        )


def test_camera_is_indexed_per_token_not_per_batch() -> None:
    """Catch the one mistake that is silent in production.

    ``cam`` is ``[M, K]`` per token, while ``shift``/``scale`` are ``[B, K]``
    broadcast across ``M / B`` rows. Indexing ``cam`` by the batch row instead
    compiles, runs, raises nothing, and applies token 0's camera to every token
    -- at ``B == 1``, a uniform wrong answer with no NaN and no shape mismatch.

    Giving every token a distinct constant makes that failure visible: the wrong
    indexing would make all rows differ from the reference by the same amount.
    """
    extension = _extension()
    batch, per_batch, width = 2, 3, 128
    rows = batch * per_batch
    x, shift, scale = _inputs(batch, per_batch, width)

    # Row r carries the constant r + 1, so no two tokens share a camera value
    # and token 0 of each batch is not representative of the others.
    cam = torch.arange(1, rows + 1, device="cuda", dtype=torch.bfloat16)
    cam = cam.unsqueeze(1).expand(rows, width).contiguous()

    result = _run(extension, x, shift, scale, cam, batch=batch, variant="plain")
    torch.cuda.synchronize()
    expected = _reference(x, shift, scale, cam, batch=batch)
    assert _relative_error(result["y"], expected) < _BF16_TOL

    # Directly reject the b_row-indexed alternative, so this test fails loudly
    # rather than merely drifting out of tolerance if the indexing regresses.
    broadcast_cam = cam[::per_batch].repeat_interleave(per_batch, dim=0)
    wrong = _reference(x, shift, scale, broadcast_cam, batch=batch)
    assert _relative_error(result["y"], wrong) > _BF16_TOL, (
        "output matches a per-batch-broadcast camera; cam is indexed by b_row"
    )


def test_rejects_camera_with_the_wrong_row_count() -> None:
    """A ``[B, K]`` camera must be refused rather than silently broadcast."""
    extension = _extension()
    batch, per_batch, width = 2, 3, 128
    x, shift, scale = _inputs(batch, per_batch, width)
    wrong = torch.zeros(batch, width, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="per token"):
        _run(extension, x, shift, scale, wrong, batch=batch, variant="plain")
