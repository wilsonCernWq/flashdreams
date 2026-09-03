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

"""Camera embeddings reach the right block, through the real DiT forward.

The kernel arithmetic is pinned by ``test_cosmos_modulate_camera.py`` and the
buffer's lifetime by ``test_camera_buffer_lifetime.py``. What is left, and what
these tests cover, is the plumbing in between: that the bridge hands block *i*
slab *i*, and that per-token layout survives the trip from Python.

A bridge that sliced wrongly -- handing every block slab 0, say -- would produce
plausible output and pass every kernel-level test, because the kernel faithfully
adds whatever pointer it is given.

These run on the native bf16 path with random weights and no checkpoint, so they
work on any Blackwell part including sm_121, where the FP8 DiT cannot run.

**On the shape of the slicing test.** It sets a camera on exactly one block and
leaves the rest zero. An earlier version permuted all the slabs instead and was
killed by mutation testing: forcing every block to read slab 0 still passed it,
because permuting also changes slab 0. Single-block excitation is what actually
distinguishes correct slicing from a constant offset.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch

from omnidreams.native import omnidreams_singleview as native

pytestmark = pytest.mark.ci_gpu

_RUN = os.environ.get("OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST") == "1"
_NUM_BLOCKS = 4
_LAT = 32
_TEXT_LEN = 128


@pytest.fixture(scope="module")
def rollout():
    """A small native-bf16 DiT with random weights, primed for repeated forwards.

    ``predict_flow`` does not mutate the KV cache -- ``finalize_kv_cache`` does --
    so many calls inside one ``start``/``finalize`` pair are comparable to each
    other, which is what lets these tests vary only the camera.
    """
    if not _RUN:
        pytest.skip(
            "Set OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST=1 to build the native extension."
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    extension = native.load_extension()
    assert extension is not None, native.extension_load_error()
    if not hasattr(extension, "optimized_dit_supports_camera"):
        pytest.skip("extension predates camera support")

    from flashdreams.infra.config import derive_config
    from omnidreams.config import (
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF as preset,
    )

    config = derive_config(
        preset.diffusion_model.transformer,
        checkpoint_path=None,
        native_dit_acceleration="required",
        native_dit_backend="bf16",
        native_dit_attention_backend="cudnn",
    )
    config = dataclasses.replace(
        config, network=dataclasses.replace(config.network, num_blocks=_NUM_BLOCKS)
    )

    torch.manual_seed(1234)
    with torch.device("cuda"):
        transformer = config.setup()

    executor = transformer._optimized_dit_executor
    assert executor is not None and transformer._optimized_dit_selection.enabled
    assert executor.supports_camera()

    net = config.network
    gen = torch.Generator(device="cuda").manual_seed(7)
    kw = {"device": "cuda", "dtype": config.dtype, "generator": gen}
    text = torch.randn(1, 1, _TEXT_LEN, net.crossattn_proj_in_channels, **kw) * 0.02
    image = torch.randn(1, config.num_views, 1, net.in_channels, _LAT, _LAT, **kw) * 0.02

    with torch.no_grad():
        cache = transformer.initialize_autoregressive_cache(
            height=_LAT, width=_LAT, text_embeddings=text, image_embeddings=image
        )
        latent = torch.randn(*transformer.latent_shape, **kw)
        hdmap = transformer.patchify_and_maybe_split_cp(
            torch.randn(
                1, config.num_views, config.len_t, net.additional_concat_ch,
                _LAT, _LAT, **kw
            )
        )
        timestep = torch.tensor(1000.0, device="cuda")
        cache.start(0)

        def forward() -> torch.Tensor:
            out = transformer.predict_flow(
                noisy_latent=latent, timestep=timestep, cache=cache, input=hdmap
            )
            torch.cuda.synchronize()
            return out.detach().clone()

        baseline = forward()
        buffer = executor.ensure_camera_buffer(
            num_blocks=_NUM_BLOCKS,
            batch=1,
            tokens=baseline.shape[-2],
            channels=net.model_channels,
            device=torch.device("cuda"),
            dtype=config.dtype,
        )
        yield forward, buffer, baseline


def test_zero_camera_is_bit_identical_to_no_camera(rollout) -> None:
    """Adding zero must change nothing, exactly."""
    forward, buffer, baseline = rollout
    buffer.zero_()
    assert torch.equal(forward(), baseline)


def test_nonzero_camera_changes_the_output(rollout) -> None:
    """The most basic guard: the term is not silently dropped."""
    forward, buffer, baseline = rollout
    buffer.zero_()
    buffer.fill_(0.05)
    assert not torch.equal(forward(), baseline)


@pytest.mark.parametrize("block", [0, _NUM_BLOCKS - 1])
def test_camera_on_one_block_alone_changes_the_output(rollout, block: int) -> None:
    """Per-block slicing: exciting any single slab must register.

    The ``block == _NUM_BLOCKS - 1`` case is the discriminating one. A bridge
    that always reads slab 0 sees only zeros here and reproduces the baseline;
    mutation testing confirms this parametrisation turns red for exactly that
    block and stays green for block 0.
    """
    forward, buffer, baseline = rollout
    buffer.zero_()
    buffer[block].fill_(0.05)
    assert not torch.equal(forward(), baseline)


def test_the_same_value_in_different_slabs_differs(rollout) -> None:
    """Pins *which* block moved, not merely that something did."""
    forward, buffer, _ = rollout
    buffer.zero_()
    buffer[0].fill_(0.05)
    first = forward()
    buffer.zero_()
    buffer[_NUM_BLOCKS - 1].fill_(0.05)
    assert not torch.equal(first, forward())


def test_per_token_camera_differs_from_a_per_block_constant(rollout) -> None:
    """Per-token layout survives Python -> config dict -> bridge -> kernel."""
    forward, buffer, _ = rollout
    tokens, channels = buffer.shape[2], buffer.shape[3]
    ramp = torch.arange(tokens, device="cuda", dtype=buffer.dtype) * 0.001

    buffer.zero_()
    for i in range(_NUM_BLOCKS):
        buffer[i, 0] = ramp.unsqueeze(1).expand(tokens, channels)
    varying = forward()

    buffer.zero_()
    buffer.fill_(ramp[0])
    assert not torch.equal(varying, forward())


def test_buffer_address_is_stable_across_every_forward(rollout) -> None:
    """The CUDA graph captured this address; nothing above may have moved it."""
    forward, buffer, _ = rollout
    before = buffer.data_ptr()
    buffer.zero_()
    forward()
    assert buffer.data_ptr() == before
