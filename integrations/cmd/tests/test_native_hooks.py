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

"""The five hooks that route CMD through omnidreams' native DiT.

CMD carries no kernels of its own: it loads the same ``omnidreams_singleview``
extension, instantiates the same ``OptimizedDiTExecutor``, and only supplies
tensors in the bridge's layout. These tests cover the Python seam, on CPU, with
a stub executor -- the kernels themselves are verified in the omnidreams tests.

Two of the hooks can fail *silently*, and those are what most of this file is
about:

* ``input`` means HDMap to omnidreams and *camera* to CMD. Forwarding CMD's
  camera unchanged would feed it to the HDMap branch, which accepts it without
  complaint and produces plausible, wrong output.
* ``compile_network`` wraps the network before the executor can snapshot it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from flashdreams_cmd.config import CMD_CONFIGS
from flashdreams_cmd.transformer import CMDTransformer, CMDTransformerConfig

pytestmark = pytest.mark.ci_cpu


class _StubExecutor:
    """Records what the hooks hand it, without touching CUDA."""

    def __init__(self) -> None:
        self.predict_calls: list[dict[str, Any]] = []
        self.after_init_calls = 0
        self.after_finalize_calls = 0
        self.camera_buffer: torch.Tensor | None = None
        self.ensure_calls = 0
        self._camera_key: Any = None

    def predict_flow(self, **kwargs: Any) -> torch.Tensor:
        self.predict_calls.append(kwargs)
        # The real executor returns the bridge layout [B, V, T, HW, D]; the hook
        # translates it back. Echoing the shape it was handed keeps that round
        # trip exercised instead of stubbed past.
        return torch.zeros_like(kwargs["noisy_latent"])

    def after_initialize_autoregressive_cache(self, cache: Any) -> None:
        self.after_init_calls += 1

    def after_finalize_kv_cache(self) -> None:
        self.after_finalize_calls += 1

    # --- camera transport ---
    def supports_camera(self) -> bool:
        return True

    def camera_fill_needed(self, key: Any) -> bool:
        if self._camera_key == key:
            return False
        self._camera_key = key
        return True

    def ensure_camera_buffer(self, **kwargs: Any) -> torch.Tensor:
        self.ensure_calls += 1
        return self.camera_buffer


class _StubCache:
    """Only the AR index matters to the hooks under test."""

    def __init__(self, *, ar_index: int) -> None:
        self.autoregressive_index = ar_index


def _transformer_with(executor: Any, *, camera_dim: int | None = None) -> CMDTransformer:
    """A CMDTransformer shell with the hooks live and nothing else built.

    Bypasses ``__init__`` deliberately: the real one builds a 2 B-parameter
    network, and every hook under test reads only ``_optimized_dit_executor``
    and ``config``.
    """
    transformer = object.__new__(CMDTransformer)
    config = CMD_CONFIGS["cmd-chunk1-camera-i2v" if camera_dim else "cmd-chunk1-short-i2v"]
    transformer.config = config.diffusion_model.transformer
    transformer._optimized_dit_executor = executor
    transformer._optimized_dit_selection = None
    # Normally resolved in initialize_autoregressive_cache from the rollout's
    # height/width. Must multiply out to the L of the latents these tests pass
    # ([4, 8], so L=4) -- to_bridge_latent checks that rather than reshaping
    # blindly, which is the point of it.
    transformer._native_grid = (2, 2)
    return transformer


def test_config_satisfies_the_executor_contract() -> None:
    """Every field OptimizedDiTExecutor reads must resolve on a CMD config.

    Four are forwarded by CMD (num_views, num_heads, patch_spatial,
    patch_temporal); the rest it already had. A missing one does not fail
    politely -- it is an AttributeError several frames into setup, or worse, a
    wrong token grid handed to a kernel.

    **On how this list was derived, and how it was wrong.** The first version
    came from ``grep 'self\\.config\\.'`` and found nine fields. It missed
    ``num_views``, which the executor reads through a local alias as
    ``config.num_views`` -- and the first real run failed on exactly that. The
    pattern below covers both spellings and finds twelve:

        grep -oP '(self\\.config|(?<![.\\w])config)\\.\\K[a-z_]+(\\.[a-z_]+)?' \\
            omnidreams_singleview/python/optimized_dit.py

    Still not proof: a field reached via ``getattr``, or through a config object
    passed under another name, would evade both. This test pins what is known,
    and the honest backstop remains actually running the thing.
    """
    for name in CMD_CONFIGS:
        config = CMD_CONFIGS[name].diffusion_model.transformer
        assert config.dtype is not None
        assert int(config.num_heads) > 0
        assert int(config.num_views) == 1, "the native path is single-view only"
        assert int(config.patch_spatial) > 0
        assert int(config.patch_temporal) > 0
        assert config.use_cuda_graph is not None
        assert int(config.cuda_graph_warmup_iters) >= 0
        assert int(config.network.adaln_lora_dim) > 0
        assert int(config.network.model_channels) > 0
        assert int(config.network.num_blocks) > 0
        assert int(config.network.num_heads) > 0


def test_forwarded_fields_track_network_patch_size() -> None:
    """The three forwards must derive from ``network``, not shadow it.

    A second stored copy could drift from ``patch_size``; these are properties
    precisely so they cannot.
    """
    config = CMD_CONFIGS["cmd-chunk1-short-i2v"].diffusion_model.transformer
    patch_t, patch_h, patch_w = config.network.patch_size
    assert (config.patch_temporal, config.patch_spatial) == (patch_t, patch_h)
    assert patch_h == patch_w, "the native path assumes square spatial patches"
    assert config.num_heads == config.network.num_heads


def test_config_defaults_leave_the_native_path_off() -> None:
    """Adding the fields must not change any shipped preset's behaviour."""
    for name in CMD_CONFIGS:
        transformer = CMD_CONFIGS[name].diffusion_model.transformer
        assert transformer.native_dit_acceleration == "disabled"


def test_compile_network_with_native_is_refused() -> None:
    """The base __init__ compiles the network before the executor exists.

    Rather than reorder the base or mutate config behind its back, the
    combination is refused -- which is also the documented operational
    requirement, since running both builds a torch.compile graph that is then
    discarded.
    """
    config = CMDTransformerConfig(
        native_dit_acceleration="required", compile_network=True
    )
    with pytest.raises(ValueError, match="compile_network=False"):
        CMDTransformer(config)


def test_camera_is_not_forwarded_as_hdmap() -> None:
    """The silent-failure guard: CMD's ``input`` is camera, not HDMap.

    A camera-free model must never see a non-empty tensor reach the executor's
    ``input``, because the executor treats it as ``hdmap_patched``.
    """
    executor = _StubExecutor()
    transformer = _transformer_with(executor)
    latent = torch.zeros(4, 8)

    transformer.predict_flow(
        noisy_latent=latent, timestep=torch.zeros(()), cache=object(), input=None
    )
    assert len(executor.predict_calls) == 1
    forwarded = executor.predict_calls[0]["input"]
    # Checked as a tensor-ness question first: forwarding CMD's `input`
    # unchanged hands the executor a bare None here, and `None.numel()` would
    # fail with an AttributeError that reads like a broken test rather than a
    # broken hook.
    assert isinstance(forwarded, torch.Tensor), (
        f"the executor's HDMap slot got {type(forwarded).__name__}, not a tensor "
        "-- CMD's `input` is being forwarded verbatim instead of replaced"
    )
    assert forwarded.numel() == 0, (
        "a non-empty tensor reached the executor's HDMap slot"
    )


def test_latent_layout_round_trips() -> None:
    """CMD's ``[..., L, D]`` goes down as ``[B, V, T, HW, D]`` and comes back.

    The executor speaks the bridge's five-dimensional layout; CMD's pipeline
    speaks a flat token sequence. Both directions happen inside ``predict_flow``,
    so a mistake in either would surface as a shape error deep in the caller
    rather than here.
    """
    executor = _StubExecutor()
    transformer = _transformer_with(executor)
    latent = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)

    out = transformer.predict_flow(
        noisy_latent=latent, timestep=torch.zeros(()), cache=object(), input=None
    )
    assert out.shape == latent.shape, "the flow did not come back in CMD's layout"

    sent = executor.predict_calls[0]["noisy_latent"]
    assert sent.dim() == 5, "the executor was not handed the bridge layout"
    assert sent.shape == (1, 1, 2, 2, 8), sent.shape
    # Same storage, reinterpreted -- a copy here would be silent waste at
    # 28 blocks x every scheduler step.
    assert sent.data_ptr() == latent.data_ptr()


def test_a_camera_tensor_on_a_camera_free_model_is_refused() -> None:
    """Passing camera to a ``camera_dim=None`` model has nowhere to go."""
    transformer = _transformer_with(_StubExecutor())
    with pytest.raises(ValueError, match="nowhere to put it"):
        transformer.predict_flow(
            noisy_latent=torch.zeros(4, 8),
            timestep=torch.zeros(()),
            cache=object(),
            input=torch.zeros(4, 1536),
        )


def test_a_camera_model_without_camera_tokens_is_refused() -> None:
    """The anti-silent-drop guard, from the other side.

    A camera-conditioned checkpoint reaching the native path with ``input=None``
    has no camera to inject. The forward would succeed and render camera-blind
    video -- the exact failure this whole track was built to prevent -- so it
    must raise instead.
    """
    transformer = _transformer_with(_StubExecutor(), camera_dim=1536)
    with pytest.raises(ValueError, match="silently drop camera"):
        transformer.predict_flow(
            noisy_latent=torch.zeros(4, 8),
            timestep=torch.zeros(()),
            cache=object(),
            input=None,
        )


def test_camera_buffer_is_filled_per_block_and_in_place() -> None:
    """The producer writes cam_encoder(camera) for every block, without
    replacing the buffer the CUDA graph captured."""
    executor = _StubExecutor()
    transformer = _transformer_with(executor, camera_dim=1536)
    blocks, tokens, channels = 3, 4, 8
    weights = [torch.full((channels, 1536), float(i + 1)) for i in range(blocks)]
    transformer._cam_encoder_snapshot = weights
    buffer = torch.zeros(blocks, 1, tokens, channels)
    executor.camera_buffer = buffer

    camera = torch.ones(tokens, 1536)
    transformer.predict_flow(
        noisy_latent=torch.zeros(tokens, channels),
        timestep=torch.zeros(()),
        cache=_StubCache(ar_index=0),
        input=camera,
    )

    assert executor.camera_buffer.data_ptr() == buffer.data_ptr(), (
        "the buffer was replaced rather than filled; the CUDA graph would keep "
        "reading the old allocation"
    )
    # Block i's weight is a constant (i+1), so each row sums to 1536 * (i+1).
    for i in range(blocks):
        assert torch.allclose(buffer[i, 0], torch.full((tokens, channels), 1536.0 * (i + 1))), (
            f"block {i} did not receive its own cam_encoder projection"
        )


def test_camera_is_filled_once_per_ar_chunk() -> None:
    """The scheduler calls predict_flow several times per AR chunk against the
    same camera; refilling each time is 28 GEMMs of pure waste."""
    executor = _StubExecutor()
    transformer = _transformer_with(executor, camera_dim=1536)
    transformer._cam_encoder_snapshot = [torch.zeros(8, 1536)]
    executor.camera_buffer = torch.zeros(1, 1, 4, 8)

    for _ in range(3):
        transformer.predict_flow(
            noisy_latent=torch.zeros(4, 8),
            timestep=torch.zeros(()),
            cache=_StubCache(ar_index=0),
            input=torch.ones(4, 1536),
        )
    assert executor.ensure_calls == 1, (
        f"the buffer was filled {executor.ensure_calls} times for one AR chunk"
    )


def test_finalize_hook_runs_even_when_the_base_short_circuits() -> None:
    """``skip_finalize_kv_cache`` must not skip the executor's own cleanup.

    The executor's per-chunk state is keyed on the AR index and has to be
    dropped at every chunk boundary regardless of what the base does.
    """
    executor = _StubExecutor()
    transformer = _transformer_with(executor)
    object.__setattr__(transformer.config, "skip_finalize_kv_cache", True)
    transformer.finalize_kv_cache()
    assert executor.after_finalize_calls == 1


def test_hooks_are_inert_without_an_executor() -> None:
    """With the native path off, every hook must defer to the base."""
    transformer = _transformer_with(None)
    transformer.finalize_kv_cache()  # must not raise
    assert transformer._optimized_dit_executor is None
