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

"""Lifetime rules for the per-block camera buffer.

The buffer reaches ``optimized_dit_forward`` through the config dict, which
``CUDAGraphWrapper`` passes through verbatim rather than staging. The graph
therefore captures the buffer's *address*, and a replacement tensor on the next
AR chunk would leave the graph reading the previous one.

That failure is silent in the worst way: the caching allocator usually hands the
same block back, so the output stays plausible and the camera simply stops
responding. These tests pin the invariant that prevents it --

    the camera buffer may be reallocated only where the CUDA graph is reset

-- which in practice means ``after_initialize_autoregressive_cache`` and nowhere
else. They are CPU-only: this is object lifetime, not numerics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from omnidreams.native import omnidreams_singleview as native

pytestmark = pytest.mark.ci_cpu

_NUM_BLOCKS = 4
_BATCH = 1
_TOKENS = 8
_CHANNELS = 16


class _FakeExtension:
    """Stands in for the built extension, with the camera probe present."""

    def __init__(self, *, supports_camera: bool = True) -> None:
        if supports_camera:
            self.optimized_dit_supports_camera = lambda: True


def _executor(*, supports_camera: bool = True) -> Any:
    """An OptimizedDiTExecutor with only the attributes these methods touch.

    Built without ``__init__`` on purpose: the real constructor wants a live
    transformer and a built extension, and neither is needed to exercise buffer
    lifetime.
    """
    helper = native.load_python_module("optimized_dit")
    executor = object.__new__(helper.OptimizedDiTExecutor)
    executor._native_extension = _FakeExtension(supports_camera=supports_camera)
    executor._camera_buffer = None
    executor._camera_fill_key = None
    executor._optimized_streaming_config = {}
    return executor


def _ensure(executor: Any) -> torch.Tensor:
    return executor.ensure_camera_buffer(
        num_blocks=_NUM_BLOCKS,
        batch=_BATCH,
        tokens=_TOKENS,
        channels=_CHANNELS,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )


def test_buffer_address_is_stable_across_chunks() -> None:
    """Repeated calls must hand back the same storage, not a fresh tensor.

    This is the invariant the CUDA graph depends on. If it ever regresses, the
    graph keeps reading whatever address it captured and the camera silently
    freezes.
    """
    executor = _executor()
    first = _ensure(executor)
    addresses = {first.data_ptr()}
    for _ in range(5):
        addresses.add(_ensure(executor).data_ptr())
    assert len(addresses) == 1, "camera buffer was reallocated within a rollout"


def test_buffer_is_published_into_the_streaming_config() -> None:
    """The bridge reads the buffer from the config dict, by identity."""
    executor = _executor()
    buffer = _ensure(executor)
    published = executor._optimized_streaming_config["cosmos_cam_embed"]
    assert published is buffer
    assert published.shape == (_NUM_BLOCKS, _BATCH, _TOKENS, _CHANNELS)


def test_clearing_transient_ar_caches_keeps_the_buffer() -> None:
    """Per-AR-chunk cleanup must not touch the buffer.

    ``_clear_transient_ar_caches`` runs at every chunk boundary. Adding the
    camera buffer to it would look tidy and would break the graph.
    """
    executor = _executor()
    executor._optimized_rope_cache = {}
    executor._optimized_rope_freqs_cache = {}
    executor._optimized_hdmap_cache = {}

    before = _ensure(executor).data_ptr()
    executor._clear_transient_ar_caches()
    assert executor._camera_buffer is not None
    assert executor._camera_buffer.data_ptr() == before


def test_fill_key_suppresses_refills_within_one_chunk() -> None:
    """Four scheduler steps in a chunk should refill at most once."""
    executor = _executor()
    _ensure(executor)
    assert executor.camera_fill_needed(("ar", 0)) is True
    assert executor.camera_fill_needed(("ar", 0)) is False
    assert executor.camera_fill_needed(("ar", 1)) is True
    assert executor.camera_fill_needed(("ar", 1)) is False


def test_reallocation_resets_the_fill_key() -> None:
    """A new buffer holds stale contents, so it must be refilled."""
    executor = _executor()
    _ensure(executor)
    executor.camera_fill_needed(("ar", 0))

    # A shape change forces a genuine reallocation.
    executor.ensure_camera_buffer(
        num_blocks=_NUM_BLOCKS,
        batch=_BATCH,
        tokens=_TOKENS * 2,
        channels=_CHANNELS,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert executor.camera_fill_needed(("ar", 0)) is True


def test_refuses_when_the_extension_lacks_the_probe() -> None:
    """Fail closed on a stale build rather than rendering camera-blind output.

    Unknown config keys are ignored by the bridge, so an older extension would
    accept ``cosmos_cam_embed`` and drop it. There is no correct fallback --
    unlike hdmap, camera must be injected inside every block -- so the only safe
    response is to refuse.
    """
    executor = _executor(supports_camera=False)
    assert executor.supports_camera() is False
    with pytest.raises(RuntimeError, match="rebuild the native"):
        _ensure(executor)
