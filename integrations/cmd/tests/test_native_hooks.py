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

    def predict_flow(self, **kwargs: Any) -> torch.Tensor:
        self.predict_calls.append(kwargs)
        return torch.zeros(1)

    def after_initialize_autoregressive_cache(self, cache: Any) -> None:
        self.after_init_calls += 1

    def after_finalize_kv_cache(self) -> None:
        self.after_finalize_calls += 1


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
    return transformer


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


def test_camera_conditioned_models_are_refused_for_now() -> None:
    """The kernels and transport exist; the producer does not.

    This must stay a refusal until something fills the camera buffer, because
    the alternative is camera-blind output with no error -- the exact failure
    the whole camera track was built to prevent.
    """
    transformer = _transformer_with(_StubExecutor(), camera_dim=1536)
    with pytest.raises(NotImplementedError, match="camera producer"):
        transformer.predict_flow(
            noisy_latent=torch.zeros(4, 8),
            timestep=torch.zeros(()),
            cache=object(),
            input=None,
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
