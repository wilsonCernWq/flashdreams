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

"""Shared synchronous CMD model-session state and execution."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import StepResult, TimeWindow

from .encoder import CMDCamCtrlInput
from .pipeline import CMDInferencePipelineCache

OutputStreamFactory = Callable[[], VideoOutputStream]


class CMDModelSessionCore:
    """Own one CMD cache, AR index, and generated-output stream.

    A thin wrapper over :meth:`~.pipeline.CMDInferencePipeline.initialize_cache`
    /``generate``: :meth:`reset`'s ``camera_to_world``/``intrinsics``/
    ``expected_latent_frames`` bake a fixed camera trajectory into the cache
    once, for a non-live camera-conditioned pipeline. WebRTC's runtime
    (``webrtc/session.py``) is currently the only caller, and it's live-only
    (see its module docstring), so it always passes these as ``None`` and
    threads a per-step :class:`~.encoder.CMDCamCtrlInput` through ``step``
    instead, mirroring LingBot's equivalent. The offline ``run`` mode
    (``runner.py``) drives the pipeline directly and never constructs this
    class.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        output_stream_factory: OutputStreamFactory,
    ) -> None:
        self.pipeline = pipeline
        self._output_stream_factory = output_stream_factory
        self._output_stream = output_stream_factory()
        self._cache: CMDInferencePipelineCache | None = None
        self._step_index = 0
        self._closed = False

    @property
    def cache(self) -> CMDInferencePipelineCache:
        if self._cache is None:
            raise RuntimeError("CMD model session is not initialized.")
        return self._cache

    @property
    def step_index(self) -> int:
        return self._step_index

    def next_num_frames(self) -> int:
        self._require_open()
        return int(self.pipeline.get_num_output_frames(self._step_index))

    def reset(
        self,
        *,
        prompt: str,
        image: torch.Tensor,
        camera_to_world: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        expected_latent_frames: int | None = None,
    ) -> None:
        self._require_open()
        self._cache = None
        self._output_stream.finish()
        self._output_stream = self._output_stream_factory()
        self._cache = self.pipeline.initialize_cache(
            text=[prompt],
            image=image,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            expected_latent_frames=expected_latent_frames,
        )
        self._step_index = 0

    def step(
        self,
        camera_input: CMDCamCtrlInput | None = None,
        *,
        output_window: TimeWindow | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepResult:
        self._require_open()
        step_index = self._step_index
        expected_frames = self.next_num_frames()
        start_t = time.perf_counter()
        video_chunk = self.pipeline.generate(step_index, self.cache, input=camera_input)
        stats = self.pipeline.finalize(step_index, self.cache)
        metrics = _numeric_metrics(stats)
        metrics.setdefault("model_step_s", time.perf_counter() - start_t)
        result = self._output_stream.process(
            video_chunk,
            autoregressive_index=step_index,
            metrics=metrics,
            metadata=metadata,
            output_window=output_window,
        )
        if result.frame_count != expected_frames:
            raise RuntimeError(
                f"Expected generated chunk to contain {expected_frames} frames, "
                f"got {result.frame_count}."
            )
        self._step_index += 1
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cache = None
        self._output_stream.finish()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CMD model session is closed.")


def _numeric_metrics(stats: object) -> dict[str, float | int]:
    if not isinstance(stats, Mapping):
        return {}
    return {
        str(name): value
        for name, value in stats.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


__all__ = ["CMDModelSessionCore"]
