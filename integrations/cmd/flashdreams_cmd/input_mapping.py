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

"""CMD user-event canonicalization and canonical-to-model input mapping.

CMD's only live control is a free camera driven from the keyboard -- unlike
LingBot, there are no text events and no "fixed trace vs. live" dual mode
(CMD's existing fixed-trajectory WebRTC mode is a fully separate, already-
shipped code path that never touches this module at all).

:data:`CAMERA_COMMAND` and :class:`KeyboardToCameraCommand` are adapted
verbatim from ``lingbot.input_mapping`` (already fully model-agnostic keyboard
canonicalization); :class:`CMDInputMapping` is CMD-specific.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from flashdreams.runtime.canonical import DeviceConverterSchema
from flashdreams.runtime.inputs import (
    CanonicalInputs,
    CanonicalInputSchema,
    CanonicalModality,
    InferenceInput,
    InferenceInputSchema,
    InputField,
    TimeWindow,
    UserInputCapability,
    UserInputs,
)
from flashdreams.runtime.keyboard import DEFAULT_SUPPORTED_KEYS, KeyboardState
from flashdreams.runtime.mapping import InputMappingSchema
from flashdreams.runtime.types import StepRequest

from .controls import CameraPoseIntegrator, PoseSegment

FIELD_CAMERA_POSES = "camera_poses"
FIELD_CAMERA_INTRINSICS = "camera_intrinsics"

_AXES: tuple[str, ...] = ("move_forward", "move_right", "yaw", "pitch")

_AXIS_KEYS: Mapping[str, tuple[str, str]] = {
    # Axis -> (positive key, negative key), matching CameraPoseIntegrator's
    # own key vocabulary. Both directions are derived from this one table so
    # a rebind cannot make them disagree.
    "move_forward": ("w", "s"),
    "move_right": ("e", "q"),
    "yaw": ("a", "d"),
    "pitch": ("i", "k"),
}

_KEY_ALIASES: Mapping[str, str] = {"j": "a", "l": "d"}
"""Alternate yaw keys accepted by ``KeyboardState``, folded onto ``a``/``d``."""


CAMERA_COMMAND = CanonicalModality(
    name="camera_command",
    payload_fields=frozenset({*_AXES, "segments"}),
    description=(
        "Free-camera intent. move_forward, move_right, yaw, and pitch are in "
        "[-1, 1] and hold the level state at the end of the window. segments "
        "carries the piecewise-constant timeline inside the window as "
        "((start_s, end_s, axes), ...), so a consumer can integrate sub-window "
        "timing instead of quantizing control to the chunk boundary."
    ),
)


def _axes_from_keys(pressed: list[str] | frozenset[str]) -> dict[str, float]:
    """Return camera axis values for a resolved set of pressed keys."""
    keys = {_KEY_ALIASES.get(key, key) for key in pressed}
    axes: dict[str, float] = {}
    for axis, (positive, negative) in _AXIS_KEYS.items():
        value = 0.0
        if positive in keys:
            value += 1.0
        if negative in keys:
            value -= 1.0
        axes[axis] = value
    return axes


def _keys_from_axes(axes: Mapping[str, float]) -> frozenset[str]:
    """Return the integrator key set equivalent to ``axes``.

    Pose integration stays the single implementation in
    :class:`CameraPoseIntegrator`, which is expressed over key sets.
    Converting back here keeps live CMD trajectories driven by exactly the
    same integration math regardless of how the axes arrived.
    """
    keys: set[str] = set()
    for axis, (positive, negative) in _AXIS_KEYS.items():
        value = float(axes.get(axis, 0.0))
        if value > 0:
            keys.add(positive)
        elif value < 0:
            keys.add(negative)
    return frozenset(keys)


class KeyboardToCameraCommand:
    """Convert keyboard edges into :data:`CAMERA_COMMAND` level state."""

    def __init__(
        self,
        *,
        name: str = "keyboard-to-camera-command",
        supported_keys: frozenset[str] = DEFAULT_SUPPORTED_KEYS,
        priority: int = 0,
    ) -> None:
        self._supported_keys = supported_keys
        self._state = KeyboardState(supported_keys=supported_keys)
        self._schema = DeviceConverterSchema(
            name=name,
            produces=CAMERA_COMMAND,
            device_kind="keyboard",
            priority=priority,
            consumes=(
                UserInputCapability(
                    event_type="key_down",
                    payload_fields=frozenset({"key"}),
                ),
                UserInputCapability(
                    event_type="key_up",
                    payload_fields=frozenset({"key"}),
                ),
            ),
        )

    @property
    def schema(self) -> DeviceConverterSchema:
        return self._schema

    def reset(self) -> None:
        self._state = KeyboardState(supported_keys=self._supported_keys)

    def convert(
        self,
        user_inputs: UserInputs,
        window: TimeWindow,
    ) -> Mapping[str, Any] | None:
        segments: list[tuple[float, float, dict[str, float]]] = []
        segment_start = window.start_s
        axes = _axes_from_keys(list(self._state.resolved_effective_keys()))

        for event in user_inputs.events:
            if event.event_type not in {"key_down", "key_up"}:
                continue
            key = event.payload.get("key")
            if not isinstance(key, str):
                continue
            edge_t = min(max(float(event.timestamp_s), window.start_s), window.end_s)
            if edge_t > segment_start:
                segments.append((segment_start, edge_t, axes))
                segment_start = edge_t
            self._state.apply_event(
                event="keydown" if event.event_type == "key_down" else "keyup",
                key=key,
            )
            axes = _axes_from_keys(list(self._state.resolved_effective_keys()))

        if window.end_s > segment_start or not segments:
            segments.append((segment_start, window.end_s, axes))

        return CAMERA_COMMAND.value({**axes, "segments": tuple(segments)})


def _segments_from_command(
    command: Mapping[str, Any] | None,
    *,
    start_s: float,
    end_s: float,
) -> list[PoseSegment]:
    """Return integrator-ready segments for one step window."""
    if command is None:
        return [(start_s, end_s, frozenset())]
    raw = command.get("segments")
    if not raw:
        # A source that supplies only level state still drives the step; the
        # whole window then holds one constant command.
        return [(start_s, end_s, _keys_from_axes(command))]
    segments: list[PoseSegment] = []
    for segment_start, segment_end, axes in raw:
        if float(segment_end) <= float(segment_start):
            continue
        segments.append(
            (float(segment_start), float(segment_end), _keys_from_axes(axes))
        )
    if not segments:
        return [(start_s, end_s, _keys_from_axes(command))]
    return segments


class CMDInputMapping:
    """Build CMD live per-step camera inputs from :data:`CAMERA_COMMAND` intent.

    Unlike LingBot's equivalent, this only ever integrates a live keyboard
    trajectory -- CMD's fixed-trajectory replay is a wholly separate code
    path that never constructs this class -- so there is no "fixed trace"
    branch here at all.
    """

    def __init__(
        self,
        *,
        fps: int,
        base_intrinsics: torch.Tensor,
        len_t: int,
        frame_stride: int,
        integrator: CameraPoseIntegrator | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("CMDInputMapping.fps must be > 0.")
        self._fps = int(fps)
        self._base_intrinsics = base_intrinsics.reshape(3, 3)
        self._num_camera_frames = int(len_t) * int(frame_stride)
        self._integrator = integrator or CameraPoseIntegrator()
        self._mapping_schema = InputMappingSchema(
            name="cmd-live-input-mapping",
            consumes=(CAMERA_COMMAND,),
            produces_step=(
                InputField(
                    name=FIELD_CAMERA_POSES,
                    input_modality="c2w_sequence",
                    frequency_consumed="per_step",
                    metadata={"shape": "[T,4,4]", "frame": "camera_to_world"},
                ),
                InputField(
                    name=FIELD_CAMERA_INTRINSICS,
                    input_modality="intrinsics_matrix3_sequence",
                    frequency_consumed="per_step",
                    metadata={"shape": "[T,3,3]"},
                ),
            ),
        )

    @property
    def mapping_schema(self) -> InputMappingSchema:
        return self._mapping_schema

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema:
        """Return the modalities this mapping consumes, for adapter reporting."""
        return CanonicalInputSchema(
            modalities=self._mapping_schema.consumes,
            description="CMD live WASD camera control.",
        )

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        if canonical_schema is not None and not canonical_schema.supports(
            CAMERA_COMMAND
        ):
            raise ValueError(
                "CMD live camera control requires the camera_command canonical "
                "modality, which the selected input source cannot supply."
            )
        if inference_input_schema is not None:
            for name in (FIELD_CAMERA_POSES, FIELD_CAMERA_INTRINSICS):
                if inference_input_schema.field_for(name=name, phase="step") is None:
                    raise ValueError(
                        f"CMD live input mapping produces step input {name!r}, "
                        "which this model does not declare."
                    )

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        return inference_input

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        window = request.user_input_window
        if window is None:
            raise ValueError(
                "CMD live camera control requires a per-step user_input_window."
            )
        num_frames = self._num_camera_frames
        dt = 1.0 / self._fps
        # CMD's camera sample count is the constant len_t*frame_stride on
        # every AR step, but the window's own duration is driven by
        # next_num_frames() (the *decoded* pixel-frame count), which is one
        # frame longer at AR0 due to the decoder's prefix inflation. Always
        # take the window's trailing `num_frames` frame-intervals so the
        # integrator's continuous state stays caught up with real wall-clock
        # time while only skipping a camera token for the interval that
        # corresponds to the prefix's already-known pixel frame.
        frame_times = [
            window.end_s - (num_frames - 1 - i) * dt for i in range(num_frames)
        ]
        command = canonical_inputs.values.get(CAMERA_COMMAND.name)
        segments = _segments_from_command(
            command, start_s=window.start_s, end_s=window.end_s
        )
        poses = self._integrator.integrate_chunk(
            segments=segments, frame_times=frame_times
        )
        poses_t = torch.from_numpy(np.ascontiguousarray(poses)).to(torch.float32)
        poses_t = poses_t.reshape(num_frames, 4, 4)
        intrinsics_t = self._base_intrinsics.reshape(1, 3, 3).repeat(num_frames, 1, 1)

        step = dict(inference_input.step)
        step[FIELD_CAMERA_POSES] = poses_t
        step[FIELD_CAMERA_INTRINSICS] = intrinsics_t
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step=step,
            metadata=inference_input.metadata,
        )

    def reset(self) -> None:
        """Reset accumulated keyboard/pose state at a rollout boundary."""
        self._integrator.reset()


__all__ = [
    "CAMERA_COMMAND",
    "CMDInputMapping",
    "FIELD_CAMERA_INTRINSICS",
    "FIELD_CAMERA_POSES",
    "KeyboardToCameraCommand",
]
