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

"""CPU tests for CMD's keyboard canonicalization and live input mapping."""

from __future__ import annotations

import pytest
import torch
from flashdreams_cmd.input_mapping import (
    CAMERA_COMMAND,
    FIELD_CAMERA_INTRINSICS,
    FIELD_CAMERA_POSES,
    CMDInputMapping,
    KeyboardToCameraCommand,
)

from flashdreams.runtime import (
    InferenceInput,
    InputCanonicalizer,
    StepRequest,
    TimeWindow,
    UserInputCapability,
    UserInputEvent,
    UserInputs,
    UserInputSchema,
)

pytestmark = pytest.mark.ci_cpu

_KEYBOARD_SOURCE = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
    )
)
_BASE_INTRINSICS = torch.tensor(
    [[416.0, 0.0, 208.0], [0.0, 416.0, 120.0], [0.0, 0.0, 1.0]]
)
_LEN_T = 4
_FRAME_STRIDE = 4
_NUM_CAMERA_FRAMES = _LEN_T * _FRAME_STRIDE  # 16
_FPS = 16


def _mapping() -> CMDInputMapping:
    return CMDInputMapping(
        fps=_FPS,
        base_intrinsics=_BASE_INTRINSICS,
        len_t=_LEN_T,
        frame_stride=_FRAME_STRIDE,
    )


def _step_request(*, window: TimeWindow, step_index: int = 0) -> StepRequest:
    return StepRequest(step_index=step_index, user_input_window=window)


def test_declared_step_fields_match_camera_command_consumption() -> None:
    mapping = _mapping()
    assert mapping.mapping_schema.consumes == (CAMERA_COMMAND,)
    produced = {field.name for field in mapping.mapping_schema.produces_step}
    assert produced == {FIELD_CAMERA_POSES, FIELD_CAMERA_INTRINSICS}


def test_keyboard_events_produce_camera_poses_and_intrinsics_of_expected_shape() -> (
    None
):
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    mapping = _mapping()
    dt = 1.0 / _FPS
    window = TimeWindow(start_s=0.0, end_s=_NUM_CAMERA_FRAMES * dt)
    user_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0, event_type="key_down", payload={"key": "w"}
            ),
        )
    )
    request = _step_request(window=window)

    step_inputs = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            user_inputs, window=window, source_schema=_KEYBOARD_SOURCE
        ),
        inference_input=InferenceInput(),
        request=request,
    )

    poses = step_inputs.step[FIELD_CAMERA_POSES]
    intrinsics = step_inputs.step[FIELD_CAMERA_INTRINSICS]
    assert poses.shape == (_NUM_CAMERA_FRAMES, 4, 4)
    assert intrinsics.shape == (_NUM_CAMERA_FRAMES, 3, 3)
    torch.testing.assert_close(intrinsics[0], _BASE_INTRINSICS)
    # Holding forward has to actually move the camera along the trajectory.
    assert not torch.allclose(poses[0], poses[-1])
    assert poses[-1][:3, 3].abs().sum() > 0


def test_idle_keyboard_leaves_the_camera_stationary() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    mapping = _mapping()
    dt = 1.0 / _FPS
    window = TimeWindow(start_s=0.0, end_s=_NUM_CAMERA_FRAMES * dt)
    request = _step_request(window=window)

    step_inputs = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            UserInputs(), window=window, source_schema=_KEYBOARD_SOURCE
        ),
        inference_input=InferenceInput(),
        request=request,
    )

    poses = step_inputs.step[FIELD_CAMERA_POSES]
    torch.testing.assert_close(poses[0], poses[-1])


def test_ar0_window_uses_trailing_frame_times_not_leading() -> None:
    """AR0's window is one frame longer than the camera needs (decoder's
    causal first-frame padding inflates the decoded pixel-frame count, but
    not the camera-sample count). The mapping must sample the *trailing*
    ``num_camera_frames`` frame-intervals of the window, not the leading
    ones, or every session's camera timing silently drifts by one frame.

    Discriminator: fire a key_down partway through the one extra
    (non-camera-sampled) interval at the *end* of a correctly-leading
    window / *start* of a correctly-trailing window's gap. Only the
    trailing convention's frame_times extend far enough to see its effect.
    """
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    mapping = _mapping()
    dt = 1.0 / _FPS
    # AR0-shaped window: one extra frame beyond what the camera samples.
    window = TimeWindow(start_s=0.0, end_s=(_NUM_CAMERA_FRAMES + 1) * dt)
    # Leading frame_times would span [dt, 16*dt]; trailing (correct) frame_times
    # span [2*dt, 17*dt]. An event at 16.5*dt is invisible to leading sampling
    # (all leading samples are <= 16*dt) but visible to trailing sampling
    # (whose last sample is 17*dt > 16.5*dt).
    event_time = 16.5 * dt
    user_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=event_time, event_type="key_down", payload={"key": "w"}
            ),
        )
    )
    request = _step_request(window=window)

    step_inputs = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            user_inputs, window=window, source_schema=_KEYBOARD_SOURCE
        ),
        inference_input=InferenceInput(),
        request=request,
    )

    poses = step_inputs.step[FIELD_CAMERA_POSES]
    assert poses.shape == (_NUM_CAMERA_FRAMES, 4, 4)
    assert poses[-1][:3, 3].abs().sum() > 0, (
        "expected the trailing frame_times to observe the late key_down; "
        "camera did not move, suggesting leading (incorrect) sampling"
    )


def test_reset_clears_accumulated_pose_state() -> None:
    canonicalizer = InputCanonicalizer([KeyboardToCameraCommand()])
    mapping = _mapping()
    dt = 1.0 / _FPS
    window = TimeWindow(start_s=0.0, end_s=_NUM_CAMERA_FRAMES * dt)
    user_inputs = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.0, event_type="key_down", payload={"key": "w"}
            ),
        )
    )
    request = _step_request(window=window)

    first = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            user_inputs, window=window, source_schema=_KEYBOARD_SOURCE
        ),
        inference_input=InferenceInput(),
        request=request,
    )
    assert first.step[FIELD_CAMERA_POSES][-1][:3, 3].abs().sum() > 0

    mapping.reset()
    canonicalizer.reset()
    second = mapping.map_step_inputs(
        canonical_inputs=canonicalizer.canonicalize(
            UserInputs(), window=window, source_schema=_KEYBOARD_SOURCE
        ),
        inference_input=InferenceInput(),
        request=request,
    )
    torch.testing.assert_close(second.step[FIELD_CAMERA_POSES][0], torch.eye(4))
