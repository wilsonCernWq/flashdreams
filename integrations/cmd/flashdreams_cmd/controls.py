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

"""Keyboard-driven camera pose integration for CMD's live control mode.

This is a copy of the pose-integration piece of ``lingbot.controls`` (which
is fully model-agnostic keyboard-to-pose math with no LingBot-specific
logic). CMD is a second consumer, so a future pass could hoist this into a
shared ``flashdreams.runtime`` module the way ``inputs.py``'s image/camera
loading was factored out of ``runner.py`` -- deferred for now to keep this
feature's footprint local while the live-control model-quality question
(see ``docs/sage_attention_plan.md`` sibling planning notes) is still open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

PoseSegment = tuple[float, float, frozenset[str]]


def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    cos_t = np.float32(np.cos(angle_rad))
    sin_t = np.float32(np.sin(angle_rad))
    if axis == "x":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_t, -sin_t],
                [0.0, sin_t, cos_t],
            ],
            dtype=np.float32,
        )
    if axis == "y":
        return np.array(
            [
                [cos_t, 0.0, sin_t],
                [0.0, 1.0, 0.0],
                [-sin_t, 0.0, cos_t],
            ],
            dtype=np.float32,
        )
    if axis == "z":
        return np.array(
            [
                [cos_t, -sin_t, 0.0],
                [sin_t, cos_t, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.eye(3, dtype=np.float32)


@dataclass(slots=True)
class CameraPoseIntegrator:
    """Integrate a piecewise-constant keyboard timeline into a camera path."""

    move_speed_per_s: float = 0.8
    rotate_speed_rad_per_s: float = float(np.deg2rad(32.0))
    pitch_limit_rad: float = float(np.deg2rad(85.0))
    coordinate_system: Literal["RDF", "FLU"] = "RDF"
    _current_pose: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
    )
    _current_pitch: float = 0.0

    def __post_init__(self) -> None:
        if self.coordinate_system not in {"RDF", "FLU"}:
            raise ValueError(
                "coordinate_system must be 'RDF' (right-down-forward) "
                "or 'FLU' (forward-left-up)"
            )

    def reset(self, pose: np.ndarray | None = None) -> None:
        if pose is None:
            self._current_pose = np.eye(4, dtype=np.float32)
            self._current_pitch = 0.0
            return
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}")
        self._current_pose = pose.astype(np.float32, copy=True)
        if self.coordinate_system == "FLU":
            self._current_pitch = float(np.arcsin(np.clip(pose[2, 0], -1.0, 1.0)))
        else:
            self._current_pitch = float(np.arctan2(pose[2, 1], pose[1, 1]))

    def current_pose(self) -> np.ndarray:
        return self._current_pose.copy()

    def _advance(self, *, state: frozenset[str], duration: float) -> None:
        if duration <= 0:
            return

        yaw_rate = 0.0
        if self.coordinate_system == "FLU":
            if "a" in state or "j" in state:
                yaw_rate += self.rotate_speed_rad_per_s
            if "d" in state or "l" in state:
                yaw_rate -= self.rotate_speed_rad_per_s
        else:
            if "a" in state or "j" in state:
                yaw_rate -= self.rotate_speed_rad_per_s
            if "d" in state or "l" in state:
                yaw_rate += self.rotate_speed_rad_per_s
        pitch_rate = 0.0
        if "i" in state:
            pitch_rate += self.rotate_speed_rad_per_s
        if "k" in state:
            pitch_rate -= self.rotate_speed_rad_per_s

        yaw_delta = yaw_rate * duration
        pitch_delta = pitch_rate * duration

        new_pitch = self._current_pitch + pitch_delta
        if -self.pitch_limit_rad <= new_pitch <= self.pitch_limit_rad:
            self._current_pitch = new_pitch
        else:
            pitch_delta = 0.0

        rot = self._current_pose[:3, :3]
        trans = self._current_pose[:3, 3]
        if self.coordinate_system == "FLU":
            rot_pitch = _rotation_matrix("y", -pitch_delta)
            rot_yaw = _rotation_matrix("z", yaw_delta)
        else:
            rot_pitch = _rotation_matrix("x", pitch_delta)
            rot_yaw = _rotation_matrix("y", yaw_delta)
        rot_new = rot_yaw @ rot @ rot_pitch

        forward_rate = 0.0
        if "w" in state:
            forward_rate += self.move_speed_per_s
        if "s" in state:
            forward_rate -= self.move_speed_per_s
        right_rate = 0.0
        if "e" in state:
            right_rate += self.move_speed_per_s
        if "q" in state:
            right_rate -= self.move_speed_per_s

        if self.coordinate_system == "FLU":
            vec_forward = rot_new[:, 0]
            vec_right = -rot_new[:, 1]
            forward_flat = np.array(
                [vec_forward[0], vec_forward[1], 0.0], dtype=np.float32
            )
            right_flat = np.array([vec_right[0], vec_right[1], 0.0], dtype=np.float32)
        else:
            vec_right = rot_new[:, 0]
            vec_forward = rot_new[:, 2]
            forward_flat = np.array(
                [vec_forward[0], 0.0, vec_forward[2]], dtype=np.float32
            )
            right_flat = np.array([vec_right[0], 0.0, vec_right[2]], dtype=np.float32)
        forward_norm = np.linalg.norm(forward_flat)
        right_norm = np.linalg.norm(right_flat)
        if forward_norm > 0:
            forward_flat /= forward_norm
        if right_norm > 0:
            right_flat /= right_norm

        move_vec = forward_flat * (forward_rate * duration) + right_flat * (
            right_rate * duration
        )
        self._current_pose = np.eye(4, dtype=np.float32)
        self._current_pose[:3, :3] = rot_new
        self._current_pose[:3, 3] = trans + move_vec

    def integrate_chunk(
        self,
        *,
        segments: list[PoseSegment],
        frame_times: list[float],
    ) -> np.ndarray:
        if not segments:
            raise ValueError("segments must be non-empty")
        if not frame_times:
            raise ValueError("frame_times must be non-empty")
        chunk_start = segments[0][0]
        chunk_end = segments[-1][1]
        if any(
            frame_times[i] >= frame_times[i + 1] for i in range(len(frame_times) - 1)
        ):
            raise ValueError("frame_times must be strictly increasing")
        if frame_times[0] < chunk_start - 1e-9 or frame_times[-1] > chunk_end + 1e-9:
            raise ValueError(
                "frame_times must lie within the chunk window "
                f"[{chunk_start}, {chunk_end}]"
            )

        poses: list[np.ndarray] = []
        cur_t = chunk_start
        ft_idx = 0
        for _, seg_end, seg_state in segments:
            while ft_idx < len(frame_times) and frame_times[ft_idx] <= seg_end:
                target_t = frame_times[ft_idx]
                self._advance(state=seg_state, duration=target_t - cur_t)
                cur_t = target_t
                poses.append(self._current_pose.copy())
                ft_idx += 1
            if seg_end > cur_t:
                self._advance(state=seg_state, duration=seg_end - cur_t)
                cur_t = seg_end

        return np.stack(poses, axis=0).astype(np.float32)


__all__ = ["CameraPoseIntegrator", "PoseSegment"]
