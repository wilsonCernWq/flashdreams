// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Declares the WASD-family keys CMDInputMapping's KeyboardToCameraCommand
// actually recognizes (see flashdreams_cmd/input_mapping.py's _AXIS_KEYS /
// _KEY_ALIASES). The shared browser UI only captures and sends keydown/keyup
// for keys listed here -- without this file, allowedKeys stays empty and
// every keypress is silently dropped before it reaches the data channel.
const controls = [
  {
    label: "Move",
    keys: [
      { key: "w", label: "Forward" },
      { key: "a", label: "Yaw left" },
      { key: "s", label: "Backward" },
      { key: "d", label: "Yaw right" },
    ],
  },
  {
    label: "Strafe",
    keys: [
      { key: "q", label: "Strafe left" },
      { key: "e", label: "Strafe right" },
    ],
  },
  {
    label: "Pitch",
    keys: [
      { key: "i", label: "Pitch up" },
      { key: "k", label: "Pitch down" },
    ],
  },
  {
    // Alternate yaw keys, matching lingbot's own "Look" group -- folded onto
    // a/d server-side (see input_mapping.py's _KEY_ALIASES).
    label: "Look",
    keys: [
      { key: "j", label: "Look left" },
      { key: "l", label: "Look right" },
    ],
  },
]

export default {
  modelName: "CMD",
  controls,
}
