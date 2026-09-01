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

"""Low-precision linear helpers for SANA-WM.

Re-exports the model-agnostic FP8/FP4 ``nn.Linear`` replacement implementation
from ``flashdreams.core.quant`` (moved there so CMD and other integrations can
reuse it without depending on the ``sana`` package).
"""

from __future__ import annotations

from flashdreams.core.quant import (
    FP4_MAX_E2M1,
    FP8_MAX_E4M3,
    FP8_SCALE_EPS,
    NVFP4_BLOCK_SIZE,
    NVFP4_E4M3_SCALE_EPS,
    NVFP4_GLOBAL_SCALE_EPS,
    QuantRecipe,
    TorchScaledMMFP4Linear,
    TorchScaledMMFP4Recipe,
    TorchScaledMMFP8Linear,
    TorchScaledMMFP8Recipe,
    apply_rht16,
    nvfp4_global_scale,
    quantize_nvfp4_swizzled,
    replace_linear_with_quant,
    replace_linear_with_torch_fp4,
    replace_linear_with_torch_fp8,
)

__all__ = [
    "FP4_MAX_E2M1",
    "FP8_MAX_E4M3",
    "FP8_SCALE_EPS",
    "NVFP4_BLOCK_SIZE",
    "NVFP4_E4M3_SCALE_EPS",
    "NVFP4_GLOBAL_SCALE_EPS",
    "QuantRecipe",
    "TorchScaledMMFP4Linear",
    "TorchScaledMMFP4Recipe",
    "TorchScaledMMFP8Linear",
    "TorchScaledMMFP8Recipe",
    "apply_rht16",
    "nvfp4_global_scale",
    "quantize_nvfp4_swizzled",
    "replace_linear_with_quant",
    "replace_linear_with_torch_fp4",
    "replace_linear_with_torch_fp8",
]
