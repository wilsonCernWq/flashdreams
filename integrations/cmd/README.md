<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams CMD integration

Inference-only support for the six released
[Context-Matched Distillation (CMD)](https://github.com/nv-tlabs/cmd)
Cosmos-Predict2.5 2B checkpoints. The integration preserves CMD's independent
first-frame prefix, four-step self-forcing schedule, causal KV window, and
optional block-relative camera-ray controls.

Serving is intentionally outside this first integration.

## Presets

| Runner slug | Chunk | Latent frames | KV window | Camera |
| --- | ---: | ---: | ---: | --- |
| `cmd-chunk1-short-i2v` | 1 | 24 | 21 | No |
| `cmd-chunk1-long-i2v` | 1 | 126 | 21 | No |
| `cmd-chunk4-short-i2v` | 4 | 21 | 16 | No |
| `cmd-chunk4-long-i2v` | 4 | 121 | 16 | No |
| `cmd-chunk1-camera-i2v` | 1 | 32 | 21 | Yes |
| `cmd-chunk4-camera-i2v` | 4 | 29 | 24 | Yes |

Each latent interval after the first frame decodes to four pixel frames. For
example, the chunk-4 short preset writes 81 frames: the input prefix plus 20
generated latent intervals.

## Install

From the FlashDreams repository root:

```bash
uv sync
```

The gated Cosmos-Predict2.5 dependencies and CMD weights are downloaded from
Hugging Face on first use. Authenticate beforehand:

```bash
hf auth login
```

## Run

The presets default to CMD's public example inputs and can be run directly:

```bash
uv run flashdreams-run cmd-chunk4-short-i2v
uv run flashdreams-run cmd-chunk4-camera-i2v
```

Use local inputs or override rollout length with regular runner flags:

```bash
uv run flashdreams-run cmd-chunk4-short-i2v \
  --image-path /path/to/image.png \
  --prompt /path/to/prompt.txt \
  --num-chunks 5

uv run flashdreams-run cmd-chunk4-camera-i2v \
  --image-path /path/to/camera_image.png \
  --camera-path /path/to/camera.npz \
  --prompt "A first-person driving scene."
```

The camera NPZ must contain pixel-rate `target_w2c` (`[T, 4, 4]`) and
`target_intrinsics` (`[T, 3, 3]`) arrays. The runner consumes exactly
`1 + (latent_frames - 1) * 4` camera entries and converts world-to-camera
matrices to camera-to-world before ray construction.

Multi-GPU context parallelism follows the standard runner contract:

```bash
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
  cmd-chunk4-short-i2v
```

## Test

```bash
uv run pytest integrations/cmd/tests flashdreams/tests/test_kvcache.py
```
