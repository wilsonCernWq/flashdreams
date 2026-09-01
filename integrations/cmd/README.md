<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams CMD integration

Support for the six released
[Context-Matched Distillation (CMD)](https://github.com/nv-tlabs/cmd)
Cosmos-Predict2.5 2B checkpoints. The integration preserves CMD's independent
first-frame prefix, four-step self-forcing schedule, causal KV window, and
optional block-relative camera-ray controls. Both offline batch generation
and interactive WebRTC serving are supported.

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

## Run (WebRTC)

Serve a preset as an interactive WebRTC session: the browser drives the
camera with the keyboard and receives a live video stream back. This uses
the same shared `flashdreams.serving.webrtc` transport as the `lingbot`
integration.

```bash
# Camera-conditioned preset: live WASD camera control.
uv run flashdreams-run cmd-chunk4-camera-i2v webrtc --host 0.0.0.0 --port 8080

# Non-camera preset: streams a fixed prompt/image rollout, no camera input.
uv run flashdreams-run cmd-chunk4-short-i2v webrtc --host 0.0.0.0 --port 8080

# Override the default session's prompt/first-frame image at launch, same
# --prompt/--image-path flags as offline `run` above.
uv run flashdreams-run cmd-chunk4-camera-i2v webrtc \
  --prompt "A drone flying over a neon-lit cyberpunk city at night." \
  --image-path /path/to/first_frame.png \
  --host 0.0.0.0 --port 8080
```

`--camera-path` is still accepted (it's a shared `CMDRunnerConfig` field with
offline `run`) but has **no effect** under `webrtc` -- there is no
fixed-trajectory camera replay in WebRTC mode, only live keyboard control for
camera-conditioned presets (see below).

For a reproducible checked-in configuration, use
`--manifest configs/launch_manifest/cmd_webrtc.yaml`; the manifest's runner
must match the selected runner slug.

Then open:

- [http://localhost:8080/request_session](http://localhost:8080/request_session)
- [http://localhost:8080/healthz](http://localhost:8080/healthz) (`runtime_ready` indicates preload completion)

### Runtime requirements

Same checkpoint / HuggingFace requirements as offline `run` above, plus a
CUDA-capable GPU reachable from the serving process (no CPU fallback for
serving).

### DataChannel message format

Browser -> server:

```json
{
  "type": "action",
  "action": {
    "event": "keydown",
    "key": "w"
  }
}
```

- Supported `event` values: `keydown` / `keyup` (requires `key` in
  `w,a,s,d,q,e,i,j,k,l`). There is no `step` action: a session generates
  continuously until the client disconnects, with no chunk-count cutoff
  (that stays exclusive to the offline `run` mode above).
- Key mapping: `w/s` forward/backward, `a/d` (or `j/l`) yaw left/right,
  `q/e` strafe left/right, `i/k` pitch up/down.
- Camera control only applies to camera-conditioned presets
  (`cmd-*-camera-i2v`); other presets ignore keyboard input and stream a
  fixed rollout instead.
- The prompt and first-frame image are the runner config's own CLI-level
  defaults (`--prompt`, `--image-path`), resolved once at server start and
  reused for every connecting session -- there is no per-session override
  and no text-event protocol (unlike lingbot's compatibility server).

Server -> browser:

```json
{
  "type": "chunk_done",
  "chunk_index": 3,
  "num_frames": 4,
  "enqueued_frames": 4
}
```

## Test

```bash
uv run pytest integrations/cmd/tests flashdreams/tests/test_kvcache.py
```
