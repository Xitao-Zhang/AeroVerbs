# AeroVerbs Visual Dataset Design

**Date:** 2026-08-31

## 1. Goal

Build a standalone, reproducible visual dataset at
`/home/abc/aeroverbs_visual_dataset` on the server. The dataset derives two
time-aligned visual views from the existing AeroVerbs expert trajectories:

- a 10 Hz view for OpenVLA, preserving the current AeroVerbs control and
  evaluation frequency;
- a 16 Hz view for WorldVLN, matching its default `fps=16`, `step=16`, and
  `num_frames=49` training and inference contract.

Both views are rendered independently by Habitat-Sim from one continuous
expert-pose curve. They share one immutable train/validation/test split and
one explicit coordinate convention. The source trajectory dataset remains
read-only.

WorldVLN `latents.pt` files are outside this build. They are model-versioned
derived artifacts and will be generated in a later stage from the completed
WorldVLN clip index and a specified WorldVLN/InfinityStar checkpoint.

## 2. Inputs and Non-Goals

### 2.1 Source data

The server source root is:

```text
/home/abc/AeroVerbs/data/trajctorys
```

Each source episode must provide:

```text
<skill>/<scene_id>/
├── frames.json
├── meta.json
├── trajectory.json
├── initial_rgb.png
└── _render_scene/<scene_id>_with_ground.glb
```

The five supported skills are `around_left`, `around_right`, `through`,
`over`, and `under`.

### 2.2 Non-goals

This build does not:

- modify source `frames.json`, metadata, trajectories, radar files, or GLBs;
- collect depth, radar, heatmaps, masks, IMU, velocity, or acceleration;
- generate WorldVLN `latents.pt` files;
- train OpenVLA or WorldVLN;
- perform WorldVLN action-aware GRPO rollouts;
- change either model's inference server.

## 3. Chosen Approach

Construct a continuous pose curve from each source episode, sample it at the
two target frequencies, and render every target pose with Habitat-Sim.

The rejected alternatives are:

1. selecting approximate 10 Hz frames from a 16 Hz render, because the two
   clocks do not remain aligned and RGB would not match the action pose;
2. video-frame interpolation from 10 Hz to 16 Hz, because interpolated pixels
   are not a true observation from the labeled 3D camera pose.

Habitat-Sim 0.3.1 in the server's `habitat-render` Conda environment is the
reference renderer. RGB resolution is 640 by 480. The renderer must retain
the existing AeroVerbs collection conventions: z-up to Habitat coordinates,
front-center camera rotation, fixed camera-relative light, horizontal flip,
and white compositing for empty black pixels.

## 4. Dataset Layout

```text
/home/abc/aeroverbs_visual_dataset/
├── dataset_info.json
├── splits/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── episodes/<split>/<skill>/<scene_id>/
│   ├── metadata.json
│   ├── pose_10hz.json
│   ├── pose_16hz.json
│   ├── rgb_10hz/000000.png
│   ├── rgb_16hz/000000.png
│   ├── episode_16hz.mp4
│   └── .render_complete.json
├── openvla/<split>/<episode_id>/
│   ├── 000000.png
│   └── log.json
├── worldvln/
│   ├── backbone/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── clips/<split>/<clip_id>/
│   │   ├── images/frame_000000.png
│   │   └── preprocessed_logs.json
│   └── clips_index.jsonl
└── logs/
    ├── build_state.jsonl
    ├── errors.jsonl
    └── validation_summary.json
```

`episode_id` is `<skill>__<scene_id>`. `clip_id` is
`<episode_id>__s<start_frame:06d>`.

OpenVLA images are hard links to `rgb_10hz`. WorldVLN action-decoder clip
images are hard links to `rgb_16hz`. All paths reside on the same filesystem;
if hard-link creation fails, the build fails that episode rather than silently
copying data.

## 5. Split Policy

The split is fixed before rendering or clip creation. It reuses AeroVerbs'
deterministic object-family stratification with train fraction 0.8 and
validation fraction 0.1; test receives the remainder. Episodes from source
families with fewer than three samples remain in train, matching the current
dataset behavior.

All derived data from an episode inherits that episode's split. No clip,
hard link, video window, or model-ready export may cross split boundaries.
The split files record source episode paths, skill, scene ID, object family,
and split-assignment algorithm version.

## 6. Pose and Time Synchronization

### 6.1 Source pose

For each source frame, read:

- `timestamp` in seconds;
- `camera_position_std` in world metres, z-up;
- `camera_forward_std` as the horizontal world forward vector.

Timestamps must be finite and strictly increasing. Positions and forward
vectors must be finite. A zero-length horizontal forward vector invalidates
the episode.

### 6.2 10 Hz view

The 10 Hz view uses source timestamps and source poses without interpolation.
This preserves the exact supervision and control frequency currently used by
AeroVerbs and OpenVLA rollout.

### 6.3 Continuous curve and 16 Hz view

Compute source yaw as `atan2(forward_y, forward_x)`, unwrap it over time, and
linearly interpolate position and unwrapped yaw against source timestamps.
Sample at:

```text
t[k] = t[0] + k / 16
```

for every non-negative integer `k` satisfying `t[k] <= t[-1] + 1e-9`.
Reconstruct horizontal forward as `[cos(yaw), sin(yaw), 0]`. Roll and pitch
remain zero. No extrapolation or repeated terminal frame is allowed.

### 6.4 Stored coordinate representations

Each pose entry stores both:

1. world render pose: position in metres and yaw in radians;
2. model pose in the first-frame local coordinate system.

The local coordinate convention is:

```text
x = initial forward
y = initial left
z = world up
position unit = centimetres
angle unit = degrees
pose order = [x, y, z, roll, yaw, pitch]
```

The local origin and initial yaw are zero. Roll and pitch are zero. The
convention is written into every episode's metadata and top-level
`dataset_info.json`. Model adapters must not infer the y-axis convention from
WorldVLN client comments.

## 7. Rendering

Rendering operates per complete episode. One worker initializes one
Habitat-Sim instance for the episode GLB, renders all required poses, and then
closes the instance.

The worker forms the union of 10 Hz and 16 Hz timestamps. When timestamps and
world poses agree within `1e-9` seconds and `1e-7` metres/radians, the frame is
rendered once and hard-linked into both frequency directories. Other frames
are rendered independently.

Every PNG is written to a temporary sibling path, opened and checked as an RGB
640 by 480 image, and atomically installed with `os.replace`. Existing PNGs
are skipped only when they decode successfully and have the required mode and
dimensions.

The default is two worker processes on GPU device 0. Workers receive episode
jobs rather than individual frames, so a scene is never initialized once per
frame. Worker count and GPU device are CLI options.

An episode is complete only after all expected images pass validation and
`.render_complete.json` is atomically written. The marker records image
counts, source hashes, renderer version, resolution, frequencies, coordinate
schema version, and completion time.

## 8. OpenVLA Export

Each OpenVLA episode contains top-level image hard links named
`000000.png`, `000001.png`, and so on, plus `log.json`.

`log.json` contains:

```json
{
  "id": "<episode_id>",
  "raw_logs": [[0, 0, 0, 0, 0, 0]],
  "preprocessed_logs": [[0, 0, 0, 0, 0, 0]],
  "instruction": "<canonical instruction>",
  "instruction_unified": "<canonical instruction>",
  "length": 1
}
```

The illustrated arrays contain one placeholder row only to show field order;
the actual arrays contain every 10 Hz model pose and have length equal to the
image count. `raw_logs` and `preprocessed_logs` are identical first-frame-local
poses. A rigid world-to-local transform does not change the body-frame
relative action computed by UAV-Flow's dataset loader.

Canonical instructions are:

- `around_left`: `Fly around the object on its left side.`
- `around_right`: `Fly around the object on its right side.`
- `through`: `Fly through the opening.`
- `over`: `Fly over the object.`
- `under`: `Fly under the object.`

## 9. WorldVLN Export

### 9.1 Full episode video

Encode `rgb_16hz` once per episode as H.264, 16 FPS, CRF 18, yuv420p. Frame
order and frame count must match `pose_16hz.json`. Encoding requires ffmpeg;
missing ffmpeg is a preflight error.

### 9.2 Backbone windows

Each backbone JSONL record references the full episode MP4 and one inclusive
49-frame window:

```json
{
  "video_path": "/absolute/path/to/episode_16hz.mp4",
  "begin_frame_id": 0,
  "end_frame_id": 48,
  "fps": 16.0,
  "frame_idxs": [0, 1, 2],
  "sample_frames": 49,
  "tarsier2_caption": "<canonical instruction>"
}
```

The illustrated `frame_idxs` is abbreviated. The actual field lists all 49
absolute frame indices for the window.

Window starts are `0, 16, 32, ...` while a full 49-frame window fits. If the
last regular window does not end at the episode's final frame, append one
window starting at `N - 49`. Do not duplicate an existing start. Episodes with
fewer than 49 frames produce no WorldVLN windows and are reported, not padded.

### 9.3 Action-decoder clips

For each backbone window, create a clip directory containing 49 image hard
links named `frame_000000.png` through `frame_000048.png` and a
`preprocessed_logs.json` list of the matching 49 local poses. The pose schema
is `[x, y, z, roll, yaw, pitch]` in centimetres/degrees.

`clips_index.jsonl` records clip ID, split, skill, source episode, images
directory, pose path, expected future latent path, frame range, instruction,
and coordinate schema. It is the input contract for the later latent
extraction stage. No empty or fake latent file is created.

## 10. Build Phases and CLI

The single entry point is `scripts/build_visual_dataset.py` with four
subcommands:

```text
prepare   discover source episodes, assign splits, and write pose plans
render    render double-frequency RGB with Habitat-Sim
export    create model-ready links, logs, MP4 files, clips, and manifests
validate  perform complete consistency validation without rendering
```

Common options include source root, output root, selected skills, selected
episode IDs, maximum episodes, and resume mode. Render options include worker
count, GPU device, width, and height. Destructive replacement requires an
explicit `--overwrite`; it is never implied by disabling resume.

The build writes append-only state events to `logs/build_state.jsonl` and
per-episode failures to `logs/errors.jsonl`. One episode failure does not stop
other episodes. `--retry-errors` selects the latest failed episodes for a new
attempt.

## 11. Validation and Acceptance

Validation checks:

- image count equals pose count at each frequency;
- every image decodes as RGB 640 by 480;
- MP4 reports 16 FPS and the expected frame count;
- OpenVLA image files are hard links to the corresponding 10 Hz images;
- every WorldVLN clip has exactly 49 image hard links and 49 poses;
- all JSON/JSONL schema fields and referenced paths are valid;
- train, validation, and test episode sets are pairwise disjoint;
- no source path is underneath the output root;
- no output path is underneath the source root;
- all five canonical instructions map to the correct skill;
- local/world pose round trips stay within numerical tolerance;
- action magnitudes and yaw deltas are finite and summarized by split/skill.

Server acceptance proceeds in gates:

1. Render one episode. Compare its rendered 10 Hz frame zero to the source
   `initial_rgb.png`; require matching dimensions and a mean absolute RGB
   error no greater than 2.0. Visually inspect orientation once.
2. Render one episode from each skill and run full validation.
3. Run the complete dataset with two workers and resume enabled.
4. Run full validation and write `logs/validation_summary.json` with episode,
   frame, clip, failure, and per-split/per-skill counts.

The complete build is accepted only when there are no failed episodes among
the selected source set, all model-ready manifests validate, and the summary
reports zero split leakage.

## 12. Test Strategy

Unit tests cover:

- source 10 Hz poses are preserved exactly;
- 16 Hz timestamp count and terminal-boundary behavior;
- yaw interpolation across the `-pi`/`pi` boundary;
- first-frame-local coordinate conversion and inverse conversion;
- 49-frame regular and final-tail window starts;
- episodes shorter than 49 frames are reported without padding;
- object-family split determinism and absence of leakage;
- canonical instruction mapping;
- atomic PNG installation and corrupted-image detection;
- resume skips valid frames but replaces invalid frames;
- hard-link identity checks;
- OpenVLA and WorldVLN schema generation.

An integration test uses a fake renderer and a temporary two-episode source to
exercise `prepare`, `render`, `export`, and `validate` without Habitat-Sim.
Real Habitat-Sim rendering is verified only in the server smoke gates because
it requires GPU/EGL and a real GLB.

## 13. Operational Constraints

- Server renderer environment: Conda `habitat-render`, Python 3.9,
  Habitat-Sim 0.3.1.
- Source root: `/home/abc/AeroVerbs/data/trajctorys`.
- Output root: `/home/abc/aeroverbs_visual_dataset`.
- Default render workers: 2.
- Default GPU device: 0.
- RGB format: PNG, 640 by 480.
- OpenVLA frequency: source 10 Hz timestamps.
- WorldVLN frequency: 16 Hz.
- WorldVLN clip length: 49 frames.
- WorldVLN clip stride: 16 frames plus one non-duplicate tail-aligned window.
- WorldVLN MP4: H.264 CRF 18, yuv420p, 16 FPS.
- Source data is read-only and never mutated.
