# AeroVerbs Two Visual Collectors Design

**Date:** 2026-08-31

## Goal

Create two independent, resumable Habitat-Sim collection scripts that read
expert trajectories from `/home/abc/AeroVerbs/data/trajctorys` and write
model-ready visual datasets beneath `/home/abc/aeroverbs_visual_dataset`:

- OpenVLA at the source 10 Hz poses;
- WorldVLN at interpolated 16 Hz poses and 49-frame clips.

The source dataset is read-only. The collectors do not train models and do
not generate WorldVLN `latents.pt` files.

## Entrypoints

The repository adds exactly two user-facing scripts:

```text
scripts/collect_openvla_visual_dataset.py
scripts/collect_worldvln_visual_dataset.py
```

They share tested utilities under `aeroverbs/visual_dataset/` for episode
discovery, deterministic splits, pose interpolation, Habitat rendering,
atomic files, and validation. Each script can run without the other and only
writes its own model subtree plus the common immutable split manifests.

## Inputs and split policy

Each source episode must contain `frames.json`, `meta.json`, and
`_render_scene/<scene_id>_with_ground.glb`. Supported skills are
`around_left`, `around_right`, `through`, `over`, and `under`.

Both scripts use the same deterministic AeroVerbs object-family 80/10/10
train/val/test split. The split is created under `splits/` with a schema
version and source-root fingerprint. A collector refuses to use an existing
split manifest if the source fingerprint or split settings differ.

## Output layout

```text
/home/abc/aeroverbs_visual_dataset/
├── dataset_info.json
├── splits/{train,val,test}.json
├── openvla/<split>/<skill>__<scene_id>/
│   ├── 000000.png
│   ├── 000001.png
│   ├── log.json
│   └── .complete.json
├── worldvln/
│   ├── episodes/<split>/<skill>__<scene_id>/
│   │   ├── images/frame_000000.png
│   │   ├── preprocessed_logs.json
│   │   ├── metadata.json
│   │   ├── episode_16hz.mp4
│   │   └── .complete.json
│   ├── clips/<split>/<clip_id>/
│   │   ├── images/frame_000000.png
│   │   └── preprocessed_logs.json
│   ├── backbone/{train,val,test}.jsonl
│   └── clips_index.jsonl
└── logs/{openvla,worldvln}/
```

WorldVLN clip images are hard links to their 16 Hz episode images. OpenVLA
and WorldVLN render independently as requested; no image is linked across
the two model subtrees.

## Pose contract

Source position is metres in z-up world coordinates. Source yaw is
`atan2(forward_y, forward_x)` and is unwrapped before interpolation.

Model poses use first-frame-local coordinates:

```text
x = initial forward
y = initial left
z = world up
pose = [x_cm, y_cm, z_cm, roll_deg, yaw_deg, pitch_deg]
```

Roll and pitch are zero because the stored AeroVerbs camera forward is
horizontal. Every pose must be finite and timestamps must be strictly
increasing.

## OpenVLA collector

OpenVLA uses every source frame without temporal interpolation. For each
frame it renders one RGB PNG at 640x480. `log.json` contains equal-length
`raw_logs`, `preprocessed_logs`, and image lists; raw and preprocessed logs
are the same first-frame-local pose sequence. The instruction is the
canonical English command for the episode skill.

## WorldVLN collector

WorldVLN samples each expert trajectory at `t0 + k/16` while the sample time
does not exceed the source terminal timestamp. Position and unwrapped yaw
are linearly interpolated; no terminal padding or extrapolation is used.

Each episode is rendered to `frame_%06d.png` and encoded as H.264, 16 FPS,
CRF 18, yuv420p. Full 49-frame windows start at `0,16,32,...`; when needed,
one non-duplicate tail window starts at `N-49`. Episodes shorter than 49
frames are retained as rendered episodes but produce no training clip.

Backbone JSONL records contain absolute video path, inclusive frame range,
all 49 explicit frame indices, `fps=16.0`, `sample_frames=49`, and the
canonical instruction as `tarsier2_caption`. `clips_index.jsonl` records the
future expected latent path but does not create a fake latent file.

## Reliability and CLI

Both scripts provide:

```text
--source-root --output-root --skills --scene-id --max-episodes
--num-workers --gpu-device-id --resume --validate-only
```

Defaults are the server paths above, two workers, and GPU 0. Rendering is
episode-scoped so Habitat initializes once per scene. PNG and JSON files are
written to temporary siblings and installed with `os.replace`. A valid
`.complete.json` is written only after all expected files decode and counts
match. Resume skips a complete valid episode and repairs missing or corrupt
frames in incomplete episodes.

Per-episode failures are appended to a model-specific JSONL error log and do
not stop unrelated episodes. The process exits nonzero when any selected
episode remains invalid.

## Acceptance gates

1. Unit tests prove 10 Hz preservation, 16 Hz interpolation across yaw wrap,
   local coordinates, split determinism, clip windows, resume, and schemas.
2. Server smoke test renders one OpenVLA episode and one WorldVLN episode.
3. Frame zero is compared with source `initial_rgb.png`; dimensions must
   match and mean RGB absolute error must be at most 2.0.
4. One episode per skill passes full validation before full collection.
5. Full collection completes with zero invalid episodes and no split overlap.

