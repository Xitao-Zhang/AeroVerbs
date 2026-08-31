# AeroVerbs Two Visual Collectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy two independently runnable, process-parallel Habitat-Sim collectors for 10 Hz OpenVLA data and interpolated 16 Hz WorldVLN data, stopping after one-episode smoke tests rather than starting full collection.

**Architecture:** Shared modules under `aeroverbs.visual_dataset` own immutable splits, pose conversion, process-pool rendering, atomic output, and validation. Two thin CLI scripts select the 10 Hz OpenVLA or 16 Hz WorldVLN contract. Each worker opens one Habitat-Sim scene and renders one complete episode, so workers never share simulator state.

**Tech Stack:** Python 3.9, NumPy, Pillow, SciPy, Habitat-Sim 0.3.1, `concurrent.futures.ProcessPoolExecutor`, ffmpeg/ffprobe, unittest.

**Spec:** `docs/superpowers/specs/2026-08-31-aeroverbs-two-visual-collectors-design.md`

## Global Constraints

- Source root is read-only: `/home/abc/AeroVerbs/data/trajctorys`.
- Output root is `/home/abc/aeroverbs_visual_dataset`.
- Add exactly two user-facing scripts: `collect_openvla_visual_dataset.py` and `collect_worldvln_visual_dataset.py`.
- OpenVLA renders unmodified source poses at nominal 10 Hz.
- WorldVLN linearly interpolates position and unwrapped yaw at exactly 16 Hz.
- Both CLIs support process parallelism with `--num-workers`; default is 2 and every worker owns its Habitat-Sim instance.
- This implementation run must stop after `--max-episodes 1` smoke tests for each collector. Do not launch full collection.
- RGB is PNG at 640x480 on GPU 0.
- Model pose is first-frame local `[x_cm,y_left_cm,z_cm,roll_deg,yaw_deg,pitch_deg]`.
- WorldVLN uses 49-frame windows with stride 16 and one non-duplicate tail window.
- Do not create fake WorldVLN latent files.
- Preserve all unrelated working-tree changes.

---

### Task 1: Shared pose and split contract

**Files:**
- Create: `aeroverbs/visual_dataset/__init__.py`
- Create: `aeroverbs/visual_dataset/core.py`
- Test: `tests/test_visual_dataset_core.py`

**Interfaces:**
- Produces `PoseSequence`, `Episode`, `load_pose_sequence(Path)`, `resample_poses(PoseSequence, float)`, `to_initial_local_poses(PoseSequence)`, `discover_episodes(Path)`, `assign_splits(list[Episode])`, and `ensure_split_manifests(Path, Path)`.
- `PoseSequence` contains finite `timestamps_s[T]`, `positions_world_m[T,3]`, and unwrapped-compatible `yaw_world_rad[T]`.

- [ ] **Step 1: Write failing tests for exact 10 Hz preservation, 16 Hz boundaries, yaw wrap, left-positive local coordinates, deterministic disjoint splits, and source/output nesting rejection.**

- [ ] **Step 2: Run `conda run -n visfly python -m unittest tests.test_visual_dataset_core -v` and confirm failure because `aeroverbs.visual_dataset.core` is missing.**

- [ ] **Step 3: Implement validated dataclasses, JSON loading, interpolation, coordinate conversion, discovery, object-family splitting, source fingerprinting, and atomic split JSON writes.**

- [ ] **Step 4: Re-run the focused test and confirm all cases pass.**

- [ ] **Step 5: Commit only Task 1 files with message `feat: add visual dataset pose and split core`.**

---

### Task 2: Atomic process-parallel Habitat renderer

**Files:**
- Create: `aeroverbs/visual_dataset/rendering.py`
- Test: `tests/test_visual_dataset_rendering.py`

**Interfaces:**
- Produces `RenderJob`, `RenderResult`, `is_valid_png(Path, int, int)`, `atomic_save_png(np.ndarray, Path)`, `render_episode(RenderJob)`, and `run_parallel(jobs, num_workers, worker_fn=render_episode)`.
- `run_parallel` preserves input result order, isolates per-job exceptions, and uses serial execution only when `num_workers == 1`.

- [ ] **Step 1: Write failing tests using a fake renderer for corrupt-frame repair, valid-frame resume, atomic output, worker exception isolation, ordered results, and proof that two jobs execute through two distinct worker PIDs when `num_workers=2`.**

- [ ] **Step 2: Run `conda run -n visfly python -m unittest tests.test_visual_dataset_rendering -v` and confirm the missing-module failure.**

- [ ] **Step 3: Implement PNG validation/atomic writes, lazy Habitat import inside the worker, one simulator context per episode, completion markers, and a spawn-safe `ProcessPoolExecutor`.**

- [ ] **Step 4: Run rendering tests and `tests.test_rgb_renderer`; confirm all pass.**

- [ ] **Step 5: Commit Task 2 files with message `feat: add parallel Habitat episode renderer`.**

---

### Task 3: OpenVLA 10 Hz collector CLI

**Files:**
- Create: `aeroverbs/visual_dataset/openvla.py`
- Create: `scripts/collect_openvla_visual_dataset.py`
- Test: `tests/test_collect_openvla_visual_dataset.py`

**Interfaces:**
- Produces `build_openvla_jobs(...)`, `write_openvla_log(...)`, `validate_openvla_episode(...)`, `parse_args(...)`, and `main(...)`.
- CLI options include `--source-root`, `--output-root`, `--skills`, `--scene-id`, `--max-episodes`, `--num-workers`, `--gpu-device-id`, `--resume`, and `--validate-only`.

- [ ] **Step 1: Write failing parser/schema tests proving default server paths, default two workers, source-pose preservation, exact image/log length equality, canonical instructions, and `--max-episodes 1` selection.**

- [ ] **Step 2: Run `conda run -n visfly python -m unittest tests.test_collect_openvla_visual_dataset -v` and confirm failure because the script is missing.**

- [ ] **Step 3: Implement the OpenVLA job builder, 10 Hz rendering orchestration, `log.json`, completion validation, machine-readable summary, and nonzero exit on invalid selected episodes.**

- [ ] **Step 4: Run OpenVLA collector tests together with shared core/rendering tests.**

- [ ] **Step 5: Commit Task 3 files with message `feat: collect OpenVLA visual trajectories`.**

---

### Task 4: WorldVLN 16 Hz collector CLI

**Files:**
- Create: `aeroverbs/visual_dataset/worldvln.py`
- Create: `scripts/collect_worldvln_visual_dataset.py`
- Test: `tests/test_collect_worldvln_visual_dataset.py`

**Interfaces:**
- Produces `worldvln_window_starts(int, length=49, stride=16)`, `build_worldvln_jobs(...)`, `encode_episode_video(...)`, `export_worldvln_clips(...)`, `validate_worldvln_episode(...)`, `parse_args(...)`, and `main(...)`.
- The CLI exposes the same filtering, parallelism, resume, and validation options as OpenVLA.

- [ ] **Step 1: Write failing tests for 16 Hz pose count, no extrapolation, 49-frame regular/tail windows, short-episode handling, hard-linked clip images, pose alignment, JSONL fields, and checked ffmpeg argument construction.**

- [ ] **Step 2: Run `conda run -n visfly python -m unittest tests.test_collect_worldvln_visual_dataset -v` and confirm failure because the script is missing.**

- [ ] **Step 3: Implement 16 Hz orchestration, episode images/poses, H.264 encoding, hard-linked action clips, backbone split JSONL, clip index, validation, and machine-readable summary.**

- [ ] **Step 4: Run WorldVLN collector and all shared visual-dataset tests.**

- [ ] **Step 5: Commit Task 4 files with message `feat: collect WorldVLN visual trajectories`.**

---

### Task 5: Local regression verification and server deployment

**Files:**
- Modify only if tests expose a collector defect; otherwise no production files change.

**Interfaces:**
- Deploys the committed collector package/scripts to `/home/abc/AeroVerbs` without overwriting unrelated server changes.

- [ ] **Step 1: Run `conda run -n visfly python -m unittest discover -s tests -p 'test_visual_dataset_*.py' -v`.**

- [ ] **Step 2: Run both collector-specific test modules plus existing `tests.test_rgb_renderer` and `tests.test_rollout_openvla`.**

- [ ] **Step 3: Compare local and server Git/status state, transfer only new collector files, and compile them in `habitat-render`.**

- [ ] **Step 4: Run both CLIs with `--help` in the server `habitat-render` environment and verify defaults show process parallelism.**

- [ ] **Step 5: Confirm ffmpeg/ffprobe availability and perform no rendering yet.**

---

### Task 6: One-episode server smoke tests only

**Files:**
- Outputs only under `/home/abc/aeroverbs_visual_dataset`.

**Interfaces:**
- Validates real Habitat/EGL rendering and both model schemas without authorizing full collection.

- [ ] **Step 1: Run OpenVLA collector with one explicit scene, `--max-episodes 1 --num-workers 1 --resume`, then validate its PNG count, `log.json`, and completion marker.**

- [ ] **Step 2: Compare OpenVLA frame zero with source `initial_rgb.png`; require 640x480 RGB and mean absolute error at most 2.0.**

- [ ] **Step 3: Run WorldVLN collector on the same explicit scene with `--max-episodes 1 --num-workers 1 --resume`, then validate 16 Hz frame count, MP4, clip count, pose count, and JSONL references.**

- [ ] **Step 4: Run both scripts in `--validate-only` mode for only that scene and save summaries.**

- [ ] **Step 5: Stop. Report smoke-test results and the exact future full-collection commands, but do not execute those commands.**

