# AeroVerbs Visual Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a resumable Habitat-Sim pipeline that derives synchronized 10 Hz OpenVLA data and 16 Hz WorldVLN data from existing AeroVerbs expert trajectories.

**Architecture:** A new `aeroverbs.visual_dataset` package separates pose construction, layout/splits, rendering, export, and validation. A single CLI runs `prepare`, `render`, `export`, and `validate`; model-ready files use hard links to immutable canonical episode images. Real Habitat rendering runs only on the server, while all deterministic logic and orchestration are tested locally with a fake renderer.

**Tech Stack:** Python 3.9+, NumPy, Pillow, SciPy, Habitat-Sim 0.3.1, ffmpeg/ffprobe, unittest, multiprocessing.

**Spec:** `docs/superpowers/specs/2026-08-31-aeroverbs-visual-dataset-design.md`

## Global Constraints

- Source root is read-only: `/home/abc/AeroVerbs/data/trajctorys`.
- Output root is separate: `/home/abc/aeroverbs_visual_dataset`.
- Renderer environment is Conda `habitat-render`, Python 3.9, Habitat-Sim 0.3.1.
- RGB is PNG, 640x480; server rendering defaults to two workers on GPU 0.
- OpenVLA uses unchanged source timestamps/poses (nominal 10 Hz).
- WorldVLN uses interpolated 16 Hz poses, 49-frame windows, stride 16, plus one non-duplicate tail window.
- Coordinates are first-frame local `x=forward, y=left, z=up`; positions are centimetres and angles degrees for model files.
- WorldVLN MP4 is H.264 CRF 18, yuv420p, 16 FPS.
- No fake `latents.pt` is created.
- Existing unrelated working-tree changes must remain untouched.

---

### Task 1: Pose construction and coordinate contract

**Files:**
- Create: `aeroverbs/visual_dataset/__init__.py`
- Create: `aeroverbs/visual_dataset/trajectory.py`
- Create: `tests/test_visual_dataset_trajectory.py`

**Interfaces:**
- Produces `PoseSequence`, `load_source_pose_sequence()`, `resample_pose_sequence()`, `world_to_initial_local_pose()`, and `build_episode_pose_payloads()`.
- `PoseSequence.timestamps_s`, `.positions_world_m`, `.yaw_world_rad` are NumPy arrays with lengths `T`, `T x 3`, and `T`.
- `build_episode_pose_payloads(sample_dir)` returns `(pose_10hz_dict, pose_16hz_dict, source_hashes)` used by prepare/render/export.

- [ ] **Step 1: Write failing pose tests**

```python
class VisualTrajectoryTest(unittest.TestCase):
    def test_source_poses_are_preserved(self):
        seq = PoseSequence(
            timestamps_s=np.array([0.0, 0.1, 0.2]),
            positions_world_m=np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3]], dtype=float),
            yaw_world_rad=np.array([0.0, 0.1, 0.2]),
        )
        out = resample_pose_sequence(seq, hz=None)
        np.testing.assert_array_equal(out.timestamps_s, seq.timestamps_s)
        np.testing.assert_array_equal(out.positions_world_m, seq.positions_world_m)

    def test_16hz_stops_without_extrapolation(self):
        seq = make_sequence([0.0, 0.1, 0.2])
        out = resample_pose_sequence(seq, hz=16.0)
        np.testing.assert_allclose(out.timestamps_s, [0.0, 0.0625, 0.125, 0.1875])
        self.assertLessEqual(out.timestamps_s[-1], 0.2)

    def test_yaw_unwrap_interpolates_short_way(self):
        seq = make_sequence([0.0, 1.0], yaw_deg=[179.0, -179.0])
        out = resample_pose_sequence(seq, hz=2.0)
        self.assertAlmostEqual(np.degrees(out.yaw_world_rad[1]), 180.0, places=5)

    def test_initial_local_pose_uses_left_positive(self):
        pose = world_to_initial_local_pose(
            position_world_m=np.array([-1.0, 2.0, 4.0]),
            yaw_world_rad=np.pi,
            origin_world_m=np.array([0.0, 0.0, 3.0]),
            initial_yaw_world_rad=np.pi / 2,
        )
        np.testing.assert_allclose(pose, [200.0, 100.0, 100.0, 0.0, 90.0, 0.0])
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_trajectory -v`

Expected: import failure for `aeroverbs.visual_dataset.trajectory`.

- [ ] **Step 3: Implement validated pose dataclasses and interpolation**

```python
@dataclass(frozen=True)
class PoseSequence:
    timestamps_s: np.ndarray
    positions_world_m: np.ndarray
    yaw_world_rad: np.ndarray

    def validate(self) -> None:
        if self.timestamps_s.ndim != 1 or len(self.timestamps_s) == 0:
            raise ValueError("timestamps must be a non-empty vector")
        if self.positions_world_m.shape != (len(self.timestamps_s), 3):
            raise ValueError("positions must have shape (T,3)")
        if self.yaw_world_rad.shape != (len(self.timestamps_s),):
            raise ValueError("yaw must have shape (T,)")
        if not all(np.all(np.isfinite(x)) for x in (self.timestamps_s, self.positions_world_m, self.yaw_world_rad)):
            raise ValueError("pose arrays must be finite")
        if np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("timestamps must be strictly increasing")

def resample_pose_sequence(source: PoseSequence, hz: Optional[float]) -> PoseSequence:
    source.validate()
    if hz is None:
        return PoseSequence(source.timestamps_s.copy(), source.positions_world_m.copy(), source.yaw_world_rad.copy())
    count = int(np.floor((source.timestamps_s[-1] - source.timestamps_s[0]) * hz + 1e-9)) + 1
    target_t = source.timestamps_s[0] + np.arange(count, dtype=np.float64) / hz
    positions = np.column_stack([np.interp(target_t, source.timestamps_s, source.positions_world_m[:, i]) for i in range(3)])
    yaw = np.interp(target_t, source.timestamps_s, np.unwrap(source.yaw_world_rad))
    return PoseSequence(target_t, positions, yaw)
```

Implement source JSON loading, forward normalization, SHA-256 hashing of source metadata, local/world pose serialization, and inverse conversion for validation.

- [ ] **Step 4: Run pose tests**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_trajectory -v`

Expected: all pose tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add -f aeroverbs/visual_dataset/__init__.py aeroverbs/visual_dataset/trajectory.py tests/test_visual_dataset_trajectory.py
git commit -m "feat: add visual dataset pose synchronization"
```

---

### Task 2: Episode discovery, deterministic splits, and clip layout

**Files:**
- Create: `aeroverbs/visual_dataset/layout.py`
- Create: `tests/test_visual_dataset_layout.py`

**Interfaces:**
- Consumes `build_episode_pose_payloads()` from Task 1.
- Produces `EpisodeRecord`, `discover_episodes()`, `assign_splits()`, `worldvln_window_starts()`, `prepare_dataset()` and atomic JSON/JSONL helpers.
- `prepare_dataset(source_root, output_root, skills, episode_ids, max_episodes)` writes split files, episode metadata, both pose files, and `dataset_info.json`; it does not render.

- [ ] **Step 1: Write failing discovery, split, and window tests**

```python
def test_window_starts_include_non_duplicate_tail():
    self.assertEqual(worldvln_window_starts(49), [0])
    self.assertEqual(worldvln_window_starts(66), [0, 16, 17])
    self.assertEqual(worldvln_window_starts(81), [0, 16, 32])
    self.assertEqual(worldvln_window_starts(48), [])

def test_episode_derivatives_never_cross_splits():
    records = make_family_records(count=10)
    assigned = assign_splits(records, train_frac=0.8, val_frac=0.1)
    paths = {name: {r.episode_id for r in values} for name, values in assigned.items()}
    self.assertFalse(paths["train"] & paths["val"])
    self.assertFalse(paths["train"] & paths["test"])
    self.assertFalse(paths["val"] & paths["test"])
```

Include tests that families with fewer than three episodes stay in train, only five skills are discovered, filters are deterministic, and output-under-source/source-under-output are rejected.

- [ ] **Step 2: Run layout tests and verify failure**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_layout -v`

Expected: import failure for `aeroverbs.visual_dataset.layout`.

- [ ] **Step 3: Implement layout and prepare phase**

```python
@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    skill: str
    scene_id: str
    object_family: str
    sample_dir: Path
    split: Optional[str] = None

def worldvln_window_starts(num_frames: int, length: int = 49, stride: int = 16) -> list[int]:
    if num_frames < length:
        return []
    starts = list(range(0, num_frames - length + 1, stride))
    tail = num_frames - length
    if tail not in starts:
        starts.append(tail)
    return starts
```

Reuse the family-name rule from `aeroverbs.data.dataset`, but keep split serialization inside this package so server prepare does not depend on training-only imports. Use temporary sibling files and `os.replace` for every JSON/JSONL write.

- [ ] **Step 4: Run trajectory and layout tests together**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_trajectory tests.test_visual_dataset_layout -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add -f aeroverbs/visual_dataset/layout.py tests/test_visual_dataset_layout.py
git commit -m "feat: prepare visual dataset layout and splits"
```

---

### Task 3: Resumable Habitat rendering

**Files:**
- Create: `aeroverbs/visual_dataset/renderer.py`
- Create: `tests/test_visual_dataset_renderer.py`
- Modify: `aeroverbs/env/rgb_renderer.py`

**Interfaces:**
- Consumes prepared episode metadata and pose JSON from Task 2.
- Produces `is_valid_rgb()`, `atomic_save_rgb()`, `render_episode()`, and `render_dataset()`.
- `render_episode(job, renderer_factory=HabitatRGBRenderer)` returns `RenderResult(episode_id, rendered, linked, skipped, error)`.

- [ ] **Step 1: Write failing atomic/resume tests with a fake renderer**

```python
class FakeRenderer:
    calls = 0
    def __init__(self, scene_path, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def render(self, position_std, forward_std):
        type(self).calls += 1
        return np.full((480, 640, 3), 17, dtype=np.uint8)

def test_resume_skips_valid_and_replaces_corrupt_png():
    job = prepared_job_with_shared_t0()
    render_episode(job, renderer_factory=FakeRenderer)
    first_calls = FakeRenderer.calls
    corrupt_one_output(job)
    result = render_episode(job, renderer_factory=FakeRenderer)
    self.assertEqual(result.rendered, 1)
    self.assertGreater(FakeRenderer.calls, first_calls)
    self.assertTrue(is_valid_rgb(corrupt_path, width=640, height=480))

def test_shared_timestamp_renders_once_and_hardlinks_both_views():
    result = render_episode(prepared_job_with_only_shared_t0(), renderer_factory=FakeRenderer)
    self.assertEqual(result.rendered, 1)
    self.assertTrue(os.path.samefile(result.path_10hz, result.path_16hz))
```

- [ ] **Step 2: Run renderer tests and verify failure**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_renderer -v`

Expected: import failure for `aeroverbs.visual_dataset.renderer`.

- [ ] **Step 3: Implement safe frame rendering and job execution**

Implement:

```python
def atomic_save_rgb(array: np.ndarray, target: Path) -> None:
    if array.dtype != np.uint8 or array.shape != (480, 640, 3):
        raise ValueError("renderer must return uint8 RGB 480x640")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.png")
    Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
    if not is_valid_rgb(temporary, 640, 480):
        temporary.unlink(missing_ok=True)
        raise ValueError("temporary PNG validation failed")
    os.replace(temporary, target)
```

Merge target timestamps by tolerance, hard-link exactly shared poses, process one complete episode per worker, append state/error events under a file lock, and write `.render_complete.json` only after recounting all images. Extend the renderer error message to name `habitat-render` as the supported server environment without changing camera behavior.

- [ ] **Step 4: Run renderer and existing RGB tests**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_renderer tests.test_rgb_renderer -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add -f aeroverbs/visual_dataset/renderer.py aeroverbs/env/rgb_renderer.py tests/test_visual_dataset_renderer.py
git commit -m "feat: add resumable dual-rate Habitat rendering"
```

---

### Task 4: OpenVLA and WorldVLN exporters

**Files:**
- Create: `aeroverbs/visual_dataset/exporters.py`
- Create: `tests/test_visual_dataset_exporters.py`

**Interfaces:**
- Consumes completed canonical episode directories from Tasks 2-3.
- Produces `canonical_instruction()`, `export_openvla_episode()`, `encode_worldvln_video()`, `export_worldvln_episode()`, and `export_dataset()`.
- `encode_worldvln_video(rgb_dir, output_path, ffmpeg_bin="ffmpeg")` invokes ffmpeg with a checked argument list, never a shell string.

- [ ] **Step 1: Write failing model-format tests**

```python
def test_openvla_export_links_images_and_writes_matching_pose_lengths():
    result = export_openvla_episode(prepared_rendered_episode())
    log = json.loads((result.output_dir / "log.json").read_text())
    self.assertEqual(log["length"], len(log["raw_logs"]))
    self.assertEqual(log["raw_logs"], log["preprocessed_logs"])
    self.assertTrue(os.path.samefile(result.output_dir / "000000.png", result.rgb_source / "000000.png"))

def test_worldvln_export_uses_49_aligned_images_and_poses():
    result = export_worldvln_episode(prepared_episode(num_frames_16hz=66), ffmpeg_runner=fake_ffmpeg)
    self.assertEqual(result.window_starts, [0, 16, 17])
    for clip in result.clips:
        poses = json.loads((clip / "preprocessed_logs.json").read_text())
        self.assertEqual(len(poses), 49)
        self.assertEqual(len(list((clip / "images").glob("frame_*.png"))), 49)
```

Test all five instruction strings, clips shorter than 49, absolute `frame_idxs`, expected latent paths without latent file creation, and ffmpeg failure propagation.

- [ ] **Step 2: Run exporter tests and verify failure**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_exporters -v`

Expected: import failure for `aeroverbs.visual_dataset.exporters`.

- [ ] **Step 3: Implement OpenVLA and WorldVLN exports**

Use `os.link` and fail on cross-device/link errors. Encode video with:

```python
command = [
    ffmpeg_bin, "-y", "-framerate", "16", "-i", str(rgb_dir / "%06d.png"),
    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output_path),
]
subprocess.run(command, check=True, capture_output=True, text=True)
```

Write backbone JSONL records with 49 absolute `frame_idxs`, `begin_frame_id`, inclusive `end_frame_id`, `fps=16.0`, and canonical caption. Write `clips_index.jsonl` entries containing an expected latent path such as `<clip>/latents.pt` but do not create the file.

- [ ] **Step 4: Run exporter tests plus earlier unit tests**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_trajectory tests.test_visual_dataset_layout tests.test_visual_dataset_renderer tests.test_visual_dataset_exporters -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add -f aeroverbs/visual_dataset/exporters.py tests/test_visual_dataset_exporters.py
git commit -m "feat: export OpenVLA and WorldVLN datasets"
```

---

### Task 5: Full dataset validation

**Files:**
- Create: `aeroverbs/visual_dataset/validation.py`
- Create: `tests/test_visual_dataset_validation.py`

**Interfaces:**
- Consumes the complete output tree.
- Produces `ValidationIssue`, `ValidationSummary`, `validate_episode()`, and `validate_dataset()`.
- `validate_dataset(...).ok` is true only when there are zero error-severity issues; warnings include source episodes too short for WorldVLN.

- [ ] **Step 1: Write failing corruption/leakage/video tests**

```python
def test_validation_rejects_split_leakage_and_non_hardlink_exports():
    root = complete_fake_dataset()
    inject_duplicate_episode_across_splits(root)
    replace_hardlink_with_copy(root)
    summary = validate_dataset(root, ffprobe_runner=fake_ffprobe)
    codes = {issue.code for issue in summary.issues}
    self.assertIn("split_leakage", codes)
    self.assertIn("openvla_not_hardlinked", codes)
    self.assertFalse(summary.ok)

def test_validation_accepts_short_worldvln_episode_as_warning():
    summary = validate_dataset(complete_fake_dataset(worldvln_frames=48), ffprobe_runner=fake_ffprobe)
    self.assertTrue(summary.ok)
    self.assertIn("worldvln_too_short", {issue.code for issue in summary.issues})
```

- [ ] **Step 2: Run validation tests and verify failure**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_validation -v`

Expected: import failure for `aeroverbs.visual_dataset.validation`.

- [ ] **Step 3: Implement validators and summary aggregation**

Check PNG mode/size/count, pose count, SHA metadata, `os.path.samefile`, JSONL paths, clip length, ffprobe `avg_frame_rate` and `nb_read_frames`, split disjointness, path nesting, pose round trip, finite action distributions, and canonical instructions. Write summary atomically to `logs/validation_summary.json` with per-split/per-skill counts and issue records.

- [ ] **Step 4: Run all visual dataset unit tests**

Run: `conda run -n visfly python -m unittest discover -s tests -p 'test_visual_dataset_*.py' -v`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add -f aeroverbs/visual_dataset/validation.py tests/test_visual_dataset_validation.py
git commit -m "feat: validate shared visual dataset artifacts"
```

---

### Task 6: CLI orchestration and fake-renderer integration test

**Files:**
- Create: `scripts/build_visual_dataset.py`
- Create: `tests/test_build_visual_dataset.py`
- Modify: `aeroverbs/visual_dataset/__init__.py`

**Interfaces:**
- Exposes CLI subcommands `prepare`, `render`, `export`, and `validate`.
- All subcommands accept `--source-root`, `--output-root`, repeated/comma-separated skill filters, episode filters, and `--max-episodes`.
- Render adds `--num-workers`, `--gpu-device-id`, `--resume`, `--retry-errors`, and `--overwrite`.
- Process exit is zero only when the selected phase completes without error-level failures.

- [ ] **Step 1: Write failing parser and end-to-end tests**

```python
def test_cli_defaults_match_server_contract():
    args = parse_args(["render"])
    self.assertEqual(args.source_root, "/home/abc/AeroVerbs/data/trajctorys")
    self.assertEqual(args.output_root, "/home/abc/aeroverbs_visual_dataset")
    self.assertEqual(args.num_workers, 2)
    self.assertEqual(args.gpu_device_id, 0)

def test_fake_pipeline_prepare_render_export_validate():
    with TemporaryDirectory() as tmp:
        source = create_two_episode_source(Path(tmp) / "source")
        output = Path(tmp) / "visual"
        self.assertEqual(main(["prepare", "--source-root", str(source), "--output-root", str(output)]), 0)
        self.assertEqual(run_render_with_factory(output, FakeRenderer), 0)
        self.assertEqual(run_export_with_fake_ffmpeg(output), 0)
        self.assertEqual(run_validate_with_fake_ffprobe(output), 0)
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `conda run -n visfly python -m unittest tests.test_build_visual_dataset -v`

Expected: import or missing-entry-point failure.

- [ ] **Step 3: Implement CLI without importing Habitat for non-render phases**

Add the repository root to `sys.path` like existing scripts. Lazily import `HabitatRGBRenderer` only inside `render`. Print one machine-readable final JSON object containing phase, selected episodes, successes, failures, warnings, output root, and log paths. Reject `--overwrite` together with `--resume`.

- [ ] **Step 4: Run focused and existing regression tests**

Run:

```bash
conda run -n visfly python -m unittest discover -s tests -p 'test_visual_dataset_*.py' -v
conda run -n visfly python -m unittest tests.test_build_visual_dataset tests.test_rgb_renderer tests.test_rollout_openvla -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add -f scripts/build_visual_dataset.py tests/test_build_visual_dataset.py aeroverbs/visual_dataset/__init__.py
git commit -m "feat: add visual dataset build CLI"
```

---

### Task 7: Server launcher, deployment, and real Habitat smoke gates

**Files:**
- Create: `scripts/run_visual_dataset_server.sh`
- Create: `tests/test_visual_dataset_server_script.py`
- Modify: `README.md`

**Interfaces:**
- Launcher consumes `PHASE`, `SOURCE_ROOT`, `OUTPUT_ROOT`, `NUM_WORKERS`, `GPU_DEVICE_ID`, and trailing CLI arguments.
- It runs `/home/abc/miniconda3/bin/conda run -n habitat-render python scripts/build_visual_dataset.py` from `/home/abc/AeroVerbs`.

- [ ] **Step 1: Write a failing launcher contract test**

```python
def test_server_launcher_has_safe_defaults_and_shell_syntax():
    script = REPO_ROOT / "scripts" / "run_visual_dataset_server.sh"
    completed = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    self.assertEqual(completed.returncode, 0, completed.stderr)
    text = script.read_text()
    self.assertIn("SOURCE_ROOT=${SOURCE_ROOT:-/home/abc/AeroVerbs/data/trajctorys}", text)
    self.assertIn("OUTPUT_ROOT=${OUTPUT_ROOT:-/home/abc/aeroverbs_visual_dataset}", text)
    self.assertNotIn("rm -rf", text)
```

- [ ] **Step 2: Run the launcher test and verify failure**

Run: `conda run -n visfly python -m unittest tests.test_visual_dataset_server_script -v`

Expected: failure because the launcher does not exist.

- [ ] **Step 3: Implement launcher and document exact commands**

Launcher core:

```bash
#!/usr/bin/env bash
set -euo pipefail
PHASE=${PHASE:-validate}
SOURCE_ROOT=${SOURCE_ROOT:-/home/abc/AeroVerbs/data/trajctorys}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/abc/aeroverbs_visual_dataset}
NUM_WORKERS=${NUM_WORKERS:-2}
GPU_DEVICE_ID=${GPU_DEVICE_ID:-0}
cd /home/abc/AeroVerbs
exec /home/abc/miniconda3/bin/conda run -n habitat-render python scripts/build_visual_dataset.py \
  "$PHASE" --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" \
  --num-workers "$NUM_WORKERS" --gpu-device-id "$GPU_DEVICE_ID" "$@"
```

Make the CLI accept render-only options for render and omit them for other phases in the launcher via a Bash case statement. Document prepare, one-episode smoke, five-skill smoke, full render, export, validate, retry, and resume commands.

- [ ] **Step 4: Run all local verification**

Run:

```bash
conda run -n visfly python -m unittest discover -s tests -p 'test_visual_dataset_*.py' -v
conda run -n visfly python -m unittest tests.test_build_visual_dataset tests.test_visual_dataset_server_script tests.test_rgb_renderer tests.test_rollout_openvla -v
git diff --check
```

Expected: all tests pass and `git diff --check` emits no output.

- [ ] **Step 5: Commit Task 7**

```bash
git add -f scripts/run_visual_dataset_server.sh tests/test_visual_dataset_server_script.py README.md
git commit -m "docs: add visual dataset server workflow"
```

- [ ] **Step 6: Check server prerequisites without changing data**

Run over SSH:

```bash
/home/abc/miniconda3/bin/conda run -n habitat-render python -c "import habitat_sim, numpy, PIL, scipy; print(habitat_sim.__file__)"
ffmpeg -version
ffprobe -version
df -h /home/abc
```

Expected: imports succeed, ffmpeg and ffprobe exit zero, and at least 25 GiB is free.

- [ ] **Step 7: Deploy only required code, preserving server data**

Run locally:

```bash
rsync -a --relative \
  aeroverbs/visual_dataset \
  aeroverbs/env/rgb_renderer.py \
  scripts/build_visual_dataset.py \
  scripts/run_visual_dataset_server.sh \
  zxt@a6k-pro6000.local:/home/abc/AeroVerbs/
```

Do not use `--delete`. Verify remote paths and imports after transfer.

- [ ] **Step 8: Run prepare and one-episode real rendering gate**

On the server:

```bash
cd /home/abc/AeroVerbs
PHASE=prepare bash scripts/run_visual_dataset_server.sh --max-episodes 1
PHASE=render bash scripts/run_visual_dataset_server.sh --max-episodes 1 --num-workers 1 --resume
PHASE=validate bash scripts/run_visual_dataset_server.sh --max-episodes 1
```

Then compute frame-zero MAE against the source `initial_rgb.png`; expected RGB MAE is at most 2.0 and dimensions are 640x480. Inspect the two images once with local download or server-generated comparison artifact.

- [ ] **Step 9: Run five-skill gate**

Prepare/render/export/validate one explicit episode per skill with one worker. Expected: five successful episodes, no error-level validation issues, OpenVLA hard links valid, and every eligible WorldVLN clip contains 49 aligned images/poses.

- [ ] **Step 10: Run full build with resume**

On the server:

```bash
cd /home/abc/AeroVerbs
PHASE=prepare bash scripts/run_visual_dataset_server.sh
PHASE=render NUM_WORKERS=2 bash scripts/run_visual_dataset_server.sh --resume
PHASE=export bash scripts/run_visual_dataset_server.sh --resume
PHASE=validate bash scripts/run_visual_dataset_server.sh
```

Expected: `validation_summary.json` reports all selected source episodes processed, zero failed episodes, zero split leakage, and valid per-split/per-skill frame and clip counts. If errors exist, run render/export with `--retry-errors`, then validate again before claiming completion.
