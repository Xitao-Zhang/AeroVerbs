# DAgger Clearance and Region-Aware Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the hard collision radius at 0.15 m while restoring BC-equivalent DAgger losses and giving the online Fast-Planner teacher an independent preferred clearance plus a soft, non-binding use of the precomputed passable region.

**Architecture:** The collector continues to use 0.15 m for collision termination and SDF hard inflation. `FastPlannerTeacherConfig` gains a raw-surface `preferred_clearance`; the B-spline ESDF target is derived as `preferred_clearance - collision_radius`. The region point cloud selects a soft semantic guide near the expert gate crossing and high-clearance region points, but no post-plan check requires the crossing to lie inside the imperfect region.

**Tech Stack:** Python 3.10, NumPy, PyTorch, unittest, Python Fast-Planner port.

**Spec:** User-approved bounded design in the 2026-08-26 Codex task; no separate spec file.

## Global Constraints

- Hard collision radius remains exactly 0.15 m by default.
- DAgger loss defaults inherit the loaded BC checkpoint unless the CLI explicitly overrides them.
- The passable region is a soft guide source, never a hard crossing-membership validator.
- Existing precomputed SDF files are reused; no SDF recomputation is allowed.

---

### Task 1: Restore BC-equivalent DAgger loss defaults

**Files:**
- Modify: `scripts/train_dagger.py`
- Modify: `scripts/dagger_iterate.py`
- Test: `tests/test_train_dagger_loss_defaults.py`
- Test: `tests/test_dagger_iterate_fast_planner.py`

**Interfaces:**
- Consumes: loss weights stored in the starting checkpoint config.
- Produces: optional CLI overrides where omission preserves checkpoint values.

- [ ] Write tests proving omitted DAgger loss arguments resolve to the checkpoint values and explicit values still override them.
- [ ] Run the focused tests and verify they fail because current defaults force completion 0, clearance 0, and temporal smoothness 0.2.
- [ ] Change the three CLI defaults to `None`, apply only non-`None` overrides, and forward them from `dagger_iterate.py` only when supplied.
- [ ] Run the focused tests and verify they pass.

### Task 2: Add independent Teacher preferred clearance

**Files:**
- Modify: `aeroverbs/data/fast_planner_teacher.py`
- Modify: `scripts/dagger_collect.py`
- Modify: `scripts/dagger_iterate.py`
- Test: `tests/test_fast_planner_teacher.py`
- Test: `tests/test_dagger_collect_fast_planner.py`
- Test: `tests/test_dagger_iterate_fast_planner.py`

**Interfaces:**
- Consumes: `collision_radius: float` and `preferred_clearance: float` in raw surface-distance meters.
- Produces: `FastPlannerConfig.optimization_dist0 = max(preferred_clearance - collision_radius, 0.0)` and persisted planner statistics.

- [ ] Write tests proving 0.15 m hard collision and 0.50 m preferred clearance yield a 0.35 m optimizer target and are forwarded through collector/iterator commands.
- [ ] Run focused tests and verify the missing option/config behavior fails.
- [ ] Add validation, CLI forwarding, planner derivation, and stats fields without changing hard SDF inflation.
- [ ] Run focused tests and verify they pass.

### Task 3: Use the passable region as a soft guide

**Files:**
- Modify: `aeroverbs/data/fast_planner_teacher.py`
- Test: `tests/test_fast_planner_teacher.py`

**Interfaces:**
- Consumes: region point cloud, gate summary plane, expert trajectory, and precomputed SDF.
- Produces: one semantic guide selected from finite region candidates using proximity to the expert plane crossing and preferred-clearance reward; falls back to expert crossing, then summary center.

- [ ] Write tests proving the guide favors a region candidate near the expert crossing with adequate clearance, and proving no hard region-membership validation rejects a valid planner path.
- [ ] Run the focused test and verify current centroid selection fails the expected guide.
- [ ] Refactor guide loading into geometry/region loading followed by SDF-aware soft candidate scoring.
- [ ] Run focused tests and verify they pass.

### Task 4: Integration verification and documentation

**Files:**
- Modify: `PROJECT_STATUS.md`

**Interfaces:**
- Consumes: all previous behavior and real precomputed dataset SDF.
- Produces: verified CLI pipeline and a recorded project-state update.

- [ ] Run the full server unittest suite.
- [ ] Run compileall on modified modules.
- [ ] Instantiate the Teacher on real around/through samples and verify hard radius, preferred clearance, region guide source, and finite waypoints.
- [ ] Print the generated collect and train commands to verify the full DAgger orchestration.
- [ ] Append the implemented behavior, defaults, tests, and remaining limitations to `PROJECT_STATUS.md`.
