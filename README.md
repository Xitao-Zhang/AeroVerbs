# AeroVerbs

AeroVerbs is a LiDAR-based UAV navigation policy.

## Project structure

- `aeroverbs/` contains the policy, data loaders, training code, and simulation utilities.
- `configs/` contains YAML configuration files and `scripts/` contains the training, evaluation, and rollout commands.
- `eval_manifests/` contains evaluation scenario lists. `docs/` and `exp/` contain workflow notes and experiment results.
- `final.pt` is the trained policy checkpoint. `test_data_254.zip` contains 254 test trajectories with recorded LiDAR frames and scene meshes.

## Setup

Run the following from the repository root with Python 3.10 or newer and Git LFS installed.

```bash
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For training or offline evaluation, set `data.data_root` in [`configs/default.yaml`](configs/default.yaml) to your prepared trajectory dataset, organized as `<skill>/<scene_id>/`. The config also controls the data split, device, and output directory.

## Usage

To verify the LiDAR data for one bundled test scene, extract the archive and compare its first recorded frame against a fresh raycast.

```bash
unzip test_data_254.zip -d data
python -m pip install trimesh rtree scipy
python scripts/verify_radar.py \
  --sample-dir data/test_data_254/through/auto_door_opening_00111
```

With a prepared dataset configured, train a policy or evaluate the included checkpoint on the configured test split.

```bash
python scripts/train.py --config configs/default.yaml
python scripts/eval.py --config configs/default.yaml --checkpoint final.pt --split test
```

Training saves checkpoints and metrics under `outputs/<experiment>/<run_name>/` by default.
