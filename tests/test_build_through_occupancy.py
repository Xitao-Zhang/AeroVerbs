import json
import sys
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_through_occupancy import build_scene_occupancy


def test_build_scene_occupancy_writes_under_compatible_npz(tmp_path):
    scene_id = "auto_door_opening_test"
    scene_dir = tmp_path / "_scenes" / "through" / scene_id
    scene_dir.mkdir(parents=True)

    mesh = trimesh.creation.box(extents=(0.2, 2.0, 3.0))
    mesh.apply_translation((5.0, 0.0, 1.5))
    mesh.export(scene_dir / f"{scene_id}.glb")
    (scene_dir / "metadata.json").write_text(
        json.dumps({"objects": [{"id": "door_0", "type": "door"}]}),
        encoding="utf-8",
    )

    output = build_scene_occupancy(scene_dir, voxel_size=0.2, overwrite=True)

    assert output == scene_dir / "occupancy" / "door_0_voxels.npz"
    payload = np.load(output)
    assert set(["voxel_size", "origin", "occupied_indices", "bbox_min", "bbox_max", "metadata"]).issubset(payload.files)
    assert payload["occupied_indices"].ndim == 2
    assert payload["occupied_indices"].shape[1] == 3
    assert payload["occupied_indices"].shape[0] > 0
