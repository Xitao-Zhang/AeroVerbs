#!/usr/bin/env python3
"""Export the fixed 10 m x 10 m procedural scene used by experiments."""

import argparse
from pathlib import Path

from uav_maneuver_planning.habitat_assets import create_habitat_scene, create_render_scene_with_ground
from uav_maneuver_planning.scenes import create_test_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_scenes/procedural_test_scene"),
        help="Directory for the object-only GLB, metadata, and occupancy files.",
    )
    args = parser.parse_args()

    asset = create_habitat_scene(create_test_scene(), args.output_dir)
    render_glb = args.output_dir / f"{asset.scene_glb.stem}_with_ground.glb"
    create_render_scene_with_ground(asset.scene_glb, render_glb)
    print(f"scene_glb={asset.scene_glb}")
    print(f"render_scene_glb={render_glb}")
    print(f"metadata={asset.metadata_json}")


if __name__ == "__main__":
    main()
