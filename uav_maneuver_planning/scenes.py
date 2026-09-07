from typing import Iterable, Optional

from .interfaces import SemanticScene, Vector3
from .registry import ObjectRegistry, SemanticObject


def create_scene(
    scene_id: str = "semantic_scene",
    uav_id: str = "uav_0",
    start: Optional[Vector3] = (1.0, 0.0, 1.0),
    goal: Optional[Vector3] = (9.0, 0.0, 1.0),
    objects: Optional[Iterable[SemanticObject]] = None,
) -> SemanticScene:
    return SemanticScene(
        scene_id=scene_id,
        uav_id=uav_id,
        start=start,
        goal=goal,
        registry=ObjectRegistry(tuple(objects or ())),
    )


def create_demo_scene() -> SemanticScene:
    return create_scene(
        scene_id="demo_scene",
        objects=(
            SemanticObject(
                object_id="pillar_0",
                object_type="pillar",
                center=(5.0, 0.0, 1.5),
                radius=0.35,
                height=3.0,
            ),
        ),
    )


def create_test_scene(
    scene_id: str = "procedural_test_scene",
    uav_id: str = "uav_0",
    start: Optional[Vector3] = None,
    goal: Optional[Vector3] = None,
) -> SemanticScene:
    """Create the 10 m x 10 m procedural scene used for task experiments.

    The standard frame is z-up, with x in [0, 10] and y in [-5, 5].
    Route endpoints are deliberately caller-supplied because each maneuver
    task can require a different start and goal.
    """
    return create_scene(
        scene_id=scene_id,
        uav_id=uav_id,
        start=start,
        goal=goal,
        objects=(
            SemanticObject(
                object_id="door_0",
                object_type="door",
                center=(8.0, -3.0, 1.1),
                width=1.2,
                height=2.2,
                thickness=0.15,
                opening_center=(8.0, -3.0, 1.1),
                bbox_min=(7.325, -3.075, 0.0),
                bbox_max=(8.675, -2.925, 2.2),
                voxel_size=0.08,
            ),
            SemanticObject(
                object_id="pillar_0",
                object_type="pillar",
                center=(3.0, 3.0, 1.5),
                radius=0.35,
                height=3.0,
                voxel_size=0.08,
            ),
            SemanticObject(
                object_id="table_0",
                object_type="table",
                center=(3.0, -3.0, 0.0),
                bbox_min=(2.4, -3.4, 0.0),
                bbox_max=(3.6, -2.6, 0.75),
                voxel_size=0.08,
            ),
        ),
    )
