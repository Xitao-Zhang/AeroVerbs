from typing import Iterable, Optional

from .interfaces import SemanticScene, Vector3
from .registry import ObjectRegistry, SemanticObject


def create_scene(
    scene_id: str = "semantic_scene",
    uav_id: str = "uav_0",
    start: Vector3 = (1.0, 0.0, 1.0),
    goal: Vector3 = (9.0, 0.0, 1.0),
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
