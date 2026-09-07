from .labels import (
    skill_to_id,
    body_frame_basis,
    compute_waypoint_labels,
    compute_completion_labels,
)
__all__ = [
    "ManeuverDataset",
    "TrajectoryDataset",
    "build_sample_index",
    "split_samples",
    "preprocess_radar",
    "skill_to_id",
    "body_frame_basis",
    "compute_waypoint_labels",
    "compute_completion_labels",
]


def __getattr__(name):
    if name in {"ManeuverDataset", "TrajectoryDataset", "build_sample_index", "split_samples"}:
        from . import dataset

        return getattr(dataset, name)
    if name == "preprocess_radar":
        from . import transforms

        return getattr(transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
