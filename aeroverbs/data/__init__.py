from .labels import (
    skill_to_id,
    body_frame_basis,
    compute_waypoint_labels,
    compute_completion_labels,
)
from .heatmap import bbox_to_heatmap

__all__ = [
    "ManeuverDataset",
    "TrajectoryDataset",
    "build_sample_index",
    "split_samples",
    "preprocess_radar",
    "resize_heatmap",
    "skill_to_id",
    "body_frame_basis",
    "compute_waypoint_labels",
    "compute_completion_labels",
    "bbox_to_heatmap",
]


def __getattr__(name):
    if name in {"ManeuverDataset", "TrajectoryDataset", "build_sample_index", "split_samples"}:
        from . import dataset

        return getattr(dataset, name)
    if name in {"preprocess_radar", "resize_heatmap"}:
        from . import transforms

        return getattr(transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
