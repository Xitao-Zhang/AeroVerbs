from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .coordinates import RegionFormatError, parse_region_text, sample_metrics


def evaluate_prediction(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "skill": row["skill"],
        "prediction": prediction,
        "target": row["regions"],
        "valid": False,
    }
    try:
        parsed = parse_region_text(
            prediction,
            skill=row["skill"],
            width=int(row["width"]),
            height=int(row["height"]),
        )
        result.update(sample_metrics(
            parsed,
            row["regions"],
            skill=row["skill"],
            width=int(row["width"]),
            height=int(row["height"]),
        ))
        result["parsed"] = parsed
        result["valid"] = True
    except RegionFormatError as exc:
        result["error"] = str(exc)
    return result


def aggregate_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = records
    for record in records:
        groups[record["skill"]].append(record)
    output = {}
    for name, items in groups.items():
        valid = [item for item in items if item.get("valid")]
        output[name] = {
            "count": len(items),
            "valid_count": len(valid),
            "valid_rate": len(valid) / len(items) if items else 0.0,
            "mean_iou": sum(x["iou"] for x in valid) / len(valid) if valid else 0.0,
            "mean_center_error_px": sum(x["center_error_px"] for x in valid) / len(valid) if valid else None,
            "mean_vertex_error_px": sum(x["vertex_error_px"] for x in valid) / len(valid) if valid else None,
        }
    return output
