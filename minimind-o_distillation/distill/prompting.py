from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=4)
def _load_annotation_module(script_path: str) -> ModuleType:
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("aeroverbs_teacher_prompt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load annotation script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_prompt"):
        raise AttributeError(f"{path} does not define build_prompt")
    return module


def build_teacher_prompt(
    annotation_script: str | Path,
    *,
    width: int,
    height: int,
    skill: str,
    target_object: str,
) -> str:
    """Recreate the exact semantic prompt used to produce the labels."""
    module = _load_annotation_module(str(Path(annotation_script).expanduser().resolve()))
    return module.build_prompt(
        int(width),
        int(height),
        task=str(skill),
        target_object=str(target_object),
        target_marking="none",
    )


def build_student_prompt(*, width: int, height: int, skill: str, target_object: str) -> str:
    """Compact deployment-style prompt used to train and run MiniMind-O."""
    side = {"around_left": "LEFT", "around_right": "RIGHT"}.get(skill)
    coordinates = (
        f"Image size: {width}x{height}. The origin is the top-left; x increases rightward and y downward. "
        "Consider perspective and do not unnecessarily shrink distant free space.\n\n"
    )
    ordering = (
        "Return the four corners in this order: top-left, top-right, bottom-right, bottom-left. Output only:\n"
    )
    if side:
        key = side.lower()
        return (
            f"A small UAV must fly around the {target_object} on its {side} side. Locate the specified object "
            f"and annotate the clear flyable region immediately to its {key}.\n\n"
            + coordinates
            + "The rectangle must not overlap the target or other obstacles. Its height should cover the target's "
            "full visible height. Its width should be approximately 1/6 to 1/4 of the image width, trimmed only "
            "when required for safety.\n\n"
            + ordering
            + f'{{"{key}":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}}'
        )
    if skill == "through":
        return (
            f"A small UAV must fly through the {target_object}. Locate the specified doorway, window, gate, ring, "
            "or other opening and annotate the clear flyable region inside it.\n\n"
            + coordinates
            + "Cover the opening's full usable width and height, including its bottom edge. Keep the rectangle "
            "inside the opening and exclude its frame, walls, ceiling, and other obstacles.\n\n"
            + ordering
            + '{"passage":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'
        )
    if skill == "over":
        return (
            f"A small UAV must fly OVER the {target_object}. Locate the specified object and annotate the clear "
            "flyable airspace immediately above it.\n\n"
            + coordinates
            + "The rectangle's lower edge should start just above the object's highest point. Its width should be "
            "about 1.5 to 2 times the object's visible width and centered above it. Extend upward through the "
            "available clear air, while avoiding all obstacles.\n\n"
            + ordering
            + '{"over":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'
        )
    if skill == "under":
        return (
            f"A small UAV must fly UNDER the {target_object}. Locate the specified object and annotate the clear "
            "flyable opening between its underside and the ground.\n\n"
            + coordinates
            + "Cover the opening's full usable height, from the ground up to the object's underside, and its full "
            "clear width. Keep the rectangle inside the opening and exclude legs, supports, and other obstacles.\n\n"
            + ordering
            + '{"under":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}'
        )
    raise ValueError(f"Unsupported skill: {skill}")


def build_user_content(prompt: str) -> str:
    return "<image>\n" + prompt.strip()
