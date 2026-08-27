"""Environment interfaces for closed-loop rollout.

The radar renderer here is self-contained (trimesh raycast, no Habitat) so
rollout can run headless on any machine with ``trimesh`` installed.
"""

from .radar import load_scene_mesh, render_scene_glb, sample_lidar_raycast

__all__ = ["load_scene_mesh", "render_scene_glb", "sample_lidar_raycast"]
