import json
import os
import sys
import tempfile
import unittest


SERVER_PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SERVER_SCRIPTS = os.path.join(SERVER_PROJECT_ROOT, "scripts")
if SERVER_SCRIPTS not in sys.path:
    sys.path.insert(0, SERVER_SCRIPTS)

import dagger_collect


class DaggerCollectSamplingTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _sample(self, name, *, asset=None, object_type=None):
        path = os.path.join(self.tempdir.name, "around_left", name)
        os.makedirs(path, exist_ok=True)
        obj = {"id": "object_0", "type": object_type}
        if asset is not None:
            obj["asset_path"] = f"/assets/{asset}.glb"
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "skill": "around_left",
                    "scene_id": name,
                    "object_id": f"{object_type or 'object'}_0",
                    "objects": [obj],
                },
                f,
            )
        return path

    def test_round_robin_covers_categories_before_repeating_one(self):
        samples = [
            self._sample("bike_0", asset="bike"),
            self._sample("bike_1", asset="bike"),
            self._sample("plant_0", asset="plant"),
            self._sample("plant_1", asset="plant"),
            self._sample("lamp_0", asset="lamp"),
            self._sample("lamp_1", asset="lamp"),
        ]

        ordered = dagger_collect._stratified_sample_order(samples, seed=42)
        first_categories = [dagger_collect._sample_category(path) for path in ordered[:3]]

        self.assertEqual(set(first_categories), {"bike", "plant", "lamp"})

    def test_fixed_seed_is_reproducible_and_another_seed_changes_order(self):
        samples = [
            self._sample(f"bike_{i}", asset="bike") for i in range(4)
        ] + [
            self._sample(f"plant_{i}", asset="plant") for i in range(4)
        ] + [
            self._sample(f"lamp_{i}", asset="lamp") for i in range(4)
        ]

        first = dagger_collect._stratified_sample_order(samples, seed=42)
        repeated = dagger_collect._stratified_sample_order(samples, seed=42)
        different = dagger_collect._stratified_sample_order(samples, seed=43)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_generated_opening_type_is_used_when_asset_path_is_missing(self):
        samples = [
            self._sample("door_0", object_type="door"),
            self._sample("window_0", object_type="window"),
            self._sample("ring_0", object_type="ring"),
        ]

        categories = {dagger_collect._sample_category(path) for path in samples}

        self.assertEqual(categories, {"door", "window", "ring"})

    def test_generic_obstacle_type_uses_procedural_scene_category(self):
        sample = self._sample(
            "auto_around_left_column_00116", object_type="obstacle"
        )

        self.assertEqual(dagger_collect._sample_category(sample), "column")


if __name__ == "__main__":
    unittest.main()
