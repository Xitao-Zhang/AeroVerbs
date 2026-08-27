import json
import os
import subprocess
import sys
import tempfile
import unittest


SERVER_PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SERVER_SCRIPTS = os.path.join(SERVER_PROJECT_ROOT, "scripts")
if SERVER_SCRIPTS not in sys.path:
    sys.path.insert(0, SERVER_SCRIPTS)

import build_eval_manifests


class BuildEvalManifestsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _sample(self, skill, scene_id, category):
        path = os.path.join(self.tempdir.name, skill, scene_id)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "frames.json"), "w", encoding="utf-8") as f:
            json.dump([], f)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "skill": skill,
                    "scene_id": scene_id,
                    "objects": [
                        {
                            "id": f"{category}_0",
                            "type": "obstacle",
                            "asset_path": f"/assets/{category}.glb",
                        }
                    ],
                },
                f,
            )
        return path

    def _test_samples(self):
        return [
            self._sample("around_left", f"left_{category}_{i}", category)
            for category in ("bike", "plant")
            for i in range(2)
        ] + [
            self._sample("through", f"through_{category}_{i}", category)
            for category in ("door", "ring")
            for i in range(1)
        ]

    def test_make_batches_is_disjoint_complete_and_has_requested_counts(self):
        samples = self._test_samples()
        counts = {"around_left": [2, 2], "through": [1, 1]}

        batches = build_eval_manifests.make_batches(samples, counts, sample_seed=42)

        self.assertEqual([len(batch) for batch in batches], [3, 3])
        self.assertEqual(
            [[item["skill"] for item in batch] for batch in batches],
            [
                ["around_left", "around_left", "through"],
                ["around_left", "around_left", "through"],
            ],
        )
        all_ids = [(item["skill"], item["scene_id"]) for batch in batches for item in batch]
        expected_ids = {
            (os.path.basename(os.path.dirname(path)), os.path.basename(path))
            for path in samples
        }
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), expected_ids)

    def test_fixed_seed_is_reproducible(self):
        samples = self._test_samples()
        counts = {"around_left": [2, 2], "through": [1, 1]}

        first = build_eval_manifests.make_batches(samples, counts, sample_seed=42)
        repeated = build_eval_manifests.make_batches(samples, counts, sample_seed=42)

        self.assertEqual(first, repeated)

    def test_rare_category_is_spread_across_batches(self):
        samples = [
            self._sample("around_left", f"rare_{i}", "rare") for i in range(2)
        ] + [
            self._sample("around_left", f"common_{i}", "common") for i in range(8)
        ]

        batches = build_eval_manifests.make_batches(
            samples,
            {"around_left": [5, 5]},
            sample_seed=42,
        )

        rare_counts = [
            sum(item["category"] == "rare" for item in batch) for batch in batches
        ]
        self.assertEqual(rare_counts, [1, 1])

    def test_write_manifests_produces_readable_index_and_batches(self):
        samples = self._test_samples()
        counts = {"around_left": [2, 2], "through": [1, 1]}
        batches = build_eval_manifests.make_batches(samples, counts, sample_seed=42)
        output_dir = os.path.join(self.tempdir.name, "manifests")

        index_path = build_eval_manifests.write_manifests(
            batches,
            output_dir,
            data_root="/dataset",
            split_seed=0,
            sample_seed=42,
        )

        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual(index["total_samples"], 6)
        self.assertEqual(index["batch_sizes"], [3, 3])
        self.assertEqual(len(index["manifests"]), 2)
        for entry in index["manifests"]:
            with open(os.path.join(output_dir, entry["file"]), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["num_samples"], 3)
            self.assertEqual(len(manifest["samples"]), 3)

    def test_make_batches_rejects_count_mismatch(self):
        samples = self._test_samples()
        counts = {"around_left": [1, 1], "through": [1, 1]}

        with self.assertRaisesRegex(ValueError, "around_left"):
            build_eval_manifests.make_batches(samples, counts, sample_seed=42)

    def test_module_can_be_imported_from_project_root(self):
        result = subprocess.run(
            [sys.executable, "-c", "import scripts.build_eval_manifests"],
            cwd=SERVER_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
