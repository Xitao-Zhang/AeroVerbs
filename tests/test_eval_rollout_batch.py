import json
import os
import sys
import tempfile
import unittest


PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import eval_rollout_batch


class EvalRolloutBatchTests(unittest.TestCase):
    def _manifest(self):
        return {
            "schema_version": 1,
            "batch_id": 1,
            "data_root": "/dataset",
            "num_samples": 3,
            "samples": [
                {"skill": "around_left", "scene_id": "scene_a", "relative_path": "around_left/scene_a"},
                {"skill": "around_left", "scene_id": "scene_b", "relative_path": "around_left/scene_b"},
                {"skill": "through", "scene_id": "scene_c", "relative_path": "through/scene_c"},
            ],
        }

    def test_load_manifest_preserves_order_and_rejects_duplicate_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "batch.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._manifest(), f)

            loaded = eval_rollout_batch.load_manifest(path)

            self.assertEqual([s["scene_id"] for s in loaded["samples"]], ["scene_a", "scene_b", "scene_c"])
            duplicate = self._manifest()
            duplicate["samples"][2] = dict(duplicate["samples"][0])
            with open(path, "w", encoding="utf-8") as f:
                json.dump(duplicate, f)
            with self.assertRaisesRegex(ValueError, "duplicate manifest sample"):
                eval_rollout_batch.load_manifest(path)

    def test_load_latest_results_uses_last_record_and_ignores_only_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"skill": "around_left", "scene_id": "scene_a", "status": "infrastructure_error"}) + "\n")
                f.write(json.dumps({"skill": "around_left", "scene_id": "scene_a", "status": "success"}) + "\n")
                f.write('{"skill":"through"')

            latest = eval_rollout_batch.load_latest_results(path)

            self.assertEqual(latest[("around_left", "scene_a")]["status"], "success")
            self.assertEqual(len(latest), 1)

    def test_repair_truncated_results_removes_partial_tail_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.jsonl")
            complete = {"skill": "over", "scene_id": "scene_x", "status": "collision"}
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(complete) + "\n")
                f.write('{"skill":"under"')

            repaired = eval_rollout_batch.repair_truncated_jsonl(path)

            self.assertTrue(repaired)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read(), json.dumps(complete) + "\n")

    def test_pending_samples_supports_optional_infrastructure_retry(self):
        manifest = self._manifest()
        latest = {
            ("around_left", "scene_a"): {"status": "success"},
            ("around_left", "scene_b"): {"status": "infrastructure_error"},
        }

        default_pending = eval_rollout_batch.pending_samples(manifest["samples"], latest)
        retry_pending = eval_rollout_batch.pending_samples(
            manifest["samples"], latest, retry_infrastructure_errors=True
        )

        self.assertEqual([s["scene_id"] for s in default_pending], ["scene_c"])
        self.assertEqual([s["scene_id"] for s in retry_pending], ["scene_b", "scene_c"])

    def test_summary_uses_success_status_and_manifest_denominators(self):
        manifest = self._manifest()
        latest = {
            ("around_left", "scene_a"): {"skill": "around_left", "scene_id": "scene_a", "status": "success", "elapsed_s": 2.0},
            ("around_left", "scene_b"): {"skill": "around_left", "scene_id": "scene_b", "status": "collision", "elapsed_s": 3.0},
        }

        summary = eval_rollout_batch.summarize_results(manifest, latest)

        self.assertEqual(summary["expected_samples"], 3)
        self.assertEqual(summary["processed_samples"], 2)
        self.assertEqual(summary["missing_samples"], 1)
        self.assertEqual(summary["status_counts"], {"collision": 1, "success": 1})
        self.assertAlmostEqual(summary["success_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["success_rate_processed"], 0.5)
        self.assertEqual(summary["per_skill"]["around_left"]["success"], 1)
        self.assertAlmostEqual(summary["per_skill"]["around_left"]["success_rate"], 0.5)
        self.assertEqual(summary["per_skill"]["through"]["expected"], 1)
        self.assertEqual(summary["per_skill"]["through"]["success_rate"], 0.0)
        self.assertEqual(summary["rollout_elapsed_s"], 5.0)

    def test_rollout_namespace_always_disables_rendering(self):
        args = eval_rollout_batch.build_rollout_namespace(
            data_root="/dataset", output_dir="/exp/batch_1/bc"
        )

        self.assertTrue(args.no_render)
        self.assertEqual(args.drone_radius, 0.15)
        self.assertEqual(args.max_steps, 200)
        self.assertEqual(args.output_dir, "/exp/batch_1/bc")


if __name__ == "__main__":
    unittest.main()
