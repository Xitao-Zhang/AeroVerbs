import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_PATH = os.environ.get(
    "DAGGER_ITERATE_SCRIPT", "/root/AeroVerbs/scripts/dagger_iterate.py"
)
SPEC = importlib.util.spec_from_file_location("dagger_iterate_under_test", SCRIPT_PATH)
dagger_iterate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dagger_iterate)


class DaggerIterateFastPlannerTests(unittest.TestCase):
    def _parse(self, extra=None):
        argv = [
            "dagger_iterate.py",
            "--start-checkpoint",
            "student.pt",
            "--output-root",
            "dagger_out",
        ] + list(extra or [])
        with mock.patch.object(sys, "argv", argv):
            return dagger_iterate.parse_args()

    def test_fast_planner_defaults_match_collector(self):
        args = self._parse()

        self.assertAlmostEqual(args.drone_radius, 0.15)
        self.assertIsNone(args.teacher_preferred_clearance)
        self.assertEqual(args.sample_seed, 42)
        self.assertEqual(args.teacher_replan_interval, 5)
        self.assertAlmostEqual(args.teacher_timeout, 2.0)
        self.assertEqual(args.teacher_max_iter, 2000)
        self.assertIsNone(args.lambda_completion)
        self.assertIsNone(args.lambda_clearance)
        self.assertIsNone(args.lambda_temporal_smooth)
        self.assertEqual(args.start_round, 1)
        self.assertIsNone(args.seed_dagger_root)

    def test_collect_command_forwards_new_options_and_drops_legacy_teacher_options(self):
        args = self._parse(
            [
                "--push-distance",
                "0.7",
                "--priority-results",
                "old_results.jsonl",
                "--teacher-timeout",
                "3.5",
            ]
        )

        command = dagger_iterate._build_collect_cmd(
            args,
            checkpoint="student.pt",
            sample_root="round1_samples",
        )

        joined = " ".join(command)
        self.assertIn("--drone-radius 0.15", joined)
        self.assertIn("--sample-seed 42", joined)
        self.assertNotIn("--teacher-preferred-clearance", command)
        self.assertIn("--teacher-replan-interval 5", joined)
        self.assertIn("--teacher-timeout 3.5", joined)
        self.assertIn("--teacher-max-iter 2000", joined)
        for legacy in (
            "--danger-threshold",
            "--push-distance",
            "--push-waypoint-count",
            "--push-blend-width",
            "--priority-results",
        ):
            self.assertNotIn(legacy, command)

        override_args = self._parse(["--teacher-preferred-clearance", "0.4"])
        override_command = dagger_iterate._build_collect_cmd(
            override_args,
            checkpoint="student.pt",
            sample_root="round1_samples",
        )
        override_index = override_command.index("--teacher-preferred-clearance")
        self.assertEqual(override_command[override_index + 1], "0.4")

    def test_train_command_only_forwards_explicit_loss_overrides(self):
        args = self._parse()
        command = dagger_iterate._build_train_cmd(
            args,
            checkpoint="student.pt",
            expert_data_root="expert_data",
            all_dagger_root="all_dagger",
            train_experiment="round1",
            train_name="train1",
        )

        self.assertNotIn("--lambda-completion", command)
        self.assertNotIn("--lambda-clearance", command)
        self.assertNotIn("--lambda-temporal-smooth", command)

        explicit = self._parse(["--lambda-clearance", "0.5"])
        explicit_command = dagger_iterate._build_train_cmd(
            explicit,
            checkpoint="student.pt",
            expert_data_root="expert_data",
            all_dagger_root="all_dagger",
            train_experiment="round1",
            train_name="train1",
        )
        index = explicit_command.index("--lambda-clearance")
        self.assertEqual(explicit_command[index + 1], "0.5")

    def test_round_training_path_uses_bc_style_timestamp_directory(self):
        experiment, run_name, checkpoint = dagger_iterate._round_training_paths(
            "dagger_out", round_idx=1, timestamp="20260826-123456"
        )

        self.assertEqual(experiment, "round1")
        self.assertEqual(run_name, "20260826-123456")
        self.assertEqual(
            checkpoint,
            os.path.join(
                "dagger_out", "round1", "20260826-123456", "final.pt"
            ),
        )

    def test_seed_history_is_linked_once_before_a_continuation_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_root = os.path.join(tmp, "round1_relabel")
            sample = os.path.join(seed_root, "around_left", "scene_001")
            os.makedirs(sample)
            all_root = os.path.join(tmp, "all_dagger")

            first = dagger_iterate._seed_all_dagger(
                seed_root, all_root, ["around_left"]
            )
            second = dagger_iterate._seed_all_dagger(
                seed_root, all_root, ["around_left"]
            )

            linked = os.path.join(all_root, "around_left", "scene_001")
            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertTrue(os.path.islink(linked))
            self.assertEqual(os.path.realpath(linked), os.path.realpath(sample))

    def test_continuation_round_numbers_start_at_requested_round(self):
        args = self._parse(["--start-round", "2", "--num-rounds", "1"])

        self.assertEqual(list(dagger_iterate._round_indices(args)), [2])


if __name__ == "__main__":
    unittest.main()
