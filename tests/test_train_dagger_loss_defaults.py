import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


SERVER_PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SCRIPT_PATH = os.path.join(SERVER_PROJECT_ROOT, "scripts", "train_dagger.py")
SPEC = importlib.util.spec_from_file_location("train_dagger_under_test", SCRIPT_PATH)
train_dagger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_dagger)


class TrainDaggerLossDefaultsTests(unittest.TestCase):
    def _parse(self, extra=None):
        argv = [
            "train_dagger.py",
            "--checkpoint",
            "student.pt",
            "--dagger-data-root",
            "dagger_data",
        ] + list(extra or [])
        with mock.patch.object(sys, "argv", argv):
            return train_dagger.parse_args()

    def test_omitted_loss_options_preserve_checkpoint_configuration(self):
        args = self._parse()
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                lambda_completion=0.05,
                lambda_clearance=1.0,
                lambda_temporal_smooth=0.0,
                lambda_action_consistency=0.1,
                completion_pos_weight=3.0,
            ),
            data=SimpleNamespace(use_sdf=True),
        )

        train_dagger._apply_loss_overrides(cfg, args)

        self.assertAlmostEqual(cfg.train.lambda_completion, 0.05)
        self.assertAlmostEqual(cfg.train.lambda_clearance, 1.0)
        self.assertAlmostEqual(cfg.train.lambda_temporal_smooth, 0.0)
        self.assertAlmostEqual(cfg.train.lambda_action_consistency, 0.0)
        self.assertTrue(cfg.data.use_sdf)

    def test_explicit_loss_options_override_checkpoint_configuration(self):
        args = self._parse(
            [
                "--lambda-completion",
                "0.1",
                "--lambda-clearance",
                "0.5",
                "--lambda-temporal-smooth",
                "0.2",
            ]
        )
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                lambda_completion=0.05,
                lambda_clearance=1.0,
                lambda_temporal_smooth=0.0,
                lambda_action_consistency=0.1,
                completion_pos_weight=3.0,
            ),
            data=SimpleNamespace(use_sdf=True),
        )

        train_dagger._apply_loss_overrides(cfg, args)

        self.assertAlmostEqual(cfg.train.lambda_completion, 0.1)
        self.assertAlmostEqual(cfg.train.lambda_clearance, 0.5)
        self.assertAlmostEqual(cfg.train.lambda_temporal_smooth, 0.2)
        self.assertAlmostEqual(cfg.train.lambda_action_consistency, 0.0)


if __name__ == "__main__":
    unittest.main()
