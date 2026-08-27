import os
import sys
import unittest
from unittest import mock


PROJECT_ROOT = os.environ.get("AEROVERBS_PROJECT_ROOT", "/root/AeroVerbs")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import train_dagger


class TrainDaggerDatasetApiTests(unittest.TestCase):
    def test_dataset_builder_uses_current_two_argument_trajectory_dataset_api(self):
        data_cfg = object()
        calls = []

        def fake_dataset(samples, cfg):
            calls.append((samples, cfg))
            return tuple(samples)

        with mock.patch.object(train_dagger, "TrajectoryDataset", side_effect=fake_dataset):
            datasets = train_dagger._make_trajectory_datasets(
                train_samples=["train"],
                val_samples=["val"],
                test_samples=["test"],
                data_cfg=data_cfg,
            )

        self.assertEqual(datasets, (("train",), ("val",), ("test",)))
        self.assertEqual(
            calls,
            [(["train"], data_cfg), (["val"], data_cfg), (["test"], data_cfg)],
        )


if __name__ == "__main__":
    unittest.main()
