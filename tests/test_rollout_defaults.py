import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import rollout


class RolloutDefaultsTest(unittest.TestCase):
    def test_rollout_defaults_to_physical_drone_radius(self):
        with patch.object(sys, "argv", ["rollout.py", "--checkpoint", "model.pt"]):
            args = rollout.parse_args()

        self.assertEqual(args.drone_radius, 0.15)


if __name__ == "__main__":
    unittest.main()
