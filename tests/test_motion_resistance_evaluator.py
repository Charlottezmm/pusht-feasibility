import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "experiments" / "motion_resistance_evaluator.py"
SPEC = importlib.util.spec_from_file_location("motion_resistance_evaluator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MotionResistanceEvaluatorTests(unittest.TestCase):
    def test_parse_seed_spec(self):
        self.assertEqual(MODULE.parse_seed_spec("0,3,5-7"), [0, 3, 5, 6, 7])
        with self.assertRaisesRegex(ValueError, "descending"):
            MODULE.parse_seed_spec("3-1")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            MODULE.parse_seed_spec("0,0")

    def test_action_sequence_is_deterministic_and_seeded(self):
        actions_a, seed_a, hash_a = MODULE.generate_action_sequence(3, 12)
        actions_b, seed_b, hash_b = MODULE.generate_action_sequence(3, 12)
        actions_c, _, hash_c = MODULE.generate_action_sequence(4, 12)

        self.assertEqual(seed_a, 100_003)
        self.assertEqual(seed_a, seed_b)
        self.assertEqual(hash_a, hash_b)
        self.assertNotEqual(hash_a, hash_c)
        np.testing.assert_array_equal(actions_a, actions_b)
        self.assertTrue(all(action.dtype == np.float32 for action in actions_a))
        self.assertEqual(len(actions_c), 12)

    def test_aggregate_preserves_declared_primary_and_paired_diagnostics(self):
        episodes = []
        trajectories = {}
        final_coverages = {
            "low_resistance": 0.1,
            "medium_resistance": 0.2,
            "high_resistance": 0.3,
        }
        for level_name, damping in MODULE.DAMPING_LEVELS:
            episodes.append(
                {
                    "level": level_name,
                    "damping": damping,
                    "initial_coverage": 0.0,
                    "final_coverage": final_coverages[level_name],
                    "coverage_change": final_coverages[level_name],
                    "max_coverage": final_coverages[level_name],
                    "episode_return": final_coverages[level_name],
                    "contact_steps": 1,
                    "observation_space_violation_count": 0,
                    "success": False,
                }
            )
        trajectories[(0, "low_resistance")] = [[0.0, 0.0], [1.0, 0.0]]
        trajectories[(0, "medium_resistance")] = [[0.0, 0.0], [0.5, 0.0]]
        trajectories[(0, "high_resistance")] = [[0.0, 0.0], [0.0, 0.0]]

        summary = MODULE.aggregate(episodes, [0], trajectories)

        self.assertEqual(summary["primary_metric"], "mean final_coverage")
        self.assertAlmostEqual(
            summary["by_level"]["high_resistance"]["mean_final_coverage"],
            0.3,
        )
        self.assertAlmostEqual(
            summary["motion_response_by_level"]["medium_resistance"][
                "mean_final_block_position_distance"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            summary["motion_response_by_level"]["high_resistance"][
                "mean_final_block_position_distance"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
