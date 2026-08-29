import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "experiments" / "diffusion_controller_evaluator.py"
SPEC = importlib.util.spec_from_file_location("diffusion_controller_evaluator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def episode(initial, maximum, final, success=False, steps=300):
    return {
        "initial_coverage": initial,
        "max_coverage": maximum,
        "final_coverage": final,
        "success": success,
        "logged_step_count": steps,
        "steps": steps,
        "invalid_action_count": 0,
        "non_finite_value_count": 0,
    }


class DiffusionControllerEvaluatorTests(unittest.TestCase):
    def test_parse_seed_spec(self):
        self.assertEqual(MODULE.parse_seed_spec("1000,1002-1004"), [1000, 1002, 1003, 1004])
        with self.assertRaisesRegex(ValueError, "descending"):
            MODULE.parse_seed_spec("3-1")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            MODULE.parse_seed_spec("1000,1000")

    def test_aggregate_separates_maximum_gate_from_final_diagnostic(self):
        summary = MODULE.aggregate(
            [
                episode(0.0, 0.42, 0.10),
                episode(0.0, 0.61, 0.50),
                episode(0.0, 0.49, 0.20),
            ]
        )

        self.assertAlmostEqual(summary["median_max_coverage"], 0.49)
        self.assertAlmostEqual(summary["mean_final_coverage"], (0.10 + 0.50 + 0.20) / 3)
        self.assertTrue(summary["non_floor_task_performance_variation"])
        self.assertFalse(summary["strong_entry_gate_pass"])
        self.assertEqual(summary["metric_roles"]["gate_metric"], "maximum coverage")
        self.assertEqual(summary["metric_roles"]["endpoint_diagnostic"], "final coverage")

    def test_one_success_passes_strong_entry_gate(self):
        summary = MODULE.aggregate(
            [
                episode(0.0, 0.96, 0.20, success=True),
                episode(0.0, 0.10, 0.10),
                episode(0.0, 0.20, 0.20),
            ]
        )
        self.assertTrue(summary["strong_entry_gate_pass"])

    def test_constant_floor_is_not_non_floor_variation(self):
        summary = MODULE.aggregate(
            [episode(0.0, 0.0, 0.0), episode(0.0, 0.0, 0.0), episode(0.0, 0.0, 0.0)]
        )
        self.assertFalse(summary["improved_above_initial"])
        self.assertFalse(summary["non_floor_task_performance_variation"])

    def test_load_policy_uses_registry_base_then_requires_diffusion_config(self):
        decoded_config = object.__new__(MODULE.DiffusionConfig)
        loaded_policy = mock.Mock()
        with (
            mock.patch.object(
                MODULE.PreTrainedConfig,
                "from_pretrained",
                return_value=decoded_config,
            ) as config_loader,
            mock.patch.object(
                MODULE.DiffusionPolicy,
                "from_pretrained",
                return_value=loaded_policy,
            ) as policy_loader,
        ):
            result = MODULE.load_policy(Path("checkpoint"), Path("compat"), "mps")

        self.assertIs(result, loaded_policy)
        config_loader.assert_called_once_with(Path("compat"), local_files_only=True)
        policy_loader.assert_called_once_with(
            Path("checkpoint"),
            config=decoded_config,
            local_files_only=True,
            map_location="cpu",
            strict=True,
        )
        loaded_policy.to.assert_called_once_with("mps")
        loaded_policy.eval.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
