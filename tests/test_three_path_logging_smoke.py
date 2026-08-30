import importlib.util
import sys
import unittest
from copy import deepcopy
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "experiments" / "three_path_logging_smoke.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("three_path_logging_smoke", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_records():
    shared = {
        "status": "succeeded",
        "debug_only": True,
        "environment_seed": 1000,
        "policy_seed": 101000,
        "configured_damping": 0.0,
        "max_steps": 300,
        "initial_state_sha256": "initial",
        "prefix_action_sha256": "prefix",
        "pre_probe_state_sha256": "pre-probe",
        "logged_step_count": 300,
        "steps": 300,
        "invalid_action_count": 0,
        "non_finite_value_count": 0,
    }
    return [
        {
            **shared,
            "path": "fixed",
            "probe_action_sha256": None,
            "post_probe_state_sha256": None,
            "response_used_count": 0,
            "adjusted_scale_count": 0,
            "probe_measurement_sha256": None,
        },
        {
            **shared,
            "path": "probe-no-adjust",
            "probe_action_sha256": "probe",
            "post_probe_state_sha256": "post-probe",
            "response_used_count": 0,
            "adjusted_scale_count": 0,
            "probe_measurement_sha256": "measurement",
        },
        {
            **shared,
            "path": "probe-adjust",
            "probe_action_sha256": "probe",
            "post_probe_state_sha256": "post-probe",
            "response_used_count": 275,
            "adjusted_scale_count": 275,
            "probe_measurement_sha256": "measurement",
        },
    ]


class ThreePathLoggingSmokeTests(unittest.TestCase):
    def test_step_phase_labels_shared_budget(self):
        self.assertEqual(MODULE.step_phase("fixed", 20, 20, 5), "prefix")
        self.assertEqual(MODULE.step_phase("fixed", 21, 20, 5), "continuation_window")
        self.assertEqual(MODULE.step_phase("fixed", 26, 20, 5), "post_window")
        self.assertEqual(MODULE.step_phase("probe-adjust", 20, 20, 5), "prefix")
        self.assertEqual(MODULE.step_phase("probe-adjust", 21, 20, 5), "probe")
        self.assertEqual(MODULE.step_phase("probe-adjust", 25, 20, 5), "probe")
        self.assertEqual(MODULE.step_phase("probe-adjust", 26, 20, 5), "post_probe")

    def test_pairing_validator_passes_complete_debug_records(self):
        result = MODULE.validate_three_path_records(valid_records())
        self.assertTrue(result["passed"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["checks"]["shared_probe"])
        self.assertTrue(result["checks"]["shared_post_probe_state"])
        self.assertTrue(result["checks"]["shared_measurement"])
        self.assertTrue(result["checks"]["response_isolation"])
        self.assertTrue(result["checks"]["adjustment_branch_exercised"])

    def test_pairing_validator_rejects_probe_hash_mismatch(self):
        records = valid_records()
        records[2]["probe_action_sha256"] = "different-probe"
        result = MODULE.validate_three_path_records(records)
        self.assertFalse(result["passed"])
        self.assertIn("probe action hashes differ", result["errors"])

    def test_pairing_validator_rejects_hidden_non_debug_record(self):
        records = valid_records()
        records[1]["debug_only"] = False
        result = MODULE.validate_three_path_records(records)
        self.assertFalse(result["passed"])
        self.assertIn("every record must be debug_only", result["errors"])

    def test_pairing_validator_rejects_response_use_in_no_adjust(self):
        records = valid_records()
        records[1]["response_used_count"] = 1
        result = MODULE.validate_three_path_records(records)
        self.assertFalse(result["passed"])
        self.assertIn("probe-no-adjust consumed response", result["errors"])

    def test_pairing_validator_rejects_unexercised_adjustment_branch(self):
        records = valid_records()
        records[2]["adjusted_scale_count"] = 0
        result = MODULE.validate_three_path_records(records)
        self.assertFalse(result["passed"])
        self.assertIn("probe-adjust never executed adjusted scale", result["errors"])

    def test_pairing_validator_rejects_incomplete_or_non_finite_run(self):
        for field, value, error in (
            ("logged_step_count", 299, "step logs are incomplete"),
            ("invalid_action_count", 1, "an invalid action was recorded"),
            ("non_finite_value_count", 1, "a non-finite value was recorded"),
        ):
            with self.subTest(field=field):
                records = deepcopy(valid_records())
                records[0][field] = value
                result = MODULE.validate_three_path_records(records)
                self.assertFalse(result["passed"])
                self.assertIn(error, result["errors"])


if __name__ == "__main__":
    unittest.main()
