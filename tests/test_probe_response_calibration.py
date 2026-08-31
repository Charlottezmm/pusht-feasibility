import importlib.util
import math
import sys
import unittest
from copy import deepcopy
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "experiments" / "probe_response_calibration.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("probe_response_calibration", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(seed, damping, response, valid=True, invalid_reasons=()):
    return {
        "status": "succeeded",
        "environment_seed": seed,
        "policy_seed": MODULE.POLICY_SEED_OFFSET + seed,
        "configured_damping": damping,
        "initial_state_sha256": f"initial-{seed}-{damping}",
        "pre_probe_state_sha256": f"pre-{seed}-{damping}",
        "post_probe_state_sha256": f"post-{seed}-{damping}",
        "probe_action_sha256": f"probe-{seed}-{damping}",
        "probe_measurement": {
            "signed_response": response,
            "path_length": abs(response),
            "contact_steps": 1 if valid else 0,
            "valid": valid,
            "invalid_reasons": list(invalid_reasons),
            "measurement_source": "simulator_info_block_pose",
        },
    }


def complete_records():
    records = []
    for damping in MODULE.CALIBRATION_DAMPINGS:
        for seed in MODULE.CALIBRATION_SEEDS:
            response = float(seed - 95) + damping
            records.append(record(seed, damping, response))
    return records


class ProbeResponseCalibrationTests(unittest.TestCase):
    def test_manifest_is_exact_balanced_and_disjoint_from_evaluation(self):
        manifest = MODULE.build_calibration_manifest()
        identities = {
            (item["environment_seed"], item["configured_damping"])
            for item in manifest
        }
        expected = {
            (seed, damping)
            for damping in MODULE.CALIBRATION_DAMPINGS
            for seed in MODULE.CALIBRATION_SEEDS
        }
        self.assertEqual(identities, expected)
        self.assertEqual(len(manifest), 20)
        self.assertTrue(
            {item["environment_seed"] for item in manifest}.isdisjoint(
                MODULE.EVALUATION_SEEDS
            )
        )
        MODULE.validate_calibration_manifest(manifest)

    def test_manifest_rejects_evaluation_seed_or_missing_attempt(self):
        manifest = MODULE.build_calibration_manifest()
        with self.assertRaisesRegex(ValueError, "evaluation seed"):
            invalid = deepcopy(manifest)
            invalid[0]["environment_seed"] = 20
            MODULE.validate_calibration_manifest(invalid)
        with self.assertRaisesRegex(ValueError, "exact planned identities"):
            MODULE.validate_calibration_manifest(manifest[:-1])

    def test_summary_uses_only_valid_values_and_even_count_median(self):
        records = complete_records()
        for index, response in enumerate((4.0, 10.0, -2.0, 18.0)):
            records[index]["probe_measurement"]["signed_response"] = response
        for index in range(4, len(records)):
            records[index]["probe_measurement"]["valid"] = False
            records[index]["probe_measurement"]["invalid_reasons"] = [
                "no_probe_contact"
            ]
        summary = MODULE.summarize_calibration_records(records)
        self.assertEqual(summary["status"], "patch")
        self.assertIsNone(summary["tau"])
        self.assertEqual(summary["valid_count"], 4)
        self.assertEqual(summary["valid_x_sorted"], [-2.0, 4.0, 10.0, 18.0])
        self.assertEqual(summary["candidate_median"], 7.0)

    def test_five_or_more_valid_values_freeze_tau(self):
        records = complete_records()
        valid_values = [-2.0, 4.0, 7.0, 10.0, 18.0]
        for index, item in enumerate(records):
            if index < len(valid_values):
                item["probe_measurement"]["signed_response"] = valid_values[index]
            else:
                item["probe_measurement"]["valid"] = False
                item["probe_measurement"]["invalid_reasons"] = [
                    "no_probe_contact"
                ]
        summary = MODULE.summarize_calibration_records(records)
        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(summary["tau"], 7.0)
        self.assertEqual(summary["candidate_median"], 7.0)
        self.assertEqual(summary["valid_count"], 5)
        self.assertEqual(summary["invalid_count"], 15)

    def test_summary_rejects_duplicate_or_missing_identity(self):
        records = complete_records()
        duplicate = deepcopy(records)
        duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, "exact planned identities"):
            MODULE.summarize_calibration_records(duplicate)
        with self.assertRaisesRegex(ValueError, "exact planned identities"):
            MODULE.summarize_calibration_records(records[:-1])

    def test_summary_rejects_non_finite_valid_response(self):
        records = complete_records()
        records[0]["probe_measurement"]["signed_response"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.summarize_calibration_records(records)

    def test_summary_reports_counts_by_condition(self):
        records = complete_records()
        records[0]["probe_measurement"]["valid"] = False
        records[0]["probe_measurement"]["invalid_reasons"] = [
            "no_probe_contact"
        ]
        summary = MODULE.summarize_calibration_records(records)
        self.assertEqual(
            summary["by_condition"]["0.0"],
            {"attempted": 10, "valid": 9, "invalid": 1},
        )
        self.assertEqual(
            summary["by_condition"]["1.0"],
            {"attempted": 10, "valid": 10, "invalid": 0},
        )
        self.assertEqual(summary["valid_count"], 19)
        self.assertEqual(summary["invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
