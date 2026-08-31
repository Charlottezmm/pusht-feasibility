import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "experiments" / "three_path_evaluator.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("three_path_evaluator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_set(seed=20, damping=0.0, response=10.0, tau=12.0):
    shared = {
        "status": "succeeded",
        "formal_evaluation": True,
        "debug_only": False,
        "environment_seed": seed,
        "policy_seed": MODULE.POLICY_SEED_OFFSET + seed,
        "configured_damping": damping,
        "max_steps": 300,
        "initial_state_sha256": "initial",
        "prefix_action_sha256": "prefix",
        "pre_probe_state_sha256": "pre-probe",
        "logged_step_count": 300,
        "steps": 300,
        "invalid_action_count": 0,
        "non_finite_value_count": 0,
        "tau": tau,
        "final_coverage": 0.2,
        "max_coverage": 0.4,
        "success": False,
        "stop_reason": "environment_truncated",
    }
    measurement = {
        "signed_response": response,
        "path_length": abs(response),
        "contact_steps": 1,
        "valid": True,
        "invalid_reasons": [],
        "measurement_source": "simulator_info_block_pose",
    }
    adjustment_exercised = response < tau
    return [
        {
            **shared,
            "path": "fixed",
            "probe_action_sha256": None,
            "post_probe_state_sha256": None,
            "response_used_count": 0,
            "adjusted_scale_count": 0,
            "probe_measurement": None,
            "probe_measurement_sha256": None,
            "final_coverage": 0.3,
        },
        {
            **shared,
            "path": "probe-no-adjust",
            "probe_action_sha256": "probe",
            "post_probe_state_sha256": "post-probe",
            "response_used_count": 0,
            "adjusted_scale_count": 0,
            "probe_measurement": measurement,
            "probe_measurement_sha256": "measurement",
            "final_coverage": 0.2,
        },
        {
            **shared,
            "path": "probe-adjust",
            "probe_action_sha256": "probe",
            "post_probe_state_sha256": "post-probe",
            "response_used_count": 275,
            "adjusted_scale_count": 275 if adjustment_exercised else 0,
            "probe_measurement": measurement,
            "probe_measurement_sha256": "measurement",
            "final_coverage": 0.35,
        },
    ]


def all_records():
    records = []
    for damping in MODULE.EVALUATION_DAMPINGS:
        for seed in MODULE.EVALUATION_SEEDS:
            paths = valid_set(seed=seed, damping=damping, response=10.0, tau=12.0)
            paths[1]["final_coverage"] = seed / 100.0
            paths[2]["final_coverage"] = seed / 100.0 + (0.1 if damping == 0.0 else -0.05)
            records.extend(paths)
    return records


class ThreePathEvaluatorTests(unittest.TestCase):
    def test_manifest_is_exact_60_episodes_and_disjoint_from_calibration(self):
        manifest = MODULE.build_evaluation_manifest()
        identities = {
            (item["environment_seed"], item["configured_damping"], item["path"])
            for item in manifest
        }
        expected = {
            (seed, damping, path)
            for damping in MODULE.EVALUATION_DAMPINGS
            for seed in MODULE.EVALUATION_SEEDS
            for path in MODULE.PATHS
        }
        self.assertEqual(identities, expected)
        self.assertEqual(len(manifest), 60)
        self.assertTrue(
            {item["environment_seed"] for item in manifest}.isdisjoint(
                MODULE.CALIBRATION_SEEDS
            )
        )
        MODULE.validate_evaluation_manifest(manifest)

    def test_manifest_rejects_calibration_seed_or_duplicate(self):
        manifest = MODULE.build_evaluation_manifest()
        calibration_seed = deepcopy(manifest)
        calibration_seed[0]["environment_seed"] = 100
        with self.assertRaisesRegex(ValueError, "calibration seed"):
            MODULE.validate_evaluation_manifest(calibration_seed)
        duplicate = deepcopy(manifest)
        duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, "exact planned identities"):
            MODULE.validate_evaluation_manifest(duplicate)

    def test_pair_validator_accepts_triggered_and_non_triggered_valid_rules(self):
        triggered = MODULE.validate_formal_set(valid_set(response=10.0, tau=12.0))
        not_triggered = MODULE.validate_formal_set(valid_set(response=14.0, tau=12.0))
        self.assertTrue(triggered["passed"])
        self.assertTrue(triggered["checks"]["adjustment_rule_matches"])
        self.assertTrue(not_triggered["passed"])
        self.assertTrue(not_triggered["checks"]["adjustment_rule_matches"])

    def test_pair_validator_accepts_invalid_probe_fallback(self):
        records = valid_set()
        for item in records[1:]:
            item["probe_measurement"] = {
                "signed_response": 0.0,
                "path_length": 0.0,
                "contact_steps": 0,
                "valid": False,
                "invalid_reasons": ["no_probe_contact"],
                "measurement_source": "simulator_info_block_pose",
            }
            item["probe_measurement_sha256"] = "invalid-measurement"
        records[2]["response_used_count"] = 0
        records[2]["adjusted_scale_count"] = 0
        result = MODULE.validate_formal_set(records)
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["invalid_fallback_matches"])

    def test_pair_validator_rejects_probe_mismatch_or_hidden_response_use(self):
        mismatch = valid_set()
        mismatch[2]["probe_action_sha256"] = "different"
        result = MODULE.validate_formal_set(mismatch)
        self.assertFalse(result["passed"])
        self.assertIn("probe action hashes differ", result["errors"])

        leak = valid_set()
        leak[1]["response_used_count"] = 1
        result = MODULE.validate_formal_set(leak)
        self.assertFalse(result["passed"])
        self.assertIn("probe-no-adjust consumed response", result["errors"])

    def test_aggregate_uses_adjust_minus_no_adjust_and_preserves_negative_result(self):
        summary = MODULE.aggregate_formal_records(all_records())
        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(summary["planned_episode_count"], 60)
        self.assertEqual(summary["primary_pair_count"], 20)
        self.assertAlmostEqual(
            summary["primary_by_condition"]["0.0"]["mean_paired_difference"],
            0.1,
        )
        self.assertAlmostEqual(
            summary["primary_by_condition"]["1.0"]["mean_paired_difference"],
            -0.05,
        )
        self.assertAlmostEqual(summary["primary_pooled"]["mean_paired_difference"], 0.025)
        self.assertEqual(summary["primary_by_condition"]["0.0"]["positive_count"], 10)
        self.assertEqual(summary["primary_by_condition"]["1.0"]["negative_count"], 10)
        self.assertIn("fixed_reference", summary)
        self.assertNotIn("fixed", summary["primary_pooled"])

    def test_aggregate_rejects_missing_or_duplicate_episode(self):
        records = all_records()
        with self.assertRaisesRegex(ValueError, "exact planned identities"):
            MODULE.aggregate_formal_records(records[:-1])
        duplicate = deepcopy(records)
        duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, "exact planned identities"):
            MODULE.aggregate_formal_records(duplicate)

    def test_load_frozen_calibration_checks_status_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            attempts_path = root / "attempts.jsonl"
            summary_path = root / "summary.json"
            config_path = root / "config.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            attempts_path.write_text("{}\n", encoding="utf-8")
            summary = {
                "status": "succeeded",
                "tau": 12.4,
                "valid_count": 20,
                "invalid_count": 0,
                "minimum_valid_probes": 5,
                "manifest_sha256": sha256(manifest_path),
                "attempts_sha256": sha256(attempts_path),
            }
            summary_path.write_text(
                json.dumps(summary, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            config = {
                "status": "succeeded",
                "summary_sha256": sha256(summary_path),
                "checkpoint_weights_sha256": "weights",
                "compat_config_sha256": "compat",
                "protocol_sha256": "protocol",
                "wrapper_sha256": "wrapper",
            }
            config_path.write_text(
                json.dumps(config, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            frozen = MODULE.load_frozen_calibration(
                root,
                expected_identity={
                    "checkpoint_weights_sha256": "weights",
                    "compat_config_sha256": "compat",
                    "protocol_sha256": "protocol",
                    "wrapper_sha256": "wrapper",
                },
            )
            self.assertEqual(frozen["tau"], 12.4)

            summary["status"] = "patch"
            summary_path.write_text(
                json.dumps(summary, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            config["summary_sha256"] = sha256(summary_path)
            config_path.write_text(
                json.dumps(config, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "status is not succeeded"):
                MODULE.load_frozen_calibration(
                    root,
                    expected_identity={
                        "checkpoint_weights_sha256": "weights",
                        "compat_config_sha256": "compat",
                        "protocol_sha256": "protocol",
                        "wrapper_sha256": "wrapper",
                    },
                )


if __name__ == "__main__":
    unittest.main()
