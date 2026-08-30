import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "experiments" / "probe_adjust_wrapper.py"
SPEC = importlib.util.spec_from_file_location("probe_adjust_wrapper", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProbeAdjustWrapperTests(unittest.TestCase):
    def test_signed_response_uses_net_displacement_projection(self):
        direction = np.asarray([0.6, 0.8])
        self.assertAlmostEqual(
            MODULE.signed_response([3.0, 4.0], [6.0, 8.0], direction),
            5.0,
        )
        self.assertAlmostEqual(
            MODULE.signed_response([1.0, 1.0], [0.0, 1.0], [1.0, 0.0]),
            -1.0,
        )
        self.assertEqual(
            MODULE.signed_response([1.0, 1.0], [1.0, 1.0], [1.0, 0.0]),
            0.0,
        )

        for invalid in ([np.nan, 1.0], [np.inf, 1.0]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    MODULE.signed_response([0.0, 0.0], invalid, [1.0, 0.0])

    def test_path_length_distinguishes_stationary_straight_and_backtracking(self):
        self.assertEqual(MODULE.trajectory_path_length([[1.0, 1.0]]), 0.0)
        self.assertAlmostEqual(
            MODULE.trajectory_path_length([[0.0, 0.0], [3.0, 4.0]]),
            5.0,
        )
        self.assertAlmostEqual(
            MODULE.trajectory_path_length(
                [[3.0, 4.0], [4.0, 4.0], [4.0, 7.0], [6.0, 8.0]]
            ),
            4.0 + np.sqrt(5.0),
        )

    def test_probe_plan_has_expected_direction_target_and_stable_hash(self):
        first = MODULE.build_probe_plan(
            agent_position=[0.0, 0.0],
            block_position=[3.0, 4.0],
            action_low=[0.0, 0.0],
            action_high=[512.0, 512.0],
            target_offset=20.0,
            probe_length=5,
        )
        second = MODULE.build_probe_plan(
            agent_position=[0.0, 0.0],
            block_position=[3.0, 4.0],
            action_low=[0.0, 0.0],
            action_high=[512.0, 512.0],
            target_offset=20.0,
            probe_length=5,
        )

        np.testing.assert_allclose(first.direction, [0.6, 0.8])
        np.testing.assert_allclose(first.target, [15.0, 20.0])
        self.assertEqual(first.actions.shape, (5, 2))
        np.testing.assert_array_equal(first.actions, second.actions)
        self.assertEqual(first.sha256, second.sha256)
        self.assertTrue(first.direction_valid)

    def test_zero_distance_probe_is_marked_invalid(self):
        plan = MODULE.build_probe_plan(
            agent_position=[4.0, 7.0],
            block_position=[4.0, 7.0],
            action_low=[0.0, 0.0],
            action_high=[512.0, 512.0],
            target_offset=20.0,
            probe_length=5,
        )

        self.assertFalse(plan.direction_valid)
        np.testing.assert_array_equal(plan.direction, [0.0, 0.0])
        np.testing.assert_array_equal(plan.target, [4.0, 7.0])

    def test_measurement_records_response_path_length_and_validity(self):
        measurement = MODULE.measure_probe_response(
            block_positions=[[3.0, 4.0], [4.0, 4.0], [4.0, 7.0], [6.0, 8.0]],
            probe_direction=[0.6, 0.8],
            direction_valid=True,
            contact_steps=3,
            measurement_source=MODULE.PRIVILEGED_MEASUREMENT_SOURCE,
        )

        self.assertTrue(measurement.valid)
        self.assertAlmostEqual(measurement.signed_response, 5.0)
        self.assertAlmostEqual(measurement.path_length, 4.0 + np.sqrt(5.0))
        self.assertEqual(measurement.invalid_reasons, ())

    def test_invalid_probe_falls_back_even_when_response_is_below_tau(self):
        measurement = MODULE.measure_probe_response(
            block_positions=[[2.0, 3.0], [5.0, 7.0]],
            probe_direction=[0.0, 1.0],
            direction_valid=True,
            contact_steps=0,
            measurement_source=MODULE.PRIVILEGED_MEASUREMENT_SOURCE,
        )

        self.assertFalse(measurement.valid)
        self.assertEqual(measurement.signed_response, 4.0)
        self.assertIn("no_probe_contact", measurement.invalid_reasons)
        self.assertEqual(
            MODULE.select_scale(MODULE.PROBE_ADJUST, measurement, tau=5.0),
            1.0,
        )

    def test_only_adjust_path_consumes_valid_response(self):
        measurement = MODULE.measure_probe_response(
            block_positions=[[2.0, 3.0], [2.0, 4.0]],
            probe_direction=[0.0, 1.0],
            direction_valid=True,
            contact_steps=1,
            measurement_source=MODULE.PRIVILEGED_MEASUREMENT_SOURCE,
        )

        self.assertEqual(
            MODULE.select_scale(MODULE.PROBE_NO_ADJUST, measurement, tau=2.0),
            1.0,
        )
        self.assertEqual(
            MODULE.select_scale(MODULE.PROBE_ADJUST, measurement, tau=2.0),
            1.25,
        )
        self.assertEqual(MODULE.select_scale(MODULE.FIXED, None, tau=2.0), 1.0)

        no_adjust = MODULE.decide_executed_action(
            path=MODULE.PROBE_NO_ADJUST,
            agent_position=[10.0, 10.0],
            base_target=[14.0, 18.0],
            action_low=[0.0, 0.0],
            action_high=[100.0, 100.0],
            measurement=measurement,
            tau=2.0,
        )
        adjust = MODULE.decide_executed_action(
            path=MODULE.PROBE_ADJUST,
            agent_position=[10.0, 10.0],
            base_target=[14.0, 18.0],
            action_low=[0.0, 0.0],
            action_high=[100.0, 100.0],
            measurement=measurement,
            tau=2.0,
        )

        np.testing.assert_array_equal(no_adjust.executed_target, [14.0, 18.0])
        np.testing.assert_array_equal(adjust.executed_target, [15.0, 20.0])
        self.assertFalse(no_adjust.response_used)
        self.assertTrue(adjust.response_used)

    def test_adjust_target_scales_then_clamps_absolute_target(self):
        np.testing.assert_array_equal(
            MODULE.adjust_target(
                agent_position=[10.0, 10.0],
                base_target=[14.0, 18.0],
                scale=1.0,
                action_low=[0.0, 0.0],
                action_high=[19.0, 19.0],
            ),
            [14.0, 18.0],
        )
        np.testing.assert_array_equal(
            MODULE.adjust_target(
                agent_position=[10.0, 10.0],
                base_target=[14.0, 18.0],
                scale=1.25,
                action_low=[0.0, 0.0],
                action_high=[19.0, 19.0],
            ),
            [15.0, 19.0],
        )

    def test_phase_budget_counts_probe_or_continuation_inside_shared_total(self):
        fixed = MODULE.phase_budget(MODULE.FIXED, total_steps=300, probe_start=20, probe_length=5)
        no_adjust = MODULE.phase_budget(
            MODULE.PROBE_NO_ADJUST,
            total_steps=300,
            probe_start=20,
            probe_length=5,
        )
        adjust = MODULE.phase_budget(
            MODULE.PROBE_ADJUST,
            total_steps=300,
            probe_start=20,
            probe_length=5,
        )

        for budget in (fixed, no_adjust, adjust):
            self.assertEqual(sum(budget.values()), 300)
            self.assertEqual(budget["prefix_steps"], 20)
            self.assertEqual(budget["post_window_steps"], 275)
        self.assertEqual(fixed["continuation_steps"], 5)
        self.assertEqual(fixed["probe_steps"], 0)
        self.assertEqual(no_adjust["continuation_steps"], 0)
        self.assertEqual(no_adjust["probe_steps"], 5)
        self.assertEqual(adjust, no_adjust)

    def test_episode_schema_is_shared_and_separates_access_roles(self):
        records = [MODULE.new_episode_record(path) for path in MODULE.PATHS]
        expected_top_level = set(records[0])
        for record in records:
            self.assertEqual(set(record), expected_top_level)
            self.assertEqual(
                set(record["controller_visible"]),
                {"ordinary_observation", "base_target"},
            )
            self.assertEqual(
                set(record["evaluator_only"]),
                {"probe", "response", "metrics", "hashes", "hidden_setting"},
            )
            self.assertNotIn("hidden_setting", record["controller_visible"])

    def test_protocol_core_action_apis_do_not_accept_damping(self):
        action_apis = (
            MODULE.build_probe_plan,
            MODULE.measure_probe_response,
            MODULE.select_scale,
            MODULE.adjust_target,
            MODULE.decide_executed_action,
            MODULE.phase_budget,
        )
        for function in action_apis:
            with self.subTest(function=function.__name__):
                self.assertNotIn("damping", inspect.signature(function).parameters)


if __name__ == "__main__":
    unittest.main()
