"""Formal paired PushT evaluation for fixed, probe-no-adjust, and probe-adjust.

The evaluator requires a succeeded calibration artifact, uses only the frozen evaluation manifest,
and reports the paired probe-adjust minus probe-no-adjust final-coverage difference. It does not
claim general adaptation, robustness, real friction identification, or real-robot transfer.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import gym_pusht  # Registers gym_pusht/PushT-v0 with Gymnasium.
import numpy as np
import torch

from diffusion_controller_evaluator import (
    ENV_ID,
    OBS_TYPE,
    POLICY_SEED_OFFSET,
    RENDER_MODE,
    compact_state,
    file_sha256,
    get_git_state,
    load_policy,
    package_versions,
    validate_observation,
)
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.utils import set_global_seed
from probe_adjust_wrapper import (
    DEFAULT_ADJUSTED_SCALE,
    FIXED,
    PATHS,
    PRIVILEGED_MEASUREMENT_SOURCE,
    PROBE_ADJUST,
    PROBE_NO_ADJUST,
    adjust_target,
    build_probe_plan,
    decide_executed_action,
    measure_probe_response,
    probe_array_sha256,
)


EVALUATION_VERSION = "0.1"
CALIBRATION_SEEDS = tuple(range(100, 110))
EVALUATION_SEEDS = tuple(range(20, 30))
EVALUATION_DAMPINGS = (0.0, 1.0)
DEFAULT_MAX_STEPS = 300
DEFAULT_PROBE_START = 20
DEFAULT_PROBE_LENGTH = 5
DEFAULT_TARGET_OFFSET = 20.0
NUMERICAL_TOLERANCE = 1e-12


def canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_evaluation_manifest():
    return [
        {
            "environment_seed": seed,
            "policy_seed": POLICY_SEED_OFFSET + seed,
            "configured_damping": damping,
            "path": path,
        }
        for damping in EVALUATION_DAMPINGS
        for seed in EVALUATION_SEEDS
        for path in PATHS
    ]


def _identity(item):
    try:
        return (
            int(item["environment_seed"]),
            float(item["configured_damping"]),
            str(item["path"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Every evaluation item requires seed, damping, and path.") from error


def validate_evaluation_manifest(manifest):
    if not isinstance(manifest, list):
        raise ValueError("Evaluation manifest must be a list.")
    calibration_overlap = {
        _identity(item)[0] for item in manifest
    }.intersection(CALIBRATION_SEEDS)
    if calibration_overlap:
        raise ValueError(
            f"Evaluation manifest contains a calibration seed: {sorted(calibration_overlap)}"
        )
    identities = [_identity(item) for item in manifest]
    expected = {
        (seed, damping, path)
        for damping in EVALUATION_DAMPINGS
        for seed in EVALUATION_SEEDS
        for path in PATHS
    }
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Evaluation manifest must contain the exact planned identities.")
    for item in manifest:
        seed, _, path = _identity(item)
        if path not in PATHS:
            raise ValueError(f"Unknown evaluation path: {path}")
        if item.get("policy_seed") != POLICY_SEED_OFFSET + seed:
            raise ValueError("Evaluation manifest contains an invalid policy seed.")
    return True


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Required frozen artifact is missing: {path}") from error


def load_frozen_calibration(calibration_dir, expected_identity):
    calibration_dir = Path(calibration_dir)
    manifest_path = calibration_dir / "manifest.json"
    attempts_path = calibration_dir / "attempts.jsonl"
    summary_path = calibration_dir / "summary.json"
    config_path = calibration_dir / "config.json"
    summary = _read_json(summary_path)
    config = _read_json(config_path)

    if summary.get("status") != "succeeded":
        raise ValueError("Frozen calibration status is not succeeded.")
    if config.get("status") != "succeeded":
        raise ValueError("Frozen calibration config status is not succeeded.")
    tau = summary.get("tau")
    if not isinstance(tau, (int, float)) or not math.isfinite(tau):
        raise ValueError("Frozen calibration tau must be finite.")
    if summary.get("valid_count", 0) < summary.get("minimum_valid_probes", 5):
        raise ValueError("Frozen calibration does not meet the minimum valid-probe Gate.")

    actual_hashes = {
        "manifest_sha256": file_sha256(manifest_path),
        "attempts_sha256": file_sha256(attempts_path),
        "summary_sha256": file_sha256(summary_path),
    }
    if summary.get("manifest_sha256") != actual_hashes["manifest_sha256"]:
        raise ValueError("Frozen calibration manifest hash mismatch.")
    if summary.get("attempts_sha256") != actual_hashes["attempts_sha256"]:
        raise ValueError("Frozen calibration attempts hash mismatch.")
    if config.get("summary_sha256") != actual_hashes["summary_sha256"]:
        raise ValueError("Frozen calibration summary hash mismatch.")
    for field, expected_value in expected_identity.items():
        if config.get(field) != expected_value:
            raise ValueError(f"Frozen calibration identity mismatch: {field}")

    return {
        "tau": float(tau),
        "valid_count": int(summary["valid_count"]),
        "invalid_count": int(summary["invalid_count"]),
        "manifest_sha256": actual_hashes["manifest_sha256"],
        "attempts_sha256": actual_hashes["attempts_sha256"],
        "summary_sha256": actual_hashes["summary_sha256"],
        "config_sha256": file_sha256(config_path),
        "calibration_dir": str(calibration_dir.resolve()),
    }


def step_phase(path, step, probe_start, probe_length):
    if path not in PATHS:
        raise ValueError(f"Unknown wrapper path: {path}")
    if step <= probe_start:
        return "prefix"
    if step <= probe_start + probe_length:
        return "continuation_window" if path == FIXED else "probe"
    return "post_window" if path == FIXED else "post_probe"


def validate_formal_set(records):
    errors = []
    by_path = {record.get("path"): record for record in records}
    if set(by_path) != set(PATHS) or len(records) != len(PATHS):
        return {
            "passed": False,
            "checks": {},
            "errors": ["records must contain each path exactly once"],
        }
    fixed = by_path[FIXED]
    no_adjust = by_path[PROBE_NO_ADJUST]
    adjust = by_path[PROBE_ADJUST]

    def shared(field):
        return len({record.get(field) for record in records}) == 1

    measurement = adjust.get("probe_measurement")
    measurement_valid = bool(measurement and measurement.get("valid") is True)
    response = None if measurement is None else measurement.get("signed_response")
    tau = adjust.get("tau")
    has_post_probe_actions = adjust.get("steps", 0) > (
        DEFAULT_PROBE_START + DEFAULT_PROBE_LENGTH
    )
    should_use_response = measurement_valid and has_post_probe_actions
    should_adjust = (
        should_use_response
        and isinstance(response, (int, float))
        and isinstance(tau, (int, float))
        and response < tau
    )

    checks = {
        "formal_identity": all(
            record.get("formal_evaluation") is True
            and record.get("debug_only") is False
            for record in records
        ),
        "succeeded": all(record.get("status") == "succeeded" for record in records),
        "shared_seed": shared("environment_seed") and shared("policy_seed"),
        "shared_setting": shared("configured_damping"),
        "shared_budget": shared("max_steps"),
        "shared_tau": shared("tau"),
        "shared_initial_state": shared("initial_state_sha256"),
        "shared_prefix": shared("prefix_action_sha256"),
        "shared_pre_probe_state": (
            no_adjust.get("pre_probe_state_sha256") is not None
            and no_adjust.get("pre_probe_state_sha256")
            == adjust.get("pre_probe_state_sha256")
        ),
        "shared_probe": (
            no_adjust.get("probe_action_sha256") is not None
            and no_adjust.get("probe_action_sha256")
            == adjust.get("probe_action_sha256")
        ),
        "shared_post_probe_state": (
            no_adjust.get("post_probe_state_sha256") is not None
            and no_adjust.get("post_probe_state_sha256")
            == adjust.get("post_probe_state_sha256")
        ),
        "shared_measurement": (
            no_adjust.get("probe_measurement_sha256") is not None
            and no_adjust.get("probe_measurement_sha256")
            == adjust.get("probe_measurement_sha256")
        ),
        "complete_step_logs": all(
            record.get("logged_step_count") == record.get("steps")
            for record in records
        ),
        "all_actions_valid": all(
            record.get("invalid_action_count") == 0 for record in records
        ),
        "all_values_finite": all(
            record.get("non_finite_value_count") == 0 for record in records
        ),
        "response_isolation": (
            fixed.get("response_used_count") == 0
            and no_adjust.get("response_used_count") == 0
            and bool(adjust.get("response_used_count", 0) > 0) == should_use_response
        ),
        "adjustment_rule_matches": (
            fixed.get("adjusted_scale_count") == 0
            and no_adjust.get("adjusted_scale_count") == 0
            and bool(adjust.get("adjusted_scale_count", 0) > 0) == should_adjust
        ),
        "invalid_fallback_matches": (
            measurement_valid
            or (
                adjust.get("response_used_count") == 0
                and adjust.get("adjusted_scale_count") == 0
            )
        ),
    }
    messages = {
        "formal_identity": "records are not formal evaluation records",
        "succeeded": "a path did not succeed",
        "shared_seed": "seed or policy seed differs",
        "shared_setting": "simulator setting differs",
        "shared_budget": "step budget differs",
        "shared_tau": "frozen tau differs",
        "shared_initial_state": "initial states differ",
        "shared_prefix": "prefix action hashes differ",
        "shared_pre_probe_state": "pre-probe states differ",
        "shared_probe": "probe action hashes differ",
        "shared_post_probe_state": "post-probe states differ",
        "shared_measurement": "probe measurements differ",
        "complete_step_logs": "step logs are incomplete",
        "all_actions_valid": "an invalid action was recorded",
        "all_values_finite": "a non-finite value was recorded",
        "response_isolation": "probe-no-adjust consumed response",
        "adjustment_rule_matches": "executed adjustment does not match frozen rule",
        "invalid_fallback_matches": "invalid probe did not fall back to identity scale",
    }
    for check, passed in checks.items():
        if not passed:
            errors.append(messages[check])
    return {"passed": not errors, "checks": checks, "errors": errors}


def _distribution(values):
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "mean_paired_difference": None,
            "median_paired_difference": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "positive_count": 0,
            "zero_count": 0,
            "negative_count": 0,
        }
    return {
        "count": len(values),
        "mean_paired_difference": statistics.fmean(values),
        "median_paired_difference": statistics.median(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "positive_count": sum(value > NUMERICAL_TOLERANCE for value in values),
        "zero_count": sum(abs(value) <= NUMERICAL_TOLERANCE for value in values),
        "negative_count": sum(value < -NUMERICAL_TOLERANCE for value in values),
    }


def _path_summary(records):
    return {
        "episode_count": len(records),
        "mean_final_coverage": statistics.fmean(
            float(record["final_coverage"]) for record in records
        ),
        "mean_max_coverage": statistics.fmean(
            float(record["max_coverage"]) for record in records
        ),
        "success_count": sum(bool(record["success"]) for record in records),
        "success_rate": statistics.fmean(
            float(bool(record["success"])) for record in records
        ),
    }


def aggregate_formal_records(records):
    if not isinstance(records, list):
        raise ValueError("Formal evaluation records must be a list.")
    identities = [_identity(record) for record in records]
    expected = {
        (seed, damping, path)
        for damping in EVALUATION_DAMPINGS
        for seed in EVALUATION_SEEDS
        for path in PATHS
    }
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Formal records must contain the exact planned identities.")

    indexed = {_identity(record): record for record in records}
    pairs = []
    validation_rows = []
    for damping in EVALUATION_DAMPINGS:
        for seed in EVALUATION_SEEDS:
            formal_set = [indexed[(seed, damping, path)] for path in PATHS]
            validation = validate_formal_set(formal_set)
            validation_rows.append(
                {
                    "environment_seed": seed,
                    "configured_damping": damping,
                    **validation,
                }
            )
            if not validation["passed"]:
                raise ValueError(
                    f"Formal pair validation failed for seed={seed} damping={damping}: "
                    + "; ".join(validation["errors"])
                )
            by_path = {record["path"]: record for record in formal_set}
            measurement = by_path[PROBE_ADJUST]["probe_measurement"]
            pairs.append(
                {
                    "environment_seed": seed,
                    "configured_damping": damping,
                    "probe_valid": bool(measurement and measurement["valid"]),
                    "probe_response": (
                        None if measurement is None else measurement["signed_response"]
                    ),
                    "tau": by_path[PROBE_ADJUST]["tau"],
                    "no_adjust_final_coverage": by_path[PROBE_NO_ADJUST][
                        "final_coverage"
                    ],
                    "adjust_final_coverage": by_path[PROBE_ADJUST]["final_coverage"],
                    "paired_difference": (
                        float(by_path[PROBE_ADJUST]["final_coverage"])
                        - float(by_path[PROBE_NO_ADJUST]["final_coverage"])
                    ),
                }
            )

    primary_by_condition = {}
    fixed_reference = {}
    for damping in EVALUATION_DAMPINGS:
        key = str(damping)
        condition_pairs = [
            pair for pair in pairs if pair["configured_damping"] == damping
        ]
        primary_by_condition[key] = _distribution(
            pair["paired_difference"] for pair in condition_pairs
        )
        condition_records = [
            record for record in records if record["configured_damping"] == damping
        ]
        path_records = {
            path: [record for record in condition_records if record["path"] == path]
            for path in PATHS
        }
        fixed_reference[key] = {
            "paths": {
                path: _path_summary(path_records[path]) for path in PATHS
            },
            "mean_no_adjust_minus_fixed": statistics.fmean(
                indexed[(seed, damping, PROBE_NO_ADJUST)]["final_coverage"]
                - indexed[(seed, damping, FIXED)]["final_coverage"]
                for seed in EVALUATION_SEEDS
            ),
            "mean_adjust_minus_fixed": statistics.fmean(
                indexed[(seed, damping, PROBE_ADJUST)]["final_coverage"]
                - indexed[(seed, damping, FIXED)]["final_coverage"]
                for seed in EVALUATION_SEEDS
            ),
        }

    valid_pairs = [pair for pair in pairs if pair["probe_valid"]]
    return {
        "status": "succeeded",
        "claim_boundary": (
            "Bounded paired simulator evaluation only; no general adaptation, robustness, "
            "real-friction, or real-robot claim."
        ),
        "planned_episode_count": len(expected),
        "observed_episode_count": len(records),
        "primary_pair_count": len(pairs),
        "primary_pairs": pairs,
        "primary_by_condition": primary_by_condition,
        "primary_pooled": _distribution(
            pair["paired_difference"] for pair in pairs
        ),
        "valid_probe_pair_count": len(valid_pairs),
        "invalid_probe_pair_count": len(pairs) - len(valid_pairs),
        "valid_probe_only_auxiliary": _distribution(
            pair["paired_difference"] for pair in valid_pairs
        ),
        "fixed_reference": fixed_reference,
        "pair_validations": validation_rows,
    }


def policy_action(policy, observation, device):
    batched_observation = {
        key: np.expand_dims(value, axis=0) for key, value in observation.items()
    }
    policy_observation = preprocess_observation(batched_observation)
    policy_observation = {
        key: value.to(device, non_blocking=True)
        for key, value in policy_observation.items()
    }
    started_at = time.perf_counter()
    with torch.inference_mode():
        action_tensor = policy.select_action(policy_observation)
    seconds = time.perf_counter() - started_at
    return action_tensor.to("cpu").numpy()[0], seconds


def run_path_episode(
    policy,
    device,
    path,
    environment_seed,
    configured_damping,
    tau,
    calibration_summary_sha256,
    step_log_path,
):
    policy_seed = POLICY_SEED_OFFSET + environment_seed
    env = gym.make(
        ENV_ID,
        obs_type=OBS_TYPE,
        render_mode=RENDER_MODE,
        damping=configured_damping,
        max_episode_steps=DEFAULT_MAX_STEPS,
    )
    started_at = time.perf_counter()

    try:
        set_global_seed(policy_seed)
        policy.reset()
        observation, reset_info = env.reset(seed=environment_seed)
        validate_observation(observation, env.observation_space, "Reset observation")
        current_info = env.unwrapped._get_info()
        current_coverage = float(env.unwrapped._get_coverage())
        initial_state = compact_state(observation, current_info, current_coverage)
        initial_state_sha256 = canonical_sha256(initial_state)

        max_coverage = current_coverage
        final_coverage = current_coverage
        episode_return = 0.0
        steps = 0
        logged_step_count = 0
        invalid_action_count = 0
        non_finite_value_count = 0
        response_used_count = 0
        adjusted_scale_count = 0
        clamped_action_count = 0
        stop_reason = "step_budget"
        final_info = dict(reset_info)

        prefix_actions = []
        pre_probe_state_sha256 = None
        post_probe_state_sha256 = None
        probe_plan = None
        probe_positions = []
        probe_contact_steps = 0
        measurement = None

        with step_log_path.open("x", encoding="utf-8") as step_log:
            for step in range(1, DEFAULT_MAX_STEPS + 1):
                phase = step_phase(
                    path,
                    step,
                    DEFAULT_PROBE_START,
                    DEFAULT_PROBE_LENGTH,
                )
                state_before = compact_state(
                    observation,
                    current_info,
                    current_coverage,
                )
                base_action = None
                scale = 1.0
                response_used = False
                inference_seconds = 0.0

                if phase == "probe":
                    if probe_plan is None:
                        probe_plan = build_probe_plan(
                            agent_position=observation["agent_pos"],
                            block_position=np.asarray(current_info["block_pose"])[:2],
                            action_low=env.action_space.low,
                            action_high=env.action_space.high,
                            target_offset=DEFAULT_TARGET_OFFSET,
                            probe_length=DEFAULT_PROBE_LENGTH,
                        )
                        pre_probe_state_sha256 = canonical_sha256(state_before)
                        probe_positions = [
                            np.asarray(current_info["block_pose"], dtype=np.float64)[:2]
                        ]
                    probe_index = step - DEFAULT_PROBE_START - 1
                    executed_action = probe_plan.actions[probe_index]
                else:
                    action, inference_seconds = policy_action(policy, observation, device)
                    base_action = action.astype(env.action_space.dtype, copy=False)
                    if phase == "post_probe":
                        decision = decide_executed_action(
                            path=path,
                            agent_position=observation["agent_pos"],
                            base_target=base_action,
                            action_low=env.action_space.low,
                            action_high=env.action_space.high,
                            measurement=measurement,
                            tau=tau,
                        )
                        executed_action = decision.executed_target
                        scale = decision.scale
                        response_used = decision.response_used
                        response_used_count += int(response_used)
                        adjusted_scale_count += int(scale == DEFAULT_ADJUSTED_SCALE)
                        clamped_action_count += int(decision.clamped)
                    else:
                        executed_action = adjust_target(
                            agent_position=observation["agent_pos"],
                            base_target=base_action,
                            scale=1.0,
                            action_low=env.action_space.low,
                            action_high=env.action_space.high,
                        )

                executed_action = np.asarray(
                    executed_action,
                    dtype=env.action_space.dtype,
                )
                if not np.all(np.isfinite(executed_action)):
                    non_finite_value_count += 1
                    raise ValueError(
                        f"Non-finite formal action at path={path} seed={environment_seed} "
                        f"damping={configured_damping} step={step}."
                    )
                if not env.action_space.contains(executed_action):
                    invalid_action_count += 1
                    raise ValueError(
                        f"Invalid formal action at path={path} seed={environment_seed} "
                        f"damping={configured_damping} step={step}: {executed_action}"
                    )

                next_observation, reward, terminated, truncated, info = env.step(
                    executed_action
                )
                validate_observation(
                    next_observation,
                    env.observation_space,
                    f"Formal observation at path={path} seed={environment_seed} step={step}",
                )
                coverage = float(info["coverage"])
                if not np.all(np.isfinite([reward, coverage])):
                    non_finite_value_count += 1
                    raise ValueError(
                        f"Non-finite reward/coverage at path={path} "
                        f"seed={environment_seed} step={step}."
                    )
                state_after = compact_state(next_observation, info, coverage)

                if step <= DEFAULT_PROBE_START:
                    prefix_actions.append(executed_action.copy())
                if step == DEFAULT_PROBE_START:
                    pre_probe_state_sha256 = canonical_sha256(state_after)
                if phase == "probe":
                    probe_positions.append(
                        np.asarray(info["block_pose"], dtype=np.float64)[:2]
                    )
                    probe_contact_steps += int(int(info["n_contacts"]) > 0)

                step_log.write(
                    json.dumps(
                        {
                            "formal_evaluation": True,
                            "path": path,
                            "environment_seed": environment_seed,
                            "policy_seed": policy_seed,
                            "configured_damping": configured_damping,
                            "tau": tau,
                            "step": step,
                            "phase": phase,
                            "controller_visible": {
                                "ordinary_observation": {
                                    "agent_pos": np.asarray(
                                        observation["agent_pos"]
                                    ).tolist(),
                                    "pixels_sha256": state_before["pixels_sha256"],
                                },
                                "base_target": (
                                    None
                                    if base_action is None
                                    else np.asarray(base_action).tolist()
                                ),
                            },
                            "action_decision": {
                                "scale": scale,
                                "response_used": response_used,
                                "executed_target": executed_action.tolist(),
                            },
                            "evaluator_only": {
                                "state_before": state_before,
                                "state_after": state_after,
                                "coverage": coverage,
                                "contacts": int(info["n_contacts"]),
                            },
                            "reward": float(reward),
                            "inference_seconds": inference_seconds,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                step_log.flush()

                episode_return += float(reward)
                max_coverage = max(max_coverage, coverage)
                final_coverage = coverage
                final_info = dict(info)
                current_info = dict(info)
                current_coverage = coverage
                observation = next_observation
                steps = step
                logged_step_count += 1

                if phase == "probe" and step == (
                    DEFAULT_PROBE_START + DEFAULT_PROBE_LENGTH
                ):
                    measurement = measure_probe_response(
                        block_positions=probe_positions,
                        probe_direction=probe_plan.direction,
                        direction_valid=probe_plan.direction_valid,
                        contact_steps=probe_contact_steps,
                        measurement_source=PRIVILEGED_MEASUREMENT_SOURCE,
                    )
                    post_probe_state_sha256 = canonical_sha256(state_after)
                    policy.reset()

                if terminated:
                    stop_reason = "terminated_success"
                    break
                if truncated:
                    stop_reason = "environment_truncated"
                    break

        prefix_action_sha256 = probe_array_sha256(np.asarray(prefix_actions))
        measurement_record = None
        if measurement is not None:
            measurement_record = {
                "signed_response": measurement.signed_response,
                "path_length": measurement.path_length,
                "contact_steps": measurement.contact_steps,
                "valid": measurement.valid,
                "invalid_reasons": list(measurement.invalid_reasons),
                "measurement_source": measurement.measurement_source,
            }

        return {
            "status": "succeeded",
            "formal_evaluation": True,
            "debug_only": False,
            "path": path,
            "environment_seed": environment_seed,
            "policy_seed": policy_seed,
            "configured_damping": configured_damping,
            "space_damping": env.unwrapped.space.damping,
            "max_steps": DEFAULT_MAX_STEPS,
            "steps": steps,
            "logged_step_count": logged_step_count,
            "initial_state": initial_state,
            "initial_state_sha256": initial_state_sha256,
            "prefix_action_sha256": prefix_action_sha256,
            "pre_probe_state_sha256": pre_probe_state_sha256,
            "post_probe_state_sha256": post_probe_state_sha256,
            "probe_action_sha256": None if probe_plan is None else probe_plan.sha256,
            "probe_measurement": measurement_record,
            "probe_measurement_sha256": (
                None if measurement_record is None else canonical_sha256(measurement_record)
            ),
            "tau": tau,
            "calibration_summary_sha256": calibration_summary_sha256,
            "response_used_count": response_used_count,
            "adjusted_scale_count": adjusted_scale_count,
            "clamped_action_count": clamped_action_count,
            "initial_coverage": initial_state["coverage"],
            "max_coverage": max_coverage,
            "final_coverage": final_coverage,
            "episode_return": episode_return,
            "success": bool(final_info.get("is_success", False)),
            "stop_reason": stop_reason,
            "invalid_action_count": invalid_action_count,
            "non_finite_value_count": non_finite_value_count,
            "runtime_seconds": time.perf_counter() - started_at,
            "step_log": f"steps/{step_log_path.name}",
            "step_log_sha256": file_sha256(step_log_path),
        }
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the frozen paired PushT three-path formal evaluation."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--compat-config-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _condition_token(damping):
    return str(float(damping)).replace("-", "m").replace(".", "p")


def main():
    args = parse_args()
    if not args.checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
    if not args.compat_config_dir.is_dir():
        raise ValueError(
            f"Compatibility config directory does not exist: {args.compat_config_dir}"
        )

    script_path = Path(__file__).resolve()
    wrapper_path = script_path.with_name("probe_adjust_wrapper.py")
    protocol_path = script_path.with_name(
        "2026-08-30-probe-adjust-wrapper-protocol-v0.2.md"
    )
    weights_path = args.checkpoint_dir / "model.safetensors"
    compat_path = args.compat_config_dir / "config.json"
    current_identity = {
        "checkpoint_weights_sha256": file_sha256(weights_path),
        "compat_config_sha256": file_sha256(compat_path),
        "protocol_sha256": file_sha256(protocol_path),
        "wrapper_sha256": file_sha256(wrapper_path),
    }
    frozen_calibration = load_frozen_calibration(
        args.calibration_dir,
        expected_identity=current_identity,
    )
    tau = frozen_calibration["tau"]
    manifest = build_evaluation_manifest()
    validate_evaluation_manifest(manifest)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    steps_directory = args.output_dir / "steps"
    validations_directory = args.output_dir / "pair-validations"
    steps_directory.mkdir()
    validations_directory.mkdir()
    manifest_path = args.output_dir / "manifest.json"
    config_path = args.output_dir / "config.json"
    episodes_path = args.output_dir / "episodes.jsonl"
    summary_path = args.output_dir / "summary.json"
    device = torch.device(args.device)

    manifest_document = {
        "evaluation_version": EVALUATION_VERSION,
        "seeds": list(EVALUATION_SEEDS),
        "calibration_seeds_excluded": list(CALIBRATION_SEEDS),
        "dampings": list(EVALUATION_DAMPINGS),
        "paths": list(PATHS),
        "max_steps": DEFAULT_MAX_STEPS,
        "probe_start": DEFAULT_PROBE_START,
        "probe_length": DEFAULT_PROBE_LENGTH,
        "target_offset": DEFAULT_TARGET_OFFSET,
        "tau": tau,
        "calibration_summary_sha256": frozen_calibration["summary_sha256"],
        "episodes": manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "status": "started",
        "formal_evaluation": True,
        "debug_only": False,
        "claim_boundary": (
            "Bounded paired simulator evaluation only; excludes general adaptation, "
            "robustness, real-friction, and real-robot claims."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_version": EVALUATION_VERSION,
        "environment_id": ENV_ID,
        "observation_type": OBS_TYPE,
        "device": str(device),
        "tau": tau,
        "frozen_calibration": frozen_calibration,
        "manifest_sha256": file_sha256(manifest_path),
        **current_identity,
        "formal_evaluator_sha256": file_sha256(script_path),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": package_versions(),
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "git": get_git_state(),
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        policy_load_started_at = time.perf_counter()
        policy = load_policy(args.checkpoint_dir, args.compat_config_dir, device)
        policy_load_seconds = time.perf_counter() - policy_load_started_at
        records = []
        with episodes_path.open("x", encoding="utf-8") as episodes_log:
            for damping in EVALUATION_DAMPINGS:
                for seed in EVALUATION_SEEDS:
                    formal_set = []
                    for path in PATHS:
                        step_log_path = steps_directory / (
                            f"seed-{seed}-damping-{_condition_token(damping)}-{path}.jsonl"
                        )
                        record = run_path_episode(
                            policy=policy,
                            device=device,
                            path=path,
                            environment_seed=seed,
                            configured_damping=damping,
                            tau=tau,
                            calibration_summary_sha256=frozen_calibration[
                                "summary_sha256"
                            ],
                            step_log_path=step_log_path,
                        )
                        records.append(record)
                        formal_set.append(record)
                        episodes_log.write(json.dumps(record, sort_keys=True) + "\n")
                        episodes_log.flush()
                        print(
                            f"path={path} seed={seed} damping={damping} "
                            f"final={record['final_coverage']:.6f} "
                            f"success={record['success']} steps={record['steps']}",
                            flush=True,
                        )

                    validation = validate_formal_set(formal_set)
                    validation_path = validations_directory / (
                        f"seed-{seed}-damping-{_condition_token(damping)}.json"
                    )
                    validation_path.write_text(
                        json.dumps(validation, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if not validation["passed"]:
                        raise RuntimeError(
                            f"Formal validation failed for seed={seed} damping={damping}: "
                            + "; ".join(validation["errors"])
                        )

        summary = aggregate_formal_records(records)
        summary.update(
            {
                "policy_load_seconds": policy_load_seconds,
                "tau": tau,
                "calibration_summary_sha256": frozen_calibration["summary_sha256"],
                "manifest_sha256": file_sha256(manifest_path),
                "episodes_sha256": file_sha256(episodes_path),
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config["status"] = summary["status"]
        config["summary_sha256"] = file_sha256(summary_path)
        config["episodes_sha256"] = file_sha256(episodes_path)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"output_dir={args.output_dir}", flush=True)
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config["status"] = "failed"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
