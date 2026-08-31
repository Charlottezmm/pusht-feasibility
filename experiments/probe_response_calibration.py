"""Calibrate the frozen PushT probe-response threshold before formal evaluation.

This runner executes only the shared prefix and probe-measurement path. It never applies an
adjustment, runs an evaluation seed, or produces effectiveness evidence.
"""

import argparse
import hashlib
import json
import math
import os
import platform
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
    PRIVILEGED_MEASUREMENT_SOURCE,
    build_probe_plan,
    measure_probe_response,
    probe_array_sha256,
)


CALIBRATION_VERSION = "0.1"
CALIBRATION_SEEDS = tuple(range(100, 110))
EVALUATION_SEEDS = tuple(range(20, 30))
CALIBRATION_DAMPINGS = (0.0, 1.0)
MINIMUM_VALID_PROBES = 5
DEFAULT_PROBE_START = 20
DEFAULT_PROBE_LENGTH = 5
DEFAULT_TARGET_OFFSET = 20.0


def canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_calibration_manifest():
    return [
        {
            "environment_seed": seed,
            "policy_seed": POLICY_SEED_OFFSET + seed,
            "configured_damping": damping,
        }
        for damping in CALIBRATION_DAMPINGS
        for seed in CALIBRATION_SEEDS
    ]


def _identity(item):
    try:
        return (int(item["environment_seed"]), float(item["configured_damping"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Every calibration item requires a valid seed and damping.") from error


def validate_calibration_manifest(manifest):
    if not isinstance(manifest, list):
        raise ValueError("Calibration manifest must be a list.")
    evaluation_overlap = {
        _identity(item)[0] for item in manifest
    }.intersection(EVALUATION_SEEDS)
    if evaluation_overlap:
        raise ValueError(
            f"Calibration manifest contains an evaluation seed: {sorted(evaluation_overlap)}"
        )

    identities = [_identity(item) for item in manifest]
    expected = {
        (seed, damping)
        for damping in CALIBRATION_DAMPINGS
        for seed in CALIBRATION_SEEDS
    }
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Calibration manifest must contain the exact planned identities.")
    for item in manifest:
        seed, _ = _identity(item)
        if item.get("policy_seed") != POLICY_SEED_OFFSET + seed:
            raise ValueError("Calibration manifest contains an invalid policy seed.")
    return True


def _validate_record(record):
    required_identity_fields = (
        "initial_state_sha256",
        "pre_probe_state_sha256",
        "post_probe_state_sha256",
        "probe_action_sha256",
    )
    if record.get("status") != "succeeded":
        raise ValueError("Every calibration record must have status=succeeded.")
    seed, _ = _identity(record)
    if record.get("policy_seed") != POLICY_SEED_OFFSET + seed:
        raise ValueError("Calibration record contains an invalid policy seed.")
    for field in required_identity_fields:
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"Calibration record is missing identity field: {field}")
    measurement = record.get("probe_measurement")
    if not isinstance(measurement, dict):
        raise ValueError("Calibration record is missing probe_measurement.")
    if not isinstance(measurement.get("valid"), bool):
        raise ValueError("Probe measurement valid flag must be a bool.")
    if measurement.get("measurement_source") != PRIVILEGED_MEASUREMENT_SOURCE:
        raise ValueError("Probe measurement source is not the frozen privileged source.")
    response = measurement.get("signed_response")
    if not isinstance(response, (int, float)) or not math.isfinite(response):
        raise ValueError("Probe signed response must be finite.")
    invalid_reasons = measurement.get("invalid_reasons")
    if not isinstance(invalid_reasons, list):
        raise ValueError("Probe invalid_reasons must be a list.")
    if measurement["valid"] and invalid_reasons:
        raise ValueError("A valid probe cannot contain invalid reasons.")
    if not measurement["valid"] and not invalid_reasons:
        raise ValueError("An invalid probe must contain at least one invalid reason.")


def summarize_calibration_records(records):
    if not isinstance(records, list):
        raise ValueError("Calibration records must be a list.")
    identities = [_identity(record) for record in records]
    expected = {
        (seed, damping)
        for damping in CALIBRATION_DAMPINGS
        for seed in CALIBRATION_SEEDS
    }
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("Calibration records must contain the exact planned identities.")

    valid_values = []
    invalid_reason_counts = {}
    by_condition = {
        str(damping): {"attempted": 0, "valid": 0, "invalid": 0}
        for damping in CALIBRATION_DAMPINGS
    }
    for record in records:
        _validate_record(record)
        condition_key = str(float(record["configured_damping"]))
        counts = by_condition[condition_key]
        counts["attempted"] += 1
        measurement = record["probe_measurement"]
        if measurement["valid"]:
            counts["valid"] += 1
            valid_values.append(float(measurement["signed_response"]))
        else:
            counts["invalid"] += 1
            for reason in measurement["invalid_reasons"]:
                invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1

    valid_x_sorted = sorted(valid_values)
    candidate_median = (
        None if not valid_x_sorted else float(np.median(np.asarray(valid_x_sorted)))
    )
    enough_valid = len(valid_x_sorted) >= MINIMUM_VALID_PROBES
    return {
        "status": "succeeded" if enough_valid else "patch",
        "claim_boundary": (
            "Threshold-selection evidence only; no formal evaluation or effectiveness claim."
        ),
        "planned_attempt_count": len(expected),
        "observed_attempt_count": len(records),
        "valid_count": len(valid_x_sorted),
        "invalid_count": len(records) - len(valid_x_sorted),
        "minimum_valid_probes": MINIMUM_VALID_PROBES,
        "valid_x_sorted": valid_x_sorted,
        "candidate_median": candidate_median,
        "tau": candidate_median if enough_valid else None,
        "by_condition": by_condition,
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
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


def run_calibration_attempt(
    policy,
    device,
    environment_seed,
    configured_damping,
    probe_start,
    probe_length,
    target_offset,
    step_log_path,
):
    policy_seed = POLICY_SEED_OFFSET + environment_seed
    max_steps = probe_start + probe_length
    env = gym.make(
        ENV_ID,
        obs_type=OBS_TYPE,
        render_mode=RENDER_MODE,
        damping=configured_damping,
        max_episode_steps=max_steps,
    )
    started_at = time.perf_counter()

    try:
        set_global_seed(policy_seed)
        policy.reset()
        observation, _ = env.reset(seed=environment_seed)
        validate_observation(observation, env.observation_space, "Reset observation")
        current_info = env.unwrapped._get_info()
        current_coverage = float(env.unwrapped._get_coverage())
        initial_state = compact_state(observation, current_info, current_coverage)
        initial_state_sha256 = canonical_sha256(initial_state)

        prefix_actions = []
        probe_plan = None
        probe_positions = []
        probe_contact_steps = 0
        pre_probe_state_sha256 = None
        post_probe_state_sha256 = None
        logged_step_count = 0

        with step_log_path.open("x", encoding="utf-8") as step_log:
            for step in range(1, max_steps + 1):
                state_before = compact_state(
                    observation,
                    current_info,
                    current_coverage,
                )
                inference_seconds = 0.0
                if step <= probe_start:
                    action, inference_seconds = policy_action(policy, observation, device)
                    executed_action = action.astype(env.action_space.dtype, copy=False)
                    phase = "prefix"
                else:
                    phase = "probe"
                    if probe_plan is None:
                        probe_plan = build_probe_plan(
                            agent_position=observation["agent_pos"],
                            block_position=np.asarray(current_info["block_pose"])[:2],
                            action_low=env.action_space.low,
                            action_high=env.action_space.high,
                            target_offset=target_offset,
                            probe_length=probe_length,
                        )
                        pre_probe_state_sha256 = canonical_sha256(state_before)
                        probe_positions = [
                            np.asarray(current_info["block_pose"], dtype=np.float64)[:2]
                        ]
                    probe_index = step - probe_start - 1
                    executed_action = probe_plan.actions[probe_index].astype(
                        env.action_space.dtype,
                        copy=False,
                    )

                executed_action = np.asarray(executed_action, dtype=env.action_space.dtype)
                if not np.all(np.isfinite(executed_action)):
                    raise ValueError(
                        f"Non-finite calibration action at seed={environment_seed} "
                        f"damping={configured_damping} step={step}."
                    )
                if not env.action_space.contains(executed_action):
                    raise ValueError(
                        f"Invalid calibration action at seed={environment_seed} "
                        f"damping={configured_damping} step={step}: {executed_action}"
                    )

                next_observation, reward, terminated, truncated, info = env.step(
                    executed_action
                )
                validate_observation(
                    next_observation,
                    env.observation_space,
                    f"Calibration observation at seed={environment_seed} step={step}",
                )
                coverage = float(info["coverage"])
                if not np.all(np.isfinite([reward, coverage])):
                    raise ValueError(
                        f"Non-finite reward/coverage at seed={environment_seed} step={step}."
                    )
                state_after = compact_state(next_observation, info, coverage)

                if phase == "prefix":
                    prefix_actions.append(executed_action.copy())
                else:
                    probe_positions.append(
                        np.asarray(info["block_pose"], dtype=np.float64)[:2]
                    )
                    probe_contact_steps += int(int(info["n_contacts"]) > 0)

                step_log.write(
                    json.dumps(
                        {
                            "environment_seed": environment_seed,
                            "policy_seed": policy_seed,
                            "configured_damping": configured_damping,
                            "step": step,
                            "phase": phase,
                            "controller_visible": {
                                "ordinary_observation": {
                                    "agent_pos": np.asarray(
                                        observation["agent_pos"]
                                    ).tolist(),
                                    "pixels_sha256": state_before["pixels_sha256"],
                                }
                            },
                            "executed_target": executed_action.tolist(),
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
                logged_step_count += 1

                current_info = dict(info)
                current_coverage = coverage
                observation = next_observation

                if phase == "probe" and step == max_steps:
                    post_probe_state_sha256 = canonical_sha256(state_after)
                elif terminated or truncated:
                    raise RuntimeError(
                        f"Calibration attempt ended before probe measurement at "
                        f"seed={environment_seed} damping={configured_damping} step={step}."
                    )

        measurement = measure_probe_response(
            block_positions=probe_positions,
            probe_direction=probe_plan.direction,
            direction_valid=probe_plan.direction_valid,
            contact_steps=probe_contact_steps,
            measurement_source=PRIVILEGED_MEASUREMENT_SOURCE,
        )
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
            "calibration_only": True,
            "environment_seed": environment_seed,
            "policy_seed": policy_seed,
            "configured_damping": configured_damping,
            "space_damping": env.unwrapped.space.damping,
            "prefix_steps": probe_start,
            "probe_steps": probe_length,
            "logged_step_count": logged_step_count,
            "initial_state_sha256": initial_state_sha256,
            "prefix_action_sha256": probe_array_sha256(np.asarray(prefix_actions)),
            "pre_probe_state_sha256": pre_probe_state_sha256,
            "post_probe_state_sha256": post_probe_state_sha256,
            "probe_action_sha256": probe_plan.sha256,
            "probe_measurement": measurement_record,
            "runtime_seconds": time.perf_counter() - started_at,
            "step_log": f"steps/{step_log_path.name}",
            "step_log_sha256": file_sha256(step_log_path),
        }
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate the frozen PushT probe-response threshold."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--compat-config-dir", type=Path, required=True)
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

    manifest = build_calibration_manifest()
    validate_calibration_manifest(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    steps_directory = args.output_dir / "steps"
    steps_directory.mkdir()

    script_path = Path(__file__).resolve()
    wrapper_path = script_path.with_name("probe_adjust_wrapper.py")
    protocol_path = script_path.with_name(
        "2026-08-30-probe-adjust-wrapper-protocol-v0.2.md"
    )
    weights_path = args.checkpoint_dir / "model.safetensors"
    compat_path = args.compat_config_dir / "config.json"
    manifest_path = args.output_dir / "manifest.json"
    config_path = args.output_dir / "config.json"
    attempts_path = args.output_dir / "attempts.jsonl"
    summary_path = args.output_dir / "summary.json"
    device = torch.device(args.device)

    manifest_document = {
        "calibration_version": CALIBRATION_VERSION,
        "seeds": list(CALIBRATION_SEEDS),
        "evaluation_seeds_excluded": list(EVALUATION_SEEDS),
        "dampings": list(CALIBRATION_DAMPINGS),
        "probe_start": DEFAULT_PROBE_START,
        "probe_length": DEFAULT_PROBE_LENGTH,
        "target_offset": DEFAULT_TARGET_OFFSET,
        "minimum_valid_probes": MINIMUM_VALID_PROBES,
        "attempts": manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = {
        "status": "started",
        "calibration_only": True,
        "claim_boundary": (
            "Threshold-selection evidence only; excludes evaluation and effectiveness claims."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_version": CALIBRATION_VERSION,
        "environment_id": ENV_ID,
        "observation_type": OBS_TYPE,
        "device": str(device),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_weights_sha256": file_sha256(weights_path),
        "compat_config_sha256": file_sha256(compat_path),
        "protocol_sha256": file_sha256(protocol_path),
        "wrapper_sha256": file_sha256(wrapper_path),
        "calibration_runner_sha256": file_sha256(script_path),
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
        with attempts_path.open("x", encoding="utf-8") as attempts_log:
            for item in manifest:
                seed = item["environment_seed"]
                damping = item["configured_damping"]
                step_log_path = steps_directory / (
                    f"seed-{seed}-damping-{_condition_token(damping)}.jsonl"
                )
                record = run_calibration_attempt(
                    policy=policy,
                    device=device,
                    environment_seed=seed,
                    configured_damping=damping,
                    probe_start=DEFAULT_PROBE_START,
                    probe_length=DEFAULT_PROBE_LENGTH,
                    target_offset=DEFAULT_TARGET_OFFSET,
                    step_log_path=step_log_path,
                )
                records.append(record)
                attempts_log.write(json.dumps(record, sort_keys=True) + "\n")
                attempts_log.flush()
                print(
                    f"seed={seed} damping={damping} "
                    f"valid={record['probe_measurement']['valid']} "
                    f"x={record['probe_measurement']['signed_response']:.6f}",
                    flush=True,
                )

        summary = summarize_calibration_records(records)
        summary.update(
            {
                "policy_load_seconds": policy_load_seconds,
                "manifest_sha256": file_sha256(manifest_path),
                "attempts_sha256": file_sha256(attempts_path),
                "config_identity_sha256": canonical_sha256(
                    {
                        key: config[key]
                        for key in (
                            "calibration_version",
                            "device",
                            "manifest_sha256",
                            "checkpoint_weights_sha256",
                            "compat_config_sha256",
                            "protocol_sha256",
                            "wrapper_sha256",
                            "calibration_runner_sha256",
                        )
                    }
                ),
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config["status"] = summary["status"]
        config["summary_sha256"] = file_sha256(summary_path)
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
