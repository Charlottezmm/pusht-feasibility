"""Debug-only runtime adapter for the PushT three-path wrapper.

This script connects the frozen Diffusion Policy and PushT environment to the pure wrapper core.
Its single-seed output is interface and logging evidence only, never calibration or effectiveness
evidence.
"""

import argparse
import hashlib
import json
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
    SUCCESS_COVERAGE_THRESHOLD,
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


ADAPTER_VERSION = "0.1"
DEFAULT_MAX_STEPS = 300
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


def step_phase(path, step, probe_start, probe_length):
    if path not in PATHS:
        raise ValueError(f"Unknown wrapper path: {path}")
    if step <= probe_start:
        return "prefix"
    if step <= probe_start + probe_length:
        return "continuation_window" if path == FIXED else "probe"
    return "post_window" if path == FIXED else "post_probe"


def validate_three_path_records(records):
    errors = []
    by_path = {record.get("path"): record for record in records}
    if set(by_path) != set(PATHS) or len(records) != len(PATHS):
        errors.append("records must contain each path exactly once")
        return {"passed": False, "checks": {}, "errors": errors}

    fixed = by_path[FIXED]
    no_adjust = by_path[PROBE_NO_ADJUST]
    adjust = by_path[PROBE_ADJUST]

    def shared(field):
        return len({record.get(field) for record in records}) == 1

    checks = {
        "debug_only": all(record.get("debug_only") is True for record in records),
        "succeeded": all(record.get("status") == "succeeded" for record in records),
        "shared_seed": shared("environment_seed") and shared("policy_seed"),
        "shared_setting": shared("configured_damping"),
        "shared_budget": shared("max_steps"),
        "shared_initial_state": shared("initial_state_sha256"),
        "shared_prefix": shared("prefix_action_sha256"),
        "shared_pre_probe_state": shared("pre_probe_state_sha256"),
        "shared_probe": (
            no_adjust.get("probe_action_sha256") is not None
            and no_adjust.get("probe_action_sha256") == adjust.get("probe_action_sha256")
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
            record.get("logged_step_count") == record.get("steps") for record in records
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
            and adjust.get("response_used_count", 0) > 0
        ),
        "adjustment_branch_exercised": (
            fixed.get("adjusted_scale_count") == 0
            and no_adjust.get("adjusted_scale_count") == 0
            and adjust.get("adjusted_scale_count", 0) > 0
        ),
    }

    messages = {
        "debug_only": "every record must be debug_only",
        "succeeded": "every path must succeed",
        "shared_seed": "seed or policy seed differs",
        "shared_setting": "simulator setting differs",
        "shared_budget": "step budget differs",
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
        "adjustment_branch_exercised": "probe-adjust never executed adjusted scale",
    }
    for check, passed in checks.items():
        if not passed:
            errors.append(messages[check])
    return {"passed": not errors, "checks": checks, "errors": errors}


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
    if device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - started_at
    return action_tensor.to("cpu").numpy()[0], seconds


def run_path_episode(
    policy,
    device,
    path,
    environment_seed,
    configured_damping,
    max_steps,
    probe_start,
    probe_length,
    target_offset,
    debug_tau,
    step_log_path,
):
    policy_seed = POLICY_SEED_OFFSET + environment_seed
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
            for step in range(1, max_steps + 1):
                phase = step_phase(path, step, probe_start, probe_length)
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
                            target_offset=target_offset,
                            probe_length=probe_length,
                        )
                        pre_probe_state_sha256 = canonical_sha256(state_before)
                        probe_positions = [
                            np.asarray(current_info["block_pose"], dtype=np.float64)[:2]
                        ]
                    probe_index = step - probe_start - 1
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
                            tau=debug_tau,
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
                        f"Non-finite executed action at path={path} step={step}: "
                        f"{executed_action}"
                    )
                if not env.action_space.contains(executed_action):
                    invalid_action_count += 1
                    raise ValueError(
                        f"Invalid executed action at path={path} step={step}: {executed_action}"
                    )

                next_observation, reward, terminated, truncated, info = env.step(
                    executed_action
                )
                validate_observation(
                    next_observation,
                    env.observation_space,
                    f"Environment observation at path={path} step={step}",
                )
                coverage = float(info["coverage"])
                if not np.all(np.isfinite([reward, coverage])):
                    non_finite_value_count += 1
                    raise ValueError(
                        f"Non-finite reward/coverage at path={path} step={step}."
                    )
                state_after = compact_state(next_observation, info, coverage)

                if step <= probe_start:
                    prefix_actions.append(executed_action.copy())
                if step == probe_start:
                    pre_probe_state_sha256 = canonical_sha256(state_after)
                if phase == "probe":
                    probe_positions.append(
                        np.asarray(info["block_pose"], dtype=np.float64)[:2]
                    )
                    probe_contact_steps += int(int(info["n_contacts"]) > 0)

                step_log.write(
                    json.dumps(
                        {
                            "path": path,
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

                if phase == "probe" and step == probe_start + probe_length:
                    measurement = measure_probe_response(
                        block_positions=probe_positions,
                        probe_direction=probe_plan.direction,
                        direction_valid=probe_plan.direction_valid,
                        contact_steps=probe_contact_steps,
                        measurement_source=PRIVILEGED_MEASUREMENT_SOURCE,
                    )
                    post_probe_state_sha256 = canonical_sha256(state_after)
                    policy.reset()  # Clear stale pre-probe observations and queued targets.

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
            "debug_only": True,
            "path": path,
            "environment_seed": environment_seed,
            "policy_seed": policy_seed,
            "configured_damping": configured_damping,
            "space_damping": env.unwrapped.space.damping,
            "max_steps": max_steps,
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
            "debug_tau": debug_tau,
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
        description="Run a debug-only one-seed PushT three-path logging smoke."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--compat-config-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    parser.add_argument("--damping", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--probe-start", type=int, default=DEFAULT_PROBE_START)
    parser.add_argument("--probe-length", type=int, default=DEFAULT_PROBE_LENGTH)
    parser.add_argument("--target-offset", type=float, default=DEFAULT_TARGET_OFFSET)
    parser.add_argument("--debug-tau", type=float, required=True)
    parser.add_argument("--debug-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.debug_only:
        raise ValueError("--debug-only is required; this script cannot produce evaluation evidence.")
    if not np.isfinite(args.debug_tau):
        raise ValueError("--debug-tau must be finite.")
    if args.max_steps <= 0 or args.max_steps > DEFAULT_MAX_STEPS:
        raise ValueError(f"max_steps must be between 1 and {DEFAULT_MAX_STEPS}.")
    if args.probe_start < 0 or args.probe_length <= 0:
        raise ValueError("probe_start must be non-negative and probe_length positive.")
    if args.probe_start + args.probe_length >= args.max_steps:
        raise ValueError("probe window must leave at least one post-probe step.")
    if not args.checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
    if not args.compat_config_dir.is_dir():
        raise ValueError(f"Compatibility config directory does not exist: {args.compat_config_dir}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    steps_directory = args.output_dir / "steps"
    steps_directory.mkdir()
    config_path = args.output_dir / "config.json"
    script_path = Path(__file__).resolve()
    wrapper_path = script_path.with_name("probe_adjust_wrapper.py")
    protocol_path = script_path.with_name(
        "2026-08-30-probe-adjust-wrapper-protocol-v0.2.md"
    )
    weights_path = args.checkpoint_dir / "model.safetensors"
    compat_path = args.compat_config_dir / "config.json"
    device = torch.device(args.device)

    config = {
        "status": "started",
        "debug_only": True,
        "claim_boundary": (
            "Interface/logging evidence only; excluded from calibration, evaluation, and "
            "effectiveness claims."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_version": ADAPTER_VERSION,
        "environment_id": ENV_ID,
        "environment_arguments": {
            "obs_type": OBS_TYPE,
            "render_mode": RENDER_MODE,
            "damping": args.damping,
            "max_episode_steps": args.max_steps,
        },
        "environment_seed": args.seed,
        "policy_seed": POLICY_SEED_OFFSET + args.seed,
        "paths": list(PATHS),
        "probe_start": args.probe_start,
        "probe_length": args.probe_length,
        "target_offset": args.target_offset,
        "debug_tau": args.debug_tau,
        "adjusted_scale": DEFAULT_ADJUSTED_SCALE,
        "device": str(device),
        "checkpoint_weights_sha256": file_sha256(weights_path),
        "compat_config_sha256": file_sha256(compat_path),
        "protocol_sha256": file_sha256(protocol_path),
        "wrapper_sha256": file_sha256(wrapper_path),
        "adapter_sha256": file_sha256(script_path),
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
        episodes_path = args.output_dir / "episodes.jsonl"
        with episodes_path.open("x", encoding="utf-8") as episodes_log:
            for path in PATHS:
                record = run_path_episode(
                    policy=policy,
                    device=device,
                    path=path,
                    environment_seed=args.seed,
                    configured_damping=args.damping,
                    max_steps=args.max_steps,
                    probe_start=args.probe_start,
                    probe_length=args.probe_length,
                    target_offset=args.target_offset,
                    debug_tau=args.debug_tau,
                    step_log_path=steps_directory / f"{path}.jsonl",
                )
                records.append(record)
                episodes_log.write(json.dumps(record, sort_keys=True) + "\n")
                episodes_log.flush()
                print(
                    f"path={path} final_coverage={record['final_coverage']:.6f} "
                    f"max_coverage={record['max_coverage']:.6f} "
                    f"success={record['success']} steps={record['steps']} "
                    f"response_used={record['response_used_count']} "
                    f"adjusted_steps={record['adjusted_scale_count']}",
                    flush=True,
                )

        validation = validate_three_path_records(records)
        (args.output_dir / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "status": "succeeded" if validation["passed"] else "failed_validation",
            "debug_only": True,
            "claim_boundary": config["claim_boundary"],
            "policy_load_seconds": policy_load_seconds,
            "validation_passed": validation["passed"],
            "paths": {
                record["path"]: {
                    "steps": record["steps"],
                    "max_coverage": record["max_coverage"],
                    "final_coverage": record["final_coverage"],
                    "success": record["success"],
                    "probe_measurement": record["probe_measurement"],
                    "response_used_count": record["response_used_count"],
                    "adjusted_scale_count": record["adjusted_scale_count"],
                    "runtime_seconds": record["runtime_seconds"],
                }
                for record in records
            },
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config["status"] = summary["status"]
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"output_dir={args.output_dir}", flush=True)
        if not validation["passed"]:
            raise RuntimeError(
                "Three-path debug validation failed: " + "; ".join(validation["errors"])
            )
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
