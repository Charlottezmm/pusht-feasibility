import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import gym_pusht  # Registers gym_pusht/PushT-v0 with Gymnasium.
import numpy as np
import torch

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import set_global_seed
from lerobot.configs.policies import PreTrainedConfig


ENV_ID = "gym_pusht/PushT-v0"
EVALUATOR_VERSION = "0.1"
OBS_TYPE = "pixels_agent_pos"
RENDER_MODE = "rgb_array"
DEFAULT_MAX_STEPS = 300
POLICY_SEED_OFFSET = 100_000
SUCCESS_COVERAGE_THRESHOLD = 0.95
STRONG_MEDIAN_MAX_COVERAGE_THRESHOLD = 0.5
NUMERICAL_TOLERANCE = 1e-6


def parse_seed_spec(seed_spec):
    seeds = []
    for item in seed_spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending seed range: {item}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed specification contains duplicates.")
    return seeds


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_state():
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_paths = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "worktree_dirty": bool(dirty_paths), "dirty_paths": dirty_paths}


def package_versions():
    names = (
        "lerobot",
        "torch",
        "torchvision",
        "diffusers",
        "huggingface-hub",
        "gym-pusht",
        "gymnasium",
        "numpy",
        "pymunk",
        "safetensors",
        "shapely",
    )
    return {name: importlib.metadata.version(name) for name in names}


def validate_observation(observation, observation_space, context):
    if set(observation) != {"agent_pos", "pixels"}:
        raise ValueError(f"{context} has unexpected keys: {sorted(observation)}")
    if not observation_space.contains(observation):
        raise ValueError(f"{context} is outside the declared observation space.")
    if not np.all(np.isfinite(observation["agent_pos"])):
        raise ValueError(f"{context} contains non-finite agent_pos values.")


def compact_state(observation, info, coverage):
    pixels = np.asarray(observation["pixels"])
    return {
        "agent_pos": np.asarray(observation["agent_pos"]).tolist(),
        "block_pose": np.asarray(info["block_pose"]).tolist(),
        "goal_pose": np.asarray(info["goal_pose"]).tolist(),
        "coverage": float(coverage),
        "pixels_shape": list(pixels.shape),
        "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
    }


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(episodes):
    max_coverages = [row["max_coverage"] for row in episodes]
    initial_coverages = [row["initial_coverage"] for row in episodes]
    success_count = sum(row["success"] for row in episodes)
    max_coverage_range = float(max(max_coverages) - min(max_coverages))
    improved_above_initial = any(
        maximum > initial + NUMERICAL_TOLERANCE
        for maximum, initial in zip(max_coverages, initial_coverages, strict=True)
    )
    non_floor_variation = (
        improved_above_initial and max_coverage_range > NUMERICAL_TOLERANCE
    )
    median_max_coverage = float(np.median(np.asarray(max_coverages, dtype=np.float64)))
    return {
        "episode_count": len(episodes),
        "mean_initial_coverage": mean(initial_coverages),
        "mean_max_coverage": mean(max_coverages),
        "median_max_coverage": median_max_coverage,
        "mean_final_coverage": mean([row["final_coverage"] for row in episodes]),
        "max_coverage_range": max_coverage_range,
        "improved_above_initial": improved_above_initial,
        "non_floor_task_performance_variation": non_floor_variation,
        "success_count": success_count,
        "success_rate": mean([row["success"] for row in episodes]),
        "strong_entry_gate_pass": bool(
            success_count >= 1
            or median_max_coverage >= STRONG_MEDIAN_MAX_COVERAGE_THRESHOLD
        ),
        "all_step_logs_complete": all(
            row["logged_step_count"] == row["steps"] for row in episodes
        ),
        "all_actions_valid": all(row["invalid_action_count"] == 0 for row in episodes),
        "all_values_finite": all(row["non_finite_value_count"] == 0 for row in episodes),
        "metric_roles": {
            "gate_metric": "maximum coverage",
            "endpoint_diagnostic": "final coverage",
            "success_rule": f"coverage > {SUCCESS_COVERAGE_THRESHOLD}",
        },
    }


def load_policy(checkpoint_dir, compat_config_dir, device):
    config = PreTrainedConfig.from_pretrained(
        compat_config_dir,
        local_files_only=True,
    )
    if not isinstance(config, DiffusionConfig):
        raise TypeError(
            f"Expected DiffusionConfig, but compatibility config decoded to {type(config).__name__}."
        )
    policy = DiffusionPolicy.from_pretrained(
        checkpoint_dir,
        config=config,
        local_files_only=True,
        map_location="cpu",
        strict=True,
    )
    policy.to(device)
    policy.eval()
    return policy


def run_episode(policy, device, damping, environment_seed, max_steps, steps_directory):
    policy_seed = POLICY_SEED_OFFSET + environment_seed
    step_log_name = f"damping-{damping:g}-seed-{environment_seed}.jsonl"
    step_log_path = steps_directory / step_log_name
    env = gym.make(
        ENV_ID,
        obs_type=OBS_TYPE,
        render_mode=RENDER_MODE,
        damping=damping,
        max_episode_steps=max_steps,
    )
    started_at = time.perf_counter()

    try:
        set_global_seed(policy_seed)
        policy.reset()
        observation, reset_info = env.reset(seed=environment_seed)
        validate_observation(observation, env.observation_space, "Reset observation")

        initial_coverage = float(env.unwrapped._get_coverage())
        initial_info = env.unwrapped._get_info()
        initial_state = compact_state(observation, initial_info, initial_coverage)
        max_coverage = initial_coverage
        max_reward = float(
            np.clip(initial_coverage / SUCCESS_COVERAGE_THRESHOLD, 0.0, 1.0)
        )
        episode_return = 0.0
        contact_steps = 0
        stop_reason = "step_budget"
        final_info = dict(reset_info)
        final_coverage = initial_coverage
        steps = 0
        logged_step_count = 0
        invalid_action_count = 0
        non_finite_value_count = 0

        with step_log_path.open("x", encoding="utf-8") as step_log:
            for step in range(1, max_steps + 1):
                batched_observation = {
                    key: np.expand_dims(value, axis=0)
                    for key, value in observation.items()
                }
                policy_observation = preprocess_observation(batched_observation)
                policy_observation = {
                    key: value.to(device, non_blocking=True)
                    for key, value in policy_observation.items()
                }

                inference_started_at = time.perf_counter()
                with torch.inference_mode():
                    action_tensor = policy.select_action(policy_observation)
                if device.type == "mps":
                    torch.mps.synchronize()
                inference_seconds = time.perf_counter() - inference_started_at

                action = action_tensor.to("cpu").numpy()[0]
                action = action.astype(env.action_space.dtype, copy=False)
                if not np.all(np.isfinite(action)):
                    non_finite_value_count += 1
                    raise ValueError(f"Policy produced non-finite action at step {step}: {action}")
                if not env.action_space.contains(action):
                    invalid_action_count += 1
                    raise ValueError(f"Policy produced invalid action at step {step}: {action}")

                state_before = compact_state(observation, initial_info if step == 1 else final_info, final_coverage)
                next_observation, reward, terminated, truncated, info = env.step(action)
                validate_observation(
                    next_observation,
                    env.observation_space,
                    f"Environment observation at step {step}",
                )
                coverage = float(info["coverage"])
                numeric_values = np.asarray([reward, coverage], dtype=np.float64)
                if not np.all(np.isfinite(numeric_values)):
                    non_finite_value_count += 1
                    raise ValueError(f"Non-finite reward/coverage at step {step}: {numeric_values}")

                state_after = compact_state(next_observation, info, coverage)
                episode_return += float(reward)
                max_coverage = max(max_coverage, coverage)
                max_reward = max(max_reward, float(reward))
                contact_steps += int(int(info["n_contacts"]) > 0)
                steps = step
                final_info = dict(info)
                final_coverage = coverage

                step_log.write(
                    json.dumps(
                        {
                            "step": step,
                            "state_before": state_before,
                            "action": action.tolist(),
                            "state_after": state_after,
                            "reward": float(reward),
                            "coverage": coverage,
                            "contacts": int(info["n_contacts"]),
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
                observation = next_observation

                if terminated:
                    stop_reason = "terminated_success"
                    break
                if truncated:
                    stop_reason = "environment_truncated"
                    break

        runtime_seconds = time.perf_counter() - started_at
        return {
            "damping": damping,
            "configured_damping": env.unwrapped.damping,
            "space_damping": env.unwrapped.space.damping,
            "environment_seed": environment_seed,
            "policy_seed": policy_seed,
            "device": str(device),
            "steps": steps,
            "logged_step_count": logged_step_count,
            "initial_state": initial_state,
            "final_state": compact_state(observation, final_info, final_coverage),
            "initial_coverage": initial_coverage,
            "max_coverage": max_coverage,
            "final_coverage": final_coverage,
            "coverage_drop_after_maximum": max_coverage - final_coverage,
            "max_reward": max_reward,
            "episode_return": episode_return,
            "contact_steps": contact_steps,
            "success": bool(final_info.get("is_success", False)),
            "stop_reason": stop_reason,
            "runtime_seconds": runtime_seconds,
            "invalid_action_count": invalid_action_count,
            "non_finite_value_count": non_finite_value_count,
            "step_log": f"steps/{step_log_name}",
            "step_log_sha256": file_sha256(step_log_path),
        }
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the pinned LeRobot Diffusion Policy on PushT with auditable logs."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--compat-config-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    parser.add_argument("--damping", type=float, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = parse_seed_spec(args.seeds)
    if args.max_steps <= 0 or args.max_steps > DEFAULT_MAX_STEPS:
        raise ValueError(
            f"max_steps must be between 1 and {DEFAULT_MAX_STEPS}, inclusive."
        )
    if not args.checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
    if not args.compat_config_dir.is_dir():
        raise ValueError(f"Compatibility config directory does not exist: {args.compat_config_dir}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    steps_directory = args.output_dir / "steps"
    steps_directory.mkdir()
    device = torch.device(args.device)
    script_path = Path(__file__).resolve()
    weights_path = args.checkpoint_dir / "model.safetensors"
    config_path = args.compat_config_dir / "config.json"

    config_record = {
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "environment_id": ENV_ID,
        "environment_arguments": {
            "obs_type": OBS_TYPE,
            "render_mode": RENDER_MODE,
            "damping": args.damping,
            "max_episode_steps": args.max_steps,
        },
        "environment_seeds": seeds,
        "policy_seed_rule": "policy_seed = 100000 + environment_seed",
        "device": str(device),
        "weight_load_staging_device": "cpu",
        "use_amp": False,
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_weights_sha256": file_sha256(weights_path),
        "compat_config_dir": str(args.compat_config_dir.resolve()),
        "compat_config_sha256": file_sha256(config_path),
        "max_steps": args.max_steps,
        "gate_definition": {
            "required_episode_count_for_project_reference": 3,
            "gate_metric": "maximum coverage",
            "endpoint_diagnostic": "final coverage",
            "non_floor_variation": (
                "At least one maximum coverage exceeds its initial coverage by more than 1e-6, "
                "and the range of maximum coverage across episodes exceeds 1e-6."
            ),
            "strong_entry_gate": (
                "At least one success, or median maximum coverage >= 0.5."
            ),
        },
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch_mps_built": torch.backends.mps.is_built(),
        "torch_mps_available": torch.backends.mps.is_available(),
        "git": get_git_state(),
        "evaluator_sha256": file_sha256(script_path),
    }
    config_path_out = args.output_dir / "config.json"
    config_path_out.write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        policy_load_started_at = time.perf_counter()
        policy = load_policy(
            args.checkpoint_dir,
            args.compat_config_dir,
            device,
        )
        policy_load_seconds = time.perf_counter() - policy_load_started_at
        parameter = next(policy.parameters())

        episodes = []
        episodes_path = args.output_dir / "episodes.jsonl"
        with episodes_path.open("x", encoding="utf-8") as episodes_log:
            for seed in seeds:
                episode = run_episode(
                    policy=policy,
                    device=device,
                    damping=args.damping,
                    environment_seed=seed,
                    max_steps=args.max_steps,
                    steps_directory=steps_directory,
                )
                episodes.append(episode)
                episodes_log.write(json.dumps(episode, sort_keys=True) + "\n")
                episodes_log.flush()
                print(
                    f"seed={seed} damping={args.damping:g} "
                    f"max_coverage={episode['max_coverage']:.6f} "
                    f"final_coverage={episode['final_coverage']:.6f} "
                    f"success={episode['success']} steps={episode['steps']} "
                    f"runtime_seconds={episode['runtime_seconds']:.3f}",
                    flush=True,
                )

        summary = aggregate(episodes)
        summary.update(
            {
                "status": "succeeded",
                "damping": args.damping,
                "device": str(device),
                "policy_load_seconds": policy_load_seconds,
                "policy_parameter_device": str(parameter.device),
                "policy_parameter_dtype": str(parameter.dtype),
                "policy_parameter_count": sum(p.numel() for p in policy.parameters()),
            }
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_record["status"] = "succeeded"
        config_path_out.write_text(
            json.dumps(config_record, indent=2, sort_keys=True) + "\n",
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
        config_record["status"] = "failed"
        config_path_out.write_text(
            json.dumps(config_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
