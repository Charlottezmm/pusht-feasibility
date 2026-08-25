import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import gym_pusht  # Registers gym_pusht/PushT-v0 with Gymnasium.
import numpy as np


ENV_ID = "gym_pusht/PushT-v0"
EVALUATOR_VERSION = "0.2"
DAMPING = 1.0
OBS_TYPE = "state"
RENDER_MODE = "rgb_array"
DEFAULT_MAX_STEPS = 300
RANDOM_ACTION_SEED_OFFSET = 100_000
POLICIES = ("random", "block_chasing")


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


def get_git_state():
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "worktree_dirty": dirty}


def package_versions():
    names = ("gym-pusht", "gymnasium", "numpy", "pymunk", "shapely")
    return {name: importlib.metadata.version(name) for name in names}


def make_action(policy_name, observation, env):
    if policy_name == "random":
        return env.action_space.sample(), False
    if policy_name == "block_chasing":
        block_center = np.asarray(observation[2:4], dtype=env.action_space.dtype)
        action = np.clip(
            block_center,
            env.action_space.low,
            env.action_space.high,
        ).astype(env.action_space.dtype, copy=False)
        return action, bool(not np.array_equal(action, block_center))
    raise ValueError(f"Unknown policy: {policy_name}")


def validate_observation_structure(observation, observation_space, context):
    observation_array = np.asarray(observation)
    if observation_array.shape != observation_space.shape:
        raise ValueError(
            f"{context} has shape {observation_array.shape}; "
            f"expected {observation_space.shape}."
        )
    if not np.all(np.isfinite(observation_array)):
        raise ValueError(f"{context} contains non-finite values: {observation_array}")


def run_episode(policy_name, environment_seed, max_steps, steps_directory):
    action_seed = RANDOM_ACTION_SEED_OFFSET + environment_seed
    env = gym.make(
        ENV_ID,
        obs_type=OBS_TYPE,
        render_mode=RENDER_MODE,
        damping=DAMPING,
    )
    step_log_path = steps_directory / f"{policy_name}-seed-{environment_seed}.jsonl"

    try:
        observation, reset_info = env.reset(seed=environment_seed)
        env.action_space.seed(action_seed)
        validate_observation_structure(
            observation, env.observation_space, "Reset observation"
        )

        initial_observation = observation.copy()
        initial_observation_in_declared_space = bool(
            env.observation_space.contains(observation)
        )
        initial_coverage = float(env.unwrapped._get_coverage())
        episode_return = 0.0
        max_coverage = initial_coverage
        contact_steps = 0
        stop_reason = "step_budget"
        final_info = dict(reset_info)
        steps = 0
        observation_space_violation_steps = []
        projected_action_steps = []

        with step_log_path.open("x", encoding="utf-8") as step_log:
            for step in range(1, max_steps + 1):
                observation_in_declared_space = bool(
                    env.observation_space.contains(observation)
                )
                action, action_was_projected = make_action(
                    policy_name, observation, env
                )
                if action_was_projected:
                    projected_action_steps.append(step)
                if not env.action_space.contains(action):
                    raise ValueError(
                        f"Policy {policy_name} produced invalid action at step {step}: {action}"
                    )

                next_observation, reward, terminated, truncated, info = env.step(action)
                validate_observation_structure(
                    next_observation,
                    env.observation_space,
                    f"Environment observation at step {step}",
                )
                next_observation_in_declared_space = bool(
                    env.observation_space.contains(next_observation)
                )
                if not next_observation_in_declared_space:
                    observation_space_violation_steps.append(step)

                coverage = float(info["coverage"])
                contacts = int(info["n_contacts"])
                episode_return += float(reward)
                max_coverage = max(max_coverage, coverage)
                contact_steps += int(contacts > 0)
                steps = step
                final_info = dict(info)

                step_log.write(
                    json.dumps(
                        {
                            "step": step,
                            "observation": observation.tolist(),
                            "observation_in_declared_space": observation_in_declared_space,
                            "action": action.tolist(),
                            "action_was_projected": action_was_projected,
                            "next_observation": next_observation.tolist(),
                            "next_observation_in_declared_space": (
                                next_observation_in_declared_space
                            ),
                            "reward": float(reward),
                            "coverage": coverage,
                            "contacts": contacts,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                observation = next_observation

                if terminated:
                    stop_reason = "terminated_success"
                    break
                if truncated:
                    stop_reason = "environment_truncated"
                    break

        final_coverage = float(final_info["coverage"])
        return {
            "policy": policy_name,
            "environment_seed": environment_seed,
            "action_seed": action_seed if policy_name == "random" else None,
            "steps": steps,
            "initial_observation": initial_observation.tolist(),
            "initial_observation_in_declared_space": (
                initial_observation_in_declared_space
            ),
            "observation_space_violation_count": len(
                observation_space_violation_steps
            ),
            "observation_space_violation_steps": observation_space_violation_steps,
            "projected_action_count": len(projected_action_steps),
            "projected_action_steps": projected_action_steps,
            "initial_coverage": initial_coverage,
            "final_coverage": final_coverage,
            "coverage_change": final_coverage - initial_coverage,
            "max_coverage": max_coverage,
            "episode_return": episode_return,
            "contact_steps": contact_steps,
            "success": bool(final_info["is_success"]),
            "stop_reason": stop_reason,
            "step_log": str(step_log_path),
        }
    finally:
        env.close()


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(episodes, seeds, policies):
    by_policy = {}
    for policy_name in policies:
        rows = [row for row in episodes if row["policy"] == policy_name]
        by_policy[policy_name] = {
            "episode_count": len(rows),
            "mean_initial_coverage": mean([row["initial_coverage"] for row in rows]),
            "mean_final_coverage": mean([row["final_coverage"] for row in rows]),
            "mean_coverage_change": mean([row["coverage_change"] for row in rows]),
            "mean_max_coverage": mean([row["max_coverage"] for row in rows]),
            "mean_episode_return": mean([row["episode_return"] for row in rows]),
            "mean_contact_steps": mean([row["contact_steps"] for row in rows]),
            "observation_space_violation_count": sum(
                row["observation_space_violation_count"] for row in rows
            ),
            "episodes_with_observation_space_violations": sum(
                row["observation_space_violation_count"] > 0 for row in rows
            ),
            "projected_action_count": sum(
                row["projected_action_count"] for row in rows
            ),
            "episodes_with_projected_actions": sum(
                row["projected_action_count"] > 0 for row in rows
            ),
            "success_count": sum(row["success"] for row in rows),
            "success_rate": mean([row["success"] for row in rows]),
        }

    paired_final_coverage = []
    if set(policies) == set(POLICIES):
        row_lookup = {
            (row["environment_seed"], row["policy"]): row for row in episodes
        }
        for seed in seeds:
            random_final = row_lookup[(seed, "random")]["final_coverage"]
            heuristic_final = row_lookup[(seed, "block_chasing")]["final_coverage"]
            paired_final_coverage.append(
                {
                    "environment_seed": seed,
                    "random": random_final,
                    "block_chasing": heuristic_final,
                    "block_chasing_minus_random": heuristic_final - random_final,
                }
            )

    return {
        "by_policy": by_policy,
        "paired_final_coverage": paired_final_coverage,
        "descriptive_h1_supported": (
            by_policy["block_chasing"]["mean_final_coverage"]
            > by_policy["random"]["mean_final_coverage"]
            if set(policies) == set(POLICIES)
            else None
        ),
        "interpretation_boundary": (
            "Descriptive result for the declared seeds and configuration; "
            "not a population-level or statistical-generalization claim."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate paired seeded-random and block-chasing PushT baselines."
    )
    parser.add_argument(
        "--seeds",
        default="0-9",
        help="Comma-separated seeds or inclusive ranges, for example 0 or 0-9.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
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

    args.output_dir.mkdir(parents=True, exist_ok=False)
    steps_directory = args.output_dir / "steps"
    steps_directory.mkdir()

    script_path = Path(__file__).resolve()
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "environment_id": ENV_ID,
        "environment_arguments": {
            "obs_type": OBS_TYPE,
            "render_mode": RENDER_MODE,
            "damping": DAMPING,
        },
        "environment_seeds": seeds,
        "random_action_seed_rule": (
            f"action_seed = {RANDOM_ACTION_SEED_OFFSET} + environment_seed"
        ),
        "policies": args.policies,
        "block_chasing_action_rule": (
            "Cast observation[2:4] to the action dtype, then project each coordinate "
            "to the nearest value inside the declared action-space bounds."
        ),
        "max_steps": args.max_steps,
        "primary_metric": "mean final_coverage",
        "auxiliary_metrics": [
            "success_rate",
            "mean max_coverage",
            "mean episode_return",
            "mean contact_steps",
            "observation-space violation count",
            "projected block-chasing action count",
        ],
        "observation_validation": (
            "Wrong-shaped or non-finite observations abort the run. Values outside the "
            "environment's declared Box are preserved and explicitly counted because "
            "gym-pusht 0.1.6 can transiently return such finite physical states."
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": get_git_state(),
        "evaluator_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    episodes = []
    episodes_path = args.output_dir / "episodes.jsonl"
    with episodes_path.open("x", encoding="utf-8") as episodes_log:
        for seed in seeds:
            for policy_name in args.policies:
                result = run_episode(
                    policy_name=policy_name,
                    environment_seed=seed,
                    max_steps=args.max_steps,
                    steps_directory=steps_directory,
                )
                episodes.append(result)
                episodes_log.write(json.dumps(result, sort_keys=True) + "\n")
                print(
                    f"seed={seed} policy={policy_name} "
                    f"final_coverage={result['final_coverage']:.6f} "
                    f"success={result['success']} steps={result['steps']}"
                )

    summary = aggregate(episodes, seeds, args.policies)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["by_policy"], indent=2, sort_keys=True))
    print(f"descriptive_h1_supported={summary['descriptive_h1_supported']}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
