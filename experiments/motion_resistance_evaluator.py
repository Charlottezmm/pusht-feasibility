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
EVALUATOR_VERSION = "0.1"
OBS_TYPE = "state"
RENDER_MODE = "rgb_array"
DEFAULT_MAX_STEPS = 300
RANDOM_ACTION_SEED_OFFSET = 100_000
DAMPING_LEVELS = (
    ("low_resistance", 1.0),
    ("medium_resistance", 0.7),
    ("high_resistance", 0.4),
)
REFERENCE_LEVEL = "low_resistance"


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


def validate_observation_structure(observation, observation_space, context):
    observation_array = np.asarray(observation)
    if observation_array.shape != observation_space.shape:
        raise ValueError(
            f"{context} has shape {observation_array.shape}; "
            f"expected {observation_space.shape}."
        )
    if not np.all(np.isfinite(observation_array)):
        raise ValueError(f"{context} contains non-finite values: {observation_array}")


def action_sequence_sha256(actions):
    action_array = np.asarray(actions, dtype="<f4")
    return hashlib.sha256(action_array.tobytes()).hexdigest()


def generate_action_sequence(environment_seed, max_steps):
    action_space = gym.spaces.Box(
        low=0.0,
        high=512.0,
        shape=(2,),
        dtype=np.float32,
    )
    action_seed = RANDOM_ACTION_SEED_OFFSET + environment_seed
    action_space.seed(action_seed)
    actions = [action_space.sample() for _ in range(max_steps)]
    return actions, action_seed, action_sequence_sha256(actions)


def run_episode(
    level_name,
    damping,
    environment_seed,
    actions,
    action_seed,
    action_hash,
    steps_directory,
):
    env = gym.make(
        ENV_ID,
        obs_type=OBS_TYPE,
        render_mode=RENDER_MODE,
        damping=damping,
    )
    step_log_name = f"{level_name}-seed-{environment_seed}.jsonl"
    step_log_path = steps_directory / step_log_name

    try:
        observation, reset_info = env.reset(seed=environment_seed)
        validate_observation_structure(
            observation,
            env.observation_space,
            "Reset observation",
        )

        initial_observation = np.asarray(observation).copy()
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
        block_trajectory = [initial_observation[2:4].tolist()]

        with step_log_path.open("x", encoding="utf-8") as step_log:
            for step, action in enumerate(actions, start=1):
                observation_in_declared_space = bool(
                    env.observation_space.contains(observation)
                )
                if not env.action_space.contains(action):
                    raise ValueError(
                        f"Generated action is invalid at step {step}: {action}"
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
                block_trajectory.append(np.asarray(next_observation)[2:4].tolist())

                step_log.write(
                    json.dumps(
                        {
                            "step": step,
                            "observation": np.asarray(observation).tolist(),
                            "observation_in_declared_space": (
                                observation_in_declared_space
                            ),
                            "action": np.asarray(action).tolist(),
                            "next_observation": np.asarray(next_observation).tolist(),
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

        final_observation = np.asarray(observation)
        final_coverage = float(final_info["coverage"])
        episode = {
            "level": level_name,
            "damping": damping,
            "environment_seed": environment_seed,
            "action_seed": action_seed,
            "planned_action_count": len(actions),
            "planned_action_sha256": action_hash,
            "steps": steps,
            "initial_observation": initial_observation.tolist(),
            "initial_observation_in_declared_space": (
                initial_observation_in_declared_space
            ),
            "final_agent_position": final_observation[0:2].tolist(),
            "final_block_position": final_observation[2:4].tolist(),
            "observation_space_violation_count": len(
                observation_space_violation_steps
            ),
            "observation_space_violation_steps": (
                observation_space_violation_steps
            ),
            "initial_coverage": initial_coverage,
            "final_coverage": final_coverage,
            "coverage_change": final_coverage - initial_coverage,
            "max_coverage": max_coverage,
            "episode_return": episode_return,
            "contact_steps": contact_steps,
            "success": bool(final_info["is_success"]),
            "stop_reason": stop_reason,
            "step_log": f"steps/{step_log_name}",
        }
        return episode, block_trajectory
    finally:
        env.close()


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate(episodes, seeds, trajectories):
    by_level = {}
    for level_name, damping in DAMPING_LEVELS:
        rows = [row for row in episodes if row["level"] == level_name]
        by_level[level_name] = {
            "damping": damping,
            "episode_count": len(rows),
            "mean_initial_coverage": mean(
                [row["initial_coverage"] for row in rows]
            ),
            "mean_final_coverage": mean([row["final_coverage"] for row in rows]),
            "mean_coverage_change": mean(
                [row["coverage_change"] for row in rows]
            ),
            "mean_max_coverage": mean([row["max_coverage"] for row in rows]),
            "mean_episode_return": mean(
                [row["episode_return"] for row in rows]
            ),
            "mean_contact_steps": mean([row["contact_steps"] for row in rows]),
            "observation_space_violation_count": sum(
                row["observation_space_violation_count"] for row in rows
            ),
            "success_count": sum(row["success"] for row in rows),
            "success_rate": mean([row["success"] for row in rows]),
        }

    paired_vs_reference = []
    for seed in seeds:
        reference = np.asarray(
            trajectories[(seed, REFERENCE_LEVEL)],
            dtype=np.float64,
        )
        for level_name, damping in DAMPING_LEVELS:
            if level_name == REFERENCE_LEVEL:
                continue
            candidate = np.asarray(
                trajectories[(seed, level_name)],
                dtype=np.float64,
            )
            shared_positions = min(len(reference), len(candidate))
            distances = np.linalg.norm(
                candidate[:shared_positions] - reference[:shared_positions],
                axis=1,
            )
            paired_vs_reference.append(
                {
                    "environment_seed": seed,
                    "level": level_name,
                    "damping": damping,
                    "shared_position_count": shared_positions,
                    "mean_block_trajectory_distance": mean(distances),
                    "max_block_trajectory_distance": float(np.max(distances)),
                    "final_block_position_distance": float(
                        np.linalg.norm(candidate[-1] - reference[-1])
                    ),
                }
            )

    motion_response_by_level = {}
    for level_name, damping in DAMPING_LEVELS:
        if level_name == REFERENCE_LEVEL:
            continue
        rows = [
            row for row in paired_vs_reference if row["level"] == level_name
        ]
        motion_response_by_level[level_name] = {
            "damping": damping,
            "paired_seed_count": len(rows),
            "mean_block_trajectory_distance": mean(
                [row["mean_block_trajectory_distance"] for row in rows]
            ),
            "mean_final_block_position_distance": mean(
                [row["final_block_position_distance"] for row in rows]
            ),
        }

    return {
        "by_level": by_level,
        "paired_vs_low_resistance": paired_vs_reference,
        "motion_response_by_level": motion_response_by_level,
        "primary_metric": "mean final_coverage",
        "interpretation_boundary": (
            "Coverage metrics describe task overlap for the declared fixed action "
            "sequences. Block-position distances diagnose simulator response "
            "differences; neither establishes adaptation, policy robustness, or "
            "real-world friction."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay identical seeded open-loop actions across three PushT "
            "motion-resistance settings."
        )
    )
    parser.add_argument(
        "--seeds",
        default="0-9",
        help="Comma-separated seeds or inclusive ranges, for example 0,3 or 0-9.",
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

    action_sequences = {}
    action_metadata = {}
    for seed in seeds:
        actions, action_seed, action_hash = generate_action_sequence(
            seed,
            args.max_steps,
        )
        action_sequences[seed] = actions
        action_metadata[str(seed)] = {
            "action_seed": action_seed,
            "planned_action_count": len(actions),
            "planned_action_sha256": action_hash,
        }

    script_path = Path(__file__).resolve()
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "environment_id": ENV_ID,
        "environment_arguments": {
            "obs_type": OBS_TYPE,
            "render_mode": RENDER_MODE,
        },
        "damping_levels": [
            {"level": level_name, "damping": damping}
            for level_name, damping in DAMPING_LEVELS
        ],
        "environment_seeds": seeds,
        "action_sequences": action_metadata,
        "action_rule": (
            "For each environment seed, generate one Box(0, 512, (2,), "
            "float32) sequence with action_seed = 100000 + environment_seed, "
            "then replay the exact arrays in all three damping levels."
        ),
        "max_steps": args.max_steps,
        "primary_metric": "mean final_coverage",
        "auxiliary_metrics": [
            "mean max_coverage",
            "mean coverage_change",
            "mean episode_return",
            "mean contact_steps",
            "success_rate",
            "observation-space violation count",
            "block trajectory distance relative to damping=1.0",
            "final block-position distance relative to damping=1.0",
        ],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": get_git_state(),
        "evaluator_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    episodes = []
    trajectories = {}
    episodes_path = args.output_dir / "episodes.jsonl"
    with episodes_path.open("x", encoding="utf-8") as episodes_log:
        for seed in seeds:
            actions = action_sequences[seed]
            action_seed = action_metadata[str(seed)]["action_seed"]
            action_hash = action_metadata[str(seed)]["planned_action_sha256"]
            paired_initial_observation = None

            for level_name, damping in DAMPING_LEVELS:
                episode, block_trajectory = run_episode(
                    level_name=level_name,
                    damping=damping,
                    environment_seed=seed,
                    actions=actions,
                    action_seed=action_seed,
                    action_hash=action_hash,
                    steps_directory=steps_directory,
                )
                initial_observation = np.asarray(episode["initial_observation"])
                if paired_initial_observation is None:
                    paired_initial_observation = initial_observation
                elif not np.array_equal(
                    initial_observation,
                    paired_initial_observation,
                ):
                    raise ValueError(
                        f"Seed {seed} did not produce identical initial "
                        "observations across damping levels."
                    )

                episodes.append(episode)
                trajectories[(seed, level_name)] = block_trajectory
                episodes_log.write(json.dumps(episode, sort_keys=True) + "\n")
                print(
                    f"seed={seed} level={level_name} damping={damping} "
                    f"final_coverage={episode['final_coverage']:.6f} "
                    f"success={episode['success']} steps={episode['steps']}"
                )

    summary = aggregate(episodes, seeds, trajectories)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["by_level"], indent=2, sort_keys=True))
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
