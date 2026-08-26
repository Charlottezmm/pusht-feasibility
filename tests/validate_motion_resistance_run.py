import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_LEVELS = {
    "low_resistance": 1.0,
    "medium_resistance": 0.7,
    "high_resistance": 0.4,
}
REFERENCE_LEVEL = "low_resistance"
ATOL = 1e-12


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def action_hash(actions):
    action_array = np.asarray(actions, dtype="<f4")
    return hashlib.sha256(action_array.tobytes()).hexdigest()


def assert_close(actual, expected, context):
    if not np.isclose(actual, expected, rtol=0.0, atol=ATOL):
        raise AssertionError(f"{context}: actual={actual}, expected={expected}")


def validate_episode(output_dir, episode):
    steps = read_jsonl(output_dir / episode["step_log"])
    if len(steps) != episode["steps"]:
        raise AssertionError(
            f"{episode['level']} seed {episode['environment_seed']} step count mismatch"
        )
    if not steps:
        raise AssertionError("Episode has no step rows")

    actions = [row["action"] for row in steps]
    applied_hash = action_hash(actions)
    planned_hash = episode["planned_action_sha256"]
    if len(steps) == episode["planned_action_count"] and applied_hash != planned_hash:
        raise AssertionError("Complete episode action hash does not match planned hash")

    assert_close(
        sum(row["reward"] for row in steps),
        episode["episode_return"],
        "episode_return",
    )
    assert_close(steps[-1]["coverage"], episode["final_coverage"], "final_coverage")
    assert_close(
        max([episode["initial_coverage"]] + [row["coverage"] for row in steps]),
        episode["max_coverage"],
        "max_coverage",
    )
    if sum(row["contacts"] > 0 for row in steps) != episode["contact_steps"]:
        raise AssertionError("contact_steps mismatch")
    if (
        sum(not row["next_observation_in_declared_space"] for row in steps)
        != episode["observation_space_violation_count"]
    ):
        raise AssertionError("observation-space violation count mismatch")
    np.testing.assert_allclose(
        steps[-1]["next_observation"][2:4],
        episode["final_block_position"],
        rtol=0.0,
        atol=ATOL,
    )
    return steps


def recompute_summary(episodes, grouped_steps, seeds):
    by_level = {}
    for level_name, damping in EXPECTED_LEVELS.items():
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

    paired = []
    for seed in seeds:
        reference_episode = next(
            row
            for row in episodes
            if row["environment_seed"] == seed and row["level"] == REFERENCE_LEVEL
        )
        reference_steps = grouped_steps[(seed, REFERENCE_LEVEL)]
        reference = np.asarray(
            [reference_episode["initial_observation"][2:4]]
            + [row["next_observation"][2:4] for row in reference_steps],
            dtype=np.float64,
        )
        for level_name, damping in EXPECTED_LEVELS.items():
            if level_name == REFERENCE_LEVEL:
                continue
            candidate_episode = next(
                row
                for row in episodes
                if row["environment_seed"] == seed and row["level"] == level_name
            )
            candidate_steps = grouped_steps[(seed, level_name)]
            candidate = np.asarray(
                [candidate_episode["initial_observation"][2:4]]
                + [row["next_observation"][2:4] for row in candidate_steps],
                dtype=np.float64,
            )
            shared = min(len(reference), len(candidate))
            distances = np.linalg.norm(candidate[:shared] - reference[:shared], axis=1)
            paired.append(
                {
                    "environment_seed": seed,
                    "level": level_name,
                    "damping": damping,
                    "shared_position_count": shared,
                    "mean_block_trajectory_distance": mean(distances),
                    "max_block_trajectory_distance": float(np.max(distances)),
                    "final_block_position_distance": float(
                        np.linalg.norm(candidate[-1] - reference[-1])
                    ),
                }
            )
    return by_level, paired


def compare_nested(actual, expected, context):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{context} keys differ")
        for key in expected:
            compare_nested(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{context} lengths differ")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_nested(actual_item, expected_item, f"{context}[{index}]")
    elif isinstance(expected, float):
        assert_close(actual, expected, context)
    elif actual != expected:
        raise AssertionError(f"{context}: actual={actual}, expected={expected}")


def validate_run(output_dir):
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(output_dir / "episodes.jsonl")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    seeds = config["environment_seeds"]
    expected_count = len(seeds) * len(EXPECTED_LEVELS)
    if len(episodes) != expected_count:
        raise AssertionError(
            f"Expected {expected_count} episodes, found {len(episodes)}"
        )

    grouped_steps = {}
    for seed in seeds:
        seed_rows = [row for row in episodes if row["environment_seed"] == seed]
        if {row["level"]: row["damping"] for row in seed_rows} != EXPECTED_LEVELS:
            raise AssertionError(f"Seed {seed} damping levels differ from contract")
        initial_observations = [row["initial_observation"] for row in seed_rows]
        for observation in initial_observations[1:]:
            np.testing.assert_array_equal(observation, initial_observations[0])
        planned_hashes = {row["planned_action_sha256"] for row in seed_rows}
        if len(planned_hashes) != 1:
            raise AssertionError(f"Seed {seed} planned action hashes differ")

        step_rows = {}
        for episode in seed_rows:
            steps = validate_episode(output_dir, episode)
            grouped_steps[(seed, episode["level"])] = steps
            step_rows[episode["level"]] = steps
        shortest = min(len(rows) for rows in step_rows.values())
        reference_actions = [
            row["action"] for row in step_rows[REFERENCE_LEVEL][:shortest]
        ]
        for level_name, rows in step_rows.items():
            candidate_actions = [row["action"] for row in rows[:shortest]]
            np.testing.assert_array_equal(candidate_actions, reference_actions)

    expected_by_level, expected_paired = recompute_summary(
        episodes,
        grouped_steps,
        seeds,
    )
    compare_nested(summary["by_level"], expected_by_level, "by_level")
    compare_nested(
        summary["paired_vs_low_resistance"],
        expected_paired,
        "paired_vs_low_resistance",
    )
    print(
        json.dumps(
            {
                "status": "valid",
                "episode_count": len(episodes),
                "seed_count": len(seeds),
                "levels": EXPECTED_LEVELS,
            },
            sort_keys=True,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_run(args.output_dir)
