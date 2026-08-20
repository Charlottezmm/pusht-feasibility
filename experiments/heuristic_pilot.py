import gymnasium as gym
import gym_pusht  # Registers gym_pusht/PushT-v0 with Gymnasium.

ENV_ID = "gym_pusht/PushT-v0"
SEED = 0
PILOT_STEPS = 20
DAMPING = 1.0
OBS_TYPE = "state"
RENDER_MODE = "rgb_array"


def block_chasing_policy(observation):
    return observation[2:4].astype("float32")


def main():
    env = gym.make(
        ENV_ID,
        obs_type=OBS_TYPE,
        render_mode=RENDER_MODE,
        damping=DAMPING,
    )

    try:
        observation, info = env.reset(seed=SEED)
        initial_observation = observation.copy()
        initial_coverage = env.unwrapped._get_coverage()
        max_coverage = initial_coverage

        episode_return = 0.0
        contact_steps = 0
        steps = 0
        stop_reason = "pilot_step_limit"

        print("initial_observation", initial_observation)
        print("initial_coverage", initial_coverage)

        for step in range(1, PILOT_STEPS + 1):
            action = block_chasing_policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            image = env.render()

            steps = step
            episode_return += reward
            max_coverage = max(max_coverage, info["coverage"])
            contact_steps += int(info["n_contacts"] > 0)

            print(
                "step", step,
                "action", action,
                "reward", reward,
                "coverage", info["coverage"],
                "contacts", info["n_contacts"],
                "terminated", terminated,
                "truncated", truncated,
            )

            if terminated:
                stop_reason = "terminated"
                break
            if truncated:
                stop_reason = "truncated"
                break

        final_coverage = info["coverage"]
        coverage_change = final_coverage - initial_coverage

        print("env_id", ENV_ID)
        print("seed", SEED)
        print("pilot_steps", PILOT_STEPS)
        print("damping", DAMPING)
        print("obs_type", OBS_TYPE)
        print("render_mode", RENDER_MODE)
        print("policy", block_chasing_policy.__name__)
        print("steps", steps)
        print("episode_return", episode_return)
        print("initial_coverage", initial_coverage)
        print("final_coverage", final_coverage)
        print("coverage_change", coverage_change)
        print("max_coverage", max_coverage)
        print("contact_steps", contact_steps)
        print("success", info["is_success"])
        print("stop_reason", stop_reason)
        print("render_shape", image.shape)
        print("render_dtype", image.dtype)
    finally:
        env.close()
        print("closed")


if __name__ == "__main__":
    main()
