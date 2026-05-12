from __future__ import annotations

import argparse

from .discretizer import discretize_observation
from .q_table_agent import QTableAgent
from .train import make_env
from .utils import load_config


def evaluate(config_path: str) -> None:
    config = load_config(config_path)

    env = make_env(config)
    agent = QTableAgent(
        n_actions=int(env.action_space.n),
        alpha=float(config["alpha"]),
        gamma=float(config["gamma"]),
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
    )
    agent.load(config["q_table_path"])
    agent.epsilon = 0.0

    obs = env.reset()
    state = discretize_observation(obs)
    total_reward = 0.0
    collisions = 0
    max_steps = int(config["max_steps_per_episode"])

    for _ in range(max_steps):
        action = agent.select_action(state)
        next_obs, reward, done, info = env.step(action)
        state = discretize_observation(next_obs)
        total_reward += float(reward)

        if isinstance(info, dict):
            collisions += int(info.get("collision", 0))

        if done:
            break

    print(f"Evaluation total reward: {total_reward}")
    if collisions:
        print(f"Collision-related metric from info dict: {collisions}")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tabular Q-learning policy for ns3-gym cognitive radio")
    parser.add_argument("--config", default="configs/qlearning.yaml", help="Path to YAML config")
    args = parser.parse_args()

    evaluate(args.config)


if __name__ == "__main__":
    main()
