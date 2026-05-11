from __future__ import annotations

import argparse

from tqdm import trange

from .discretizer import discretize_observation
from .q_table_agent import QTableAgent
from .utils import load_config, set_seed


def make_env():
    """Create ns3-gym environment.

    TODO: Fill exact env args for local cognitive-radio setup once ns3-gym is installed.
    """
    try:
        from ns3gym import ns3env
    except ImportError as exc:
        raise ImportError(
            "ns3gym is not installed. Build/install ns-3 + ns3-gym first (see scripts/setup_ns3gym.sh)."
        ) from exc

    # TODO: Replace placeholder kwargs based on your ns3-gym cognitive radio example wiring.
    return ns3env.Ns3Env()


def train(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["seed"]))

    env = make_env()
    n_actions = int(env.action_space.n)

    agent = QTableAgent(
        n_actions=n_actions,
        alpha=float(config["alpha"]),
        gamma=float(config["gamma"]),
        epsilon_start=float(config["epsilon_start"]),
        epsilon_min=float(config["epsilon_min"]),
        epsilon_decay=float(config["epsilon_decay"]),
    )

    episodes = int(config["episodes"])
    max_steps = int(config["max_steps_per_episode"])

    for _ in trange(episodes, desc="Training"):
        obs = env.reset()
        state = discretize_observation(obs)

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, _info = env.step(action)
            next_state = discretize_observation(next_obs)
            agent.update(state, action, float(reward), next_state, bool(done))
            state = next_state
            if done:
                break

        agent.decay_epsilon()

    agent.save(config["q_table_path"])
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning agent for ns3-gym cognitive radio")
    parser.add_argument("--config", default="configs/qlearning.yaml", help="Path to YAML config")
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
