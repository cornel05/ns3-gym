from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

from tqdm import trange

from .discretizer import discretize_observation
from .q_table_agent import QTableAgent
from .utils import load_config, set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = REPO_ROOT / "ns-allinone-3.40" / "ns-3.40" / "contrib" / "opengym" / "examples" / "interference-pattern"


@contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class _ScenarioCwdEnv:
    def __init__(self, env, scenario_dir: Path):
        self._env = env
        self._scenario_dir = scenario_dir

    def __getattr__(self, name):
        value = getattr(self._env, name)
        if not callable(value):
            return value

        def call_in_scenario_dir(*args, **kwargs):
            with _pushd(self._scenario_dir):
                return value(*args, **kwargs)

        return call_in_scenario_dir


def _resolve_scenario_dir(value: str | None) -> Path:
    scenario_dir = Path(value).expanduser() if value else DEFAULT_SCENARIO_DIR
    if not scenario_dir.is_absolute():
        scenario_dir = REPO_ROOT / scenario_dir
    return scenario_dir.resolve()


def make_env(config: dict | None = None):
    """Create ns3-gym environment."""
    try:
        from ns3gym import ns3env
    except ImportError as exc:
        import os
        import sys

        raise ImportError(
            "ns3gym is not installed in this interpreter.\n"
            f"python={sys.executable}\n"
            f"VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')}\n"
            "Install ns3gym into this env from ns-allinone-3.40/ns-3.40/contrib/opengym/model/ns3gym "
            "(see scripts/setup_ns3gym.sh)."
        ) from exc

    config = config or {}
    env_config = dict(config.get("env", {}))
    scenario_dir = _resolve_scenario_dir(env_config.pop("scenario_dir", None))
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"ns3-gym scenario directory does not exist: {scenario_dir}")

    sim_args = dict(env_config.pop("sim_args", {}))
    env_kwargs = {
        "port": int(env_config.pop("port", 0)),
        "stepTime": float(env_config.pop("step_time", 0.1)),
        "startSim": bool(env_config.pop("start_sim", True)),
        "simSeed": int(env_config.pop("sim_seed", config.get("seed", 0))),
        "simArgs": sim_args,
        "debug": bool(env_config.pop("debug", False)),
    }
    if env_config:
        unknown = ", ".join(sorted(env_config))
        raise ValueError(f"Unknown env config key(s): {unknown}")

    with _pushd(scenario_dir):
        env = ns3env.Ns3Env(**env_kwargs)
    return _ScenarioCwdEnv(env, scenario_dir)


def train(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(int(config["seed"]))

    env = make_env(config)
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
