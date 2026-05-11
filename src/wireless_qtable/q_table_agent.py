from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


State = Tuple[int, ...]


class QTableAgent:
    def __init__(
        self,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[State, np.ndarray] = {}

    def _ensure_state(self, state: State) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=float)
        return self.q_table[state]

    def select_action(self, state: State) -> int:
        q_values = self._ensure_state(state)
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(q_values))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        current_q = self._ensure_state(state)
        next_q = self._ensure_state(next_state)
        target = reward if done else reward + self.gamma * float(np.max(next_q))
        current_q[action] += self.alpha * (target - current_q[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table, allow_pickle=True)

    def load(self, path: str | Path) -> None:
        loaded = np.load(Path(path), allow_pickle=True).item()
        self.q_table = {tuple(k): np.asarray(v, dtype=float) for k, v in loaded.items()}
