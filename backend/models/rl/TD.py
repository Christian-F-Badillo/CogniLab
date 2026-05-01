import numpy as np
from typing import Dict, Any
from ...core.baseAgent import Agent
from ...schemas.data import ModelConfig
from .policies import BasePolicy


class QLearningAgent(Agent):
    def __init__(self, config: ModelConfig, policy: BasePolicy):
        super().__init__(config)
        self.Q = np.zeros(shape=(self._nstates, self._nactions), dtype=np.float64)
        self.policy = policy
        self.lr = self._params.get("lr", 0.1)
        self.gamma = self._params.get("gamma", 0.9)

    def update(self, input: Dict[str, Any]) -> None:
        current_state = input["current_state"]
        next_state = input["next_state"]
        reward = input["reward"]
        action = input["action"]

        is_done = input.get("done", False)
        best_action_q = 0.0 if is_done else np.max(self.Q[next_state, :])

        self.Q[current_state, action] += self.lr * (
            reward + (self.gamma * best_action_q) - self.Q[current_state, action]
        )

    def act(self, input: Dict[str, Any]) -> int:
        state = input["current_state"]

        action_values = self.Q[state, :]

        action = self.policy.select_action(action_values)

        return action


class SarsaAgent(Agent):
    def __init__(self, config: ModelConfig, policy: BasePolicy):
        super().__init__(config)
        self.Q = np.zeros(shape=(self._nstates, self._nactions), dtype=np.float64)
        self.policy = policy
        self.lr = self._params.get("lr", 0.1)
        self.gamma = self._params.get("gamma", 0.9)

    def update(self, input: Dict[str, Any]) -> None:
        current_state = input["current_state"]
        next_state = input["next_state"]
        reward = input["reward"]
        action = input["action"]

        is_done = input.get("done", False)
        q_next_action = (
            0.0
            if is_done
            else self.Q[next_state, self.act({"current_state": next_state})]
        )

        self.Q[current_state, action] += self.lr * (
            reward + (self.gamma * q_next_action) - self.Q[current_state, action]
        )

    def act(self, input: Dict[str, Any]) -> int:
        state = input["current_state"]

        action_values = self.Q[state, :]

        action = self.policy.select_action(action_values)

        return action
