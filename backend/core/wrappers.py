from abc import ABC, abstractmethod
import gymnasium as gym
from ..types.data import EnvStep
from typing import Any


class BaseUIWrapper(ABC):
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[Any, EnvStep]:
        pass

    @abstractmethod
    def step(self, action: int | float | list) -> tuple[Any, EnvStep]:
        pass
