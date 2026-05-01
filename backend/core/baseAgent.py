from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..schemas.data import ModelConfig


class Agent(ABC):
    def __init__(self, config: ModelConfig):
        "Base Agent Class"
        self._params = config.parameters
        self._name = config.name
        self._type = config.type
        self._nactions = config.n_actions
        self._nstates = config.n_states

    @abstractmethod
    def act(self, input: Dict[str, Any]) -> int | float | List[Any]:
        "Method of RL agent to act given the env state"
        pass

    @abstractmethod
    def update(self, input: Dict[str, Any]) -> None:
        "Take the environment feedback to update the model parameters"

    @property
    def name(self) -> str:
        "Get the model name (id)"
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        "Set the custom model name (id)"
        if not value:
            raise ValueError("Agent::Model name empty")

        self._name = value

    @property
    def type(self) -> str:
        "Type of Model"
        return self._type
