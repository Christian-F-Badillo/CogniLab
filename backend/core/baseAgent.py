from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
from ..types.data import ModelConfig, EnvStep
import numpy as np


class Agent(ABC):
    def __init__(self, config: Optional[ModelConfig] = None):
        "Base Agent Class"
        self._config = config

    @abstractmethod
    def act(
        self, state: Union[np.ndarray, List[float]]
    ) -> Union[int, float, List[Any]]:
        "Method of RL agent to act given the env state"
        pass

    @abstractmethod
    def update(self, step: EnvStep) -> None:
        "Take the environment feedback to update the model parameters"

    @property
    @abstractmethod
    def name(self) -> str:
        "Get the model name (id)"
        pass

    @name.setter
    @abstractmethod
    def name(self, value: str) -> None:
        "Set the custom model name (id)"
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        "Type of Model"
        pass
