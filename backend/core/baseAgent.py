from abc import ABC, abstractmethod
from typing import Any, Dict, List
from ..schemas.data import ModelConfig
import pickle


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

    def save_state(self, filepath: str) -> None:
        """
        Serializa el estado matemático completo del agente (pesos, matrices, hiperparámetros).
        Utiliza el dict interno para ser agnóstico a implementaciones futuras (RL, DDM, etc.).
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_state(self, filepath: str) -> None:
        """
        Rehidrata el estado del agente desde el sistema de archivos.
        Garantiza que la Propiedad de Markov se mantenga entre sesiones pausadas.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.__dict__.update(state)

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
