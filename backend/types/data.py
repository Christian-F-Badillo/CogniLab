from typing import NamedTuple, Dict, Any, List, Union
import numpy as np


# RL Env Data
class EnvStep(NamedTuple):
    action: Union[int, float, List[Any]]
    state: Union[np.ndarray, List[float]]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class EnvData(NamedTuple):
    type: str
    data: Dict[str, List[Any]]


# Model Config for UI
class ModelConfigUI(NamedTuple):
    name: str
    type: str
    parameters: Dict[
        str, Dict[str, Any]
    ]  # example: {'alpha': {'min': 0, 'max': 1, 'default': 0.1}}


# Model Config for backend
class ModelConfig(NamedTuple):
    name: str
    type: str
    parameters: Dict[str, Any]


# Results for Comparation in RL Experiments
class ExperimentResultRL(NamedTuple):
    model_id: str
    rewards_history: List[float]
    steps_per_episode: List[int]
    hyperparameters: Dict[str, Any]
    timestamp: str
