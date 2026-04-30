from typing import NamedTuple, Dict, Any, List
from pydantic import BaseModel


# RL Env Data
class EnvStep(NamedTuple):
    action: Dict[str, Any] | List[Any] | float | int
    state: Dict[str, Any]
    reward: float | int | Any
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class EnvData(NamedTuple):
    type: str
    data: Dict[str, List[Any]]


class ParameterSchema(BaseModel):
    label: str
    type: str  # 'float', 'int', 'boolean'
    min: float | None = None
    max: float | None = None
    default: float | int | bool
    step: float | None = None


class ModelDefinition(BaseModel):
    id: str
    name: str
    description: str
    parameters: Dict[str, ParameterSchema]


# Model Config for backend
class ModelConfig(NamedTuple):
    name: str
    type: str
    parameters: Dict[str, Any]
    n_actions: int
    n_states: int


# Results for Comparation in RL Experiments
class ExperimentResultRL(NamedTuple):
    model_id: str
    rewards_history: List[float]
    steps_per_episode: List[int]
    hyperparameters: Dict[str, Any]
    timestamp: str
