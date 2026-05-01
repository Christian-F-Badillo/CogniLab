from pathlib import Path
import json

from .factory import ExperimentFactory
from backend.models.rl.TD import QLearningAgent, SarsaAgent
from backend.models.rl.policies import EpsilonGreedyPolicy, GreedyPolicy, SoftmaxPolicy
from backend.envs.gridworld.mazes import MazeEnv
from backend.envs.wrappers import DiscreteEnvUIWrapper



def register_simulation_dependencies():
    """
    Fase de Bootstrap (Inversión de Control).
    Registramos las clases disponibles en el sistema dentro del contenedor Factory.
    La API delega el conocimiento de 'cómo' se construyen los objetos.
    """
    # Registrar Modelos
    ExperimentFactory.register_model("q_learning", QLearningAgent)
    ExperimentFactory.register_model("sarsa", SarsaAgent)

    # Registrar Políticas
    ExperimentFactory.register_policy("epsilon_greedy", EpsilonGreedyPolicy)
    ExperimentFactory.register_policy("greedy", GreedyPolicy)
    ExperimentFactory.register_policy("softmax", SoftmaxPolicy)

    # Registrar Ambientes y Wrappers
    ExperimentFactory.register_env("gridworld_maze", MazeEnv)
    ExperimentFactory.register_wrapper("discrete_ui", DiscreteEnvUIWrapper)

    config_dir = Path(__file__).parent.parent / "config"

    # Cargar y registrar esquemas
    try:
        with open(config_dir / "envs_rl_schema.json", "r") as f:
            ExperimentFactory.register_schema("environment", json.load(f))

        with open(config_dir / "models_schema.json", "r") as f:
            ExperimentFactory.register_schema("model", json.load(f))

        with open(config_dir / "policies_rl_schema.json", "r") as f:
            ExperimentFactory.register_schema("policy", json.load(f))
    except FileNotFoundError as e:
        print(f"Advertencia: No se encontró un esquema JSON. {e}")