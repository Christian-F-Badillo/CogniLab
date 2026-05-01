# backend/core/factory.py
from typing import Any, Dict, Type, Tuple, Optional, Callable, Awaitable
import logging

# Interfaces Base
from .baseAgent import Agent
from .wrappers import BaseUIWrapper

# Coordinadores y Monitores Base
from .coordinator import BaseExperimentCoordinator, ExperimentCoordinatorRL
from .monitors import BaseResultMonitor, RLResultMonitor

# Tipos para validación
from backend.models.rl.TD import ModelConfig


class ExperimentFactory:
    _MODELS_REGISTRY: Dict[str, Type[Any]] = {}
    _POLICIES_REGISTRY: Dict[str, Type[Any]] = {}
    _ENVS_REGISTRY: Dict[str, Type[Any]] = {}
    _WRAPPERS_REGISTRY: Dict[str, Type[Any]] = {}

    _COORDINATORS_REGISTRY: Dict[str, Type[BaseExperimentCoordinator]] = {
        "rl": ExperimentCoordinatorRL
    }

    _MONITORS_REGISTRY: Dict[str, Type[BaseResultMonitor]] = {"rl": RLResultMonitor}

    _SCHEMAS_REGISTRY: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_schema(cls, schema_type: str, schema_dict: Dict[str, Any]):
        """Registra metadatos de configuración (UI Configs)."""
        cls._SCHEMAS_REGISTRY[schema_type] = schema_dict

    @classmethod
    def register_model(cls, name: str, model_cls: Type[Any]):
        cls._MODELS_REGISTRY[name] = model_cls

    @classmethod
    def register_env(cls, name: str, env_cls: Type[Any]):
        cls._ENVS_REGISTRY[name] = env_cls

    @classmethod
    def register_policy(cls, name: str, policy_cls: Type[Any]):
        cls._POLICIES_REGISTRY[name] = policy_cls

    @classmethod
    def register_wrapper(cls, name: str, wrapper_cls: Type[Any]):
        cls._WRAPPERS_REGISTRY[name] = wrapper_cls

    @classmethod
    def register_coordinator(
        cls, name: str, coordinator_cls: Type[BaseExperimentCoordinator]
    ):
        cls._COORDINATORS_REGISTRY[name] = coordinator_cls

    @classmethod
    def register_monitor(cls, name: str, monitor_cls: Type[BaseResultMonitor]):
        cls._MONITORS_REGISTRY[name] = monitor_cls

    @classmethod
    def _validate_payload(cls, payload: Dict[str, Any], schema_type: str):
        """
        NOTA: Desactivamos jsonschema.validate() aquí porque los archivos JSON actuales
        son configuraciones de UI (Frontend), no esquemas formales de validación backend.
        Aquí a futuro se implementará validación Pydantic o JSONSchema estricto.
        """
        pass

    @classmethod
    def _build_policy(cls, policy_config: Dict[str, Any]) -> Any:
        policy_id = policy_config.get("id")
        if policy_id not in cls._POLICIES_REGISTRY:
            raise ValueError(
                f"Política no soportada: '{policy_id}'. Disponibles: {list(cls._POLICIES_REGISTRY.keys())}"
            )
        return cls._POLICIES_REGISTRY[policy_id](**policy_config.get("params", {}))

    @classmethod
    def _build_agent(
        cls, model_config: Dict[str, Any], env_info: Dict[str, Any]
    ) -> Any:
        model_id = model_config.get("id")
        if model_id not in cls._MODELS_REGISTRY:
            raise ValueError(f"Modelo no soportado: '{model_id}'.")

        params = model_config.get("params", {}).copy()

        # Inyectar política
        if "policy" in model_config:
            params["policy"] = cls._build_policy(model_config["policy"])

        # Reestructurar dependencias específicas de TDLearning / QLearningAgent
        if "config" in params:
            config_data = params.pop("config")

            if "n_states" in env_info:
                config_data["n_states"] = env_info["n_states"]
            if "n_actions" in env_info:
                config_data["n_actions"] = env_info["n_actions"]

            params["config"] = ModelConfig(**config_data)

        return cls._MODELS_REGISTRY[model_id](**params)

    @classmethod
    def _build_environment(cls, env_config: Dict[str, Any]) -> Any:
        env_id = env_config.get("id")
        if env_id not in cls._ENVS_REGISTRY:
            raise ValueError(f"Ambiente no soportado: '{env_id}'")

        base_env = cls._ENVS_REGISTRY[env_id](**env_config.get("params", {}))

        wrapper_config = env_config.get("wrapper")
        if wrapper_config:
            wrapper_id = wrapper_config.get("id")
            if wrapper_id in cls._WRAPPERS_REGISTRY:
                return cls._WRAPPERS_REGISTRY[wrapper_id](
                    base_env, **wrapper_config.get("params", {})
                )
            raise ValueError(f"Wrapper no soportado: '{wrapper_id}'")

        return base_env

    @classmethod
    def build_experiment_context(
        cls,
        request_payload: Dict[str, Any],
        experiment_type: str = "rl",
        on_step_cb: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_episode_cb: Optional[Callable[[int, float], Awaitable[None]]] = None,
    ) -> Tuple[BaseExperimentCoordinator, BaseResultMonitor]:
        if experiment_type not in cls._COORDINATORS_REGISTRY:
            raise ValueError(f"Tipo de experimento '{experiment_type}' no registrado.")

        env = None
        env_info = {}
        env_config = request_payload.get("environment")

        if env_config:
            env = cls._build_environment(env_config)

            # --- CÓDIGO LIMPIO ---
            # Gracias a que DiscreteEnvUIWrapper implementa __getattr__,
            # podemos acceder a las propiedades nativas sin romper la abstracción.
            if hasattr(env, "observation_space") and hasattr(
                env.observation_space, "n"
            ):
                env_info["n_states"] = env.observation_space.n
            else:
                logging.warning("El ambiente no expone observation_space.n")

            if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
                env_info["n_actions"] = env.action_space.n
            else:
                logging.warning("El ambiente no expone action_space.n")

        model_config = request_payload.get("model")
        if not model_config:
            raise ValueError("Configuración del modelo faltante.")

        model = cls._build_agent(model_config, env_info)

        run_config = request_payload.get("run_params", {})
        CoordinatorClass = cls._COORDINATORS_REGISTRY[experiment_type]

        coordinator_kwargs = {
            "n_episodes": run_config.get("episodes", 1),
            "step_delay": run_config.get("step_delay", 0.0),
            "on_step_cb": on_step_cb,
            "on_episode_cb": on_episode_cb,
        }

        if experiment_type == "rl":
            if env is None:
                raise ValueError("RL requiere 'environment'.")
            coordinator_kwargs["agent"] = model
            coordinator_kwargs["wrapper"] = env
        else:
            coordinator_kwargs["model"] = model
            if env is not None:
                coordinator_kwargs["environment"] = env

        coordinator = CoordinatorClass(**coordinator_kwargs)
        MonitorClass = cls._MONITORS_REGISTRY.get(experiment_type, BaseResultMonitor)
        monitor = MonitorClass()

        return coordinator, monitor
