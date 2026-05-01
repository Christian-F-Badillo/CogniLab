# backend/core/factory.py
from typing import Any, Dict, Type, Tuple, Optional, Callable, Awaitable
from .coordinator import BaseExperimentCoordinator, ExperimentCoordinatorRL
from .monitors import BaseResultMonitor, RLResultMonitor


class ExperimentFactory:
    """
    Patrón Factory generalizado para construir contextos de simulación desde solicitudes JSON/API.
    Utiliza registros dinámicos (Inversión de Control) para evitar acoplamiento rígido con
    implementaciones específicas y soportar múltiples paradigmas (RL, DDM, Bayesianos).
    """

    _MODELS_REGISTRY: Dict[str, Type[Any]] = {}
    _POLICIES_REGISTRY: Dict[str, Type[Any]] = {}
    _ENVS_REGISTRY: Dict[str, Type[Any]] = {}
    _WRAPPERS_REGISTRY: Dict[str, Type[Any]] = {}

    # Registros del Core
    _COORDINATORS_REGISTRY: Dict[str, Type[BaseExperimentCoordinator]] = {
        "rl": ExperimentCoordinatorRL
    }

    _MONITORS_REGISTRY: Dict[str, Type[BaseResultMonitor]] = {"rl": RLResultMonitor}

    # --- Métodos de Registro Dinámico ---

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
    def _build_policy(cls, policy_config: Dict[str, Any]) -> Any:
        policy_id = policy_config.get("id")
        if policy_id not in cls._POLICIES_REGISTRY:
            raise ValueError(
                f"Política no soportada o no registrada: '{policy_id}'. "
                f"Disponibles: {list(cls._POLICIES_REGISTRY.keys())}"
            )

        params = policy_config.get("params", {})
        return cls._POLICIES_REGISTRY[policy_id](**params)

    @classmethod
    def _build_agent(
        cls, model_config: Dict[str, Any], env_info: Dict[str, Any]
    ) -> Any:
        model_id = model_config.get("id")
        if model_id not in cls._MODELS_REGISTRY:
            raise ValueError(
                f"Modelo no soportado o no registrado: '{model_id}'. "
                f"Disponibles: {list(cls._MODELS_REGISTRY.keys())}"
            )

        params = model_config.get("params", {}).copy()

        # Inyectar política dinámicamente si el modelo la pide
        if "policy" in model_config:
            params["policy"] = cls._build_policy(model_config["policy"])

        # Inyectar metadatos del ambiente (ej. state_space) sólo si hay info disponible
        # Esto permite que modelos que no usan ambiente (ej. generativos) no fallen
        params.update(env_info)

        return cls._MODELS_REGISTRY[model_id](**params)

    @classmethod
    def _build_environment(cls, env_config: Dict[str, Any]) -> Any:
        env_id = env_config.get("id")
        if env_id not in cls._ENVS_REGISTRY:
            raise ValueError(f"Ambiente no soportado o no registrado: '{env_id}'")

        params = env_config.get("params", {})
        base_env = cls._ENVS_REGISTRY[env_id](**params)

        # Aplicar wrapper si la solicitud lo especifica
        wrapper_config = env_config.get("wrapper")
        if wrapper_config:
            wrapper_id = wrapper_config.get("id")
            if wrapper_id in cls._WRAPPERS_REGISTRY:
                wrapper_params = wrapper_config.get("params", {})
                return cls._WRAPPERS_REGISTRY[wrapper_id](base_env, **wrapper_params)
            else:
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
        """
        Punto de entrada genérico para construir CUALQUIER experimento.
        Delegará a coordinadores específicos según 'experiment_type'.
        """
        if experiment_type not in cls._COORDINATORS_REGISTRY:
            raise ValueError(f"Tipo de experimento '{experiment_type}' no registrado.")

        # 1. Construir Ambiente (Opcional, depende del paradigma)
        env = None
        env_info = {}
        env_config = request_payload.get("environment")

        if env_config:
            env = cls._build_environment(env_config)
            # Extracción agnóstica de propiedades para pasarlas al modelo
            if hasattr(env, "states_idx"):
                env_info["states_idx"] = env.states_idx
            if hasattr(env, "actions_idx"):
                env_info["actions_idx"] = env.actions_idx

        # 2. Construir Modelo/Agente
        model_config = request_payload.get("model")
        if not model_config:
            raise ValueError("Configuración del modelo faltante en la solicitud.")

        model = cls._build_agent(model_config, env_info)

        # 3. Preparar Coordinador
        run_config = request_payload.get("run_params", {})
        CoordinatorClass = cls._COORDINATORS_REGISTRY[experiment_type]

        coordinator_kwargs = {
            "n_episodes": run_config.get("episodes", 1),
            "step_delay": run_config.get("step_delay", 0.0),
            "on_step_cb": on_step_cb,
            "on_episode_cb": on_episode_cb,
        }

        # Inyección condicional de dependencias según el tipo de experimento
        if experiment_type == "rl":
            if env is None:
                raise ValueError("Experimento RL requiere un 'environment'.")
            coordinator_kwargs["agent"] = model
            coordinator_kwargs["wrapper"] = env
        else:
            # Para paradigmas no-RL (donde el ambiente puede ser None)
            coordinator_kwargs["model"] = model
            if env is not None:
                coordinator_kwargs["environment"] = env

        coordinator = CoordinatorClass(**coordinator_kwargs)

        # 4. Preparar Monitor
        MonitorClass = cls._MONITORS_REGISTRY.get(experiment_type, BaseResultMonitor)
        # TODO: Leer el directorio de salida (output_dir) de run_config si existe
        monitor = MonitorClass()

        return coordinator, monitor
