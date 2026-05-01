from ..core.wrappers import BaseUIWrapper
from ..core.env_interfaces import AbstractDiscreteEnv
from ..schemas.data import EnvStep


class DiscreteEnvUIWrapper(BaseUIWrapper):
    def __init__(self, env: AbstractDiscreteEnv) -> None:
        self.env = env

    def __getattr__(self, name: str):
        """
        Patrón Proxy Transparente:
        Delega atributos inexistentes al entorno base.
        Permite a la Factoría leer env.observation_space sin romper abstracción.
        """
        if name.startswith("_"):
            raise AttributeError(f"Acceso a atributo privado '{name}' denegado.")
        return getattr(self.env, name)

    def reset(self, seed: int | None = None) -> tuple[int, EnvStep]:
        """Retorna (estado_puro_para_agente, datos_visuales_para_ui)"""
        obs_int, info = self.env.reset(seed=seed)
        row, col = self.env._decode_state(obs_int)

        ui_state = {"render_type": "GRID_INIT", "agent_pos": [row, col]}
        ui_step = EnvStep(
            action=-99,
            state=ui_state,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )
        return obs_int, ui_step

    def step(self, action: int | float | list) -> tuple[int, EnvStep]:
        """Retorna (estado_puro_para_agente, datos_visuales_para_ui)"""
        obs_int, reward, term, trunc, info = self.env.step(action)
        row, col = self.env._decode_state(obs_int)

        ui_state = {"render_type": "GRID_UPDATE", "agent_pos": [row, col]}
        ui_step = EnvStep(
            action=action,
            state=ui_state,
            reward=reward,
            terminated=term,
            truncated=trunc,
            info=info,
        )
        return obs_int, ui_step
