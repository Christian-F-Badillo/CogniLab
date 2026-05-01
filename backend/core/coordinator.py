# backend/core/coordinator.py
import asyncio
from typing import Callable, Awaitable, Any, Dict, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from .wrappers import BaseUIWrapper
from .baseAgent import Agent
from ..types.data import ExperimentResultRL


class BaseExperimentCoordinator(ABC):
    """
    Clase base abstracta para orquestadores de experimentos.
    Define el contrato de ejecución, control asíncrono y manejo de callbacks.
    Independiente de si el modelo es RL, Bayesiano, Cognitivo, etc.
    """

    def __init__(
        self,
        n_episodes: int = 1,
        step_delay: float = 0.0,
        on_step_cb: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_episode_cb: Optional[Callable[[int, Any], Awaitable[None]]] = None,
    ):
        self.n_episodes = n_episodes
        self.step_delay = step_delay
        self.on_step_cb = on_step_cb
        self.on_episode_cb = on_episode_cb
        self.is_running = False

    @abstractmethod
    async def run(self) -> Any:
        """
        Ejecuta el experimento. Debe ser implementado por subclases específicas
        (ej. RL, Modelos Cognitivos, etc.) y retornar una estructura de datos tipada.
        """
        pass

    def stop(self) -> None:
        """Interrupción forzada (ej. cancelación desde la UI o monitor)"""
        self.is_running = False

    async def _handle_async_flow(self) -> None:
        """
        Gestiona el flujo concurrente cediendo el control al Event Loop de Python.
        Previene que simulaciones intensivas bloqueen el servidor asíncrono.
        """
        if self.step_delay > 0:
            await asyncio.sleep(self.step_delay)
        else:
            await asyncio.sleep(0)


class ExperimentCoordinatorRL(BaseExperimentCoordinator):
    """
    Orquestador específico para Procesos de Decisión de Markov (MDP) en Reinforcement Learning.
    Desacopla la lógica matemática del agente/ambiente de la capa de transporte.
    """

    def __init__(
        self,
        agent: Agent,
        wrapper: BaseUIWrapper,
        n_episodes: int = 1,
        step_delay: float = 0.0,
        on_step_cb: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_episode_cb: Optional[Callable[[int, float], Awaitable[None]]] = None,
    ):
        super().__init__(
            n_episodes=n_episodes,
            step_delay=step_delay,
            on_step_cb=on_step_cb,
            on_episode_cb=on_episode_cb,
        )

        self.agent = agent
        self.wrapper = wrapper

        self.rewards_history: List[float] = []
        self.steps_per_episode: List[int] = []

    async def run(self) -> ExperimentResultRL:
        """
        Ejecuta N episodios del MDP.
        """
        self.is_running = True

        for episode in range(self.n_episodes):
            if not self.is_running:
                break

            episode_reward = 0.0
            steps = 0

            # Inicialización S_0
            curr_state_int, ui_step = self.wrapper.reset()

            if self.on_step_cb:
                await self.on_step_cb(ui_step._asdict())

            while self.is_running:
                # Selección de acción
                action = self.agent.act({"current_state": curr_state_int})

                # Transición del ambiente
                next_state_int, ui_step = self.wrapper.step(action)

                # Actualización de parámetros
                self.agent.update(
                    {
                        "current_state": curr_state_int,
                        "next_state": next_state_int,
                        "reward": ui_step.reward,
                        "action": action,
                        "done": ui_step.terminated or ui_step.truncated,
                    }
                )

                # Acumulación de métricas
                episode_reward += ui_step.reward
                steps += 1

                # Emisión de eventos
                if self.on_step_cb:
                    await self.on_step_cb(ui_step._asdict())

                # Actualización de estado
                curr_state_int = next_state_int

                # Control asíncrono heredado
                await self._handle_async_flow()

                if ui_step.terminated or ui_step.truncated:
                    break

            # Registro de métricas del episodio
            self.rewards_history.append(episode_reward)
            self.steps_per_episode.append(steps)

            if self.on_episode_cb:
                await self.on_episode_cb(episode, episode_reward)

        self.is_running = False

        return ExperimentResultRL(
            model_id=self.agent.name,
            rewards_history=self.rewards_history,
            steps_per_episode=self.steps_per_episode,
            hyperparameters=self.agent._params,
            timestamp=datetime.utcnow().isoformat(),
        )

