# backend/core/coordinator.py
import asyncio
from typing import Callable, Awaitable, Any, Dict
from .wrappers import BaseUIWrapper
from .baseAgent import Agent


class ExperimentCoordinator:
    def __init__(
        self,
        agent: Agent,
        wrapper: BaseUIWrapper,
        send_callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ):
        self.agent = agent
        self.wrapper = wrapper
        self.send_callback = send_callback
        self.is_running = False

    async def run_episode(self) -> None:
        # Obtenemos el estado matemático y visual en un solo llamado
        curr_state_int, ui_step = self.wrapper.reset()
        await self.send_callback(ui_step._asdict())

        self.is_running = True

        while self.is_running:
            # Agente actua
            action = self.agent.act({"current_state": curr_state_int})

            # Transición
            next_state_int, ui_step = self.wrapper.step(action)

            # Actualiza el Agente.
            self.agent.update(
                {
                    "current_state": curr_state_int,
                    "next_state": next_state_int,
                    "reward": ui_step.reward,
                    "action": action,
                    "done": ui_step.terminated or ui_step.truncated,
                }
            )

            # Enviamos el payload
            await self.send_callback(ui_step._asdict())

            # Avanzamos el reloj
            curr_state_int = next_state_int
            await asyncio.sleep(0.05)

            if ui_step.terminated or ui_step.truncated:
                self.is_running = False
