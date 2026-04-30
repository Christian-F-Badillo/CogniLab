from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

# Importaciones de tus módulos (ajusta las rutas según tu estructura)
from backend.core.coordinator import ExperimentCoordinator
from backend.models.rl.TD import QLearningAgent
from backend.models.rl.policies import EpsilonGreedyPolicy
from backend.envs.gridworld.mazes import MazeEnv, MazeType
from backend.envs.wrappers import DiscreteEnvUIWrapper
from backend.types.data import ModelConfig

app = FastAPI(title="CogniLab API")


@app.websocket("/ws/simulate")
async def websocket_simulation_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw_message = await websocket.receive_text()
            payload = json.loads(raw_message)

            if payload.get("command") == "START_EPISODE":
                # En un sistema completo, esto usaría tus factorías
                env = MazeEnv(type=MazeType.USHAPE)

                config = ModelConfig(
                    name="Q-Learning-Maze",
                    type="RL",
                    parameters={"lr": 0.1, "gamma": 0.95},
                    n_actions=env.action_space.n,
                    n_states=env.observation_space.n,
                )

                policy = EpsilonGreedyPolicy(epsilon=0.1)
                agent = QLearningAgent(config=config, policy=policy)
                ui_wrapper = DiscreteEnvUIWrapper(env=env)

                # 2. Definición del callback inyectable
                async def send_to_frontend(data: dict):
                    await websocket.send_json(data)

                # 3. Orquestación
                coordinator = ExperimentCoordinator(
                    env=env,
                    agent=agent,
                    ui_wrapper=ui_wrapper,
                    send_callback=send_to_frontend,
                )

                # Bloquea este manejador de mensajes hasta que el episodio termine,
                # pero permite que otros websockets sigan funcionando.
                await coordinator.run_episode()

                # Enviamos señal de fin de episodio
                await websocket.send_json(
                    {"type": "SYSTEM", "message": "EPISODE_COMPLETE"}
                )

    except WebSocketDisconnect:
        print("Electron Client Disconnected")
