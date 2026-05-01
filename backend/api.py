import json
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Importación de la Factoría y Módulos Core
from backend.core.factory import ExperimentFactory

# Importación de Clases Concretas (Para registrarlas en la factoría)
from backend.models.rl.TD import QLearningAgent
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
    # TODO: Cuando SARSA se divida, regístralo aquí: ExperimentFactory.register_model("sarsa", SarsaAgent)

    # Registrar Políticas
    ExperimentFactory.register_policy("epsilon_greedy", EpsilonGreedyPolicy)
    ExperimentFactory.register_policy("greedy", GreedyPolicy)
    ExperimentFactory.register_policy("softmax", SoftmaxPolicy)

    # Registrar Ambientes y Wrappers
    ExperimentFactory.register_env("gridworld_maze", MazeEnv)
    ExperimentFactory.register_wrapper("discrete_ui", DiscreteEnvUIWrapper)

    # 1. Resolver rutas absolutas a tus esquemas
    config_dir = Path(__file__).parent / "config"

    # 2. Cargar y registrar esquemas
    try:
        with open(config_dir / "envs_rl_schema.json", "r") as f:
            ExperimentFactory.register_schema("environment", json.load(f))

        with open(config_dir / "models_schema.json", "r") as f:
            ExperimentFactory.register_schema("model", json.load(f))

        with open(config_dir / "policies_rl_schema.json", "r") as f:
            ExperimentFactory.register_schema("policy", json.load(f))
    except FileNotFoundError as e:
        print(f"Advertencia: No se encontró un esquema JSON. {e}")


# Inicializar el registro de dependencias al arrancar el servidor
register_simulation_dependencies()

app = FastAPI(title="CogniLab API")


@app.websocket("/ws/simulate")
async def websocket_simulation_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw_message = await websocket.receive_text()
            payload = json.loads(raw_message)
            command = payload.get("command")

            if command == "START_EXPERIMENT":
                await websocket.send_json(
                    {"type": "SYSTEM", "message": "INITIALIZING_EXPERIMENT"}
                )

                # 1. Definición de Observadores Asíncronos (Callbacks)
                async def on_step(step_data: Dict[str, Any]):
                    """Inyectado en el Coordinador para emitir el renderizado de la UI en tiempo real."""
                    await websocket.send_json({"type": "STEP", "data": step_data})

                async def on_episode_end(episode: int, reward: float):
                    """Inyectado para transmitir curvas de aprendizaje dinámicas."""
                    await websocket.send_json(
                        {
                            "type": "EPISODE_SUMMARY",
                            "data": {"episode": episode, "reward": reward},
                        }
                    )

                try:
                    # 2. Rehidratación del Contexto de Simulación vía Factoría
                    # Si no se especifica, asumimos por retrocompatibilidad que es RL
                    experiment_type = payload.get("experiment_type", "rl")

                    coordinator, monitor = ExperimentFactory.build_experiment_context(
                        request_payload=payload,
                        experiment_type=experiment_type,
                        on_step_cb=on_step,
                        on_episode_cb=on_episode_end,
                    )

                    # 3. Evaluación Continua (El bucle del MDP cede el Event Loop mediante asyncio.sleep(0))
                    result = await coordinator.run()

                    # 4. Agregación de Resultados y Persistencia
                    monitor.add_result(result)
                    filepath = monitor.save()
                    summary = monitor.get_summary()

                    # Notificar cierre ordenado
                    await websocket.send_json(
                        {
                            "type": "EXPERIMENT_COMPLETE",
                            "summary": summary,
                            "saved_file": filepath,
                        }
                    )

                except ValueError as e:
                    # Captura de violaciones de contrato en instanciación (ej. modelo no registrado)
                    await websocket.send_json(
                        {
                            "type": "ERROR",
                            "message": f"Configuración inválida: {str(e)}",
                        }
                    )
                except Exception as e:
                    # Captura de inestabilidades matemáticas o fallos de ejecución crudos
                    await websocket.send_json(
                        {
                            "type": "FATAL_ERROR",
                            "message": f"Error interno en simulación: {str(e)}",
                        }
                    )

            elif command == "STOP_EXPERIMENT":
                # La implementación requeriría que el objeto 'coordinator' se guarde a nivel
                # de estado de conexión WebSocket. Por ahora, reportamos.
                await websocket.send_json(
                    {
                        "type": "SYSTEM",
                        "message": "Parada de emergencia no implementada todavía.",
                    }
                )

    except WebSocketDisconnect:
        print("Cliente de UI Desconectado.")
