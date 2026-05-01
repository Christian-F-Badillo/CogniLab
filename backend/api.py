import json
import asyncio
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.core.register import register_simulation_dependencies
from backend.core.factory import ExperimentFactory

register_simulation_dependencies()

app = FastAPI(title="CogniLab API")

@app.websocket("/ws/simulate")
async def websocket_simulation_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Estado de la sesión concurrente
    active_task: asyncio.Task | None = None
    coordinator = None
    temp_file_path = Path("results/temp/agent_state.pkl")

    async def run_simulation_task(payload: Dict[str, Any], is_resume: bool):
        """
        Tarea aislada que ejecuta el MDP. Su separación permite a la API
        seguir escuchando comandos entrantes (PAUSE/STOP) simultáneamente.
        """
        nonlocal coordinator, active_task
        try:
            experiment_type = payload.get("experiment_type", "rl")

            async def on_step(step_data: Dict[str, Any]):
                await websocket.send_json({"type": "STEP", "data": step_data})

            async def on_episode_end(episode: int, reward: float):
                await websocket.send_json({"type": "EPISODE_SUMMARY", "data": {"episode": episode, "reward": reward}})

            coordinator, monitor = ExperimentFactory.build_experiment_context(
                request_payload=payload,
                experiment_type=experiment_type,
                on_step_cb=on_step,
                on_episode_cb=on_episode_end
            )

            # Extracción del modelo usando Reflexión para ser agnóstico al paradigma
            model = getattr(coordinator, 'agent', getattr(coordinator, 'model', None))

            # --- Gestión de Memoria del Agente (I/O Lógico) ---
            if is_resume and temp_file_path.exists():
                if model:
                    model.load_state(str(temp_file_path))
                    await websocket.send_json({"type": "SYSTEM", "message": "Estado de la matriz (Pesos/Q) recuperado. Reanudando aprendizaje..."})
            elif not is_resume and temp_file_path.exists():
                temp_file_path.unlink() # Purgar memoria para inicio limpio

            # --- Ejecución del Modelo ---
            result = await coordinator.run()

            # --- Lógica Post-Ejecución ---
            if getattr(coordinator, 'was_stopped', False):
                # Detención inducida por el usuario
                if model:
                    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                    model.save_state(str(temp_file_path))
                await websocket.send_json({"type": "SYSTEM", "message": "Simulación Pausada. Tensor serializado a disco seguro."})
            else:
                # Convergencia o fin natural
                monitor.add_result(result)
                filepath = monitor.save()
                summary = monitor.get_summary()
                if temp_file_path.exists():
                    temp_file_path.unlink() # Limpieza
                
                await websocket.send_json({
                    "type": "EXPERIMENT_COMPLETE",
                    "summary": summary,
                    "saved_file": filepath
                })

        except Exception as e:
            await websocket.send_json({"type": "FATAL_ERROR", "message": f"Inestabilidad matemática o error: {str(e)}"})
        finally:
            active_task = None

    # Bucle Principal de Red (Totalmente no bloqueante)
    try:
        while True:
            raw_message = await websocket.receive_text()
            payload = json.loads(raw_message)
            command = payload.get("command")

            if command == "START_EXPERIMENT":
                if active_task and not active_task.done():
                    await websocket.send_json({"type": "ERROR", "message": "El motor ya está resolviendo un experimento."})
                    continue
                await websocket.send_json({"type": "SYSTEM", "message": "INICIALIZANDO_ENTORNO_LIMPIO"})
                active_task = asyncio.create_task(run_simulation_task(payload, is_resume=False))

            elif command == "PAUSE_EXPERIMENT" or command == "STOP_EXPERIMENT":
                if coordinator and active_task and not active_task.done():
                    await websocket.send_json({"type": "SYSTEM", "message": "Señal de interrupción recibida. Deteniendo y volcando tensores..."})
                    coordinator.stop()
                    # NOTA: No hacemos 'await active_task' aquí para no bloquear el socket.
                    # La tarea terminará su ciclo actual y enviará el mensaje final de guardado.

            elif command == "RESUME_EXPERIMENT":
                if active_task and not active_task.done():
                    await websocket.send_json({"type": "ERROR", "message": "El motor ya está corriendo."})
                    continue
                await websocket.send_json({"type": "SYSTEM", "message": "REHIDRATANDO_ENTORNO"})
                active_task = asyncio.create_task(run_simulation_task(payload, is_resume=True))

    except WebSocketDisconnect:
        if coordinator and active_task and not active_task.done():
            coordinator.stop()
        print("Sesión TCP Cerrada. Recursos liberados.")