import asyncio
import websockets
import json


async def run_test():
    # URL de tu WebSocket (Ajusta el puerto si FastAPI no corre en 8000)
    uri = "ws://localhost:8000/ws/simulate"

    # El payload exacto que tu API espera ahora, validando la Inversión de Control
    payload = {
        "command": "START_EXPERIMENT",
        "experiment_type": "rl",
        "environment": {
            "id": "gridworld_maze",
            "params": {"type": "zigzag", "max_steps": 100},
            "wrapper": {"id": "discrete_ui"},
        },
        "model": {
            "id": "sarsa",
            "params": {
                "config": {
                    "name": "Sarsa Grid zigzag",
                    "type": "TD",
                    "parameters": {"lr": 0.1, "gamma": 0.8},
                    "n_actions": 4,
                    "n_states": 1000,
                }
            },
            "policy": {"id": "epsilon_greedy", "params": {"epsilon": 0.2}},
        },
        "run_params": {
            "episodes": 3,
            "step_delay": 0.01,
        },
    }

    try:
        print(f"Conectando a {uri}...")
        async with websockets.connect(uri) as websocket:
            print("Conexión establecida. Enviando solicitud START_EXPERIMENT...")

            # Enviar el comando
            await websocket.send(json.dumps(payload))

            # Bucle de escucha de respuestas
            while True:
                response_str = await websocket.recv()
                response = json.loads(response_str)

                msg_type = response.get("type")

                if msg_type == "SYSTEM":
                    print(f"[{msg_type}] {response.get('message')}")

                elif msg_type == "STEP":
                    # Solo imprimimos un fragmento para no inundar la consola
                    data = response.get("data", {})
                    state = data.get("state", "N/A")
                    reward = data.get("reward", 0)
                    print(f"  -> Step | State: {state} | Reward: {reward}")

                elif msg_type == "EPISODE_SUMMARY":
                    ep = response.get("data", {}).get("episode")
                    rew = response.get("data", {}).get("reward")
                    print(
                        f"\n[{msg_type}] Episodio {ep} finalizado. Retorno total: {rew}\n"
                    )

                elif msg_type == "EXPERIMENT_COMPLETE":
                    print(f"\n[{msg_type}] Simulación terminada exitosamente.")
                    print("Resumen:", json.dumps(response.get("summary"), indent=2))
                    print(f"Archivo guardado en: {response.get('saved_file')}")
                    break  # Salimos del bucle porque el experimento terminó

                elif msg_type in ["ERROR", "FATAL_ERROR"]:
                    print(f"\n[!!! {msg_type} !!!] {response.get('message')}")
                    break

    except ConnectionRefusedError:
        print(
            "ERROR: No se pudo conectar. ¿Está FastAPI corriendo? (uvicorn main:app --reload)"
        )
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la prueba: {e}")


if __name__ == "__main__":
    asyncio.run(run_test())
