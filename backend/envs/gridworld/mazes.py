from typing import Any, Dict, Tuple
import gymnasium as gym
import numpy as np
from ...core.env_interfaces import AbstractDiscreteEnv
from ._mazes_types import (
    MazeType,
    MAZE_FOUR_ROOMS,
    MAZE_OPEN,
    MAZE_U_SHAPE,
    MAZE_WALL,
    MAZE_ZIGZAG,
)

mazes = {
    MazeType.OPEN: MAZE_OPEN,
    MazeType.ROOMS: MAZE_FOUR_ROOMS,
    MazeType.USHAPE: MAZE_U_SHAPE,
    MazeType.WALL: MAZE_WALL,
    MazeType.ZIGZAG: MAZE_ZIGZAG,
}


class MazeEnv(AbstractDiscreteEnv):
    """
    Custom Maze Environment para Modelamiento Tabular
    Implementa funciones de transición y recompensa restrictivas.
    """

    def __init__(self, type: MazeType, max_steps: int = 100) -> None:
        # Información del Laberinto
        self.type = type
        self.max_steps = max_steps
        self.maze = self._parse_maze(mazes[type])

        # Información del Agente
        self._agent_position = np.array([-1, -1], dtype=np.int8)
        self._target_position = np.array([-1, -1], dtype=np.int8)

        # Dimensiones topológicas
        self.rows, self.cols = self.maze.shape

        # Espacios del estándar Gymnasium
        self.observation_space = gym.spaces.Discrete(self.rows * self.cols)
        self.action_space = gym.spaces.Discrete(4)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """
        Reinicializa la topología de la simulación.
        """
        if seed is not None:
            np.random.seed(seed)

        self._agent_position = np.argwhere(self.maze == 2)[0]
        self._target_position = np.argwhere(self.maze == 3)[0]

        observation = self._get_obs()
        info = self._get_info()

        # Inicialización de métricas del MDP
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.steps = 0

        return observation, info

    @property
    def maze_shape(self) -> tuple[int, int]:
        return self.maze.shape

    def _get_info(self) -> Dict[str, Any]:
        """
        Computa información auxiliar, como la distancia de Manhattan (Norma L1)
        útil para análisis cognitivos o funciones heurísticas.
        """
        return {
            "distance": np.linalg.norm(
                self._agent_position - self._target_position, ord=1
            )
        }

    def _get_obs(self) -> int:
        row, col = self._agent_position
        return self._encode_state(row, col)

    def _encode_state(self, row: int, col: int) -> int:
        """Proyección de espacio de coordenadas 2D a un escalar unidimensional."""
        return int(row * self.cols + col)

    def _decode_state(self, state: int) -> Tuple[int, int]:
        """Proyección inversa del escalar a coordenadas en la grilla."""
        return int(state // self.cols), int(state % self.cols)

    def _parse_maze(self, maze_layout: list[str]) -> np.ndarray:
        """
        Mapea el texto del laberinto a un tensor.
        0: Libre, 1: Muro, 2: Inicio, 3: Meta
        """
        mapping = {".": 0, "#": 1, "S": 2, "G": 3, " ": 0}
        maze_matrix = []
        for row in maze_layout:
            maze_matrix.append([mapping.get(char, 0) for char in row])
        return np.array(maze_matrix, dtype=np.int8)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Evalúa la transición del MDP P(s', r | s, a).
        """
        row, col = self._agent_position
        new_row, new_col = row, col

        # Mapeo cinemático: 0: Arriba, 1: Derecha, 2: Abajo, 3: Izquierda
        if action == 0:
            new_row -= 1
        elif action == 1:
            new_col += 1
        elif action == 2:
            new_row += 1
        elif action == 3:
            new_col -= 1

        # 1. Detección de Colisión Límite Lógica (Fuera de la matriz)
        hit_boundary = not (0 <= new_row < self.rows and 0 <= new_col < self.cols)

        # Estabilizador Numérico: Previene OutOfBounds en tensores de Numpy subyacentes
        new_row = max(0, min(self.rows - 1, new_row))
        new_col = max(0, min(self.cols - 1, new_col))

        # 2. Detección de Colisión Física (Muros internos)
        hit_wall = self.maze[new_row, new_col] == 1

        # 3. Lógica de Evaluación de Recompensas
        if hit_boundary or hit_wall:
            # Castigo por chocar. El estado se revierte implícitamente al no actualizar _agent_position.
            self.reward = -1.0
        else:
            # Transición de estado válida
            self._agent_position = np.array([new_row, new_col])

            if self.maze[new_row, new_col] == 3:
                # Condición de Absorción: Alcanzó la meta
                self.reward = 20.0
                self.terminated = True
            else:
                # Condición de Costo Constante: Incentivo de ruta mínima
                self.reward = -0.5

        # Mantenimiento de reloj de la simulación
        self.steps += 1
        if self.steps >= self.max_steps:
            self.truncated = True

        return (
            self._get_obs(),
            self.reward,
            self.terminated,
            self.truncated,
            self._get_info(),
        )
