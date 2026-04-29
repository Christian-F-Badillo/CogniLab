from typing import Any, Dict, Tuple
import gymnasium as gym
import numpy as np
from _mazes_types import (
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


class MazeEnv(gym.Env):
    """
    Custom Maze Environment
    """

    def __init__(self, type: MazeType, max_steps: int = 100) -> None:
        # Maze Information
        self.type = type
        self.max_steps = max_steps
        self.maze = self._parse_maze(mazes[type])

        # Agent information
        # - Get the position of the agent Spawn with a np.ndarray
        self._agent_position = np.array([-1, -1], dtype=np.int8)
        self._target_position = np.array([-1, -1], dtype=np.int8)

        # Observation Space
        self.observation_space = gym.spaces.Dict(
            {"agent": gym.spaces.Box(0, 9, shape=(2,), dtype=np.int8)}
        )

        # Actions
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),  # Right
            1: np.array([-1, 0]),  # Up
            2: np.array([0, -1]),  # Left
            3: np.array([1, 0]),  # Down
        }

        # Env Step info
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.steps = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute the action over the maze
        """

        self.steps += 1

        if self.steps == self.max_steps:
            self.truncated = True

        # Map action to direction
        direction = self._action_to_direction[action]

        # Proposed new state
        new_position = self._agent_position + direction
        bounderies_mask = (
            (new_position[0] < 0)
            | (new_position[0] > 9)
            | (new_position[1] < 0)
            | (new_position[1] > 9)
        )
        if bounderies_mask:
            out_bounderies = True
            is_free = False
        else:
            out_bounderies = False
            is_free = self.maze[new_position[0], new_position[1]] != 1

        # Checking bounderies
        if bounderies_mask or not is_free:
            out_bounderies = True
            self.reward = -2.0
            observation = self._get_obs()
            info = self._get_info()
            info["out_boundaries"] = out_bounderies
            self.terminated = False

            return observation, self.reward, self.terminated, self.truncated, info

        self._agent_position += direction
        out_bounderies = False
        observation = self._get_obs()

        self.terminated = np.array_equal(self._agent_position, self._target_position)
        self.reward = 10 if self.terminated else -1.0

        info = self._get_info()
        info["out_boundaries"] = out_bounderies

        return observation, self.reward, self.terminated, self.truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment for each maze type.
        """
        np.random.seed(seed)

        self._agent_position = np.argwhere(self.maze == 2)[0]
        self._target_position = np.argwhere(self.maze == 3)[0]

        observation = self._get_obs()
        info = self._get_info()
        # Env Step info
        self.reward = 0.0
        self.terminated = False
        self.truncated = False
        self.steps = 0

        return observation, info

    def _get_info(self) -> Dict[str, Any]:
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_position - self._target_position, ord=1
            )
        }

    def _get_obs(self) -> np.ndarray:
        return self._agent_position

    def _parse_maze(self, maze_layout: list[str]) -> np.ndarray:
        """
        Map maze text to matrix with using the standar:
        0: Free (.), 1: Obstacule (#), 2: Spawn (S), 3: Goal (G)
        """
        mapping = {".": 0, "#": 1, "S": 2, "G": 3}
        matrix = np.zeros((10, 10), dtype=np.int8)

        for i, row in enumerate(maze_layout):
            for j, char in enumerate(row):
                matrix[i, j] = mapping[char]

        return matrix
