from abc import ABC, abstractmethod
import gymnasium as gym


class AbstractDiscreteEnv(gym.Env, ABC):
    """Tabular Environment Interface of RL (GridWorlds)."""

    @property
    @abstractmethod
    def maze_shape(self) -> tuple[int, int]:
        """Get the maze shape."""
        pass

    def _encode_state(self, row: int, col: int) -> int:
        """Encode X and Y positions into unique Int."""
        _, cols = self.maze_shape
        return int(row * cols + col)

    def _decode_state(self, state: int) -> tuple[int, int]:
        """Decodify Int to X and Y position in a Maze."""
        _, cols = self.maze_shape
        return (state // cols, state % cols)
