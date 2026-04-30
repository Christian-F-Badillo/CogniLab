from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    """Policy base class."""

    @abstractmethod
    def select_action(self, values: np.ndarray) -> int:
        """Sample the policy."""
        pass

    @abstractmethod
    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        """Return the Policy distribution pi(a|s)."""
        pass


class GreedyPolicy(BasePolicy):
    def select_action(self, values: np.ndarray) -> int:
        return self._get_argmax_random_tie_break(values)

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        probs = np.zeros_like(values)
        max_value = np.argmax(values).item()
        probs[max_value] = 1.0

        return probs

    def _get_argmax_random_tie_break(self, values: np.ndarray) -> int:
        """If ties in max values, choice randomly one"""
        max_value = np.max(values)
        # np.flatnonzero devuelve los índices donde la condición es True
        return np.random.choice(np.flatnonzero(values == max_value))


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float = 0.1) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Policy::Invalid policy parameter")

        self._epsilon = epsilon

    def select_action(self, values: np.ndarray) -> int:
        probs = self.get_probabilities(values)

        return np.random.choice(values.size, p=probs)

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        n_act = values.size
        greedy_idx = self._get_argmax_random_tie_break(values)

        probs = np.full(n_act, self._epsilon / n_act, dtype=np.float64)
        probs[greedy_idx] += 1.0 - self._epsilon

        return probs

    def _get_argmax_random_tie_break(self, values: np.ndarray) -> int:
        """If there are ties in max values, choice ramdomly one"""
        max_value = np.max(values)
        # np.flatnonzero devuelve los índices donde la condición es True
        return np.random.choice(np.flatnonzero(values == max_value))


class SoftmaxPolicy(BasePolicy):
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError("Policy::Invalid policy parameter")

        self._temp = temperature

    def select_action(self, values: np.ndarray) -> int:
        probs = self.get_probabilities(values)

        return np.random.choice(values.size, p=probs)

    def get_probabilities(self, values: np.ndarray) -> np.ndarray:
        return self._softmax(values)

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        scaled_values = values / self._temp
        scaled_values -= np.max(scaled_values)

        num = np.exp(scaled_values)
        return num / np.sum(num)
