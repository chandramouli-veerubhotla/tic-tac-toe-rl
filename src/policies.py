"""Policies"""
import abc
import numpy as np
from typing import Optional, List


class Policy(abc.ABC):

    def __init__(self, Q: np.ndarray):
        if len(Q.shape) != 2:
            raise ValueError("Q-values must be a 2D array")
        # Maintain Q function
        self._Q = Q
        self._num_states = Q.shape[0]
        self._num_actions = Q.shape[1]

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def Q(self):
        return self._Q

    @abc.abstractmethod
    def __call__(self, state: int, feasible_actions: Optional[List[int]] = None) -> tuple[int, float]:
        raise NotImplementedError()


class RandomPolicy(Policy):

    def __call__(self, state: int, feasible_actions: Optional[List[int]] = None) -> tuple[int, float]:
        if feasible_actions is None:
            feasible_actions = list(range(self._num_actions))
        action = np.random.choice(feasible_actions)
        value = float(self.Q[state, action])
        return action, value


class GreedyPolicy(Policy):
    """Greedy policy: Selects the action with the highest Q-value for a given state."""

    def __call__(self, state: int, feasible_actions: Optional[List[int]] = None) -> tuple[int, float]:
        action: int = np.argmax(self.Q[state, :]).item()
        value: float = float(self.Q[state, action])
        return action, value


class EpsilonGreedyPolicy(RandomPolicy, GreedyPolicy):
    """Epsilon-greedy policy: Selects the action with the highest Q-value with probability (1 - epsilon)
    and selects a random action with probability epsilon."""

    def __init__(self, Q: np.ndarray, epsilon: float = 0.1):
        super().__init__(Q)
        # Epsilon value
        self._epsilon = None
        self.epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("Epsilon must be in the range [0, 1]")
        self._epsilon = value

    def __call__(self, state: int, feasible_actions: Optional[List[int]] = None) -> tuple[int, float]:
        if np.random.rand() < self._epsilon:
            # Random Policy
            return super(RandomPolicy, self).__call__(state, feasible_actions=feasible_actions)
        # Greedy Policy
        return super(GreedyPolicy, self).__call__(state, feasible_actions=feasible_actions)
