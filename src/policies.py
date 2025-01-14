"""Policies"""
import numpy as np
from typing import Optional, List


def random_policy(Q: np.ndarray, state: int, feasible_actions: Optional[List[int]] = None) -> tuple[int, float]:
    if feasible_actions is None:
        feasible_actions = list(range(Q.shape[1]))
    action = np.random.choice(feasible_actions)
    value = float(Q[state, action])
    return action, value


def greedy_policy(Q: np.ndarray, state: int, feasible_actions: Optional[List[int]] = None) -> tuple[int, float]:
    # Ensure feasible_actions contain indexes within the Q-values array
    if feasible_actions is None:
        feasible_actions = list(range(Q.shape[1]))

    # Create mask for actions to filter out infeasible actions
    mask = np.zeros(Q.shape[1], dtype=bool)
    mask[feasible_actions] = True
    masked_state_values = np.where(mask, Q[state, :], -np.inf)

    # Apply greedy selection and get action, value
    action: int = np.argmax(masked_state_values).item()
    value: float = float(Q[state, action])
    return action, value


def epsilon_greedy_policy(Q: np.ndarray, state: int, feasible_actions: Optional[List[int]] = None, epsilon: float = 0.1) -> tuple[int, float]:
    if np.random.rand() < epsilon:
        # Random Policy
        return random_policy(Q, state, feasible_actions=feasible_actions)
    # Greedy Policy
    return greedy_policy(Q, state, feasible_actions=feasible_actions)
