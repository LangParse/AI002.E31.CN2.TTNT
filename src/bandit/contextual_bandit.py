"""
Contextual bandit base class for AI Medication Reminder.

Provides base interface and common functionality for contextual bandit algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd


class ContextualBandit(ABC):
    """Abstract base class for contextual bandit algorithms."""

    def __init__(self, arms: List[str]):
        self.arms = arms
        self.n_arms = len(arms)
        self.history = []

    @abstractmethod
    def select_arm(self, context: Dict[str, float]) -> str:
        """
        Select an arm given the context.

        Args:
            context: Dictionary of context features

        Returns:
            Selected arm
        """
        pass

    @abstractmethod
    def update(self, arm: str, reward: float, context: Dict[str, float]) -> None:
        """
        Update the bandit with observed reward.

        Args:
            arm: Selected arm
            reward: Observed reward
            context: Context features
        """
        pass

    def get_history(self) -> pd.DataFrame:
        """
        Get the history of interactions.

        Returns:
            DataFrame with interaction history
        """
        return pd.DataFrame(self.history)

    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.history = []

    def get_cumulative_reward(self) -> float:
        """
        Get total cumulative reward.

        Returns:
            Sum of all rewards received
        """
        if not self.history:
            return 0.0
        return sum(entry["reward"] for entry in self.history)

    def get_cumulative_regret(self, optimal_rewards: Dict[str, float]) -> List[float]:
        """
        Calculate cumulative regret over time.

        Args:
            optimal_rewards: Dictionary mapping contexts to optimal rewards

        Returns:
            List of cumulative regret values
        """
        cumulative_regret = []
        regret_sum = 0.0

        for entry in self.history:
            # Find optimal reward for this context (simplified)
            optimal_reward = max(optimal_rewards.values())  # Simplified assumption
            regret = optimal_reward - entry["reward"]
            regret_sum += regret
            cumulative_regret.append(regret_sum)

        return cumulative_regret
