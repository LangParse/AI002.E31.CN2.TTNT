"""
Epsilon-Greedy contextual bandit implementation for AI Medication Reminder.

Implements epsilon-greedy strategy for channel selection according to pipeline specification A.7.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import BanditConfig


class EpsilonGreedyBandit:
    """Epsilon-Greedy contextual bandit for medication reminder channel selection."""

    def __init__(self, config: BanditConfig, arms: Optional[List[str]] = None):
        self.config = config
        self.arms = arms or config.arms
        self.n_arms = len(self.arms)

        # Initialize outcome models for each arm
        self.outcome_models = {}
        for arm in self.arms:
            self.outcome_models[arm] = LogisticRegression(
                max_iter=1000,
                solver="liblinear",  # Better for small datasets
                random_state=42,
            )

        # Track statistics
        self.arm_counts = {arm: 0 for arm in self.arms}
        self.arm_rewards = {arm: 0.0 for arm in self.arms}
        self.total_rounds = 0

        # History for model training
        self.history = []

    def _get_epsilon(self, t: int) -> float:
        """
        Calculate epsilon value based on decay strategy.

        Args:
            t: Current time step

        Returns:
            Epsilon value for exploration
        """
        if self.config.epsilon_decay == "sqrt":
            return self.config.epsilon_initial / np.sqrt(max(1, t))
        elif self.config.epsilon_decay == "linear":
            return max(0.01, self.config.epsilon_initial * (1 - t / 1000))
        else:
            return self.config.epsilon_initial

    def _train_outcome_models(self) -> None:
        """Train outcome models for each arm using historical data."""
        if len(self.history) < self.config.outcome_model_min_samples:
            return

        # Convert history to DataFrame
        df = pd.DataFrame(self.history)

        # Train model for each arm
        for arm in self.arms:
            arm_data = df[df["arm"] == arm]

            if (
                len(arm_data) >= 2
            ):  # Minimum samples per arm (reduced for faster training)
                X = np.array(arm_data[self.config.context_features].values)
                y = np.array(arm_data["reward"].values)

                # Check if we have at least 2 different classes
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    # Skip training if only one class present
                    continue

                # Also check if we have reasonable class balance
                success_rate = np.mean(y)
                min_rate = getattr(self.config, "min_success_rate", 0.05)
                max_rate = getattr(self.config, "max_success_rate", 0.95)

                if success_rate < min_rate or success_rate > max_rate:
                    # Skip training if extremely imbalanced
                    continue

                try:
                    self.outcome_models[arm].fit(X, y)
                except Exception as e:
                    print(f"Warning: Failed to train model for arm {arm}: {str(e)}")

    def select_arm(self, context: Dict[str, float]) -> str:
        """
        Select an arm using epsilon-greedy strategy.

        Args:
            context: Dictionary of context features

        Returns:
            Selected arm (channel)
        """
        self.total_rounds += 1
        epsilon = self._get_epsilon(self.total_rounds)

        # Exploration: random selection
        if np.random.random() < epsilon:
            selected_arm = np.random.choice(self.arms)
            return selected_arm

        # Exploitation: select best arm based on models
        if len(self.history) >= self.config.outcome_model_min_samples:
            # Use trained models to predict rewards
            context_array = np.array(
                [context[feat] for feat in self.config.context_features]
            ).reshape(1, -1)

            predicted_rewards = {}
            for arm in self.arms:
                try:
                    # Check if model is fitted by looking for required attributes
                    model = self.outcome_models[arm]
                    if hasattr(model, "coef_") and model.coef_ is not None:
                        # Model is fitted, use it for prediction
                        prob = model.predict_proba(context_array)[0, 1]
                        predicted_rewards[arm] = prob
                    else:
                        # Model not fitted yet, use empirical average
                        predicted_rewards[arm] = self.arm_rewards[arm] / max(
                            1, self.arm_counts[arm]
                        )
                except Exception as e:
                    # Fallback to empirical average
                    predicted_rewards[arm] = self.arm_rewards[arm] / max(
                        1, self.arm_counts[arm]
                    )
                    # Only print warning for unexpected errors, not "not fitted" errors
                    if "not fitted" not in str(e).lower():
                        print(
                            f"Warning: Failed to predict with model for arm {arm}: {str(e)}"
                        )

            # Select arm with highest predicted reward
            selected_arm = max(
                predicted_rewards.keys(), key=lambda x: predicted_rewards[x]
            )
        else:
            # Fallback to empirical averages
            arm_averages = {
                arm: self.arm_rewards[arm] / max(1, self.arm_counts[arm])
                for arm in self.arms
            }
            selected_arm = max(arm_averages.keys(), key=lambda x: arm_averages[x])

        return selected_arm

    def update(self, arm: str, reward: float, context: Dict[str, float]) -> None:
        """
        Update bandit with observed reward.

        Args:
            arm: Selected arm
            reward: Observed reward (0 or 1)
            context: Context features used for selection
        """
        # Update statistics
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward

        # Add to history
        history_entry = {"arm": arm, "reward": reward, "round": self.total_rounds}

        # Add context features
        for feat in self.config.context_features:
            history_entry[feat] = context.get(feat, 0.0)

        self.history.append(history_entry)

        # Retrain models periodically (more frequently at start)
        if len(self.history) <= 20:
            # Train more frequently at the beginning
            if len(self.history) % 5 == 0:
                self._train_outcome_models()
        elif len(self.history) % 20 == 0:  # Retrain every 20 observations
            self._train_outcome_models()

    def get_arm_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each arm.

        Returns:
            Dictionary with statistics for each arm
        """
        stats = {}
        for arm in self.arms:
            stats[arm] = {
                "count": self.arm_counts[arm],
                "total_reward": self.arm_rewards[arm],
                "average_reward": self.arm_rewards[arm] / max(1, self.arm_counts[arm]),
                "selection_rate": self.arm_counts[arm] / max(1, self.total_rounds),
            }
        return stats

    def get_regret_bounds(self) -> Dict[str, float]:
        """
        Calculate theoretical regret bounds.

        Returns:
            Dictionary with regret bound information
        """
        if self.total_rounds == 0:
            return {"theoretical_regret": 0.0, "empirical_regret": 0.0}

        # Simplified regret calculation
        # In practice, you'd need the true optimal arm to calculate real regret
        arm_stats = self.get_arm_statistics()
        best_empirical_reward = max(
            stats["average_reward"] for stats in arm_stats.values()
        )

        # Empirical regret (assuming best observed arm is optimal)
        total_reward = sum(self.arm_rewards.values())
        empirical_regret = (best_empirical_reward * self.total_rounds) - total_reward

        return {
            "theoretical_regret": np.sqrt(
                self.n_arms * self.total_rounds * np.log(self.total_rounds)
            ),
            "empirical_regret": empirical_regret,
            "best_arm_reward": best_empirical_reward,
        }

    def reset(self) -> None:
        """Reset bandit state for new simulation."""
        self.arm_counts = {arm: 0 for arm in self.arms}
        self.arm_rewards = {arm: 0.0 for arm in self.arms}
        self.total_rounds = 0

        # Reset outcome models if they exist
        if hasattr(self, "outcome_models"):
            self.outcome_models = {
                arm: LogisticRegression(
                    max_iter=1000, solver="liblinear", random_state=42
                )
                for arm in self.arms
            }
