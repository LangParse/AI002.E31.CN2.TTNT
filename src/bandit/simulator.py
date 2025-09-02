"""
Bandit simulation for AI Medication Reminder.

Simulates contextual bandit performance for channel selection optimization.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from .epsilon_greedy import EpsilonGreedyBandit


class BanditSimulator:
    """Simulates contextual bandit algorithms for medication reminder optimization."""

    def __init__(self, config: Config):
        self.config = config

    def simulate_bandit_policy(
        self, bandit, test_df: pd.DataFrame, context_features: List[str]
    ) -> Dict:
        """
        Simulate bandit policy on test data.

        Args:
            bandit: Bandit algorithm instance
            test_df: Test DataFrame with context and outcomes
            context_features: List of context feature names

        Returns:
            Dictionary with simulation results
        """
        print(f"Simulating bandit policy on {len(test_df)} interactions...")

        # Reset bandit
        bandit.reset()

        results = {
            "selected_arms": [],
            "rewards": [],
            "contexts": [],
            "cumulative_reward": [],
            "arm_selection_rates": [],
        }

        cumulative_reward = 0

        for idx, row in test_df.iterrows():
            # Extract context
            context = {feat: row[feat] for feat in context_features}

            # Select arm
            selected_arm = bandit.select_arm(context)

            # Get reward (simulate based on actual outcome if arm matches)
            if selected_arm == row["channel"]:
                reward = row["responded_within_2h"]
            else:
                # Simulate reward for counterfactual arm
                # This is simplified - in practice you'd need a more sophisticated model
                base_prob = 0.55  # Base response probability
                if selected_arm == "push":
                    prob = base_prob + 0.05
                elif selected_arm == "voice":
                    prob = base_prob - 0.05
                else:
                    prob = base_prob
                reward = np.random.binomial(1, prob)

            # Update bandit
            bandit.update(selected_arm, reward, context)

            # Record results
            results["selected_arms"].append(selected_arm)
            results["rewards"].append(reward)
            results["contexts"].append(context)

            cumulative_reward += reward
            results["cumulative_reward"].append(cumulative_reward)

            # Calculate current arm selection rates
            arm_counts = bandit.get_arm_statistics()
            selection_rates = {
                arm: stats["selection_rate"] for arm, stats in arm_counts.items()
            }
            results["arm_selection_rates"].append(selection_rates.copy())

        # Final statistics
        final_stats = bandit.get_arm_statistics()
        regret_bounds = bandit.get_regret_bounds()

        results["final_statistics"] = final_stats
        results["regret_bounds"] = regret_bounds
        results["total_reward"] = cumulative_reward
        results["average_reward"] = cumulative_reward / len(test_df)

        print(
            f"Simulation completed. Total reward: {cumulative_reward}, Average: {results['average_reward']:.4f}"
        )

        return results

    def compare_policies(
        self, test_df: pd.DataFrame, context_features: List[str]
    ) -> Dict:
        """
        Compare different bandit policies.

        Args:
            test_df: Test DataFrame
            context_features: List of context feature names

        Returns:
            Dictionary with comparison results
        """
        print("Comparing bandit policies...")

        policies = {}

        # Epsilon-Greedy with different parameters
        for epsilon in [0.1, 0.2, 0.3]:
            config = self.config.bandit
            config.epsilon_initial = epsilon

            bandit = EpsilonGreedyBandit(config)
            results = self.simulate_bandit_policy(bandit, test_df, context_features)
            policies[f"epsilon_greedy_{epsilon}"] = results

        # Random policy (baseline)
        random_results = self._simulate_random_policy(test_df)
        policies["random"] = random_results

        # Fixed policy (always use best historical channel)
        fixed_results = self._simulate_fixed_policy(test_df)
        policies["fixed_best"] = fixed_results

        # Compare results
        comparison = self._compare_policy_results(policies)

        return {"policies": policies, "comparison": comparison}

    def _simulate_random_policy(self, test_df: pd.DataFrame) -> Dict:
        """Simulate random channel selection policy."""
        arms = self.config.bandit.arms
        selected_arms = np.random.choice(arms, len(test_df))

        rewards = []
        cumulative_reward = 0
        cumulative_rewards = []

        for i, (idx, row) in enumerate(test_df.iterrows()):
            selected_arm = selected_arms[i]

            if selected_arm == row["channel"]:
                reward = row["responded_within_2h"]
            else:
                # Simulate counterfactual reward
                base_prob = 0.55
                if selected_arm == "push":
                    prob = base_prob + 0.05
                elif selected_arm == "voice":
                    prob = base_prob - 0.05
                else:
                    prob = base_prob
                reward = np.random.binomial(1, prob)

            rewards.append(reward)
            cumulative_reward += reward
            cumulative_rewards.append(cumulative_reward)

        return {
            "selected_arms": selected_arms.tolist(),
            "rewards": rewards,
            "cumulative_reward": cumulative_rewards,
            "total_reward": cumulative_reward,
            "average_reward": cumulative_reward / len(test_df),
        }

    def _simulate_fixed_policy(self, test_df: pd.DataFrame) -> Dict:
        """Simulate fixed best channel policy."""
        # Find historically best channel
        channel_performance = test_df.groupby("channel")["responded_within_2h"].mean()
        best_channel = channel_performance.idxmax()

        selected_arms = [best_channel] * len(test_df)
        rewards = []
        cumulative_reward = 0
        cumulative_rewards = []

        for i, (idx, row) in enumerate(test_df.iterrows()):
            if best_channel == row["channel"]:
                reward = row["responded_within_2h"]
            else:
                # Simulate counterfactual reward
                reward = np.random.binomial(1, channel_performance[best_channel])

            rewards.append(reward)
            cumulative_reward += reward
            cumulative_rewards.append(cumulative_reward)

        return {
            "selected_arms": selected_arms,
            "rewards": rewards,
            "cumulative_reward": cumulative_rewards,
            "total_reward": cumulative_reward,
            "average_reward": cumulative_reward / len(test_df),
            "best_channel": best_channel,
        }

    def _compare_policy_results(self, policies: Dict) -> Dict:
        """Compare results from different policies."""
        comparison = {"average_rewards": {}, "total_rewards": {}, "ranking": []}

        for policy_name, results in policies.items():
            comparison["average_rewards"][policy_name] = results["average_reward"]
            comparison["total_rewards"][policy_name] = results["total_reward"]

        # Rank policies by average reward
        ranked_policies = sorted(
            comparison["average_rewards"].items(), key=lambda x: x[1], reverse=True
        )
        comparison["ranking"] = [policy[0] for policy in ranked_policies]

        print("\nPolicy Comparison Results:")
        print("-" * 40)
        for i, (policy, avg_reward) in enumerate(ranked_policies):
            print(f"{i + 1}. {policy}: {avg_reward:.4f}")

        return comparison

    def plot_policy_comparison(
        self, policies: Dict, save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparison of different policies.

        Args:
            policies: Dictionary of policy results
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Plot cumulative rewards
            plt.subplot(2, 2, 1)
            for policy_name, results in policies.items():
                if "cumulative_reward" in results:
                    plt.plot(results["cumulative_reward"], label=policy_name)
            plt.xlabel("Time Steps")
            plt.ylabel("Cumulative Reward")
            plt.title("Cumulative Reward Over Time")
            plt.legend()
            plt.grid(True)

            # Plot average rewards
            plt.subplot(2, 2, 2)
            policy_names = list(policies.keys())
            avg_rewards = [policies[p]["average_reward"] for p in policy_names]
            plt.bar(policy_names, avg_rewards)
            plt.xlabel("Policy")
            plt.ylabel("Average Reward")
            plt.title("Average Reward by Policy")
            plt.xticks(rotation=45)

            # Plot arm selection rates (for bandit policies)
            plt.subplot(2, 2, 3)
            for policy_name, results in policies.items():
                if "final_statistics" in results:
                    arms = list(results["final_statistics"].keys())
                    selection_rates = [
                        results["final_statistics"][arm]["selection_rate"]
                        for arm in arms
                    ]
                    plt.bar(
                        [f"{policy_name}_{arm}" for arm in arms],
                        selection_rates,
                        alpha=0.7,
                    )
            plt.xlabel("Policy-Arm")
            plt.ylabel("Selection Rate")
            plt.title("Arm Selection Rates")
            plt.xticks(rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Policy comparison plot saved to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plot: {str(e)}")
