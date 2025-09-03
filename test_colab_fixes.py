#!/usr/bin/env python3
"""
Test script to verify Colab notebook fixes.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_metrics_keys():
    """Test that metrics have correct keys."""
    print("🧪 Testing Metrics Keys...")
    
    try:
        from src.evaluation import MetricsCalculator
        import numpy as np
        
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1])
        
        metrics = calc.calculate_all_metrics(y_true, y_proba)
        
        print(f"  ✅ Metrics keys: {list(metrics.keys())}")
        print(f"  ✅ F1 key exists: {'f1' in metrics}")
        print(f"  ✅ F1 value: {metrics.get('f1', 'NOT_FOUND'):.4f}")
        
        # Check for f1_score key (should not exist)
        if 'f1_score' in metrics:
            print(f"  ❌ Old f1_score key still exists!")
            return False
        else:
            print(f"  ✅ Old f1_score key correctly removed")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Metrics test failed: {e}")
        return False

def test_bandit_config():
    """Test bandit configuration."""
    print("\n🧪 Testing Bandit Config...")
    
    try:
        from src import Config
        config = Config.from_env()
        
        min_rate = getattr(config.bandit, 'min_success_rate', None)
        max_rate = getattr(config.bandit, 'max_success_rate', None)
        
        print(f"  ✅ Bandit min_success_rate: {min_rate}")
        print(f"  ✅ Bandit max_success_rate: {max_rate}")
        
        if min_rate is None or max_rate is None:
            print(f"  ❌ Config attributes missing!")
            return False
            
        if min_rate >= max_rate:
            print(f"  ❌ Invalid rate ranges!")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_bandit_class_check():
    """Test bandit class balance checking."""
    print("\n🧪 Testing Bandit Class Balance Check...")
    
    try:
        from src.bandit import EpsilonGreedyBandit
        from src import Config
        import numpy as np
        
        config = Config.from_env()
        bandit = EpsilonGreedyBandit(config.bandit)
        
        # Test with single class data (should skip training)
        bandit.history = [
            {"arm": "voice", "reward": 0, "hour": 9, "dow": 1, "ctr7": 0.5, "lag_1": 0},
            {"arm": "voice", "reward": 0, "hour": 10, "dow": 1, "ctr7": 0.5, "lag_1": 0},
            {"arm": "voice", "reward": 0, "hour": 11, "dow": 1, "ctr7": 0.5, "lag_1": 0},
        ]
        
        # This should not raise an error
        bandit._train_outcome_models()
        print(f"  ✅ Single class data handled gracefully")
        
        # Test with balanced data (should train successfully)
        bandit.history = [
            {"arm": "push", "reward": 0, "hour": 9, "dow": 1, "ctr7": 0.5, "lag_1": 0},
            {"arm": "push", "reward": 1, "hour": 10, "dow": 1, "ctr7": 0.6, "lag_1": 1},
            {"arm": "push", "reward": 0, "hour": 11, "dow": 1, "ctr7": 0.4, "lag_1": 0},
            {"arm": "push", "reward": 1, "hour": 12, "dow": 1, "ctr7": 0.7, "lag_1": 1},
        ]
        
        bandit._train_outcome_models()
        print(f"  ✅ Balanced data trains successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Bandit test failed: {e}")
        return False

def test_bandit_results_format():
    """Test bandit results format for sorting."""
    print("\n🧪 Testing Bandit Results Format...")
    
    try:
        # Simulate different result formats
        test_results = {
            "policy1": 0.5,
            "policy2": {"average_reward": 0.7},
            "policy3": 0.3,
            "policy4": {"total_reward": 10, "average_reward": 0.6}
        }
        
        # Test the sorting logic from notebook
        sorted_results = []
        for policy, reward in test_results.items():
            if isinstance(reward, (int, float)):
                sorted_results.append((policy, reward))
            elif isinstance(reward, dict) and 'average_reward' in reward:
                sorted_results.append((policy, reward['average_reward']))
            else:
                sorted_results.append((policy, 0.0))  # fallback
        
        sorted_results = sorted(sorted_results, key=lambda x: x[1], reverse=True)
        
        print(f"  ✅ Sorted results: {sorted_results}")
        
        # Check order is correct
        expected_order = ["policy2", "policy4", "policy1", "policy3"]
        actual_order = [policy for policy, _ in sorted_results]
        
        if actual_order == expected_order:
            print(f"  ✅ Sorting order correct")
            return True
        else:
            print(f"  ❌ Sorting order incorrect: {actual_order} != {expected_order}")
            return False
            
    except Exception as e:
        print(f"  ❌ Results format test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Colab Notebook Fixes")
    print("=" * 50)
    
    tests = [
        test_metrics_keys,
        test_bandit_config,
        test_bandit_class_check,
        test_bandit_results_format
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All fixes working correctly!")
        return 0
    else:
        print("❌ Some fixes need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
