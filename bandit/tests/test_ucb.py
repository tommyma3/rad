import unittest

from bandit.algorithm import UCB
from bandit.utils import get_config


class DPTUCBTests(unittest.TestCase):
    def test_warmup_visits_every_arm_in_order_regardless_of_rewards_or_seed(self):
        for seed in (0, 1, 42):
            learner = UCB(num_arms=5, seed=seed)
            for expected in range(5):
                self.assertEqual(learner.select_action(), expected)
                learner.update(expected, 100.0 if expected == 0 else -100.0)

    def test_fixed_bonus_avoids_logarithmic_overexploration(self):
        learner = UCB(num_arms=2)
        for _ in range(100):
            learner.update(0, 0.95)
        learner.update(1, 0.0)
        # DPT scores are 1.05 versus 1.0; the old logarithmic bonus picks arm 1.
        self.assertEqual(learner.select_action(), 0)
        learner.exploration_coefficient = 2.0
        self.assertEqual(learner.select_action(), 1)

    def test_exact_ties_choose_first_arm_without_consuming_rng(self):
        learner = UCB(num_arms=5, seed=42)
        for arm in range(5):
            learner.update(arm, 0.5)
        before = learner.state_dict()['rng_state']
        for _ in range(10):
            self.assertEqual(learner.select_action(), 0)
        self.assertEqual(learner.state_dict()['rng_state'], before)

    def test_configured_source_policy_matches_class_default(self):
        config = get_config('config/algorithm/ucb.yaml')
        self.assertEqual(config['algorithm'], 'dpt_ucb')
        learner = UCB(num_arms=2, exploration_coefficient=config['exploration_coefficient'])
        for arm in (0, 1):
            learner.update(arm, 0.0)
        learner.update(0, 1.0)
        self.assertEqual(learner.exploration_coefficient, UCB().exploration_coefficient)
        self.assertEqual(learner.select_action(), 0)


if __name__ == '__main__':
    unittest.main()
