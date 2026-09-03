from __future__ import annotations

import unittest

from rad_memory.check_recurrent_ppo_convergence import has_converged, parse_args
from rad_memory.recurrent_ppo import RecurrentPPOConfig


class RecurrentPPOConvergenceTest(unittest.TestCase):
    def test_requires_stable_tail_not_an_earlier_peak(self):
        evaluations = [
            {"success_rate": 0.95},
            {"success_rate": 0.92},
            {"success_rate": 0.89},
        ]
        self.assertFalse(
            has_converged(
                evaluations,
                minimum_success_rate=0.9,
                required_consecutive_evals=3,
            )
        )
        evaluations[-1]["success_rate"] = 0.91
        self.assertTrue(
            has_converged(
                evaluations,
                minimum_success_rate=0.9,
                required_consecutive_evals=3,
            )
        )

    def test_probe_defaults_match_shared_collection_configuration(self):
        args = parse_args(["--no-progress-bar"])
        config = RecurrentPPOConfig()
        self.assertEqual(args.n_steps, config.n_steps)
        self.assertEqual(args.batch_size, config.batch_size)
        self.assertEqual(args.learning_rate, config.learning_rate)
        self.assertEqual(args.gamma, config.gamma)
        self.assertEqual(args.gae_lambda, config.gae_lambda)
        self.assertEqual(args.ent_coef, config.ent_coef)
        self.assertEqual(args.n_epochs, config.n_epochs)


if __name__ == "__main__":
    unittest.main()
