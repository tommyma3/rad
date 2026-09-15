import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from bandit.algorithm import UCB
from bandit.env import BANDIT, BanditTask, sample_task
from bandit.rollout import generate_history
from bandit.test_ucb_convergence import (
    evaluate_recommendation, has_converged, main, parse_args, run_episode, same_state,
)


class UCBConvergenceTests(unittest.TestCase):
    def test_episode_matches_collection_and_preserves_state_across_episodes(self):
        task = sample_task(7)
        learner = UCB(seed=1)
        first = run_episode(task, learner, pre_steps=50, post_steps=50, delay=100,
                            reward_seed=0, distractor_seed=2)
        expected, _ = generate_history(task, delay=100, reward_seed=0,
                                        learner_seed=1, distractor_seed=2)
        for key in ("states", "actions", "rewards"):
            np.testing.assert_array_equal(first[key], expected[key])
        run_episode(task, learner, pre_steps=50, post_steps=50, delay=100,
                    reward_seed=3, distractor_seed=4)
        self.assertEqual(learner.total_pulls, 200)

    def test_delays_and_heldout_evaluation_cannot_change_training_history(self):
        task = sample_task(2)
        first, second = UCB(seed=3), UCB(seed=3)
        for episode in range(3):
            plain = run_episode(task, first, pre_steps=10, post_steps=10, delay=0,
                                reward_seed=episode, distractor_seed=episode + 10)
            delayed = run_episode(task, second, pre_steps=10, post_steps=10, delay=13,
                                  reward_seed=episode, distractor_seed=episode + 10)
            for key in ("actions", "rewards"):
                np.testing.assert_array_equal(plain[key], delayed[key][delayed["states"] == BANDIT])
            snapshot = second.state_dict()
            result = evaluate_recommendation(task, second, 100, episode)
            self.assertTrue(same_state(snapshot, second.state_dict()))
            self.assertTrue(same_state(first.state_dict(), second.state_dict()))
            self.assertIn("heldout_mean_reward", result)

    def test_only_final_consecutive_checks_count(self):
        self.assertFalse(has_converged([{"passed": True}], 3))
        self.assertFalse(has_converged([{"passed": True}] * 3 + [{"passed": False}], 3))
        self.assertTrue(has_converged([{"passed": False}] + [{"passed": True}] * 3, 3))

    def test_heldout_rewards_are_gaussian(self):
        task = BanditTask("fixed", (0.5, 0.5), "uniform", 0)
        learner = UCB(num_arms=2)
        learner.update(0, 1.0)
        result = evaluate_recommendation(task, learner, 50000, 8)
        self.assertAlmostEqual(result["heldout_mean_reward"], 0.5, delta=0.01)
        self.assertAlmostEqual(result["heldout_reward_standard_error"] * np.sqrt(50000), 0.3, delta=0.01)

    def test_actual_learning_passes_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "check"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--episodes", "100", "--seeds", "0", "--task-seeds", "0",
                               "--evaluation-pulls", "64", "--output-dir", str(output)])
            self.assertEqual(status, 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["passed"])
            result = summary["tasks"][0]["runs"][0]
            self.assertEqual(result["genuine_pulls"], 10000)
            self.assertLessEqual(result["final_evaluation"]["online_mean_regret"], 0.05)
            self.assertEqual(result["final_consecutive_checks"], [True] * 3)
            self.assertTrue((output / "task-0" / "learning_curves.png").exists())
            with np.load(output / "task-0" / "seed-0" / "history.npz") as history:
                self.assertEqual(history["actions"].shape, (100, 200))
            with self.assertRaises(FileExistsError):
                main(["--output-dir", str(output)])

    def test_pull_budget_is_exact_even_when_not_episode_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pulls"
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--pulls", "250", "--seeds", "0", "--task-seeds", "0",
                               "--eval-interval", "100", "--window-pulls", "100",
                               "--required-consecutive-evals", "1", "--max-mean-regret", "1",
                               "--max-recommendation-regret", "1", "--evaluation-pulls", "16",
                               "--output-dir", str(output)])
            self.assertEqual(status, 0)
            result = json.loads((output / "task-0" / "seed-0" / "result.json").read_text())
            self.assertEqual(result["arm_pulls"], 250)
            with np.load(output / "task-0" / "seed-0" / "history.npz") as history:
                self.assertEqual(history["episode_offsets"][-1], 450)

    def test_insufficient_learning_returns_nonzero_from_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fail"
            script = Path(__file__).resolve().parents[1] / "test_ucb_convergence.py"
            completed = subprocess.run([
                sys.executable, str(script), "--episodes", "1", "--seeds", "0", "--eval-interval", "1",
                "--window-episodes", "1", "--required-consecutive-evals", "1", "--max-mean-regret", "0",
                "--output-dir", str(output)], capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            self.assertFalse(json.loads((output / "summary.json").read_text())["passed"])

    def test_argument_validation(self):
        for options in (["--episodes", "0"], ["--seeds", "1", "1"],
                        ["--max-mean-regret", "nan"], ["--required-seed-fraction", "0"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(options)


if __name__ == "__main__":
    unittest.main()
