import json
from pathlib import Path
import tempfile
import unittest

from rad_memory.check_ppo_convergence import collection_config, main, parse_args, sample_task
from rad_memory.envs import MemoryTaskSpec
from rad_memory.ppo import PPOConfig
from rad_memory.task_pool import freeze_task, generate_pool, load_pool


class PPOConvergenceTest(unittest.TestCase):
    def test_defaults_and_saved_collection_config_match(self):
        self.assertEqual(collection_config(parse_args([])), PPOConfig())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "source_config.json")
            config = PPOConfig(n_steps=16, batch_size=8, learning_rate=0.001, n_epochs=2)
            path.write_text(json.dumps({"source_algorithm": "ppo", "ppo": config.to_dict()}))
            args = parse_args(["--ppo-config", str(path)])
            self.assertEqual(collection_config(args), config)
            # Explicit flags override the saved config.
            args = parse_args(["--ppo-config", str(path), "--batch-size", "32"])
            self.assertEqual(collection_config(args).batch_size, 32)
            path.write_text(json.dumps({"source_algorithm": "recurrent_ppo"}))
            with self.assertRaisesRegex(ValueError, "not standard PPO"):
                collection_config(parse_args(["--ppo-config", str(path)]))

    def _pool_args(self, tmp):
        path = Path(tmp, "tasks.json")
        pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=30), 4, 0.5, 0)
        path.write_text(json.dumps(pool))
        return path, pool

    def test_samples_a_task_from_the_pool(self):
        import random
        with tempfile.TemporaryDirectory() as tmp:
            path, pool = self._pool_args(tmp)
            args = parse_args(["--manifest", str(path), "--sample-seed", "3"])
            spec = sample_task(args, random.Random(args.sample_seed))
            self.assertIn(spec.to_dict(), pool["tasks"])
            # The same sample seed must redraw the same task.
            self.assertEqual(sample_task(args, random.Random(3)).to_dict(), spec.to_dict())
            # A different sample seed should be able to draw another task.
            other = sample_task(args, random.Random(4))
            self.assertIn(other.to_dict(), pool["tasks"])

    def test_pins_manifest_task_and_honors_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, pool = self._pool_args(tmp)
            target = next(task for task in pool["tasks"] if task["split"] == "test")
            args = parse_args(["--manifest", str(path), "--task-id", target["task_id"]])
            self.assertEqual(sample_task(args, None).to_dict(), target)
            args = parse_args(["--manifest", str(path), "--task-id", "missing"])
            with self.assertRaisesRegex(ValueError, "No matching task"):
                sample_task(args, None)
            import random
            args = parse_args(["--manifest", str(path), "--split", "train"])
            self.assertEqual(sample_task(args, random.Random(0)).split, "train")
            with self.assertRaises(SystemExit):
                parse_args(["--manifest", str(path), "--horizon", "10"])
            with self.assertRaises(SystemExit):
                parse_args(["--task-id", target["task_id"]])

    def test_generates_and_freezes_a_random_task(self):
        import random
        args = parse_args(["--sample-seed", "5"])
        spec = sample_task(args, random.Random(5))
        self.assertEqual(spec.env_id, "MiniGrid-MemoryS13Random-v0")
        self.assertIsNotNone(spec.configuration)
        again = sample_task(parse_args(["--sample-seed", "5"]), random.Random(5))
        self.assertEqual(again.task_id, spec.task_id)
        fixed = sample_task(parse_args(["--task-seed", "7", "--horizon", "30"]), random.Random(0))
        self.assertEqual(fixed.seed, 7)

    def test_rejects_changing_layout_saved_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "task.json")
            path.write_text(json.dumps(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train").to_dict()))
            with self.assertRaisesRegex(ValueError, "fixed configuration"):
                sample_task(parse_args(["--task-spec", str(path)]), None)

    def test_real_training_reports_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = freeze_task(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "test", horizon=1))
            task_path = root / "task.json"
            task_path.write_text(json.dumps(spec.to_dict()))
            common = ["--task-spec", str(task_path), "--seed", "0",
                      "--total-timesteps", "8", "--evaluation-interval", "8",
                      "--evaluation-episodes", "1", "--required-consecutive-evals", "1",
                      "--n-steps", "8", "--batch-size", "8", "--n-epochs", "1"]
            # No action can reach the branch from this initial pose in one step.
            with self.assertRaises(SystemExit) as failure:
                main(common + ["--output-dir", str(root / "fail"), "--no-progress-bar"])
            self.assertEqual(failure.exception.code, 1)
            summary = json.loads((root / "fail/summary.json").read_text())
            self.assertFalse(summary["passed"])
            self.assertFalse(summary["converged"])
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["task_spec"], spec.to_dict())
            self.assertEqual(summary["final_success_rate"], 0)
            self.assertTrue((root / "fail/learning-curve.png").exists())
            self.assertTrue((root / "fail/evaluations.json").exists())
            main(common + ["--output-dir", str(root / "pass"), "--minimum-success-rate", "0",
                           "--no-progress-bar"])
            self.assertTrue(json.loads((root / "pass/summary.json").read_text())["passed"])
            # A zero threshold above tests reporting only, not learned competence.


if __name__ == "__main__":
    unittest.main()
