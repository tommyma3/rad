import json
from pathlib import Path
import tempfile
import unittest

from rad_memory.check_ppo_convergence import (
    collection_settings,
    main,
    parse_args,
    sample_task,
    task_template,
)
from rad_memory.envs import MemoryTaskSpec
from rad_memory.ppo import PPOConfig
from rad_memory.task_pool import freeze_task, generate_pool, load_pool


class PPOConvergenceTest(unittest.TestCase):
    def test_defaults_and_saved_collection_config_match(self):
        config, budget, streams = collection_settings(parse_args([]))
        self.assertEqual(config, PPOConfig())
        self.assertEqual(budget["total_timesteps"], 200_000)
        self.assertEqual(budget["evaluation_interval"], 10_000)
        self.assertEqual(budget["evaluation_episodes"], 20)
        self.assertEqual(budget["minimum_success_rate"], 0.9)
        self.assertEqual(budget["required_consecutive_evals"], 2)
        self.assertEqual(streams, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "source_config.json")
            config = PPOConfig(n_steps=16, batch_size=8, learning_rate=0.001, n_epochs=2)
            path.write_text(json.dumps({"source_algorithm": "ppo", "ppo": config.to_dict()}))
            args = parse_args(["--ppo-config", str(path)])
            self.assertEqual(collection_settings(args)[0], config)
            # Explicit flags override the saved config.
            args = parse_args(["--ppo-config", str(path), "--batch-size", "32"])
            self.assertEqual(collection_settings(args)[0].batch_size, 32)
            # YAML source configs (e.g. config/source/ppo.yaml) contribute the
            # hyperparameters and the budget/criterion/stream keys.
            yaml_path = Path(tmp, "source_config.yaml")
            yaml_path.write_text(
                "source_algorithm: ppo\n"
                "ppo:\n"
                f"  n_steps: {config.n_steps}\n"
                f"  batch_size: {config.batch_size}\n"
                f"  learning_rate: {config.learning_rate}\n"
                f"  n_epochs: {config.n_epochs}\n"
                "total_timesteps: 1234\n"
                "evaluation_interval: 111\n"
                "evaluation_episodes: 7\n"
                "minimum_success_rate: 0.5\n"
                "required_consecutive_evals: 4\n"
                "streams: 3\n",
                encoding="utf-8",
            )
            loaded, budget, streams = collection_settings(parse_args(["--ppo-config", str(yaml_path)]))
            self.assertEqual(loaded, config)
            self.assertEqual(budget["total_timesteps"], 1234)
            self.assertEqual(budget["evaluation_interval"], 111)
            self.assertEqual(budget["evaluation_episodes"], 7)
            self.assertEqual(budget["minimum_success_rate"], 0.5)
            self.assertEqual(budget["required_consecutive_evals"], 4)
            self.assertEqual(streams, 3)
            # Flags still win over the file budget and streams.
            loaded, budget, streams = collection_settings(parse_args(
                ["--ppo-config", str(yaml_path), "--total-timesteps", "99", "--streams", "2"]))
            self.assertEqual(budget["total_timesteps"], 99)
            self.assertEqual(streams, 2)
            # Raw PPOConfig JSON (flat hyperparameters) is accepted too.
            raw_path = Path(tmp, "ppo_config.json")
            raw_path.write_text(json.dumps(config.to_dict()))
            self.assertEqual(collection_settings(parse_args(["--ppo-config", str(raw_path)]))[0], config)
            path.write_text(json.dumps({"source_algorithm": "recurrent_ppo"}))
            with self.assertRaisesRegex(ValueError, "not standard PPO"):
                collection_settings(parse_args(["--ppo-config", str(path)]))

    def test_rejects_invalid_budget_and_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp, "bad.yaml")
            yaml_path.write_text("source_algorithm: ppo\nstreams: 0\n")
            with self.assertRaisesRegex(ValueError, "positive"):
                collection_settings(parse_args(["--ppo-config", str(yaml_path)]))
            yaml_path.write_text("source_algorithm: ppo\nminimum_success_rate: 2\n")
            with self.assertRaisesRegex(ValueError, "Success rate"):
                collection_settings(parse_args(["--ppo-config", str(yaml_path)]))
            with self.assertRaises(SystemExit):
                parse_args(["--streams", "0"])

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

    def test_profile_supplies_collection_episode_length(self):
        import random
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp, "profile.json")
            profile_path.write_text(json.dumps({
                "recommended_horizon": 32,
                "short_context": 6,
                "task_spec": {"env_id": "MiniGrid-MemoryS17Random-v0", "controlled": True,
                              "random_length": True, "size": 17, "horizon": None,
                              "seed": 0, "split": "profile"},
            }))
            template = task_template(parse_args(["--profile", str(profile_path)]))
            self.assertEqual(template["horizon"], 32)
            self.assertEqual(template["env_id"], "MiniGrid-MemoryS17Random-v0")
            self.assertTrue(template["controlled"])
            self.assertTrue(template["random_length"])
            self.assertEqual(template["size"], 17)
            # Explicit flags still win over the profile.
            template = task_template(parse_args(
                ["--profile", str(profile_path), "--horizon", "12",
                 "--env-id", "MiniGrid-MemoryS13Random-v0"]))
            self.assertEqual(template["horizon"], 12)
            self.assertEqual(template["env_id"], "MiniGrid-MemoryS13Random-v0")
            # Generated tasks freeze the profiled episode length.
            spec = sample_task(
                parse_args(["--profile", str(profile_path), "--task-seed", "3"]),
                random.Random(0))
            self.assertEqual(spec.configuration["max_steps"], 32)
            # Saved tasks already carry the collection episode length.
            with self.assertRaises(SystemExit):
                parse_args(["--profile", str(profile_path), "--manifest", "tasks.json"])

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

    def test_streamed_probe_reports_and_converges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = freeze_task(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "test", horizon=1))
            task_path = root / "task.json"
            task_path.write_text(json.dumps(spec.to_dict()))
            main(["--task-spec", str(task_path), "--seed", "0",
                  "--total-timesteps", "16", "--evaluation-interval", "8",
                  "--evaluation-episodes", "1", "--required-consecutive-evals", "1",
                  "--n-steps", "8", "--batch-size", "16", "--n-epochs", "1",
                  "--streams", "2", "--minimum-success-rate", "0",
                  "--output-dir", str(root / "streams"), "--no-progress-bar"])
            summary = json.loads((root / "streams/summary.json").read_text())
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["streams"], 2)
            self.assertEqual(summary["training_budget"], 16)


if __name__ == "__main__":
    unittest.main()
