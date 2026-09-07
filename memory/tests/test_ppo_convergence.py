import json
from pathlib import Path
import tempfile
import unittest

from rad_memory.check_ppo_convergence import collection_config, main, parse_args, select_task
from rad_memory.envs import MemoryTaskSpec
from rad_memory.ppo import PPOConfig
from rad_memory.task_pool import freeze_task, generate_pool


class PPOConvergenceTest(unittest.TestCase):
    def test_defaults_and_saved_collection_config_match(self):
        self.assertEqual(collection_config(parse_args([])), PPOConfig())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "source_config.json")
            config = PPOConfig(n_steps=16, batch_size=8, learning_rate=0.001, n_epochs=2)
            path.write_text(json.dumps({"source_algorithm": "ppo", "ppo": config.to_dict()}))
            args = parse_args(["--ppo-config", str(path)])
            self.assertEqual(collection_config(args), config)

    def test_selects_exact_manifest_task_and_rejects_changing_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "tasks.json")
            pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=30), 2, 0.5, 0)
            path.write_text(json.dumps(pool))
            target = next(task for task in pool["tasks"] if task["split"] == "test")
            args = parse_args(["--manifest", str(path), "--task-id", target["task_id"]])
            self.assertEqual(select_task(args).to_dict(), target)
            args.task_id = "missing"
            with self.assertRaisesRegex(ValueError, "not in the manifest"):
                select_task(args)
            path.write_text(json.dumps(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train").to_dict()))
            with self.assertRaisesRegex(ValueError, "fixed configuration"):
                select_task(parse_args(["--task-spec", str(path)]))

    def test_real_training_writes_curves_and_reports_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = freeze_task(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "test", horizon=1))
            task_path = root / "task.json"
            task_path.write_text(json.dumps(spec.to_dict()))
            config_path = root / "ppo.json"
            config = PPOConfig(n_steps=8, batch_size=8, n_epochs=1)
            config_path.write_text(json.dumps(config.to_dict()))
            common = ["--task-spec", str(task_path), "--ppo-config", str(config_path), "--seeds", "0", "1",
                      "--total-timesteps", "8", "--evaluation-interval", "8", "--evaluation-episodes", "1",
                      "--required-consecutive-evals", "1"]
            # No action can reach the branch from this initial pose in one step.
            with self.assertRaises(SystemExit) as failure:
                main(common + ["--output-dir", str(root / "fail")])
            self.assertEqual(failure.exception.code, 1)
            summary = json.loads((root / "fail/summary.json").read_text())
            self.assertFalse(summary["passed"])
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["task_spec"], spec.to_dict())
            self.assertEqual(summary["ppo_config"], config.to_dict())
            self.assertEqual(len(summary["runs"]), 2)
            self.assertTrue((root / "fail/learning-curves.png").exists())
            self.assertTrue(all(run["final_success_rate"] == 0 for run in summary["runs"]))
            main(common + ["--output-dir", str(root / "pass"), "--minimum-success-rate", "0"])
            self.assertTrue(json.loads((root / "pass/summary.json").read_text())["passed"])
            # A zero threshold above tests reporting only, not learned competence.
            with self.assertRaises(FileExistsError):
                main(common + ["--output-dir", str(root / "pass")])


if __name__ == "__main__":
    unittest.main()
