"""Bounded CPU integration check: collect, all training phases, resume, evaluate."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from bandit.collect import collect_dataset
from bandit.evaluation import evaluate_suite
from bandit.tests.test_bandit import tiny_config
from bandit.training import load_checkpoint, train


class TrainingTests(unittest.TestCase):
    def test_all_phases_exact_resume_and_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            config = tiny_config("ad_short")
            config.update(gradient_accumulation_steps=2, tf_dropout=0.1)
            collect_dataset(data, config, tasks=6, validation_tasks=3)
            uninterrupted = train(config, data, root / "ad_full", cpu=True)
            partial = train(config, data, root / "ad_resumed", stop_after=2, cpu=True)
            resumed = train(config, data, root / "ad_resumed", resume=partial, cpu=True)
            first, _ = load_checkpoint(uninterrupted)
            second, _ = load_checkpoint(resumed)
            for key, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
            rad_config = tiny_config("rad")
            rad_config["train_steps"] = 2
            pretrained = train(rad_config, data, root / "pretrain", pretrain=True, cpu=True)
            rad_checkpoint = train(rad_config, data, root / "rad", pretrained=pretrained, cpu=True)
            scratch_checkpoint = train(rad_config, data, root / "rad_scratch", cpu=True)
            summary = evaluate_suite([resumed, rad_checkpoint, scratch_checkpoint], root / "evaluation", rad_config,
                                     tasks=2, delays=[0, 12], distributions=["uniform"], labels=["AD", "RAD", "RAD-scratch"],
                                     eval_seeds=2)
            self.assertEqual(len(summary), 10)
            self.assertTrue((root / "evaluation" / "delay_sweep.png").exists())
            self.assertTrue((root / "evaluation" / "cumulative_regret.png").exists())
            self.assertTrue((root / "evaluation" / "manifests.json").exists())
            self.assertEqual(summary[0]["eval_seeds"], 2)
            self.assertEqual(summary[0]["ci_unit"], "eval_seed")
            self.assertEqual(len(summary[0]["regret_curve_std"]), len(summary[0]["regret_curve"]))
            source = [r for r in summary if r["method"] == "UCB"]
            self.assertEqual(source[0]["post_return"], source[1]["post_return"])
            diagnostic = evaluate_suite([rad_checkpoint], root / "diagnostic", rad_config,
                                       manifest_path=root / "evaluation" / "manifest.json",
                                       delays=[12], shared_prefix=True)
            self.assertEqual(len(diagnostic), 3)
            self.assertEqual(diagnostic[0]["eval_seeds"], 1)
            rad_seed_config = tiny_config("rad")
            rad_seed_config["seed"] = 7
            rad_seed7 = train(rad_seed_config, data, root / "rad_seed7", cpu=True)
            multi = evaluate_suite([rad_checkpoint, rad_seed7], root / "multi_seed", rad_config,
                                   tasks=2, delays=[0, 12], distributions=["uniform"],
                                   labels=["RAD", "RAD"], eval_seeds=2)
            self.assertEqual(len(multi), 6)
            rad_rows = [r for r in multi if r["method"] == "RAD"]
            self.assertEqual(rad_rows[0]["training_runs"], 2)
            self.assertEqual(rad_rows[0]["eval_seeds"], 2)
            self.assertEqual(rad_rows[0]["ci_unit"], "training_run")
            self.assertEqual(len(rad_rows[0]["regret_curve_std"]), len(rad_rows[0]["regret_curve"]))
            bad_config = dict(config, context_steps=6)
            with self.assertRaises(ValueError):
                train(bad_config, data, root / "bad", resume=resumed, cpu=True)
            metadata = json.loads((resumed / "training.json").read_text())
            self.assertEqual(metadata["step"], 4)


if __name__ == "__main__":
    unittest.main()
