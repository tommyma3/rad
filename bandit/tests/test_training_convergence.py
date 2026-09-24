"""Regret accounting, unchanged training, resume, and paired plot completeness."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from bandit.collect import collect_dataset
from bandit.convergence_experiment import (evaluate_model, evaluation_steps, plot_results,
    resolve_pretrained, verify_pretrained)
from bandit.env import sample_task
from bandit.evaluation import make_eval_manifest
from bandit.scripts.run_training_convergence import main
from bandit.tests.test_bandit import tiny_config
from bandit.training import load_checkpoint, train
from bandit.utils import load_config


class ConstantModel:
    device = torch.device("cpu")

    def eval(self):
        return self

    def new_state(self):
        return {"compression_count": 0, "recent": None}

    def query_logits(self, state, observation):
        return torch.tensor([[1., 0., 0., 0., 0.]])

    def embed_transitions(self, *args):
        return None

    def ingest(self, *args):
        pass


class ConvergenceTests(unittest.TestCase):
    def test_pretrained_launcher_initialization_provenance_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            config = load_config("rad")
            config.update(train_steps=1, batch_size=2, validation_batches=1)
            collect_dataset(base / "data", config, tasks=4, validation_tasks=2)
            source = train(config, base / "data", base / "pretrain_s0", pretrain=True, cpu=True)
            template = str(base / "pretrain_s{seed}" / source.name)
            resolved = resolve_pretrained(template, [0], config)
            self.assertEqual(resolved["0"]["step"], 1)
            with self.assertRaisesRegex(ValueError, "architecture"):
                resolve_pretrained(template, [0], {**config, "context_steps": 51})
            root = base / "experiment"
            main(["--root", str(root), "--cpu", "--steps", "2", "--eval-interval", "1",
                  "--dataset", str(base / "data"), "--eval-tasks", "1", "--batch-size", "2",
                  "--seeds", "0", "--threads", "1", "--pretrained", template])
            plan = json.loads((root / "plan.json").read_text())
            self.assertEqual(plan["rad_pretrained"], resolved)
            manifest = json.loads((root / "evaluation_manifest.json").read_text())
            model, _ = load_checkpoint(source)
            initial = json.loads((root / "evaluations/rad_s0/step-0000000.json").read_text())
            self.assertEqual(initial["rows"], evaluate_model(model, manifest, plan["delays"]))
            for method in ("ad_short", "ad_long", "rad"):
                _, payload = load_checkpoint(root / f"{method}_s0/checkpoint-0000002")
                self.assertEqual("pretrained_source" in payload["config"], method == "rad")
            self.assertIn("pretraining updates are additional", (root / "plots/caption.txt").read_text())
            main(["--root", str(root), "--cpu", "--resume", "--threads", "1"])
            with self.assertRaisesRegex(ValueError, "compression-pretraining"):
                resolve_pretrained(str(root / "rad_s0/checkpoint-0000002"), [0], config)
            with (source / "model.pt").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                verify_pretrained(plan)

    def test_exact_regret_ignores_distractors(self):
        task = sample_task(12, num_arms=5)
        manifest = {"records": [{"task": task.to_dict(), "reward_seed": 1,
            "learner_seed": 2, "distractor_seed": 3, "policy_seed": 4}]}
        rows = evaluate_model(ConstantModel(), manifest, [0, 73], greedy=True)
        expected = np.arange(1, 101) * (max(task.means) - task.means[0])
        for row in rows:
            np.testing.assert_allclose(row["cumulative_regret_curve"], expected)
            self.assertAlmostEqual(row["cumulative_expected_regret"], expected[-1])
        self.assertEqual(evaluation_steps(5, 2), [0, 2, 4, 5])

    def test_callback_preserves_training_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config("rad")
            config.update(tf_dropout=0.1, eval_interval=2, checkpoint_interval=2)
            collect_dataset(root / "data", config, tasks=4, validation_tasks=2)
            baseline = train(config, root / "data", root / "baseline", cpu=True)
            manifest = make_eval_manifest(73, 1, ["uniform"], config)
            seen = []

            def callback(model, step):
                seen.append(step)
                torch.rand(7)  # Must not affect the following training update.
                evaluate_model(model, manifest, [6], pre_steps=6, post_steps=6)

            part = train(config, root / "data", root / "callback", cpu=True,
                         stop_after=2, evaluation_callback=callback)
            resumed = train(config, root / "data", root / "callback", cpu=True,
                            resume=part, evaluation_callback=callback)
            self.assertEqual(seen, [0, 2, 2, 4])
            first, _ = load_checkpoint(baseline)
            second, _ = load_checkpoint(resumed)
            for key, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def test_complete_launcher_and_plot_rejects_missing_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            main(["--root", str(root), "--cpu", "--steps", "2", "--eval-interval", "1",
                  "--train-tasks", "4", "--validation-tasks", "2", "--eval-tasks", "1",
                  "--batch-size", "2", "--seeds", "0", "--threads", "1"])
            self.assertTrue((root / "plots/training_convergence.pdf").exists())
            summary = json.loads((root / "plots/summary.json").read_text())
            self.assertEqual(len(summary), 9)
            self.assertEqual({row["step"] for row in summary}, {0, 1, 2})
            self.assertTrue(all(row["genuine_pulls"] == 100 for row in summary))
            before = (root / "evaluations/rad_s0/step-0000002.json").read_bytes()
            main(["--root", str(root), "--cpu", "--resume", "--threads", "1"])
            self.assertEqual(before, (root / "evaluations/rad_s0/step-0000002.json").read_bytes())
            (root / "evaluations/rad_s0/step-0000001.json").unlink()
            with self.assertRaises(FileNotFoundError):
                plot_results(root)


if __name__ == "__main__":
    unittest.main()
