"""Rollout caching for evaluate_suite: reruns reuse cached blocks, and only plotting may change."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from bandit.collect import collect_dataset
from bandit.evaluation import evaluate_suite
from bandit.tests.test_bandit import tiny_config
from bandit.training import train


class EvalCacheTests(unittest.TestCase):
    def test_cached_reruns_reuse_rollouts_and_reject_incompatible_outputs(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config("rad")
            config["train_steps"] = 2
            collect_dataset(root / "data", config, tasks=6, validation_tasks=3)
            pretrained = train(config, root / "data", root / "pretrain", pretrain=True, cpu=True)
            checkpoint = train(config, root / "data", root / "rad", pretrained=pretrained, cpu=True)
            cache, output = root / "cache", root / "eval"

            def run(**overrides):
                options = dict(tasks=2, delays=[0, 12], labels=["RAD"], eval_seeds=2,
                               cache_dir=cache)
                options.update(overrides)
                return evaluate_suite([checkpoint], output, config, **options)

            fresh = run()
            # UCB, Random, and RAD blocks for every eval seed and delay are cached.
            self.assertEqual(len(list(cache.iterdir())), 12)
            cached = run()
            self.assertEqual(fresh, cached)
            # Plotting-only settings may change on a rerun without reevaluating.
            replotted = run(reference_context=25)
            self.assertEqual(fresh, replotted)
            # rollouts.jsonl is regenerated, not appended to: 3 methods x 2 seeds x 2 delays x 2 tasks.
            self.assertEqual(len((output / "rollouts.jsonl").read_text().strip().splitlines()),
                             3 * 2 * 2 * 2)
            with self.assertRaises(FileExistsError):
                run(delays=[0])
            with self.assertRaises(FileExistsError):
                evaluate_suite([checkpoint], output, config, tasks=2, delays=[0, 12])
            self.assertEqual(run(force=True), fresh)
            self.assertEqual(len(list(cache.iterdir())), 12)


if __name__ == "__main__":
    unittest.main()
