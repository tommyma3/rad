"""Checkpoint discovery for the delay-sweep evaluation script."""
import json
import tempfile
import unittest
from pathlib import Path

from bandit.evaluation import discover_run_checkpoints


def make_run(root, name, steps, phase="distill"):
    for step in steps:
        checkpoint = Path(root) / name / f"checkpoint-{step:07d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.pt").write_bytes(b"placeholder")
        (checkpoint / "training.json").write_text(json.dumps({"phase": phase, "step": step}))


class TestDiscoverRunCheckpoints(unittest.TestCase):
    def test_families_share_a_label_and_runs_use_max_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp, "ad_long_s0", [1000, 5000, 3000])
            make_run(tmp, "ad_long_s1", [4000])
            make_run(tmp, "rad_s2", [7000, 2000])
            discovered, skipped = discover_run_checkpoints(tmp)
            self.assertEqual(skipped, [])
            self.assertEqual(sorted(discovered), ["ad_long", "rad"])
            self.assertEqual([(item["run"], item["seed"], item["step"])
                              for item in discovered["ad_long"]],
                             [("ad_long_s0", 0, 5000), ("ad_long_s1", 1, 4000)])
            self.assertEqual([(item["run"], item["seed"], item["step"])
                              for item in discovered["rad"]],
                             [("rad_s2", 2, 7000)])
            self.assertEqual(discovered["rad"][0]["checkpoint"].name, "checkpoint-0007000")

    def test_skips_runs_without_a_distilled_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp, "pretrain_s0", [2000], phase="pretrain")
            make_run(tmp, "ad_s0", [1000])
            (Path(tmp) / "notes.txt").write_text("not a run")
            (Path(tmp) / "empty_s9").mkdir()
            discovered, skipped = discover_run_checkpoints(tmp)
            self.assertEqual(skipped, ["pretrain_s0"])
            self.assertEqual(sorted(discovered), ["ad"])

    def test_run_without_seed_suffix_forms_its_own_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_run(tmp, "baseline", [1000, 2000])
            discovered, skipped = discover_run_checkpoints(tmp)
            self.assertEqual(skipped, [])
            self.assertEqual([(item["run"], item["seed"], item["step"])
                              for item in discovered["baseline"]],
                             [("baseline", None, 2000)])


if __name__ == "__main__":
    unittest.main()
