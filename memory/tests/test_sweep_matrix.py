from __future__ import annotations

import unittest

from rad_memory.run_context_sweep import build_commands


class SweepMatrixTest(unittest.TestCase):
    def test_requested_context_conditions_and_dependencies_are_present(self):
        commands = build_commands(
            horizon=100,
            short_context=30,
            seed=2,
            config="config/model/memory_base.yaml",
            data_root="datasets",
            runs_root="runs",
            evaluation={"env_id": "MiniGrid-MemoryS13-v0", "seed": 1000, "episodes": 10},
            compute_matched_steps=70000,
        )
        names = [name for name, _ in commands]
        for expected in (
            "ad-reactive",
            "ad-short",
            "ad-short-extra-compute",
            "ad-half-episode",
            "ad-full",
            "ad-2x",
            "rad-pretrain",
            "rad-short",
            "rad-no-pretrain",
        ):
            self.assertIn(expected, names)
        command_text = {name: " ".join(command) for name, command in commands}
        self.assertIn("n_transit=30", command_text["ad-short"])
        self.assertIn("n_transit=50", command_text["ad-half-episode"])
        self.assertIn("n_transit=110", command_text["ad-full"])
        self.assertIn("n_transit=200", command_text["ad-2x"])
        self.assertIn("pretrain-final.pt", command_text["rad-short"])


if __name__ == "__main__":
    unittest.main()
