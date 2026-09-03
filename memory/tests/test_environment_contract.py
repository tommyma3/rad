from __future__ import annotations

import unittest

from rad_memory.envs import MemoryTaskSpec, make_memory_env


class MemoryEnvironmentContractTest(unittest.TestCase):
    def test_controlled_reset_is_reproducible_and_cue_is_visible(self):
        spec = MemoryTaskSpec(
            env_id="MiniGrid-MemoryS13-v0",
            seed=17,
            split="train",
            horizon=100,
            controlled=True,
            size=13,
        )
        first = make_memory_env(spec)
        second = make_memory_env(spec)
        obs_a, info_a = first.reset(seed=17)
        obs_b, info_b = second.reset(seed=17)
        self.assertTrue((obs_a["image"] == obs_b["image"]).all())
        self.assertEqual(info_a["memory_cue_id"], info_b["memory_cue_id"])
        self.assertTrue(info_a["memory_cue_visible"])
        self.assertFalse(info_a["memory_decision"])
        first.close()
        second.close()

    def test_official_variant_preserves_seven_actions(self):
        spec = MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "test")
        env = make_memory_env(spec)
        env.reset()
        self.assertEqual(env.action_space.n, 7)
        env.close()


if __name__ == "__main__":
    unittest.main()
