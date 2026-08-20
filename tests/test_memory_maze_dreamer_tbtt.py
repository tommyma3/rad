from pathlib import Path
import sys
import unittest

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1] / "memory_maze"
sys.path.insert(0, str(PROJECT))

from algorithm.dreamer_tbtt import ReplayEpisode, RSSMState, SequentialTBTTReplay


def _episode(length: int) -> ReplayEpisode:
    return ReplayEpisode(
        images=np.zeros((length, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros(length, dtype=np.int64),
        rewards=np.zeros(length, dtype=np.float32),
        dones=np.zeros(length, dtype=np.float32),
    )


class DreamerTBTTContractTest(unittest.TestCase):
    def test_state_is_carried_and_detached_between_chunks(self):
        replay = SequentialTBTTReplay(1000, sequence_length=4, batch_size=2)
        replay.add(_episode(12))
        replay.add(_episode(12))
        _, initial = replay.next_batch()
        self.assertEqual(initial, [None, None])
        state = RSSMState(
            torch.ones(2, 3, requires_grad=True),
            torch.ones(2, 2, 2, requires_grad=True),
        )
        replay.update_states(state)
        _, carried = replay.next_batch()
        self.assertTrue(all(item is not None for item in carried))
        self.assertTrue(all(not item.deterministic.requires_grad for item in carried))


if __name__ == "__main__":
    unittest.main()
