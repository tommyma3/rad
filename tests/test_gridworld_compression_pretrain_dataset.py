import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'gridworld'
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('gridworld_pretrain_dataset', ROOT / 'dataset.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GridworldCompressionPretrainDatasetTest(unittest.TestCase):
    def config(self, always_use_latent_prefix=True):
        return {
            'env': 'darkroom',
            'grid_size': 1,
            'alg': 'PPO',
            'alg_seed': 0,
            'env_split_seed': 0,
            'train_env_ratio': 1.0,
            'n_transit': 4,
            'n_compress_tokens': 3,
            'always_use_latent_prefix': always_use_latent_prefix,
            'dynamics': False,
        }

    def write_fixture(self, root):
        path = root / 'history_darkroom_PPO_alg-seed0.hdf5'
        with h5py.File(path, 'w') as file:
            group = file.create_group('0')
            group['states'] = np.zeros((6, 1, 2), dtype=np.int64)
            group['actions'] = np.zeros((6, 1), dtype=np.int64)
            group['rewards'] = np.zeros((6, 1), dtype=np.float32)
            group['next_states'] = np.ones((6, 1, 2), dtype=np.int64)

    def test_null_prefix_reserves_timestep_equivalent_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)

            with_prefix = MODULE.CompressionPretrainDataset(
                self.config(True), root, 'train', n_stream=1, source_timesteps=6
            )
            self.assertEqual(with_prefix.window_size, 3)
            self.assertEqual(with_prefix[0]['states'].shape, (3, 2))

            without_prefix = MODULE.CompressionPretrainDataset(
                self.config(False), root, 'train', n_stream=1, source_timesteps=6
            )
            self.assertEqual(without_prefix.window_size, 4)
            self.assertEqual(without_prefix[0]['states'].shape, (4, 2))


if __name__ == '__main__':
    unittest.main()
