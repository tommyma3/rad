import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'metaworld'
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('metaworld_dataset', ROOT / 'dataset.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MetaWorldDatasetContractTest(unittest.TestCase):
    def config(self):
        return {
            'env': 'metaworld',
            'task': 'window-open-v3',
            'alg': 'PPO',
            'alg_seed': 0,
            'dim_obs': 3,
            'n_transit': 4,
            'dynamics': False,
            'n_compress_tokens': 3,
            'short_memory_keep': 1,
            'always_use_latent_prefix': True,
            'max_context_length': 8,
            'min_context_length': 2,
            'max_compressions': None,
            'train_n_seed': 1,
        }

    def write_fixture(self, root, split='train'):
        directory = root / 'window-open-v3'
        if split == 'test':
            directory = directory / 'test'
        directory.mkdir(parents=True)
        path = directory / 'history_window-open-v3_PPO_alg-seed0.hdf5'
        with h5py.File(path, 'w') as file:
            group = file.create_group('0')
            # Stored collection layout is time, stream, feature.
            group['states'] = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
            group['actions'] = np.zeros((6, 2, 2), dtype=np.float32)
            group['rewards'] = np.zeros((6, 2), dtype=np.float32)
            group['next_states'] = np.ones((6, 2, 3), dtype=np.float32)
            group['dones'] = np.zeros((6, 2), dtype=np.bool_)
            group['success'] = np.zeros((6, 2), dtype=np.float32)

    def test_all_datasets_share_continuous_sar_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            self.write_fixture(root, split='test')
            config = self.config()

            ad = MODULE.ADDataset(config, root, 'train', n_stream=2, source_timesteps=6)
            self.assertEqual(ad[0]['states'].shape, (4, 3))
            self.assertEqual(ad[0]['actions'].shape, (4, 2))

            rad = MODULE.RADDataset(config, root, 'train', n_stream=2, source_timesteps=6)
            item = rad[(0, 0)]
            self.assertEqual(item['states'].shape[-1], 3)
            self.assertEqual(item['actions'].shape[-1], 2)

            pretrain = MODULE.CompressionPretrainDataset(
                config, root, 'test', n_stream=1, source_timesteps=6, n_seed=1
            )
            self.assertEqual(pretrain[0]['actions'].shape, (4, 2))


if __name__ == '__main__':
    unittest.main()

