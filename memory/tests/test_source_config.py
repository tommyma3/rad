import tempfile
import unittest
from pathlib import Path

from rad_memory.ppo import PPOConfig
from rad_memory.recurrent_ppo import (
    RecurrentPPOConfig,
    load_source_config,
    source_config_from_mapping,
)
from rad_memory.train_task_pool import parse_args as parse_pool_args
from rad_memory.train_task_pool import resolve_source_args


class SourceConfigMappingTest(unittest.TestCase):
    def test_empty_mapping_gives_recurrent_defaults(self):
        algorithm, config = source_config_from_mapping({})
        self.assertEqual(algorithm, "recurrent_ppo")
        self.assertEqual(config, RecurrentPPOConfig())

    def test_ppo_section_overrides_defaults(self):
        algorithm, config = source_config_from_mapping(
            {"source_algorithm": "ppo",
             "ppo": {"n_steps": 128, "learning_rate": 1e-4}})
        self.assertEqual(algorithm, "ppo")
        self.assertEqual(config.n_steps, 128)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.batch_size, PPOConfig().batch_size)

    def test_unknown_keys_and_bad_algorithm_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown recurrent_ppo config keys"):
            source_config_from_mapping({"ppo": {"n_stepz": 8}})
        with self.assertRaisesRegex(ValueError, "source_algorithm"):
            source_config_from_mapping({"source_algorithm": "a2c"})

    def test_policy_must_match_algorithm(self):
        with self.assertRaisesRegex(ValueError, "requires policy"):
            source_config_from_mapping(
                {"source_algorithm": "ppo", "ppo": {"policy": "MlpLstmPolicy"}})
        with self.assertRaisesRegex(ValueError, "requires policy"):
            source_config_from_mapping({"ppo": {"policy": "MlpPolicy"}})


class LoadSourceConfigTest(unittest.TestCase):
    def test_round_trip_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "source.yaml")
            path.write_text(
                "source_algorithm: ppo\n"
                "ppo:\n"
                "  n_steps: 128\n"
                "total_timesteps: 5000\n",
                encoding="utf-8",
            )
            algorithm, config = load_source_config(path)
            self.assertEqual(algorithm, "ppo")
            self.assertEqual(config.n_steps, 128)
            self.assertEqual(config.policy, "MlpPolicy")

    def test_shipped_default_config_matches_dataclass(self):
        algorithm, config = load_source_config(Path("config/source/ppo.yaml"))
        self.assertEqual(algorithm, "ppo")
        self.assertEqual(config, PPOConfig())


class ResolveSourceArgsTest(unittest.TestCase):
    def _resolve(self, argv, file_values=None):
        args = parse_pool_args(argv + ["--manifest", "tasks.json", "--run-dir", "runs"])
        return resolve_source_args(args, file_values or {})

    def test_defaults_without_config(self):
        algorithm, config, budget = self._resolve([])
        self.assertEqual(algorithm, "ppo")
        self.assertEqual(config, PPOConfig())
        self.assertEqual(budget["total_timesteps"], 1_000_000)
        self.assertEqual(budget["source_seeds"], [0, 1, 2])

    def test_file_values_apply_and_flags_win(self):
        file_values = {
            "source_algorithm": "ppo",
            "ppo": {"n_steps": 512, "batch_size": 64},
            "total_timesteps": 5000,
            "source_seeds": [7],
        }
        algorithm, config, budget = self._resolve([], file_values)
        self.assertEqual(algorithm, "ppo")
        self.assertEqual((config.n_steps, config.batch_size), (512, 64))
        self.assertEqual(budget["total_timesteps"], 5000)
        self.assertEqual(budget["source_seeds"], [7])
        _, config, budget = self._resolve(
            ["--n-steps", "64", "--total-timesteps", "9"], file_values)
        self.assertEqual(config.n_steps, 64)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(budget["total_timesteps"], 9)

    def test_file_recurrent_algorithm_is_kept(self):
        algorithm, config, _ = self._resolve([], {"source_algorithm": "recurrent_ppo"})
        self.assertEqual(algorithm, "recurrent_ppo")
        self.assertIsInstance(config, RecurrentPPOConfig)
        self.assertNotIsInstance(config, PPOConfig)

    def test_conflicting_algorithm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self._resolve(["--source-algorithm", "ppo"],
                          {"source_algorithm": "recurrent_ppo"})

    def test_scalar_source_seeds_becomes_list(self):
        _, _, budget = self._resolve([], {"source_seeds": 3})
        self.assertEqual(budget["source_seeds"], [3])


if __name__ == "__main__":
    unittest.main()
