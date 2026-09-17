import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from rad_memory.dataset import ADDataset, RADDataset, collate_trajectories
from rad_memory.envs import MemoryTaskSpec, make_memory_env
from rad_memory.model import AD, RAD
from rad_memory.ppo import PPOConfig, build_ppo, evaluate_ppo
from rad_memory.recurrent_ppo import RecurrentPPOConfig
from rad_memory.task_pool import generate_pool
from rad_memory.train_task_pool import train_pool, train_task
from test_models import _config


class PPOTest(unittest.TestCase):
    def test_checkpoint_cannot_resume_on_different_source_algorithm(self):
        from rad_memory.training import validate_checkpoint_scope
        config = {"history_scope": "task", "manifest_fingerprint": "same-pool", "source_algorithm": "ppo"}
        with self.assertRaisesRegex(ValueError, "source_algorithm"):
            validate_checkpoint_scope({"config": config | {"source_algorithm": "recurrent_ppo"}}, config)

    def test_spawned_ppo_workers_and_distillation_histories(self):
        from stable_baselines3 import PPO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=3), 2, 0.5, 0)
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps(pool), encoding="utf-8")
            results = train_pool(manifest, [0, 1], root / "runs", root / "data", workers=2,
                                 source_algorithm="ppo", total_timesteps=16, evaluation_interval=8,
                                 evaluation_episodes=1, minimum_success_rate=0, required_consecutive_evals=1,
                                 ppo_config=PPOConfig(n_steps=8, batch_size=8, n_epochs=1), verbose=0)
            self.assertEqual(len(results), 2)
            self.assertEqual({r["source_algorithm"] for r in results}, {"ppo"})
            self.assertEqual(len({tuple(r["artifacts"]) for r in results}), 2)
            self.assertEqual({r["streams"] for r in results}, {1})
            spec = MemoryTaskSpec.from_dict(next(t for t in pool["tasks"] if t["split"] == "train"))
            for result in results:
                self.assertEqual(result["task_id"], spec.task_id)
                with h5py.File(result["artifacts"][0], "r") as handle:
                    self.assertEqual(handle.attrs["source_algorithm"], "ppo")
                    source = json.loads(handle.attrs["source_config"])
                    self.assertEqual(source["ppo"]["policy"], "MlpPolicy")
                    self.assertEqual(source["source_seed"], result["source_seed"])
                    run = root / "runs" / source["run_id"]
                    model = PPO.load(run / "teacher-final", device="cpu")
                    self.assertFalse(hasattr(model.policy, "lstm_actor"))
                    self.assertEqual(evaluate_ppo(model, spec, episodes=2)["episodes"], 2)
            config = _config() | {"history_scope": "task", "task_manifest": str(manifest),
                                  "source_algorithm": "ppo", "max_context_length": 12}
            for dataset_type, model_type in ((ADDataset, AD), (RADDataset, RAD)):
                with dataset_type(config, root / "data", "train") as dataset:
                    self.assertEqual(len(dataset.episodes), 2)
                    item = dataset[-1] if dataset_type is ADDataset else dataset[(len(dataset) - 1, 2)]
                    self.assertGreater(int(item["truncated"].sum()), 1)
                    output = model_type(config)(collate_trajectories([item]))
                    self.assertTrue(torch.isfinite(output["loss_total"]))
            with self.assertRaisesRegex(ValueError, "policy and algorithm"):
                train_task(spec, pool["fingerprint"], 2, root / "runs", root / "data",
                           source_algorithm="ppo", ppo_config=RecurrentPPOConfig())
            self.assertEqual(PPOConfig().n_steps, RecurrentPPOConfig().n_steps)

    def test_spawned_ppo_workers_with_multiple_streams(self):
        from stable_baselines3 import PPO
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=3), 2, 0.5, 0)
            manifest = root / "tasks.json"
            manifest.write_text(json.dumps(pool), encoding="utf-8")
            results = train_pool(manifest, [0, 1], root / "runs", root / "data", workers=2,
                                 source_algorithm="ppo", total_timesteps=16, evaluation_interval=8,
                                 evaluation_episodes=1, minimum_success_rate=0, required_consecutive_evals=1,
                                 ppo_config=PPOConfig(n_steps=8, batch_size=8, n_epochs=1),
                                 streams=2, verbose=0)
            self.assertEqual(len(results), 2)
            self.assertEqual({r["streams"] for r in results}, {2})
            self.assertEqual(len({a for r in results for a in r["artifacts"]}), 4)
            spec = MemoryTaskSpec.from_dict(next(t for t in pool["tasks"] if t["split"] == "train"))
            for result in results:
                self.assertEqual(len(result["artifacts"]), 2)
                stream_ids = set()
                for artifact in result["artifacts"]:
                    with h5py.File(artifact, "r") as handle:
                        source = json.loads(handle.attrs["source_config"])
                        stream_ids.add(source["stream_id"])
                        self.assertEqual(source["streams"], 2)
                        self.assertEqual(handle.attrs["source_algorithm"], "ppo")
                        self.assertTrue(handle.attrs["collection_complete"])
                        self.assertTrue(handle.attrs["source_converged"])
                self.assertEqual(stream_ids, {0, 1})
                run = root / "runs" / source["run_id"]
                model = PPO.load(run / "teacher-final", device="cpu")
                self.assertFalse(hasattr(model.policy, "lstm_actor"))
                self.assertEqual(evaluate_ppo(model, spec, episodes=2)["episodes"], 2)
            config = _config() | {"history_scope": "task", "task_manifest": str(manifest),
                                  "source_algorithm": "ppo", "max_context_length": 12}
            for dataset_type, model_type in ((ADDataset, AD), (RADDataset, RAD)):
                with dataset_type(config, root / "data", "train") as dataset:
                    self.assertEqual(len(dataset.episodes), 4)
                    item = dataset[-1] if dataset_type is ADDataset else dataset[(len(dataset) - 1, 2)]
                    self.assertGreaterEqual(int(item["learner_steps"][0]), 1)
                    self.assertTrue((np.diff(item["learner_steps"]) == 1).all())
                    output = model_type(config)(collate_trajectories([item]))
                    self.assertTrue(torch.isfinite(output["loss_total"]))


if __name__ == "__main__":
    unittest.main()
