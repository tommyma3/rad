from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from rad_memory.artifacts import TaskHistoryWriter
from rad_memory.dataset import ADDataset, RADDataset, collate_trajectories
from rad_memory.envs import MemoryTaskSpec, make_memory_env
from rad_memory.model import AD, RAD
from rad_memory.task_pool import freeze_task, generate_pool, load_pool
from rad_memory.train_task_pool import OnlineHistory
from test_models import _config


class FixedTaskTest(unittest.TestCase):
    def test_reset_restores_full_state_despite_seed_and_mutations(self):
        spec = freeze_task(MemoryTaskSpec("MiniGrid-MemoryS13Random-v0", 4, "train", horizon=30))
        restored = MemoryTaskSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        env = make_memory_env(restored)
        try:
            expected, _ = env.reset()
            for seed in (None, 0, 1000):
                env.unwrapped.grid.set(1, env.unwrapped.height // 2 - 1, None)
                env.step(0)
                env.unwrapped.carrying = object()
                observation, info = env.reset(seed=seed)
                self.assertEqual(env.unwrapped.grid.encode().tolist(), spec.configuration["grid"])
                self.assertEqual(env.unwrapped.agent_pos.tolist(), spec.configuration["agent_pos"])
                self.assertEqual(env.unwrapped.agent_dir, spec.configuration["agent_dir"])
                self.assertEqual(list(env.unwrapped.success_pos), spec.configuration["success_pos"])
                self.assertIsNone(env.unwrapped.carrying)
                self.assertEqual(info["memory_step"], 0)
                np.testing.assert_array_equal(observation["image"], expected["image"])
        finally:
            env.close()
        self.assertEqual(spec.task_id, replace(spec, split="test", seed=999).task_id)

    def test_pool_reproducible_disjoint_and_bounded(self):
        template = MemoryTaskSpec("MiniGrid-MemoryS13Random-v0", 0, "train", horizon=30)
        pool = generate_pool(template, 12, 0.75, 7)
        self.assertEqual(pool, generate_pool(template, 12, 0.75, 7))
        self.assertEqual(len({t["task_id"] for t in pool["tasks"]}), 12)
        self.assertEqual(sum(t["split"] == "train" for t in pool["tasks"]), 9)
        with self.assertRaisesRegex(ValueError, "unique tasks"):
            generate_pool(replace(template, controlled=True, size=7, random_length=False),
                          5, 0.8, 0, max_candidates=100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "pool.json")
            path.write_text(json.dumps(pool), encoding="utf-8")
            self.assertEqual(load_pool(path), pool)
            pool["tasks"][0]["split"] = "test"
            path.write_text(json.dumps(pool), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_pool(path)

    def test_online_histories_cross_resets_but_not_source_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 0, "train", horizon=3), 2, 0.5, 0)
            manifest = root / "pool.json"
            manifest.write_text(json.dumps(pool), encoding="utf-8")
            spec = MemoryTaskSpec.from_dict(next(t for t in pool["tasks"] if t["split"] == "train"))
            for source_seed in (0, 1):
                path = root / "train/recurrent_ppo" / f"run-{source_seed}.hdf5"
                provenance = {"manifest_fingerprint": pool["fingerprint"], "source_seed": source_seed,
                              "run_id": str(source_seed), "stream_id": 0, "history_kind": "online_training"}
                with TaskHistoryWriter(path, spec, "recurrent_ppo", provenance) as writer:
                    env = OnlineHistory(make_memory_env(spec), writer)
                    try:
                        for _ in range(4):
                            env.reset()
                            for _ in range(3):
                                env.step(0)
                        writer.handle.attrs["collection_complete"] = True
                        writer.handle.attrs["source_converged"] = True
                    finally:
                        env.close()
            config = _config() | {"n_transit": 5, "max_context_length": 12,
                                  "source_algorithm": "recurrent_ppo", "history_scope": "task",
                                  "task_manifest": str(manifest), "dataset_stride": 1}
            with ADDataset(config, root, "train") as dataset:
                self.assertEqual(len(dataset.episodes), 2)
                self.assertEqual([r.length for r in dataset.episodes], [12, 12])
                index = next(i for i, window in enumerate(dataset.windows) if window[2] == 5)
                item = dataset[index]
                self.assertEqual(item["learner_steps"].tolist(), [1, 2, 3, 4, 5])
                self.assertTrue(item["truncated"][2])
                self.assertFalse(item["truncated"][-1])
                np.testing.assert_array_equal(item["images"][0], item["images"][3])
            with RADDataset(config, root, "train") as dataset:
                index = next(i for i, window in enumerate(dataset.windows) if window[2] == 12)
                item = dataset[(index, 2)]
                self.assertEqual(len(item["actions"]), 12)
                self.assertEqual(int(item["truncated"].sum()), 4)
            with self.assertRaisesRegex(ValueError, "Unsupported trajectory"):
                ADDataset(config | {"history_scope": "episode"}, root, "train")
            with h5py.File(path, "a") as handle:
                handle.attrs["collection_complete"] = False
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                ADDataset(config, root, "train")

    def test_batch_and_streaming_agree_across_episode_boundaries(self):
        from test_artifact_dataset_contract import _episode
        steps = _episode(3) * 4
        keys = {"images": "observation", "directions": "observation", "actions": "action",
                "rewards": "reward", "terminated": "terminated", "truncated": "truncated",
                "decision": "decision", "cue_ids": "cue_id", "cue_visible": "cue_visible", "success": "success"}
        item = {key: np.asarray([s[value] for s in steps]) for key, value in keys.items()
                if key not in {"images", "directions"}}
        item.update(images=np.stack([s["observation"]["image"] for s in steps]),
                    directions=np.asarray([s["observation"]["direction"] for s in steps]), task_id="test")
        batch = collate_trajectories([item])
        for kind in (AD, RAD):
            config = _config(kind.__name__) | {"n_transit": 12 if kind is AD else 6}
            model = kind(config).eval()
            with torch.inference_mode():
                output = model(batch)
                expected = output["final_logits"] if kind is AD else output["logits_by_row"][0]
                context = model.start_context(steps[0]["observation"])
                for index, step in enumerate(steps[:-1]):
                    model.observe(context, step["action"], step["reward"], step["terminated"],
                                  step["truncated"], steps[index + 1]["observation"])
                self.assertTrue(torch.allclose(expected, model.action_logits(context), atol=1e-5))
                if kind is RAD:
                    self.assertGreater(context["num_compressions"], 0)
                    self.assertEqual(model.start_context(steps[0]["observation"])["num_compressions"], 0)

    def test_fixed_pool_end_to_end_smoke(self):
        from rad_memory.train_task_pool import train_task, train_pool
        from rad_memory.recurrent_ppo import RecurrentPPOConfig
        from rad_memory.training import train_compression, train_distillation
        from rad_memory.evaluate import evaluate_pool
        from stable_baselines3.common.vec_env import DummyVecEnv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = generate_pool(MemoryTaskSpec("MiniGrid-MemoryS7-v0", 2, "train", horizon=3), 2, 0.5, 1)
            manifest = root / "pool.json"
            manifest.write_text(json.dumps(pool), encoding="utf-8")
            spec = MemoryTaskSpec.from_dict(next(t for t in pool["tasks"] if t["split"] == "train"))
            from rad_memory.envs import FlattenMemoryObservation
            vec = DummyVecEnv([lambda: FlattenMemoryObservation(make_memory_env(spec))])
            try:
                first = vec.reset()
                for _ in range(3):
                    observation, _, done, _ = vec.step(np.asarray([0]))
                self.assertTrue(done[0])
                np.testing.assert_array_equal(first, observation)
            finally:
                vec.close()
            results = train_pool(manifest, [0, 1], root / "source", root / "data",
                                 total_timesteps=16, evaluation_interval=8, evaluation_episodes=1,
                                 minimum_success_rate=0, required_consecutive_evals=1,
                                 ppo_config=RecurrentPPOConfig(n_steps=8, batch_size=8, n_epochs=1))
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertEqual(result["task_id"], spec.task_id)
                self.assertTrue(result["converged"])  # Plumbing threshold only, not a convergence claim.
            with self.assertRaisesRegex(ValueError, "existing run"):
                train_task(spec, pool["fingerprint"], 0, root / "source", root / "data")
            config = _config() | {
                "history_scope": "task", "task_manifest": str(manifest),
                "source_algorithm": "recurrent_ppo", "max_context_length": 12,
                "seed": 0, "train_steps": 1, "pretrain_steps": 1, "train_batch_size": 2,
                "pretrain_batch_size": 2, "pretrain_lr": 0.0003, "warmup_steps": 0,
                "pretrain_warmup_steps": 0, "num_workers": 0, "mixed_precision": "no",
            }
            pretrain = train_compression(config, root / "data", root / "pretrain")
            for kind in ("AD", "RAD"):
                checkpoint = train_distillation(config, root / "data", root / kind, model_kind=kind,
                                                 pretrain_checkpoint=pretrain if kind == "RAD" else None)
                records, summary = evaluate_pool(checkpoint, manifest, 3, trials=2)
                self.assertEqual(len(records), 6)
                self.assertEqual(len(summary["adaptation_curve"]), 3)
                _, reset = evaluate_pool(checkpoint, manifest, 3, trials=1, reset_context_each_episode=True)
                self.assertTrue(reset["reset_context_each_episode"])
                self.assertEqual(summary["tasks"], 1)


if __name__ == "__main__":
    unittest.main()
