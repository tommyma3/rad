import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from bandit.collect import collect_dataset
from bandit.dataset import BanditDataset, assert_disjoint
from bandit.env import AdversarialBandit, BanditTask, BANDIT, DISTRACTOR, DelayedBandit, sample_task
from bandit.evaluation import ModelPolicy, make_eval_manifest, rollout_metrics
from bandit.model import make_model
from bandit.model.ad import masked_action_loss
from bandit.optimizer_utils import configure_trainability
from bandit.rollout import generate_history
from bandit.utils import load_config


def tiny_config(model="rad"):
    config = load_config(model)
    config.update(context_steps=5, tf_n_embd=16, tf_n_head=2, tf_n_layer=1,
                  tf_dim_feedforward=32, tf_dropout=0.0, short_memory_keep=1,
                  n_compress_tokens=3, compress_n_heads=2, compress_n_layers=1,
                  pre_steps=6, post_steps=6, train_delays=[12], eval_delays=[0, 5, 12],
                  batch_size=2, train_steps=4, warmup_steps=1, validation_batches=1,
                  log_interval=1, checkpoint_interval=100, eval_interval=100)
    return config


class EnvironmentTests(unittest.TestCase):
    def test_gap_equivalence_and_frozen_ucb(self):
        task = sample_task(15)
        direct, _ = generate_history(task, delay=0)
        delayed, audit = generate_history(task, delay=103)
        for key in ("actions", "rewards", "states"):
            np.testing.assert_array_equal(direct[key], delayed[key][delayed["loss_mask"]])
        for key in ("counts", "reward_sums"):
            np.testing.assert_array_equal(audit["before_gap"][key], audit["after_gap"][key])
        self.assertEqual(audit["before_gap"]["rng_state"], audit["after_gap"]["rng_state"])
        self.assertEqual(audit["after_gap"]["total_pulls"], 50)
        self.assertEqual(audit["final_ucb_state"]["total_pulls"], 100)

    def test_distractor_independent_of_task(self):
        first, _ = generate_history(sample_task(1), delay=19)
        second, _ = generate_history(sample_task(2), delay=19)
        gap = ~first["loss_mask"]
        for key in ("states", "actions", "rewards"):
            np.testing.assert_array_equal(first[key][gap], second[key][gap])
        self.assertTrue(np.all(first["rewards"][gap] == 0))

    def test_phase_boundaries_reset_and_task_persistence(self):
        task = sample_task(0)
        env = DelayedBandit(task, pre_steps=1, post_steps=1, delay=1)
        self.assertEqual(env.observation, BANDIT)
        self.assertEqual(env.step(0)[0], DISTRACTOR)
        self.assertEqual(env.step(0)[1], 0)
        self.assertEqual(env.observation, BANDIT)
        self.assertTrue(env.step(0)[2])
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        self.assertEqual(env.bandit.task, task)
        self.assertEqual(env.bandit.pull_count, 0)

    def test_independent_uniform_means_and_reproducibility(self):
        for seed in range(10):
            np.testing.assert_array_equal(sample_task(seed).means, np.random.default_rng(seed).uniform(0, 1, 10))
        self.assertEqual(sample_task(3), sample_task(3))
        self.assertEqual(sample_task(3).reward_std, 0.3)
        with self.assertRaises(ValueError):
            sample_task(3, "odd")

    def test_manifest_prefix_stable_when_task_count_increases(self):
        cfg = tiny_config()
        a = make_eval_manifest(3, 2, ["uniform"], cfg)
        b = make_eval_manifest(3, 4, ["uniform"], cfg)
        self.assertEqual(a["records"], b["records"][:2])

    def test_gaussian_reward_moments_and_no_clipping(self):
        task = BanditTask("fixed", (0.5, 0.5), "uniform", 0)
        env = AdversarialBandit(task, reward_seed=9, horizon=50000)
        rewards = np.array([env.pull(0) for _ in range(50000)])
        self.assertAlmostEqual(float(rewards.mean()), 0.5, delta=0.01)
        self.assertAlmostEqual(float(rewards.std()), 0.3, delta=0.01)
        self.assertLess(rewards.min(), 0.0)
        self.assertGreater(rewards.max(), 1.0)
        self.assertEqual(BanditTask.from_dict(task.to_dict()), task)
        old_task = dict(task.to_dict(), sampler="parity_conditioned_bernoulli_v1")
        with self.assertRaises(ValueError):
            BanditTask.from_dict(old_task)

    def test_metrics_exclude_gap(self):
        task = sample_task(9)
        a, _ = generate_history(task, delay=0)
        b, _ = generate_history(task, delay=100)
        self.assertEqual(rollout_metrics(task, a, 50)["post_return"],
                         rollout_metrics(task, b, 50)["post_return"])


class DatasetTests(unittest.TestCase):
    def test_collection_masks_prefixes_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            collect_dataset(directory, tiny_config(), tasks=4, validation_tasks=3)
            training = BanditDataset(Path(directory) / "train.hdf5", expected_split="train")
            validation = BanditDataset(Path(directory) / "validation.hdf5", expected_split="validation")
            assert_disjoint(training, validation)
            self.assertEqual(training.num_targets, 4 * 12)
            self.assertIn(18, training.buckets)  # The very first post-gap action.
            self.assertNotIn(12, training.buckets)  # Distractor target excluded.
            batch = training.sample_batch(4, np.random.default_rng(0))
            self.assertTrue(batch["loss_mask"].all())
            self.assertTrue((batch["query_states"] == BANDIT).all())
            genuine_rewards = np.concatenate([h["rewards"][h["loss_mask"]] for h in training.histories])
            self.assertTrue(np.any((genuine_rewards < 0) | (genuine_rewards > 1)))
            with self.assertRaises(ValueError):
                assert_disjoint(training, training)
            with self.assertRaises(FileExistsError):
                collect_dataset(directory, tiny_config(), tasks=4, validation_tasks=3)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batch(self, length=18, batch_size=2):
        return {"states": torch.cat((torch.zeros(batch_size, min(6, length), dtype=torch.long),
                                      torch.ones(batch_size, max(0, length-6), dtype=torch.long)), dim=1),
                "actions": torch.randint(0, 10, (batch_size, length)),
                "rewards": torch.rand(batch_size, length),
                "query_states": torch.zeros(batch_size, dtype=torch.long),
                "targets": torch.zeros(batch_size, dtype=torch.long),
                "loss_mask": torch.ones(batch_size, dtype=torch.bool)}

    def test_loss_mask_and_all_masked_safety(self):
        logits = torch.randn(3, 10, requires_grad=True)
        targets = torch.tensor([1, 2, 3])
        loss, _ = masked_action_loss(logits, targets, torch.tensor([True, False, True]))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(10)))
        zero, _ = masked_action_loss(logits, targets, torch.zeros(3, dtype=torch.bool))
        self.assertEqual(float(zero.detach()), 0)
        self.assertTrue(torch.isfinite(zero))

    def test_incremental_and_prefix_logits_match(self):
        for kind in ("ad_short", "rad"):
            for length in (0, 5, 6, 11, 18):
                model = make_model(tiny_config(kind)).eval()
                batch = self.batch(length)
                with torch.no_grad():
                    expected = model(batch)["logits"]
                    state = model.new_state()
                    for index in range(length):
                        tokens = model.embed_transitions(*[batch[key][:, index:index+1]
                                                           for key in ("states", "actions", "rewards")])
                        model.ingest(state, tokens)
                    actual = model.query_logits(state, batch["query_states"])
                torch.testing.assert_close(actual, expected)
                if kind == "rad":
                    self.assertEqual(state["compression_count"], model.compression_count_for_length(length))

    def test_first_post_gap_short_context_has_no_bandit_evidence(self):
        config = tiny_config("ad_short")
        model = make_model(config).eval()
        batch = self.batch(11)
        with torch.no_grad():
            logits = model(batch)["logits"]
            changed = copy.deepcopy(batch)
            changed["rewards"][:, :6] += 100
            changed["actions"][:, :6] = (changed["actions"][:, :6] + 1) % 10
            torch.testing.assert_close(logits, model(changed)["logits"])
            state = model.prefix_state(batch)
            self.assertEqual(state["recent"].shape[1], 15)

    def test_targets_do_not_enter_policy_input(self):
        for kind in ("ad_short", "rad"):
            model = make_model(tiny_config(kind)).eval()
            batch = self.batch()
            with torch.no_grad():
                logits = model(batch)["logits"]
                batch["targets"] += 1
                torch.testing.assert_close(logits, model(batch)["logits"])

    def test_rad_post_gap_loss_reaches_early_evidence(self):
        model = make_model(tiny_config()).eval()
        batch = self.batch(18)
        batch["rewards"].requires_grad_()
        model(batch)["loss"].backward()
        self.assertGreater(float(batch["rewards"].grad[:, :6].abs().sum()), 0)
        self.assertGreater(float(model.compression_transformer.compress_queries.grad.abs().sum()), 0)
        self.assertGreater(float(model.latent_gru_gate.weight.grad.abs().sum()), 0)

    def test_pretraining_and_distillation_trainability(self):
        model = make_model(tiny_config())
        configure_trainability(model, pretrain=True)
        loss = model(self.batch(5), pretrain=True)["loss"]
        loss.backward()
        self.assertIsNone(model.embed_state.weight.grad)
        self.assertIsNotNone(model.reconstruction_decoder.position_queries.grad)
        model.zero_grad(set_to_none=True)
        configure_trainability(model, pretrain=False)
        model(self.batch())["loss"].backward()
        self.assertIsNone(model.reconstruction_decoder.position_queries.grad)
        self.assertIsNotNone(model.embed_state.weight.grad)

    def test_policy_state_resets_between_tasks(self):
        model = make_model(tiny_config()).eval()
        policy = ModelPolicy(model)
        for _ in range(12):
            policy.observe(BANDIT, 1, 1.0)
        self.assertGreater(policy.compression_count, 0)
        policy.reset()
        self.assertEqual(policy.compression_count, 0)
        self.assertIsNone(policy.state["latent"])

    def test_null_prefix_and_all_latent_modes(self):
        for mode in ("replace", "residual", "multiplicative_gate", "gru_gate"):
            config = tiny_config()
            config.update(latent_update_mode=mode, always_use_latent_prefix=True)
            model = make_model(config).eval()
            model(self.batch(0))["loss"].backward()
            self.assertIsNotNone(model.null_latent_tokens.grad)
            model.zero_grad(set_to_none=True)
            state = model.prefix_state(self.batch(18))
            logits = model.query_logits(state, torch.zeros(2, dtype=torch.long))
            self.assertTrue(torch.isfinite(logits).all())
            logits.sum().backward()
            self.assertIsNone(model.null_latent_tokens.grad)
            self.assertEqual(model.max_seq_length, 3 * config["context_steps"] + 1 + config["n_compress_tokens"])


if __name__ == "__main__":
    unittest.main()
