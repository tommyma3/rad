from __future__ import annotations

import unittest

import torch

from rad_memory.model import AD, RAD
from rad_memory.optimizer_utils import build_rad_optimizer_param_groups


def _config(model: str = "AD") -> dict:
    return {
        "model": model,
        "n_transit": 6,
        "num_actions": 7,
        "tf_n_embd": 32,
        "tf_n_head": 4,
        "tf_n_layer": 2,
        "tf_dim_feedforward": 64,
        "tf_dropout": 0.0,
        "tile_embedding_dim": 4,
        "label_smoothing": 0.0,
        "n_compress_tokens": 6,
        "short_memory_keep": 1,
        "compress_n_heads": 2,
        "compress_n_layers": 1,
        "latent_update_mode": "gru_gate",
        "always_use_latent_prefix": True,
        "max_gradient_rounds": 2,
        "max_compressions": None,
        "lr": 3e-4,
        "ad_lr": 3e-4,
        "compression_lr": 1e-4,
        "latent_lr": 3e-4,
    }


def _batch(length: int = 6) -> dict[str, torch.Tensor]:
    batch = 2
    return {
        "images": torch.randint(0, 3, (batch, length, 7, 7, 3), dtype=torch.uint8),
        "directions": torch.randint(0, 4, (batch, length)),
        "actions": torch.randint(0, 7, (batch, length)),
        "rewards": torch.zeros(batch, length),
        "terminated": torch.zeros(batch, length),
        "truncated": torch.zeros(batch, length),
        "valid_mask": torch.ones(batch, length, dtype=torch.bool),
        "decision_mask": torch.zeros(batch, length, dtype=torch.bool),
    }


class ModelContractTest(unittest.TestCase):
    def test_ad_forward_and_causal_prefix(self):
        torch.manual_seed(0)
        model = AD(_config()).eval()
        batch = _batch()
        original = model(batch)["logits"][:, :3].detach()
        batch["actions"][:, 4:] = (batch["actions"][:, 4:] + 1) % 7
        changed = model(batch)["logits"][:, :3].detach()
        self.assertTrue(torch.allclose(original, changed, atol=1e-6))

    def test_rad_compresses_and_null_latent_receives_gradient(self):
        model = RAD(_config("RAD"))
        output = model(_batch(length=12))
        self.assertGreater(int(output["num_compressions"]), 0)
        output["loss_total"].backward()
        self.assertIsNotNone(model.null_latent_tokens.grad)

    def test_rad_pretraining_reconstructs_null_prefix_and_raw_tokens(self):
        model = RAD(_config("RAD"))
        output = model.forward_pretrain_compression(_batch())
        self.assertEqual(output["loss_recon"].ndim, 0)

    def test_rad_batch_final_query_matches_streaming_query(self):
        torch.manual_seed(4)
        model = RAD(_config("RAD")).eval()
        batch = _batch(length=12)
        batch = {key: value[:1] for key, value in batch.items()}
        batch_logits = model(batch)["logits_by_row"][0]
        first = {
            "image": batch["images"][0, 0].numpy(),
            "direction": int(batch["directions"][0, 0]),
        }
        context = model.start_context(first)
        for step in range(11):
            next_observation = {
                "image": batch["images"][0, step + 1].numpy(),
                "direction": int(batch["directions"][0, step + 1]),
            }
            model.observe(
                context,
                int(batch["actions"][0, step]),
                float(batch["rewards"][0, step]),
                bool(batch["terminated"][0, step]),
                bool(batch["truncated"][0, step]),
                next_observation,
            )
        streaming_logits = model.action_logits(context)
        self.assertTrue(torch.allclose(batch_logits, streaming_logits, atol=1e-5))

    def test_optimizer_groups_are_complete_and_disjoint(self):
        model = RAD(_config("RAD"))
        groups = build_rad_optimizer_param_groups(model, _config("RAD"))
        self.assertEqual({group["name"] for group in groups}, {"ad", "compression", "latent"})
        parameters = [parameter for group in groups for parameter in group["params"]]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertEqual(
            {id(parameter) for parameter in parameters},
            {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
        )


if __name__ == "__main__":
    unittest.main()
