"""Causal AD with an explicit budget of K completed transitions plus a query."""
import torch
from torch import nn
from torch.nn import functional as F

from .gpt2 import GPT2Transformer


def masked_action_loss(logits, targets, mask, label_smoothing=0.0):
    mask = mask.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0, logits.new_zeros(())
    selected = logits[mask]
    targets = targets[mask].long()
    loss = F.cross_entropy(selected, targets, label_smoothing=label_smoothing)
    accuracy = (selected.argmax(-1) == targets).float().mean()
    return loss, accuracy


class AD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context_steps = int(config["context_steps"])
        self.num_arms = int(config["num_arms"])
        if self.context_steps < 1:
            raise ValueError("context_steps must be positive")
        width = config["tf_n_embd"]
        latent_slots = config.get("n_compress_tokens", 15) if config["model"] == "RAD" else 0
        self.max_seq_length = 3 * self.context_steps + 1 + latent_slots
        self.ad_transformer = GPT2Transformer(
            width, config["tf_n_head"], config["tf_n_layer"],
            max_seq_length=self.max_seq_length,
            dim_feedforward=config["tf_dim_feedforward"], dropout=config["tf_dropout"])
        self.embed_state = nn.Embedding(2, width)
        self.embed_action = nn.Linear(self.num_arms, width)
        self.embed_reward = nn.Linear(1, width)
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, 3, width))
        self.pred_action = nn.Linear(width, self.num_arms)
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

    @property
    def device(self):
        return self.type_embedding.device

    def embed_transitions(self, states, actions, rewards):
        states = self.embed_state(states.long())
        actions = self.embed_action(F.one_hot(actions.long(), self.num_arms).to(states.dtype))
        rewards = self.embed_reward(rewards.to(states.dtype).unsqueeze(-1))
        tokens = torch.stack((states, actions, rewards), dim=2) + self.type_embedding
        return tokens.flatten(1, 2)

    def new_state(self):
        return {"recent": None, "latent": None, "compression_count": 0}

    def ingest(self, state, tokens, total_compressions=None):
        recent = state["recent"]
        recent = tokens if recent is None else torch.cat((recent, tokens), dim=1)
        state["recent"] = recent[:, -3 * self.context_steps:]
        return state

    def query_logits(self, state, observations):
        query = self.embed_state(observations.long()).unsqueeze(1) + self.type_embedding[:, :, 0]
        pieces = [query] if state["recent"] is None else [state["recent"], query]
        output = self.ad_transformer(torch.cat(pieces, dim=1), use_causal_mask=True)
        return self.pred_action(output[:, -1])

    def prefix_state(self, batch):
        tokens = self.embed_transitions(batch["states"], batch["actions"], batch["rewards"])
        return self.ingest(self.new_state(), tokens)

    def forward(self, batch, pretrain=False):
        if pretrain:
            raise ValueError("Compression pretraining requires RAD")
        state = self.prefix_state(batch)
        logits = self.query_logits(state, batch["query_states"])
        loss, accuracy = masked_action_loss(logits, batch["targets"], batch["loss_mask"],
                                            self.config.get("label_smoothing", 0.0))
        return {"loss": loss, "accuracy": accuracy, "logits": logits,
                "num_compressions": state["compression_count"]}
