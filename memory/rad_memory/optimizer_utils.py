"""Explicit RAD parameter groups for auditable learning rates."""

from __future__ import annotations


COMPRESSION_PREFIXES = ("compression_transformer.", "reconstruction_decoder.")
LATENT_PREFIXES = (
    "null_latent_tokens",
    "latent_residual_norm.",
    "latent_multiplicative_gate.",
    "latent_gru_gate.",
    "latent_gru_candidate.",
)


def _normalize_name(name: str) -> str:
    return name.replace("._orig_mod.", ".").removeprefix("_orig_mod.")


def build_rad_optimizer_param_groups(model, config: dict) -> list[dict]:
    fallback = float(config["lr"])
    rates = {
        "ad": float(config.get("ad_lr", fallback)),
        "compression": float(config.get("compression_lr", fallback)),
        "latent": float(config.get("latent_lr", fallback)),
    }
    grouped = {name: [] for name in rates}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized = _normalize_name(name)
        if normalized.startswith(COMPRESSION_PREFIXES):
            group = "compression"
        elif normalized.startswith(LATENT_PREFIXES):
            group = "latent"
        else:
            group = "ad"
        grouped[group].append(parameter)
    return [
        {"params": grouped[name], "lr": rates[name], "name": name}
        for name in ("ad", "compression", "latent")
        if grouped[name]
    ]


def freeze_reconstruction_decoder_for_finetuning(model) -> None:
    model.reconstruction_decoder.requires_grad_(False)
