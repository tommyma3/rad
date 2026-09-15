"""Explicit phase-specific trainability and optimizer groups."""


def configure_trainability(model, pretrain=False):
    for name, parameter in model.named_parameters():
        if pretrain:
            parameter.requires_grad_(name.startswith(("compression_transformer.", "reconstruction_decoder.")))
        else:
            parameter.requires_grad_(not name.startswith("reconstruction_decoder."))
            if name == "null_latent_tokens" and not model.config.get("always_use_latent_prefix", False):
                parameter.requires_grad_(False)


def optimizer_groups(model, config):
    groups = {name: [] for name in ("ad", "compression", "latent")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = "compression" if name.startswith(("compression_transformer.", "reconstruction_decoder.")) else (
            "latent" if name.startswith(("latent_", "null_latent")) else "ad")
        groups[group].append(parameter)
    return [{"params": parameters, "lr": float(config.get(f"{name}_lr", config["lr"])),
             "group_name": name} for name, parameters in groups.items() if parameters]
