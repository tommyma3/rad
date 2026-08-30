"""Optimizer parameter grouping for RAD fine-tuning."""


COMPRESSION_PARAMETER_PREFIXES = (
    'compression_transformer.',
    'reconstruction_decoder.',
)

LATENT_PARAMETER_PREFIXES = (
    'latent_type_embedding',
    'null_latent_tokens',
    'latent_residual_norm.',
    'latent_multiplicative_gate.',
    'latent_gru_gate.',
    'latent_gru_candidate.',
)


def _normalized_parameter_name(name):
    """Remove torch.compile wrappers without changing the module hierarchy."""
    return name.replace('._orig_mod.', '.').removeprefix('_orig_mod.')


def _rad_parameter_group(name):
    name = _normalized_parameter_name(name)
    if name.startswith(COMPRESSION_PARAMETER_PREFIXES):
        return 'compression'
    if name.startswith(LATENT_PARAMETER_PREFIXES):
        return 'latent'
    return 'ad'


def build_compression_pretraining_parameters(model):
    """Return the parameters used by compression reconstruction pretraining."""
    parameters = (
        list(model.compression_transformer.parameters())
        + list(model.reconstruction_decoder.parameters())
        + list(model.embed_state.parameters())
        + list(model.embed_action.parameters())
        + list(model.embed_reward.parameters())
        + [model.type_embedding]
    )
    if model.always_use_latent_prefix:
        parameters.append(model.null_latent_tokens)
    return parameters


def build_rad_optimizer_param_groups(model, config):
    """Partition every trainable RAD parameter into one learning-rate group."""
    fallback_lr = float(config['lr'])
    group_lrs = {
        'ad': float(config.get('ad_lr', fallback_lr)),
        'compression': float(config.get('compression_lr', fallback_lr)),
        'latent': float(config.get('latent_lr', fallback_lr)),
    }
    grouped_params = {name: [] for name in group_lrs}

    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            grouped_params[_rad_parameter_group(parameter_name)].append(parameter)

    parameter_groups = []
    for group_name, learning_rate in group_lrs.items():
        params = grouped_params[group_name]
        if params:
            parameter_groups.append({
                'params': params,
                'lr': learning_rate,
                'group_name': group_name,
            })

    return parameter_groups
