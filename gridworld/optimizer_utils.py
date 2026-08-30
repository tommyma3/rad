"""Optimizer parameter grouping for RAD fine-tuning."""


COMPRESSION_PARAMETER_PREFIXES = (
    'compression_transformer.',
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


def freeze_reconstruction_decoder_for_finetuning(model):
    """Keep the pretraining-only decoder out of RAD fine-tuning and DDP."""
    model.reconstruction_decoder.requires_grad_(False)


def build_compression_pretraining_parameters(model):
    """Enable exactly the parameters used by recurrent reconstruction replay."""
    model.requires_grad_(False)
    model.compression_transformer.requires_grad_(True)
    model.reconstruction_decoder.requires_grad_(True)
    model.embed_state.requires_grad_(True)
    model.embed_action.requires_grad_(True)
    model.embed_reward.requires_grad_(True)
    model.type_embedding.requires_grad_(True)
    if model.always_use_latent_prefix:
        model.null_latent_tokens.requires_grad_(True)

    if model.latent_update_mode == 'residual':
        model.latent_residual_norm.requires_grad_(True)
    elif model.latent_update_mode == 'multiplicative_gate':
        model.latent_multiplicative_gate.requires_grad_(True)
    elif model.latent_update_mode == 'gru_gate':
        model.latent_gru_gate.requires_grad_(True)
        model.latent_gru_candidate.requires_grad_(True)
    elif model.latent_update_mode != 'replace':
        raise ValueError(f'Unknown latent_update_mode: {model.latent_update_mode}')

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


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
