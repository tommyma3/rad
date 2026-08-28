"""Trainability policies and optimizer parameter grouping for RAD."""


AD_PARAMETER_PREFIXES = (
    'ad_transformer.',
    'embed_state.',
    'embed_action.',
    'embed_reward.',
    'type_embedding',
    'pred_action.',
)

COMPRESSION_PARAMETER_PREFIXES = ('compression_transformer.',)

FROZEN_FINETUNING_PARAMETER_PREFIXES = (
    'compression_transformer.compress_queries',
    'reconstruction_decoder.',
    'latent_type_embedding',
    'null_latent_tokens',
)

LATENT_UPDATE_PARAMETER_PREFIXES = {
    'replace': (),
    'residual': ('latent_residual_norm.',),
    'multiplicative_gate': ('latent_multiplicative_gate.',),
    'gru_gate': ('latent_gru_gate.', 'latent_gru_candidate.'),
}


def _normalized_parameter_name(name):
    """Remove torch.compile wrappers without changing the module hierarchy."""
    return name.replace('._orig_mod.', '.').removeprefix('_orig_mod.')


def configure_compression_pretraining(model):
    """Train only the reconstruction path and its learned token parameters."""
    model.requires_grad_(False)
    model.compression_transformer.requires_grad_(True)
    model.reconstruction_decoder.requires_grad_(True)
    model.embed_state.requires_grad_(True)
    model.embed_action.requires_grad_(True)
    model.embed_reward.requires_grad_(True)
    model.type_embedding.requires_grad_(True)
    model.null_latent_tokens.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def configure_rad_finetuning(model):
    """Apply the end-to-end RAD allowlist before compilation and DDP setup."""
    model.requires_grad_(False)
    model.ad_transformer.requires_grad_(True)
    model.embed_state.requires_grad_(True)
    model.embed_action.requires_grad_(True)
    model.embed_reward.requires_grad_(True)
    model.type_embedding.requires_grad_(True)
    model.pred_action.requires_grad_(True)
    model.compression_transformer.requires_grad_(True)
    model.compression_transformer.compress_queries.requires_grad_(False)

    mode = model.latent_update_mode
    if mode == 'residual':
        model.latent_residual_norm.requires_grad_(True)
    elif mode == 'multiplicative_gate':
        model.latent_multiplicative_gate.requires_grad_(True)
    elif mode == 'gru_gate':
        model.latent_gru_gate.requires_grad_(True)
        model.latent_gru_candidate.requires_grad_(True)
    elif mode != 'replace':
        raise ValueError(f'Unknown latent_update_mode: {mode}')


def _rad_parameter_group(name, latent_update_mode):
    name = _normalized_parameter_name(name)
    if name.startswith(FROZEN_FINETUNING_PARAMETER_PREFIXES):
        raise ValueError(f'Frozen RAD parameter unexpectedly marked trainable: {name}')
    if name.startswith(COMPRESSION_PARAMETER_PREFIXES):
        return 'compression'
    if name.startswith(LATENT_UPDATE_PARAMETER_PREFIXES[latent_update_mode]):
        return 'latent'
    if name.startswith(AD_PARAMETER_PREFIXES):
        return 'ad'
    raise ValueError(f'Unexpected trainable RAD parameter: {name}')


def build_rad_optimizer_param_groups(model, config):
    """Partition every allowed trainable RAD parameter into an LR group."""
    fallback_lr = float(config['lr'])
    group_lrs = {
        'ad': float(config.get('ad_lr', fallback_lr)),
        'compression': float(config.get('compression_lr', fallback_lr)),
        'latent': float(config.get('latent_lr', fallback_lr)),
    }
    grouped_params = {name: [] for name in group_lrs}
    latent_update_mode = getattr(model, 'latent_update_mode', config.get('latent_update_mode', 'replace'))
    if latent_update_mode not in LATENT_UPDATE_PARAMETER_PREFIXES:
        raise ValueError(f'Unknown latent_update_mode: {latent_update_mode}')

    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group_name = _rad_parameter_group(parameter_name, latent_update_mode)
            grouped_params[group_name].append(parameter)

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
