"""Protocol and names for the Darkroom long-term memory size ablation."""

PROTOCOL = 'darkroom-memory-size-v1'
SIZES = (3, 6, 15, 30, 60)
BASELINE_SIZE = 15


def run_name(size, seed, pretrain=False):
    stage = 'RAD-pretrain' if pretrain else 'RAD'
    return f'{stage}-darkroom-memory{size}-split0-train{seed}'


def validate_memory_size_config(config, pretrain=False):
    expected = dict(memory_size_comparison=PROTOCOL, env='darkroom', grid_size=9,
                    horizon=20, env_split_seed=0, collection_env_split_seed=0,
                    dataset_task_mapping='collection_order', train_env_ratio=0.9,
                    compressor_type='ae', always_use_latent_prefix=False,
                    latent_update_mode='gru_gate', short_memory_keep=5,
                    first_recent_capacity=35 if pretrain else 30,
                    recurrent_recent_capacity=35 if pretrain else 25,
                    n_transit=40 if pretrain else 30, torch_compile=False)
    if not pretrain:
        expected['save_best_model'] = True
    if config.get('compressor_comparison'):
        raise ValueError('Memory-size and compressor-comparison protocols are separate experiments')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Memory-size protocol requires {key}={value!r}')
    size = config.get('n_compress_tokens')
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or size % 3:
        raise ValueError('n_compress_tokens must be a positive multiple of three')
    if not pretrain and config.get('rad_batching_strategy') != 'compression_buckets':
        raise ValueError('Memory-size protocol requires compression_buckets')
