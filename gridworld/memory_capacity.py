"""Recent-history limits shared by RAD recurrence and dataset buckets."""


def recent_capacities(config):
    """Return token capacities before/after the first real compression.

    Explicit capacities are opt-in and expressed in environment timesteps.
    Legacy configurations continue reserving latents inside 3 * n_transit.
    """
    first = config.get('first_recent_capacity')
    recurrent = config.get('recurrent_recent_capacity')
    if first is None and recurrent is None:
        total = 3 * config['n_transit']
        latents = config.get('n_compress_tokens', 15)
        return (total - latents if config.get('always_use_latent_prefix', False) else total,
                total - latents)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (first, recurrent)):
        raise ValueError('Both recent capacities must be positive integer timesteps')
    keep = config.get('short_memory_keep', max(1, config.get('n_compress_tokens', 15) // 3))
    if not 0 <= keep < min(first, recurrent):
        raise ValueError('short_memory_keep must be smaller than both recent capacities')
    return 3 * first, 3 * recurrent


def policy_token_capacity(config):
    first, recurrent = recent_capacities(config)
    latents = config.get('n_compress_tokens', 15)
    initial_prefix = latents if config.get('always_use_latent_prefix', False) else 0
    return max(first + initial_prefix, recurrent + latents)
