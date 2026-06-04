import torch
from stable_baselines3 import PPO


class PPOWrapper(PPO):
    def __init__(self, config, env, seed, log_dir, device=None):
        policy = config['policy']
        n_steps = config['n_steps']
        batch_size = config['batch_size']
        n_epochs = config['n_epochs']
        lr = config['source_lr']
        
        # Use provided device or auto-detect
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(PPOWrapper, self).__init__(policy=policy,
                                         env=env,
                                         learning_rate=lr,
                                         n_steps=n_steps,
                                         batch_size=batch_size,
                                         n_epochs=n_epochs,
                                         verbose=0,
                                         seed=seed,
                                         device=device,
                                         tensorboard_log=log_dir if log_dir else None)