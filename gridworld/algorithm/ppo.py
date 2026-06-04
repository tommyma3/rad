import torch
from stable_baselines3 import PPO


class PPOWrapper(PPO):
    def __init__(self, config, env, seed, log_dir=None):
        policy = config['policy']
        n_steps = config['n_steps']
        batch_size = config['batch_size']
        n_epochs = config['n_epochs']
        lr = config['source_lr']
        device = config['device']
        env = env

        # Optional PPO hyperparameters (use defaults when missing)
        gamma = config.get('gamma', 0.99)
        gae_lambda = config.get('gae_lambda', 0.95)
        clip_range = config.get('clip_range', 0.2)
        ent_coef = config.get('ent_coef', 0.0)
        vf_coef = config.get('vf_coef', 0.5)
        max_grad_norm = config.get('max_grad_norm', 0.5)
        target_kl = config.get('target_kl', None)

        super(PPOWrapper, self).__init__(
            policy=policy,
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            verbose=0,
            seed=seed,
            device=device,
            tensorboard_log=log_dir,
        )