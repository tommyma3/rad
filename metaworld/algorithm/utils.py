import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HistoryLoggerCallback(BaseCallback):
    def __init__(self, env_name, env_idx, history=None, dim_obs=11):
        super(HistoryLoggerCallback, self).__init__()
        self.env_name = env_name
        self.env_idx = env_idx
        self.dim_obs = dim_obs  # Meta-world observation dimension

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.already_success = []
        self.success = []

        self.history = history

        self.episode_rewards = []
        self.episode_success = []

    def _on_step(self) -> bool:
        # Get observations - handle both vectorized and non-vectorized cases
        new_obs = self.locals["new_obs"]
        
        # Ensure new_obs is 2D (batch_size, obs_dim)
        if new_obs.ndim == 1:
            new_obs = new_obs.reshape(1, -1)
        
        # For Meta-world: obs[:, dim_obs:2*dim_obs] is the previous state, obs[:, :dim_obs] is current
        # Observation contains: current_obs (dim_obs), previous_obs (dim_obs), previous_action (4), goal (if any)
        self.states.append(new_obs[:, list(range(self.dim_obs, 2*self.dim_obs))])
        self.next_states.append(new_obs[:, list(range(self.dim_obs))])
        
        # Handle infos - can be list or single dict
        infos = self.locals['infos']
        if isinstance(infos, dict):
            infos = [infos]
        success = [info.get('success', False) for info in infos]
        self.success.append(success)
        self.episode_success.append(success)
        
        # Handle actions - ensure 2D
        actions = self.locals["actions"]
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        self.actions.append(actions)
        
        # Handle rewards - ensure 1D array
        rewards = self.locals["rewards"]
        if np.isscalar(rewards):
            rewards = np.array([rewards])
        elif isinstance(rewards, np.ndarray) and rewards.ndim == 0:
            rewards = np.array([rewards.item()])
        self.rewards.append(rewards.copy())
        
        # Handle dones - ensure 1D array
        dones = self.locals["dones"]
        if np.isscalar(dones):
            dones = np.array([dones])
        elif isinstance(dones, np.ndarray) and dones.ndim == 0:
            dones = np.array([dones.item()])
        self.dones.append(dones)
        
        self.episode_rewards.append(rewards)
        
        # Check if episode is done
        if dones[0] if isinstance(dones, np.ndarray) else dones:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
            
            mean_success_rate = np.mean((np.sum(self.episode_success, axis=0) > 0.0))
            self.logger.record('rollout/mean_success_rate', mean_success_rate)
            self.episode_success = []
            
        return True

    def _on_training_end(self):
        self.history[self.env_idx] = {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'next_states': np.array(self.next_states, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.bool_),
            'success': np.array(self.success, dtype=np.float32)
        }