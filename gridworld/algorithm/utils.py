import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HistoryLoggerCallback(BaseCallback):
    def __init__(self, env_name, env_idx, history=None, n_stack=1):
        super(HistoryLoggerCallback, self).__init__()
        self.env_name = env_name
        self.env_idx = env_idx
        self.n_stack = n_stack  # Number of stacked frames (for unstacking)

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.history = history

        self.episode_rewards = []
        self.episode_success = []

    def _unstack_obs(self, obs):
        """Extract the most recent (current) observation from stacked observations.
        
        If n_stack > 1, the observation shape is (n_envs, obs_dim * n_stack).
        We extract only the last obs_dim elements which correspond to the current frame.
        """
        if self.n_stack <= 1:
            return obs
        # obs shape: (n_envs, obs_dim * n_stack)
        # We want the last obs_dim elements (most recent frame)
        obs_dim = obs.shape[-1] // self.n_stack
        return obs[..., -obs_dim:]

    def _on_step(self) -> bool:
        # Capture state, action, and reward at each step
        obs = self.locals["obs_tensor"].cpu().numpy()
        next_obs = self.locals["new_obs"]
        
        # Unstack observations to get only the current frame
        obs = self._unstack_obs(obs)
        next_obs = self._unstack_obs(next_obs)
        
        self.states.append(obs)
        self.next_states.append(next_obs)
        self.actions.append(self.locals["actions"])

        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"])

        self.episode_rewards.append(self.locals['rewards'])
        
        if self.locals['dones'][0]:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
                        
        return True

    def _on_training_end(self):
        self.history[self.env_idx] = {
            'states': np.array(self.states, dtype=np.int32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.int32),
            'next_states': np.array(self.next_states, dtype=np.int32),
            'dones': np.array(self.dones, dtype=np.bool_)
        }