"""
Environment utilities for Meta-world environments.
"""
from gymnasium.wrappers import TimeLimit
import metaworld


def make_env(config, env_cls, task):
    """Create a Meta-world environment with the given task."""
    def _init():
        env = env_cls()
        env.set_task(task)
        return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


def get_ml1_envs(config):
    """
    Get ML1 train and test environment instances.
    
    Returns:
        train_envs: List of (env_cls, task_instance) tuples for training
        test_envs: List of (env_cls, task_instance) tuples for testing
    """
    task = config['task']
    ml1 = metaworld.ML1(env_name=task, seed=config['mw_seed'])
    
    train_envs = []
    test_envs = []
    
    # Get train environments
    for task_name, env_cls in ml1.train_classes.items():
        task_instances = [t for t in ml1.train_tasks if t.env_name == task_name]
        for task_instance in task_instances:
            train_envs.append((env_cls, task_instance))
    
    # Get test environments
    for task_name, env_cls in ml1.test_classes.items():
        task_instances = [t for t in ml1.test_tasks if t.env_name == task_name]
        for task_instance in task_instances:
            test_envs.append((env_cls, task_instance))
    
    return train_envs, test_envs


def sample_metaworld(config, shuffle=False):
    """
    Sample Meta-world environments for training and testing.
    
    Returns train_envs and test_envs as lists of (env_cls, task_instance) tuples.
    """
    return get_ml1_envs(config)


SAMPLE_ENVIRONMENT = {
    'metaworld': sample_metaworld,
}