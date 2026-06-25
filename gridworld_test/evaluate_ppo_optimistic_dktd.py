from datetime import datetime
import argparse
import os
import os.path as path

import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from algorithm import ALGORITHM
from env import make_env
from utils import get_config


def parse_location(values, name):
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f'{name} requires exactly two integers')
    return np.array(values, dtype=np.int64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate PPO + optimistic exploration on a custom dark key-to-door task.'
    )
    parser.add_argument('--key', type=int, nargs=2, required=True,
                       help='Key location as: x y')
    parser.add_argument('--door', type=int, nargs=2, required=True,
                       help='Door location as: x y')
    parser.add_argument('--seed', type=int, default=0,
                       help='Source RL seed')
    parser.add_argument('--total-timesteps', type=int, default=None,
                       help='Override source RL training timesteps')
    parser.add_argument('--n-envs', type=int, default=None,
                       help='Override number of parallel streams')
    parser.add_argument('--n-stack', type=int, default=1,
                       help='Number of frames to stack')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of true-reward evaluation episodes')
    parser.add_argument('--bonus-coef', type=float, default=None,
                       help='Override optimistic exploration bonus coefficient')
    parser.add_argument('--device', type=str, default=None,
                       help='Override SB3 device')
    parser.add_argument('--log-dir', type=str, default='./runs',
                       help='Directory for saved model and evaluation result')
    args = parser.parse_args()

    config = get_config('./config/env/dark_key_to_door.yaml')
    config.update(get_config('./config/algorithm/ppo_optimistic_dark_key_to_door.yaml'))
    config['alg_seed'] = args.seed
    config['n_stack'] = args.n_stack

    if args.total_timesteps is not None:
        config['total_source_timesteps'] = args.total_timesteps
    if args.n_envs is not None:
        config['n_stream'] = args.n_envs
    if args.bonus_coef is not None:
        config['optimistic_bonus_coef'] = args.bonus_coef
    if args.device is not None:
        config['device'] = args.device

    key = parse_location(args.key, 'key')
    door = parse_location(args.door, 'door')

    run_name = (
        f"PPOOptimistic-dark_key_to_door-key{key[0]}-{key[1]}"
        f"-door{door[0]}-{door[1]}-seed{args.seed}"
    )
    run_dir = path.join(args.log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    visit_counts = {}
    train_env = DummyVecEnv([
        make_env(
            config,
            key=key,
            goal=door,
            optimistic_exploration=True,
            visit_counts=visit_counts,
        )
        for _ in range(config['n_stream'])
    ])
    if args.n_stack > 1:
        train_env = VecFrameStack(train_env, n_stack=args.n_stack)

    alg = ALGORITHM[config['alg']](
        config,
        train_env,
        seed=args.seed,
        log_dir=run_dir,
    )

    start_time = datetime.now()
    print(f'Training started at {start_time}')
    print(f'Key: {key.tolist()}, door: {door.tolist()}')
    print(f"Timesteps: {config['total_source_timesteps']}, streams: {config['n_stream']}")
    print(f"Optimistic bonus coefficient: {config['optimistic_bonus_coef']}")

    alg.learn(
        total_timesteps=config['total_source_timesteps'],
        log_interval=None,
        reset_num_timesteps=True,
        progress_bar=True,
    )
    train_env.close()

    eval_env = DummyVecEnv([lambda: Monitor(make_env(config, key=key, goal=door)())])
    if args.n_stack > 1:
        eval_env = VecFrameStack(eval_env, n_stack=args.n_stack)

    episode_rewards, episode_lengths = evaluate_policy(
        alg,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    eval_env.close()

    episode_rewards = np.array(episode_rewards, dtype=np.float32)
    episode_lengths = np.array(episode_lengths, dtype=np.int32)
    success_rate = float(np.mean(episode_rewards >= config['max_reward']))

    model_path = path.join(run_dir, 'model.zip')
    result_path = path.join(run_dir, 'eval_result.npz')
    alg.save(model_path)
    np.savez(
        result_path,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        key=key,
        door=door,
        success_rate=success_rate,
        visit_count_states=len(visit_counts),
    )

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    print(f'Mean true return: {episode_rewards.mean():.4f}')
    print(f'Std true return: {episode_rewards.std():.4f}')
    print(f'Success rate: {success_rate:.4f}')
    print(f'Visited count-table states: {len(visit_counts)}')
    print(f'Model saved to {model_path}')
    print(f'Results saved to {result_path}')
