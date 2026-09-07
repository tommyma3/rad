"""Independent PPO/RecurrentPPO training and online collection per fixed task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym

from .artifacts import TaskHistoryWriter, transition_record
from .check_recurrent_ppo_convergence import has_converged
from .envs import FlattenMemoryObservation, MemoryTaskSpec, make_memory_env
from .recurrent_ppo import RecurrentPPOConfig, build_recurrent_ppo, evaluate_recurrent_ppo
from .task_pool import load_pool
from .ppo import PPOConfig, build_ppo, evaluate_ppo


class OnlineHistory(gym.Wrapper):
    """One chronological training stream, recorded before SB3 auto-reset/flattening."""
    def __init__(self, env, writer):
        super().__init__(env)
        self.writer = writer
        self.steps = []
        self.total_steps = 0
        self.observation = self.info = None

    def reset(self, **kwargs):
        if self.steps:
            raise RuntimeError("Cannot reset an unfinished training episode")
        self.observation, self.info = self.env.reset(**kwargs)
        return self.observation, self.info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        record = transition_record(self.observation, action, reward, terminated, truncated,
                                   observation, info, self.info)
        record["learner_step"] = self.total_steps
        self.steps.append(record)
        if terminated or truncated:
            self.writer.write_episode(self.steps, learner_step=self.total_steps)
            self.steps = []
        self.observation, self.info = observation, info
        return observation, reward, terminated, truncated, info


def train_task(spec, manifest_fingerprint, source_seed, run_dir, output_root, *,
               total_timesteps=1_000_000, evaluation_interval=50_000,
               evaluation_episodes=100, minimum_success_rate=0.9,
               required_consecutive_evals=3, ppo_config=None,
               source_algorithm="recurrent_ppo", device=None, verbose=1):
    from stable_baselines3.common.callbacks import BaseCallback

    if spec.split != "train" or spec.configuration is None:
        raise ValueError("Source training requires a fixed training task")
    if min(total_timesteps, evaluation_interval, evaluation_episodes, required_consecutive_evals) <= 0:
        raise ValueError("Training and evaluation counts must be positive")
    if not 0 <= minimum_success_rate <= 1:
        raise ValueError("minimum_success_rate must be in [0, 1]")
    if source_algorithm not in {"ppo", "recurrent_ppo"}:
        raise ValueError("source_algorithm must be ppo or recurrent_ppo")
    config_type, builder, evaluator = (
        (PPOConfig, build_ppo, evaluate_ppo) if source_algorithm == "ppo"
        else (RecurrentPPOConfig, build_recurrent_ppo, evaluate_recurrent_ppo))
    ppo_config = ppo_config or config_type()
    if ppo_config.policy != config_type().policy:
        raise ValueError("Source policy and algorithm do not match")
    device = device or ("cpu" if source_algorithm == "ppo" else "auto")
    # Preserve the established recurrent run paths while allowing both algorithms
    # to train on the same task/seed in one output root.
    run_id = f"{spec.task_id}-source-{source_seed}" + ("-ppo" if source_algorithm == "ppo" else "")
    run_dir = Path(run_dir) / run_id
    artifact = Path(output_root) / "train" / source_algorithm / f"{run_id}.hdf5"
    if run_dir.exists() or artifact.exists():
        raise ValueError(f"Refusing to append a fresh learner to existing run {run_id}")
    run_dir.mkdir(parents=True)
    provenance = {
        "manifest_fingerprint": manifest_fingerprint, "run_id": run_id,
        "source_seed": source_seed, "stream_id": 0, "history_kind": "online_training",
        "ppo": ppo_config.to_dict(), "total_timesteps": total_timesteps,
        "source_algorithm": source_algorithm, "device": device,
    }
    (run_dir / "task_spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "source_config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    evaluations = []

    def evaluate(model):
        metrics = evaluator(model, spec, episodes=evaluation_episodes)
        record = {"timesteps": int(model.num_timesteps), **metrics}
        if evaluations and evaluations[-1]["timesteps"] == record["timesteps"]:
            evaluations[-1] = record
        else:
            evaluations.append(record)
        (run_dir / "evaluations.json").write_text(json.dumps(evaluations, indent=2), encoding="utf-8")

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_evaluation = evaluation_interval

        def _on_step(self):
            if self.num_timesteps >= self.next_evaluation:
                evaluate(self.model)
                self.model.save(run_dir / f"teacher-checkpoint-{self.num_timesteps}")
                self.next_evaluation += evaluation_interval
            return True

    with TaskHistoryWriter(artifact, spec, source_algorithm, provenance) as writer:
        writer.handle.attrs["collection_complete"] = False
        recorder = OnlineHistory(make_memory_env(spec), writer)
        env = FlattenMemoryObservation(recorder)
        try:
            model = builder(env, seed=source_seed, config=ppo_config, device=device, verbose=verbose,
                            tensorboard_log=run_dir / "tensorboard")
            evaluate(model)
            model.learn(total_timesteps=total_timesteps, callback=Progress())
            model.save(run_dir / "teacher-final")
            evaluate(model)
            converged = has_converged(evaluations[1:], minimum_success_rate=minimum_success_rate,
                                      required_consecutive_evals=required_consecutive_evals)
            writer.handle.attrs["source_converged"] = converged
            writer.handle.attrs["collection_complete"] = True
            writer.handle.attrs["discarded_tail_steps"] = len(recorder.steps)
        finally:
            env.close()
    result = {"task_id": spec.task_id, "source_seed": source_seed, "converged": converged,
              "source_algorithm": source_algorithm,
              "artifact": str(artifact), "minimum_success_rate": minimum_success_rate,
              "required_consecutive_evals": required_consecutive_evals}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _train_worker(job):
    import torch
    spec, manifest_hash, seed, run_dir, output_root, threads, kwargs = job
    torch.set_num_threads(threads)
    return train_task(spec, manifest_hash, seed, run_dir, output_root, **kwargs)


def train_pool(manifest, source_seeds, run_dir, output_root, *, workers=1, torch_threads=1, **kwargs):
    if workers < 1 or torch_threads < 1:
        raise ValueError("workers and torch_threads must be positive")
    pool = load_pool(manifest)
    results = []
    jobs = []
    for value in pool["tasks"]:
        spec = MemoryTaskSpec.from_dict(value)
        if spec.split != "train":
            continue
        for seed in dict.fromkeys(source_seeds):
            jobs.append((spec, pool["fingerprint"], seed, run_dir, output_root, torch_threads, kwargs))
    def record(result):
        results.append(result)
        results.sort(key=lambda item: (item["task_id"], item["source_seed"]))
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        summary = Path(run_dir, "summary.json")
        temporary = summary.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
        temporary.replace(summary)
    if workers == 1:
        for job in jobs:
            record(_train_worker(job))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            for future in as_completed([executor.submit(_train_worker, job) for job in jobs]):
                record(future.result())
    if not results or not all(result["converged"] for result in results):
        raise RuntimeError("Source convergence failed; inspect summary.json before distillation")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-algorithm", choices=("ppo", "recurrent_ppo"), default="ppo")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--source-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-root", default="datasets-fixed")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--evaluation-interval", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--minimum-success-rate", type=float, default=0.9)
    parser.add_argument("--required-consecutive-evals", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    config_type = PPOConfig if args.source_algorithm == "ppo" else RecurrentPPOConfig
    train_pool(args.manifest, args.source_seeds, args.run_dir, args.output_root,
               source_algorithm=args.source_algorithm, workers=args.workers,
               torch_threads=args.torch_threads, device=args.device,
               total_timesteps=args.total_timesteps, evaluation_interval=args.evaluation_interval,
               evaluation_episodes=args.evaluation_episodes, minimum_success_rate=args.minimum_success_rate,
               required_consecutive_evals=args.required_consecutive_evals,
               ppo_config=config_type(n_steps=args.n_steps, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
