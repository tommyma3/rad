"""Independent RecurrentPPO training and online history collection per fixed task."""
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
               required_consecutive_evals=3, ppo_config=None):
    from stable_baselines3.common.callbacks import BaseCallback

    if spec.split != "train" or spec.configuration is None:
        raise ValueError("Source training requires a fixed training task")
    if min(total_timesteps, evaluation_interval, evaluation_episodes, required_consecutive_evals) <= 0:
        raise ValueError("Training and evaluation counts must be positive")
    if not 0 <= minimum_success_rate <= 1:
        raise ValueError("minimum_success_rate must be in [0, 1]")
    ppo_config = ppo_config or RecurrentPPOConfig()
    run_id = f"{spec.task_id}-source-{source_seed}"
    run_dir = Path(run_dir) / run_id
    artifact = Path(output_root) / "train" / "recurrent_ppo" / f"{run_id}.hdf5"
    if run_dir.exists() or artifact.exists():
        raise ValueError(f"Refusing to append a fresh learner to existing run {run_id}")
    run_dir.mkdir(parents=True)
    provenance = {
        "manifest_fingerprint": manifest_fingerprint, "run_id": run_id,
        "source_seed": source_seed, "stream_id": 0, "history_kind": "online_training",
        "ppo": ppo_config.to_dict(), "total_timesteps": total_timesteps,
    }
    (run_dir / "task_spec.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "source_config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    evaluations = []

    def evaluate(model):
        metrics = evaluate_recurrent_ppo(model, spec, episodes=evaluation_episodes)
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

    with TaskHistoryWriter(artifact, spec, "recurrent_ppo", provenance) as writer:
        writer.handle.attrs["collection_complete"] = False
        recorder = OnlineHistory(make_memory_env(spec), writer)
        env = FlattenMemoryObservation(recorder)
        try:
            model = build_recurrent_ppo(env, seed=source_seed, config=ppo_config,
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
              "artifact": str(artifact), "minimum_success_rate": minimum_success_rate,
              "required_consecutive_evals": required_consecutive_evals}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_pool(manifest, source_seeds, run_dir, output_root, **kwargs):
    pool = load_pool(manifest)
    results = []
    for value in pool["tasks"]:
        spec = MemoryTaskSpec.from_dict(value)
        if spec.split != "train":
            continue
        for seed in dict.fromkeys(source_seeds):
            results.append(train_task(spec, pool["fingerprint"], seed, run_dir, output_root, **kwargs))
            Path(run_dir, "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not results or not all(result["converged"] for result in results):
        raise RuntimeError("Source convergence failed; inspect summary.json before distillation")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
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
    train_pool(args.manifest, args.source_seeds, args.run_dir, args.output_root,
               total_timesteps=args.total_timesteps, evaluation_interval=args.evaluation_interval,
               evaluation_episodes=args.evaluation_episodes, minimum_success_rate=args.minimum_success_rate,
               required_consecutive_evals=args.required_consecutive_evals,
               ppo_config=RecurrentPPOConfig(n_steps=args.n_steps, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
