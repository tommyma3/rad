"""Check finite-budget UCB convergence on fixed tasks over arm pulls.

An episode contains pre_steps + post_steps genuine pulls and an intervening gap.
The same task and learner persist across episodes; reward draws are refreshed.
Run directly or with python -m bandit.test_ucb_convergence from the repo root.
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from pathlib import Path

import numpy as np

from bandit.algorithm import UCB
from bandit.env import BANDIT, DelayedBandit, sample_task
from bandit.utils import get_config, project_path, write_json


def independent_seed(learner_seed, task_seed, episode, stream):
    return int(np.random.SeedSequence([learner_seed, task_seed, episode, stream]).generate_state(
        1, dtype=np.uint64)[0])


def same_state(first, second):
    return (np.array_equal(first["counts"], second["counts"]) and
            np.array_equal(first["reward_sums"], second["reward_sums"]) and
            first["total_pulls"] == second["total_pulls"] and
            first["rng_state"] == second["rng_state"])


def run_episode(task, learner, *, pre_steps, post_steps, delay, reward_seed, distractor_seed,
                pull_limit=None):
    """Advance an existing learner, optionally stopping after ``pull_limit`` pulls."""
    env = DelayedBandit(task, reward_seed, pre_steps, post_steps, delay)
    distractor = np.random.default_rng(distractor_seed)
    start_pulls = learner.total_pulls
    states, actions, rewards = [], [], []
    frozen = None
    for step in range(env.sequence_length):
        if pull_limit is not None and learner.total_pulls - start_pulls >= pull_limit:
            break
        if step == pre_steps:
            frozen = learner.state_dict()
        if step == pre_steps + delay and not same_state(frozen, learner.state_dict()):
            raise AssertionError("Distractor changed UCB state")
        observation = env.observation
        action = learner.select_action() if observation == BANDIT else int(distractor.integers(env.num_arms))
        _, reward, _, _, _ = env.step(action)
        if observation == BANDIT:
            learner.update(action, reward)
        states.append(observation)
        actions.append(action)
        rewards.append(reward)
    expected_pulls = pre_steps + post_steps if pull_limit is None else min(pull_limit, pre_steps + post_steps)
    if learner.total_pulls - start_pulls != expected_pulls:
        raise AssertionError("Only genuine bandit pulls should update UCB")
    return {"states": np.asarray(states, dtype=np.int64),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float64)}


def evaluate_recommendation(task, learner, draws, seed):
    """Evaluate the empirical greedy recommendation; never update the learner.

    This is distinct from UCB's exploratory online behavior, which is measured
    separately. Hidden task means are used only by the evaluator for exact regret.
    """
    before = learner.state_dict()
    empirical = np.divide(learner.reward_sums, learner.counts,
                          out=np.full(len(learner.counts), -np.inf), where=learner.counts > 0)
    arms = np.flatnonzero(empirical == empirical.max())
    rng = np.random.default_rng(seed)
    selected = rng.choice(arms, size=draws)
    means = np.asarray(task.means)
    rewards = rng.normal(means[selected], task.reward_std)
    expected_reward = float(means[arms].mean())
    if not same_state(before, learner.state_dict()):
        raise AssertionError("Held-out evaluation changed the learner")
    return {"recommended_arms_zero_based": arms.tolist(),
            "recommended_expected_reward": expected_reward,
            "recommendation_regret": max(0.0, float(means.max()) - expected_reward),
            "heldout_mean_reward": float(rewards.mean()),
            "heldout_reward_standard_error": float(rewards.std(ddof=1) / np.sqrt(draws)) if draws > 1 else 0.0}


def has_converged(evaluations, consecutive):
    return len(evaluations) >= consecutive and all(row["passed"] for row in evaluations[-consecutive:])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulls", type=int, default=None,
                        help="Total genuine arm pulls per fixed task and learner seed")
    # Keep the old spelling as a compatibility shim for existing commands.  All
    # internal budgets and output fields use pulls.
    parser.add_argument("--episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-seeds", "--task_seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Independent UCB/reward seeds")
    parser.add_argument("--distribution", choices=("uniform",), default="uniform")
    parser.add_argument("--env", default="delayed_adversarial_bandit")
    parser.add_argument("--delay", type=int, help="Gap per episode; default uses the first configured training delay")
    parser.add_argument("--pre-steps", type=int)
    parser.add_argument("--post-steps", type=int)
    parser.add_argument("--exploration-coefficient", type=float)
    parser.add_argument("--eval-interval", "--eval-interval-pulls", type=int, default=None,
                        help="Evaluate every N arm pulls, and at the end")
    parser.add_argument("--window-pulls", type=int, default=None,
                        help="Recent arm pulls used for online metrics")
    parser.add_argument("--window-episodes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--evaluation-pulls", type=int, default=1000, help="Fresh reward draws per recommendation check")
    parser.add_argument("--max-mean-regret", type=float, default=0.05,
                        help="Maximum recent mean of mu_best - mu_selected for actual UCB pulls")
    parser.add_argument("--max-recommendation-regret", type=float, default=0.05,
                        help="Maximum exact regret of the empirical greedy recommendation")
    parser.add_argument("--min-optimal-action-rate", type=float, default=None,
                        help="Optional additional threshold on actual UCB's recent optimal-arm frequency")
    parser.add_argument("--required-consecutive-evals", type=int, default=3)
    parser.add_argument("--required-seed-fraction", type=float, default=1.0,
                        help="Required passing seed fraction separately for every task")
    parser.add_argument("--output-dir", "--output_dir", default="results/ucb_convergence")
    args = parser.parse_args(argv)
    for name in ("pulls", "episodes", "eval_interval", "window_pulls", "window_episodes",
                 "evaluation_pulls", "required_consecutive_evals"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name} must be positive")
    if args.pulls is not None and args.episodes is not None:
        parser.error("Specify --pulls or the deprecated --episodes, not both")
    if args.pulls is None and args.episodes is None:
        args.pulls = 100
    if any(seed < 0 for seed in args.task_seeds + args.seeds):
        parser.error("Seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.task_seeds)) != len(args.task_seeds):
        parser.error("Duplicate seeds would repeat identical runs")
    for name in ("max_mean_regret", "max_recommendation_regret", "min_optimal_action_rate"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"{name} must be in [0, 1]")
    if not 0 < args.required_seed_fraction <= 1:
        parser.error("required_seed_fraction must be in (0, 1]")
    if args.exploration_coefficient is not None and (
            not np.isfinite(args.exploration_coefficient) or args.exploration_coefficient < 0):
        parser.error("exploration_coefficient must be finite and nonnegative")
    return args


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(output, task, runs, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for run in runs:
        episodes = run["episodes"]
        x = [row["arm_pulls"] for row in episodes]
        label = f"seed {run['learner_seed']}"
        axes[0, 0].plot(x, [r["rolling_expected_reward"] for r in episodes], label=label)
        axes[0, 1].plot(x, [r["rolling_mean_regret"] for r in episodes], label=label)
        axes[1, 0].plot(x, [r["rolling_optimal_action_rate"] for r in episodes], label=label)
        checks = run["evaluations"]
        axes[1, 1].plot([r["arm_pulls"] for r in checks],
                        [r["recommendation_regret"] for r in checks], marker="o", label=label)
    axes[0, 0].axhline(max(task.means), color="black", linestyle="--", label="Optimal arm mean")
    axes[0, 0].axhline(np.mean(task.means), color="gray", linestyle=":", label="Random policy mean")
    axes[0, 1].axhline(args.max_mean_regret, color="black", linestyle="--", label="Pass threshold")
    axes[1, 1].axhline(args.max_recommendation_regret, color="black", linestyle="--")
    if args.min_optimal_action_rate is not None:
        axes[1, 0].axhline(args.min_optimal_action_rate, color="black", linestyle="--")
    for axis, ylabel in zip(axes.flat, ("Recent UCB expected reward", "Recent UCB mean regret",
                                       "Recent UCB optimal-action rate", "Greedy recommendation regret")):
        axis.set(xlabel="Arm pulls", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Fixed task seed {task.seed}, {task.distribution}; {args.pulls} arm pulls")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def run_check(args):
    config = get_config(f"config/env/{args.env}.yaml")
    config.update(get_config("config/algorithm/ucb.yaml"))
    config["distribution"] = args.distribution
    for key in ("pre_steps", "post_steps", "exploration_coefficient"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    delay = args.delay if args.delay is not None else config["train_delays"][0]
    if min(config["pre_steps"], config["post_steps"]) < 1 or delay < 0:
        raise ValueError("Both bandit phases must be positive and delay nonnegative")
    episode_pulls = config["pre_steps"] + config["post_steps"]
    if args.pulls is None:
        args.pulls = (args.episodes * episode_pulls) if args.episodes is not None else episode_pulls
    if args.pulls < 1:
        raise ValueError("pulls must be positive")
    if args.eval_interval is None:
        args.eval_interval = 10 * episode_pulls
    if args.window_pulls is None:
        args.window_pulls = (args.window_episodes * episode_pulls
                             if args.window_episodes is not None else 10 * episode_pulls)
    # The compatibility options above are resolved to pull units before any
    # metrics or artifacts are produced.
    if args.eval_interval < 1 or args.window_pulls < 1:
        raise ValueError("eval_interval and window_pulls must be positive")
    output = project_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"arguments": vars(args), "environment": config, "delay": delay,
               "episode_semantics": "same task and UCB state; fresh reward stream per episode",
               "budget_semantics": "pulls counts only genuine arm selections; delays and distractor steps are excluded",
               "schema": "bandit-ucb-convergence-gaussian-v3"})
    tasks = [sample_task(seed, args.distribution, config["num_arms"], reward_std=config["reward_std"])
             for seed in args.task_seeds]
    write_json(output / "tasks.json", [task.to_dict() for task in tasks])
    task_results = []
    for task in tasks:
        task_dir = output / f"task-{task.seed}"
        task_dir.mkdir()
        means = np.asarray(task.means)
        optimal = float(means.max())
        runs = []
        for seed in args.seeds:
            learner = UCB(config["num_arms"], config["exploration_coefficient"], seed)
            episodes, evaluations, histories, pull_records = [], [], [], []
            seed_dir = task_dir / f"seed-{seed}"
            seed_dir.mkdir()
            episode = 0
            next_eval = args.eval_interval
            while learner.total_pulls < args.pulls:
                episode += 1
                remaining = args.pulls - learner.total_pulls
                history = run_episode(task, learner, pre_steps=config["pre_steps"],
                                      post_steps=config["post_steps"], delay=delay,
                                      reward_seed=independent_seed(seed, task.seed, episode, 1),
                                      distractor_seed=independent_seed(seed, task.seed, episode, 2),
                                      pull_limit=remaining)
                histories.append(history)
                genuine = history["states"] == BANDIT
                rewards = history["rewards"][genuine]
                selected_means = means[history["actions"][genuine]]
                pull_records.extend({"reward": float(reward), "expected_reward": float(expected),
                                     "regret": float(optimal - expected),
                                     "optimal": bool(expected == optimal)}
                                    for reward, expected in zip(rewards, selected_means))
                row = {"episode": episode, "arm_pulls": learner.total_pulls,
                       "genuine_pulls": learner.total_pulls,
                       "sequence_steps": sum(len(h["states"]) for h in histories),
                       "return": float(rewards.sum()), "mean_reward": float(rewards.mean()),
                       "expected_reward": float(selected_means.mean()),
                       "mean_regret": float((optimal - selected_means).mean()),
                       "optimal_action_rate": float((selected_means == optimal).mean()),
                       "pre_return": float(rewards[:config["pre_steps"]].sum()),
                       "post_return": float(rewards[config["pre_steps"]:].sum())}
                window = pull_records[-args.window_pulls:]
                row["window_pulls"] = len(window)
                row["rolling_mean_reward"] = float(np.mean([r["reward"] for r in window]))
                row["rolling_expected_reward"] = float(np.mean([r["expected_reward"] for r in window]))
                row["rolling_mean_regret"] = float(np.mean([r["regret"] for r in window]))
                row["rolling_optimal_action_rate"] = float(np.mean([r["optimal"] for r in window]))
                episodes.append(row)
                if learner.total_pulls >= next_eval or learner.total_pulls == args.pulls:
                    check = {"episode": episode, "arm_pulls": learner.total_pulls,
                             "genuine_pulls": learner.total_pulls,
                             "window_pulls": len(window), "online_mean_regret": row["rolling_mean_regret"],
                             "online_optimal_action_rate": row["rolling_optimal_action_rate"],
                             **evaluate_recommendation(task, learner, args.evaluation_pulls,
                                                        independent_seed(seed, task.seed, episode, 3))}
                    check["passed"] = (len(window) == min(args.window_pulls, learner.total_pulls) and
                                       check["online_mean_regret"] <= args.max_mean_regret and
                                       check["recommendation_regret"] <= args.max_recommendation_regret and
                                       (args.min_optimal_action_rate is None or
                                        check["online_optimal_action_rate"] >= args.min_optimal_action_rate))
                    evaluations.append(check)
                    print(f"task={task.seed} seed={seed} arm_pulls={learner.total_pulls}/{args.pulls} "
                          f"regret={check['online_mean_regret']:.4f} "
                          f"greedy_regret={check['recommendation_regret']:.4f} "
                          f"check={'PASS' if check['passed'] else 'FAIL'}", flush=True)
                    while next_eval <= learner.total_pulls:
                        next_eval += args.eval_interval
            passed = has_converged(evaluations, args.required_consecutive_evals)
            state = learner.state_dict()
            state["counts"] = state["counts"].tolist()
            state["reward_sums"] = state["reward_sums"].tolist()
            write_json(seed_dir / "ucb_state.json", state)
            write_json(seed_dir / "evaluations.json", evaluations)
            write_csv(seed_dir / "episodes.csv", episodes)
            write_csv(seed_dir / "pulls.csv", episodes)
            history_lengths = [len(h["states"]) for h in histories]
            if len(set(history_lengths)) == 1:
                history_arrays = {key: np.stack([h[key] for h in histories])
                                  for key in ("states", "actions", "rewards")}
            else:
                # A pull budget need not be divisible by one episode. Preserve
                # every observation without padding the final partial episode.
                history_arrays = {key: np.concatenate([h[key] for h in histories])
                                  for key in ("states", "actions", "rewards")}
                history_arrays["episode_offsets"] = np.cumsum([0] + history_lengths)
            np.savez_compressed(seed_dir / "history.npz", **history_arrays)
            result = {"learner_seed": seed, "passed": passed, "arm_pulls": learner.total_pulls,
                      "genuine_pulls": learner.total_pulls,
                      "final_evaluation": evaluations[-1],
                      "final_consecutive_checks": [r["passed"] for r in evaluations[-args.required_consecutive_evals:]]}
            write_json(seed_dir / "result.json", result)
            runs.append({**result, "episodes": episodes, "evaluations": evaluations})
        fraction = sum(r["passed"] for r in runs) / len(runs)
        result = {"task": task.to_dict(), "optimal_mean": optimal,
                  "best_second_best_gap": float(np.sort(means)[-1] - np.sort(means)[-2]),
                  "passing_seed_fraction": fraction, "passed": fraction >= args.required_seed_fraction,
                  "runs": [{k: v for k, v in r.items() if k not in ("episodes", "evaluations")} for r in runs]}
        task_results.append(result)
        write_json(task_dir / "summary.json", result)
        plot_curves(task_dir, task, runs, args)
    summary = {"passed": all(r["passed"] for r in task_results), "arm_pulls_per_run": args.pulls,
               "genuine_pulls_per_episode": episode_pulls,
               "pull_budget": args.pulls,
               "delay": delay, "criteria": {key: getattr(args, key) for key in (
                   "max_mean_regret", "max_recommendation_regret", "min_optimal_action_rate",
                   "window_pulls", "eval_interval", "required_consecutive_evals", "required_seed_fraction")},
               "tasks": task_results,
               "interpretation": "Finite-budget near-optimality check, not a proof of asymptotic convergence. "
                                 "UCB persists across episode boundaries; results do not imply convergence within one collected history."}
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run_check(args)
    print(f"{'PASS' if summary['passed'] else 'FAIL'}: {project_path(args.output_dir) / 'summary.json'}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
