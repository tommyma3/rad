"""Train AD-short, AD-long and RAD on a GPU queue with periodic online regret."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bandit.collect import collect_dataset
from bandit.convergence_experiment import (METHODS, PROTOCOL, plot_results,
    resolve_pretrained, train_worker, verify_data, verify_held_out, verify_pretrained)
from bandit.dataset import BanditDataset, assert_disjoint
from bandit.evaluation import make_eval_manifest
from bandit.scripts.run_old_recent_evidence import resolve_devices, schedule_jobs, worker_environment
from bandit.utils import file_digest, load_config, project_path, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/training_convergence_v1")
    parser.add_argument("--dataset", help="Existing standard Gaussian collection; otherwise collect inside root")
    parser.add_argument("--pretrained", help="RAD compression-pretraining checkpoint (directory or model.pt); supports {seed}")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-tasks", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1729)
    parser.add_argument("--delays", nargs="+", type=int, default=[50])
    parser.add_argument("--train-tasks", type=int, default=10000)
    parser.add_argument("--validation-tasks", type=int, default=1000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--greedy", action="store_true", help="Argmax actions instead of sampled actions")
    parser.add_argument("--resume", action="store_true", help="Resume the frozen plan; only device/thread settings may change")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Serial CPU execution for smoke tests")
    parser.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = project_path(args.root).resolve()
    if args.worker:
        import torch
        torch.set_num_threads(args.threads)
        train_worker(root, args.worker, args.worker_seed, cpu=args.cpu, resume=args.resume)
        return
    if args.plot_only:
        print(plot_results(root))
        return
    if min(args.steps, args.eval_interval, args.eval_tasks, args.train_tasks,
           args.validation_tasks, args.batch_size, args.threads) < 1:
        parser.error("Step, interval, task, batch and thread counts must be positive")
    if (min(args.seeds + [args.eval_seed, args.data_seed]) < 0 or
            len(set(args.seeds)) != len(args.seeds) or min(args.delays) < 0 or
            len(set(args.delays)) != len(args.delays)):
        parser.error("Seeds must be nonnegative and unique; delays must be nonnegative and unique")
    devices = ["cpu"] if args.cpu else resolve_devices(args.gpus)
    if args.resume:
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        if plan.get("protocol") != PROTOCOL:
            raise ValueError("Incompatible convergence experiment")
        if args.pretrained is not None:
            requested = resolve_pretrained(args.pretrained, plan["seeds"], plan["configs"]["rad"])
            if requested != plan.get("rad_pretrained", {}):
                raise ValueError("Resume cannot change RAD's pretrained initialization; use a fresh root")
        verify_pretrained(plan)
    else:
        configs = {}
        for method in METHODS:
            config = load_config(method)
            config.update(num_arms=5, pre_steps=50, post_steps=50,
                train_steps=args.steps, batch_size=args.batch_size,
                eval_interval=args.eval_interval, checkpoint_interval=args.eval_interval,
                mixed_precision=args.mixed_precision)
            configs[method] = config
        plan = {"protocol": PROTOCOL, "configs": configs, "seeds": args.seeds,
            "delays": args.delays, "greedy": args.greedy, "eval_tasks": args.eval_tasks,
            "eval_seed": args.eval_seed, "data_seed": args.data_seed,
            "train_tasks": args.train_tasks, "validation_tasks": args.validation_tasks,
            "dataset": str(project_path(args.dataset).resolve() if args.dataset else root / "data"),
            "device_type": "cpu" if args.cpu else "cuda"}
        plan["rad_pretrained"] = resolve_pretrained(args.pretrained, args.seeds, configs["rad"])
    if plan["device_type"] != ("cpu" if args.cpu else "cuda"):
        raise ValueError("Resume must preserve CPU/CUDA device type")
    if args.cpu and plan["configs"]["ad_short"]["mixed_precision"] != "no":
        parser.error("CPU smoke tests require --mixed-precision no")
    jobs = []
    for seed in plan["seeds"]:
        for method in METHODS:
            command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                "--worker", method, "--worker-seed", str(seed), "--threads", str(args.threads)]
            if args.cpu:
                command.append("--cpu")
            if args.resume:
                command.append("--resume")
            jobs.append({"name": f"{method}_s{seed}", "command": command})
    if args.dry_run:
        print(json.dumps({"plan": plan, "devices": devices, "jobs": jobs}, indent=2))
        return
    if not args.resume and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a fresh root or --resume: {root}")
    if not args.cpu:
        probe = "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one usable CUDA GPU'; print(torch.cuda.get_device_name(0))"
        for device in devices:
            subprocess.run([sys.executable, "-c", probe], env=worker_environment(device, args.threads), check=True)
    if args.resume:
        verify_data(plan, root)
    else:
        dataset = Path(plan["dataset"])
        if args.dataset is None:
            collect_dataset(dataset, plan["configs"]["ad_short"], tasks=args.train_tasks,
                            validation_tasks=args.validation_tasks, seed=args.data_seed)
        train_data = BanditDataset(dataset / "train.hdf5", expected_split="train")
        validation = BanditDataset(dataset / "validation.hdf5", expected_split="validation")
        assert_disjoint(train_data, validation)
        collection = train_data.collection_config
        if collection != validation.collection_config:
            raise ValueError("Training/validation collection configurations differ")
        if (collection["num_arms"], collection["pre_steps"], collection["post_steps"]) != (5, 50, 50):
            raise ValueError("This experiment requires 5 arms and 50+50 genuine pulls")
        if collection.get("pre_steps_choices", [50]) != [50]:
            raise ValueError("This experiment requires a fixed 50-pull gap insertion point")
        horizon = 100 + max([*plan["delays"], *collection["train_delays"]])
        if plan["configs"]["ad_long"]["context_steps"] < horizon - 1:
            raise ValueError("AD-long context must cover the entire training/evaluation history; reduce delays")
        for config in plan["configs"].values():
            config.update({key: collection[key] for key in ("reward_std", "train_delays")})
        manifest = make_eval_manifest(args.eval_seed, args.eval_tasks, ["uniform"], plan["configs"]["ad_short"])
        verify_held_out(dataset, manifest)
        write_json(root / "evaluation_manifest.json", manifest)
        plan["data_digests"] = {split: file_digest(dataset / f"{split}.hdf5") for split in ("train", "validation")}
        plan["manifest_digest"] = file_digest(root / "evaluation_manifest.json")
        write_json(root / "plan.json", plan)
    schedule_jobs(jobs, devices, root / "logs", threads=args.threads, resume=args.resume)
    print(f"Experiment complete: {plot_results(root)}", flush=True)


if __name__ == "__main__":
    main()
