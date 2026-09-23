"""Queue independent training seeds/methods, one process per GPU, then evaluate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bandit.evidence_experiment import DEFAULT_CONFIG, METHODS, PROTOCOL, geometry, prepare_data, read_study
from bandit.utils import file_digest, project_path, write_json

REPO = Path(__file__).resolve().parents[2]


def resolve_devices(requested, environ=None):
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    allowed = [value.strip() for value in visible.split(",")] if visible is not None else None
    resolved = []
    for value in requested:
        if allowed is not None:
            if value.isdigit():
                if int(value) >= len(allowed):
                    raise ValueError(f"GPU index {value} is outside inherited CUDA_VISIBLE_DEVICES={visible}")
                value = allowed[int(value)]
            elif value not in allowed:
                raise ValueError("Requested GPU UUID is not in inherited CUDA_VISIBLE_DEVICES")
        if not value or value == "-1" or not (value.isdigit() or value.startswith(("GPU-", "MIG-"))):
            raise ValueError(f"Invalid CUDA device: {value!r}")
        resolved.append(value)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("Provide a nonempty list of distinct GPUs")
    return resolved


def worker_environment(device, threads=4, environ=None):
    source = os.environ if environ is None else environ
    distributed = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                   "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"}
    env = {key: value for key, value in source.items() if key not in distributed
           and not key.startswith(("ACCELERATE_", "TORCHELASTIC_"))}
    env.update(CUDA_VISIBLE_DEVICES="" if device == "cpu" else device,
               OMP_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    return env


def schedule_jobs(jobs, devices, logs, *, threads=4, resume=False):
    """Bounded queue; stop and reap all owned children on failure/interruption."""
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    pending, running, results = list(jobs), {}, []
    history_path = logs / "jobs.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    try:
        while pending or running:
            for device in devices:
                if device in running or not pending:
                    continue
                job = pending.pop(0)
                log_path = logs / f"{job['name']}.log"
                handle = log_path.open("a" if resume else "w", encoding="utf-8")
                try:
                    process = subprocess.Popen(job["command"], cwd=REPO,
                        env=worker_environment(device, threads), stdout=handle, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                except BaseException:
                    handle.close()
                    raise
                running[device] = (process, handle, job, log_path)
                print(f"Started {job['name']} on {device}; log: {log_path}", flush=True)
            for device, (process, handle, job, log_path) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del running[device]
                result = {"name": job["name"], "device": device, "exit_code": code, "log": str(log_path)}
                results.append(result)
                write_json(history_path, history + results)
                if code:
                    raise RuntimeError(f"{job['name']} failed with exit code {code}; see {log_path}")
                print(f"Finished {job['name']} on {device}", flush=True)
            if running:
                time.sleep(0.2)
    finally:
        for process, _, _, _ in running.values():
            if process.poll() is None:
                process.terminate()
        for process, handle, _, _ in running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                handle.close()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--root", default="runs/old_recent_evidence_v1")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="Indices within inherited CUDA visibility, or GPU UUIDs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-tasks", type=int)
    parser.add_argument("--validation-tasks", type=int)
    parser.add_argument("--test-tasks", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"))
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Serial CPU smoke test instead of GPU execution")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan without creating files or launching jobs")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args(argv)
    if (len(set(args.seeds)) != len(args.seeds) or not args.seeds or min(args.seeds) < 0 or
            len(set(args.methods)) != len(args.methods) or args.threads < 1 or args.eval_batch_size < 1):
        parser.error("Seeds/methods must be unique; seeds nonnegative; thread and batch counts positive")
    study = read_study(args.config)
    for key in ("train_tasks", "validation_tasks", "test_tasks"):
        if getattr(args, key) is not None:
            study[key] = getattr(args, key)
    for argument, key in ((args.steps, "train_steps"), (args.batch_size, "batch_size"),
                          (args.eval_interval, "eval_interval"), (args.checkpoint_interval, "checkpoint_interval"),
                          (args.mixed_precision, "mixed_precision")):
        if argument is not None:
            study["training"][key] = argument
    from bandit.evidence_experiment import validate_study
    validate_study(study)
    devices = ["cpu"] if args.cpu else resolve_devices(args.gpus)
    root = project_path(args.root).resolve()
    frozen = root / "study.json"
    jobs = []
    for seed in args.seeds:
        for method in args.methods:
            name = f"{method}_s{seed}"
            run = root / name
            command = [sys.executable, str(REPO / "bandit/train_evidence.py"), "--config", str(frozen),
                "--dataset", str(root / "data"), "--run-dir", str(run), "--method", method,
                "--seed", str(seed), "--device", "cpu" if args.cpu else "cuda", "--threads", str(args.threads)]
            if args.resume and ((run / "last.pt").exists() or (run / "config.json").exists()):
                command.append("--resume")
            jobs.append({"name": name, "command": command})
    plan = {"protocol": PROTOCOL, "study": study, "geometry": geometry(study),
            "methods": args.methods, "seeds": args.seeds}
    if args.dry_run:
        print(json.dumps({**plan, "devices": devices, "jobs": jobs}, indent=2))
        return
    if not args.cpu:
        probe = "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one usable CUDA GPU'; print(torch.cuda.get_device_name(0))"
        for device in devices:
            subprocess.run([sys.executable, "-c", probe], env=worker_environment(device, args.threads), check=True)
    if root.exists() and any(root.iterdir()):
        if not args.resume or not (root / "plan.json").exists():
            raise FileExistsError(f"Use a new experiment root or --resume: {root}")
        if json.loads((root / "plan.json").read_text(encoding="utf-8")) != plan:
            raise ValueError("Resume requires the original study, methods, and training seeds")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "plan.json", plan)
    write_json(frozen, study)
    prepare_data(root / "data", study)
    schedule_jobs(jobs, devices, root / "logs", threads=args.threads, resume=args.resume)
    if args.no_evaluate:
        return
    checkpoints = [root / job["name"] / "best-model.pt" for job in jobs]
    output = root / "evaluation"
    if args.resume and (root / "evaluation_result.json").exists():
        previous = json.loads((root / "evaluation_result.json").read_text(encoding="utf-8"))
        previous_output = Path(previous["output"])
        if (previous_output / "evaluation.json").exists():
            metadata = json.loads((previous_output / "evaluation.json").read_text(encoding="utf-8"))
            if (metadata.get("complete") and metadata["study"] == study and
                    [r["sha256"] for r in metadata["checkpoints"]] == [file_digest(p) for p in checkpoints]):
                print(f"Evaluation already complete: {previous_output}")
                return
    attempt = 0
    while output.exists() and any(output.iterdir()):
        attempt += 1
        output = root / f"evaluation-retry-{attempt:03d}"
    command = [sys.executable, str(REPO / "bandit/evaluate_evidence.py"), "--checkpoint",
        *map(str, checkpoints), "--dataset", str(root / "data"), "--output", str(output),
        "--device", "cpu" if args.cpu else "cuda", "--batch-size", str(args.eval_batch_size),
        "--threads", str(args.threads)]
    schedule_jobs([{"name": "evaluation", "command": command}], devices[:1], root / "logs",
                  threads=args.threads, resume=args.resume)
    write_json(root / "evaluation_result.json", {"output": str(output)})
    print(f"Experiment complete: {output}", flush=True)


if __name__ == "__main__":
    main()
