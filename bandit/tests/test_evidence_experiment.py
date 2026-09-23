"""Protocol, causal controls, exact resume, evaluation, and GPU queue tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from bandit.evidence_experiment import (CONDITIONS, METHODS, EvidenceDataset, geometry,
    load_evidence_checkpoint, make_split, model_config, prepare_data, read_study)
from bandit.evaluate_evidence import evaluate_runs
from bandit.model import make_model
from bandit.scripts.run_old_recent_evidence import resolve_devices, schedule_jobs, worker_environment
from bandit.train_evidence import train_run


def tiny_study():
    study = read_study()
    study.update(num_arms=4, pulls_per_arm=2, context_steps=6, short_memory_keep=1,
                 gap_compressions=2, train_tasks=8, validation_tasks=4, test_tasks=4)
    study["training"].update(train_steps=2, batch_size=2, warmup_steps=0, eval_interval=1,
        checkpoint_interval=1, log_interval=1, tf_n_embd=8, tf_n_head=2, tf_n_layer=1,
        tf_dim_feedforward=16, tf_dropout=0.1, n_compress_tokens=3,
        compress_n_layers=1, compress_n_heads=2)
    return study


class EvidenceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_balanced_independent_deterministic_data_and_empirical_labels(self):
        study = tiny_study()
        first = make_split(study, "train")
        repeated = make_split(study, "train")
        spec = geometry(study)
        for key in first:
            np.testing.assert_array_equal(first[key], repeated[key])
        self.assertEqual(first["best_is_early"].sum(), len(first["targets"]) // 2)
        for index, target in enumerate(first["targets"]):
            early = set(first["actions"][index, :spec["block_steps"]])
            recent = set(first["actions"][index, spec["recent_start"]:])
            self.assertFalse(early & recent)
            self.assertEqual(early | recent, set(range(study["num_arms"])))
            self.assertEqual(first["means"][index].argmax() in early, bool(first["best_is_early"][index]))
            genuine = first["states"][index] == 0
            counts = np.bincount(first["actions"][index, genuine], minlength=study["num_arms"])
            sums = np.bincount(first["actions"][index, genuine], weights=first["rewards"][index, genuine])
            np.testing.assert_array_equal(counts, np.full(study["num_arms"], study["pulls_per_arm"]))
            self.assertEqual(target, int(np.argmax(sums / counts + 1 / np.sqrt(counts))))
            self.assertTrue(np.all(first["rewards"][index, ~genuine] == 0))
        signatures = [set(map(tuple, make_split(study, split)["means"])) for split in ("train", "validation", "test")]
        self.assertFalse(signatures[0] & signatures[1] or signatures[0] & signatures[2] or signatures[1] & signatures[2])

    def test_default_geometry_and_invalid_geometry(self):
        self.assertEqual(geometry(read_study()), dict(block_steps=25, delay=72, recent_start=97,
                         prefix_steps=122, rad_recent_steps=30, compression_count=2))
        for change in ({"num_arms": 5}, {"pulls_per_arm": 20}, {"gap_compressions": 0}):
            with self.assertRaises(ValueError):
                geometry({**read_study(), **change})

    def test_interventions_preserve_timing_and_remove_all_transition_fields(self):
        study = tiny_study()
        with tempfile.TemporaryDirectory() as folder:
            prepare_data(folder, study)
            data = EvidenceDataset(Path(folder) / "test.npz", study)
            batches = {condition: data.batch([0, 1], condition) for condition in CONDITIONS}
            for condition, removed in (("early_only", slice(data.spec["recent_start"], None)),
                                        ("recent_only", slice(0, data.spec["block_steps"]))):
                batch = batches[condition]
                self.assertTrue(bool(torch.all(batch["states"][:, removed] == 1)))
                self.assertTrue(bool(torch.all(batch["rewards"][:, removed] == 0)))
                np.testing.assert_array_equal(batch["actions"][:, removed], data.arrays["filler_actions"][:2, removed])
                torch.testing.assert_close(batch["targets"], batches["both"]["targets"])
                self.assertEqual(batch["states"].shape, batches["both"]["states"].shape)
            self.assertEqual(set(batches["both"]), {"states", "actions", "rewards", "query_states", "targets", "loss_mask"})
            config = model_config(study, "rad", 0)
            model = make_model(config).eval()
            for batch in batches.values():
                state = model.prefix_state(batch)
                self.assertEqual(state["compression_count"], study["gap_compressions"])
                self.assertEqual(state["recent"].shape[1], 3 * data.spec["rad_recent_steps"])
                # Every recent evidence transition survives unchanged in the raw tail.
                tokens = model.embed_transitions(batch["states"], batch["actions"], batch["rewards"])
                torch.testing.assert_close(state["recent"], tokens[:, -state["recent"].shape[1]:])
            short = make_model(model_config(study, "ad_short", 0)).eval()
            first = short(data.batch([0, 1], "both", short.context_steps))["logits"]
            removed = short(data.batch([0, 1], "recent_only", short.context_steps))["logits"]
            torch.testing.assert_close(first, removed, rtol=0, atol=0)
            batch = batches["both"]
            batch["rewards"].requires_grad_()
            model(batch)["loss"].backward()
            self.assertGreater(float(batch["rewards"].grad[:, :data.spec["block_steps"]].abs().sum()), 0)

    def test_data_reuse_rejects_config_and_file_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            study = tiny_study()
            self.assertEqual(prepare_data(folder, study), prepare_data(folder, study))
            with self.assertRaises(ValueError):
                prepare_data(folder, {**study, "data_seed": 42})
            with (Path(folder) / "test.npz").open("ab") as handle:
                handle.write(b"modified")
            with self.assertRaises(ValueError):
                prepare_data(folder, study)

    def test_training_exact_resume_all_models_and_paired_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root, study = Path(folder), tiny_study()
            data = root / "data"
            prepare_data(data, study)
            checkpoints = []
            for method in METHODS:
                checkpoint = train_run(study, data, root / method, method, 0, device="cpu")
                checkpoints.append(checkpoint)
            train_run(study, data, root / "resumed", "rad", 0, device="cpu", stop_after=1)
            resumed = train_run(study, data, root / "resumed", "rad", 0, device="cpu", resume=True)
            full_model, _ = load_evidence_checkpoint(root / "rad" / "model.pt")
            resumed_model, _ = load_evidence_checkpoint(root / "resumed" / "model.pt")
            for name, tensor in full_model.state_dict().items():
                torch.testing.assert_close(tensor, resumed_model.state_dict()[name], rtol=0, atol=0)
            full_best, _ = load_evidence_checkpoint(checkpoints[0])
            resumed_best, _ = load_evidence_checkpoint(resumed)
            for name, tensor in full_best.state_dict().items():
                torch.testing.assert_close(tensor, resumed_best.state_dict()[name], rtol=0, atol=0)
            checkpoints.append(train_run(study, data, root / "rad_seed1", "rad", 1, device="cpu"))
            summary = evaluate_runs(checkpoints, data, root / "evaluation", batch_size=2)
            self.assertEqual(len(summary), 5 * 3 * 3)
            rad = [row for row in summary if row["method"] == "rad"]
            self.assertTrue(all(row["training_seeds"] == 2 and row["uncertainty_unit"] == "training_seed" for row in rad))
            effects = json.loads((root / "evaluation" / "paired_effects.json").read_text())
            short_effect = [row for row in effects if row["method"] == "ad_short" and row["condition"] == "recent_only_minus_both"]
            self.assertTrue(all(row["expected_regret"] == 0 for row in short_effect))
            self.assertTrue((root / "evaluation" / "evidence_integration.pdf").exists())
            with self.assertRaises(FileExistsError):
                train_run(study, data, root / "rad", "rad", 0, device="cpu")
            with self.assertRaises(ValueError):
                evaluate_runs([checkpoints[0], checkpoints[0]], data, root / "duplicate")
            with self.assertRaises(ValueError):
                train_run({**study, "training": {**study["training"], "batch_size": 4}}, data,
                          root / "rad", "rad", 0, device="cpu", resume=True)

    def test_recovery_before_first_checkpoint_and_best_snapshot_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            root, study = Path(folder), tiny_study()
            data = root / "data"
            train_run(study, data, root / "full", "rad", 0, device="cpu")
            train_run(study, data, root / "restart", "rad", 0, device="cpu", stop_after=1)
            # Simulate losing an uncommitted first checkpoint; config/logs exist.
            (root / "restart" / "last.pt").unlink()
            train_run(study, data, root / "restart", "rad", 0, device="cpu", resume=True)
            train_run(study, data, root / "rollback", "rad", 0, device="cpu", stop_after=1)
            # Simulate an invalid newer best file written after the last committed state.
            (root / "rollback" / "best-model.pt").write_bytes(b"uncommitted")
            with (root / "rollback" / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write('{"step":99,"validation_loss":0}\n{"partial":')
            train_run(study, data, root / "rollback", "rad", 0, device="cpu", resume=True)
            expected, _ = load_evidence_checkpoint(root / "full" / "best-model.pt")
            for directory in ("restart", "rollback"):
                actual, _ = load_evidence_checkpoint(root / directory / "best-model.pt")
                for name, value in expected.state_dict().items():
                    torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
                logs = [json.loads(line) for line in (root / directory / "metrics.jsonl").read_text().splitlines()]
                self.assertEqual([r["step"] for r in logs], [1, 2])


class IndependentGPUQueueTests(unittest.TestCase):
    def test_visibility_mapping_and_distributed_environment_removed(self):
        self.assertEqual(resolve_devices(["0", "1"], {"CUDA_VISIBLE_DEVICES": "3,7"}), ["3", "7"])
        self.assertEqual(resolve_devices(["GPU-abc"], {}), ["GPU-abc"])
        for devices, env in ((["0", "0"], {}), (["2"], {"CUDA_VISIBLE_DEVICES": "3,7"}),
                             (["0"], {"CUDA_VISIBLE_DEVICES": ""})):
            with self.assertRaises(ValueError):
                resolve_devices(devices, env)
        env = worker_environment("3", environ={"RANK": "4", "WORLD_SIZE": "8", "LOCAL_RANK": "4",
                    "ACCELERATE_USE_CPU": "true", "TORCHELASTIC_RUN_ID": "x", "PATH": "kept"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["PATH"], "kept")
        self.assertNotIn("WORLD_SIZE", env)
        self.assertNotIn("ACCELERATE_USE_CPU", env)

    def test_real_subprocess_queue_overlaps_slots_without_sharing_a_device(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = []
            for i in range(3):
                target = root / f"{i}.json"
                code = ("import json,os,time,sys; start=time.time(); time.sleep(0.8); "
                        "json.dump(dict(start=start,end=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],"
                        "world=os.environ.get('WORLD_SIZE')),open(sys.argv[1],'w'))")
                jobs.append({"name": str(i), "command": [sys.executable, "-c", code, str(target)]})
            results = schedule_jobs(jobs, ["2", "5"], root / "logs")
            rows = [json.loads((root / f"{i}.json").read_text()) for i in range(3)]
            self.assertEqual(len(results), 3)
            self.assertLess(max(rows[0]["start"], rows[1]["start"]), min(rows[0]["end"], rows[1]["end"]))
            self.assertNotEqual(rows[0]["gpu"], rows[1]["gpu"])
            self.assertTrue(all(row["world"] is None for row in rows))
            for i, first in enumerate(rows):
                for second in rows[i+1:]:
                    if first["gpu"] == second["gpu"]:
                        self.assertGreaterEqual(second["start"], first["end"])

    def test_failed_worker_stops_other_children_and_leaves_queued_job_unstarted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = [
                {"name": "fail", "command": [sys.executable, "-c", "import time; time.sleep(0.3); raise SystemExit(7)"]},
                {"name": "long", "command": [sys.executable, "-c", "import time; time.sleep(30)"]},
                {"name": "pending", "command": [sys.executable, "-c", "raise SystemExit(0)"]},
            ]
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                schedule_jobs(jobs, ["0", "1"], root)
            self.assertFalse((root / "pending.log").exists())


if __name__ == "__main__":
    unittest.main()
