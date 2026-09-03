"""Swap the initial key/ball cue while holding the later trajectory fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .model import MODEL


def _observation(image: np.ndarray, direction: int) -> dict:
    return {"image": image, "direction": int(direction)}


def _swap_key_ball(image: np.ndarray) -> np.ndarray:
    from minigrid.core.constants import OBJECT_TO_IDX

    key = OBJECT_TO_IDX["key"]
    ball = OBJECT_TO_IDX["ball"]
    result = image.copy()
    objects = result[..., 0]
    key_mask = objects == key
    ball_mask = objects == ball
    objects[key_mask] = ball
    objects[ball_mask] = key
    return result


@torch.inference_mode()
def compare_episode(model, group) -> dict | None:
    decision_indices = np.flatnonzero(group["decision"][()])
    if not len(decision_indices):
        return None
    decision = int(decision_indices[0])
    images = group["images"][()]
    directions = group["directions"][()]
    next_images = group["next_images"][()]
    next_directions = group["next_directions"][()]
    cue_visible = group["cue_visible"][()]
    contexts = []
    for swapped in (False, True):
        first_image = _swap_key_ball(images[0]) if swapped and cue_visible[0] else images[0]
        context = model.start_context(_observation(first_image, directions[0]))
        for step in range(decision):
            next_image = next_images[step]
            # Swap every observation in the contiguous cue-visible prefix. The
            # post-cue suffix, including the branch geometry, remains identical.
            if swapped and step + 1 < len(cue_visible) and cue_visible[step + 1]:
                next_image = _swap_key_ball(next_image)
            model.observe(
                context,
                int(group["actions"][step]),
                float(group["rewards"][step]),
                bool(group["terminated"][step]),
                bool(group["truncated"][step]),
                _observation(next_image, next_directions[step]),
            )
        contexts.append(context)
    original = model.action_logits(contexts[0]).softmax(-1)
    swapped = model.action_logits(contexts[1]).softmax(-1)
    return {
        "decision_step": decision,
        "original_action": int(original.argmax(-1).item()),
        "swapped_action": int(swapped.argmax(-1).item()),
        "prediction_flipped": bool(original.argmax(-1).item() != swapped.argmax(-1).item()),
        "probability_l1": float((original - swapped).abs().sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cue-swap causal diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = MODEL[config["model"]](config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    records = []
    with h5py.File(args.artifact, "r") as handle:
        for key in sorted(handle["episodes"].keys()):
            result = compare_episode(model, handle["episodes"][key])
            if result is not None:
                records.append({"episode": key, **result})
            if len(records) >= args.episodes:
                break
    if not records:
        raise ValueError("Artifact contains no branch-decision episodes")
    summary = {
        "checkpoint": args.checkpoint,
        "artifact": args.artifact,
        "episodes": len(records),
        "flip_rate": float(np.mean([record["prediction_flipped"] for record in records])),
        "mean_probability_l1": float(np.mean([record["probability_l1"] for record in records])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
