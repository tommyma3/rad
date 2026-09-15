"""Fixed effective episode length padding for AD/RAD streams."""

from __future__ import annotations

from typing import Any

import numpy as np


def configured_effective_length(config: dict[str, Any]) -> int | None:
    value = config.get("effective_episode_length")
    if value is None:
        return None
    length = int(value)
    if length < 1:
        raise ValueError("effective_episode_length must be a positive integer")
    return length


def pad_transition_tail(
    item: dict[str, np.ndarray], count: int
) -> dict[str, np.ndarray]:
    """Append no-op filler transitions so an episode occupies its effective length.

    Each dummy transition repeats the final observation, takes the noop action with zero
    reward, and carries terminated=True: the episode ended at its true terminal step and
    the dummies are uniform post-terminal filler.
    """

    if count <= 0:
        return item
    padded = dict(item)
    padded["images"] = np.concatenate(
        [item["images"], np.repeat(item["images"][-1:], count, axis=0)]
    )
    padded["directions"] = np.concatenate(
        [item["directions"], np.repeat(item["directions"][-1:], count, axis=0)]
    )
    padded["actions"] = np.concatenate(
        [item["actions"], np.zeros(count, dtype=item["actions"].dtype)]
    )
    padded["rewards"] = np.concatenate(
        [item["rewards"], np.zeros(count, dtype=item["rewards"].dtype)]
    )
    padded["terminated"] = np.concatenate(
        [item["terminated"], np.ones(count, dtype=item["terminated"].dtype)]
    )
    padded["truncated"] = np.concatenate(
        [item["truncated"], np.zeros(count, dtype=item["truncated"].dtype)]
    )
    padded["cue_ids"] = np.concatenate(
        [item["cue_ids"], np.full(count, -1, dtype=item["cue_ids"].dtype)]
    )
    padded["cue_visible"] = np.concatenate(
        [item["cue_visible"], np.zeros(count, dtype=item["cue_visible"].dtype)]
    )
    padded["decision"] = np.concatenate(
        [item["decision"], np.zeros(count, dtype=item["decision"].dtype)]
    )
    padded["success"] = np.concatenate(
        [item["success"], np.zeros(count, dtype=item["success"].dtype)]
    )
    last_step = int(item["learner_steps"][-1])
    padded["learner_steps"] = np.concatenate(
        [
            item["learner_steps"],
            np.arange(last_step + 1, last_step + 1 + count, dtype=item["learner_steps"].dtype),
        ]
    )
    return padded
