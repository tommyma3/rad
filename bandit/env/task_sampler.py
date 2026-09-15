"""Independent Uniform[0,1] arm means with Gaussian reward noise."""
from dataclasses import asdict, dataclass

import numpy as np

SAMPLER_VERSION = "iid_uniform_gaussian_v1"


@dataclass(frozen=True)
class BanditTask:
    task_id: str
    means: tuple[float, ...]
    distribution: str
    seed: int
    sampler: str = SAMPLER_VERSION
    reward_std: float = 0.3

    def __post_init__(self):
        if self.sampler != SAMPLER_VERSION or self.distribution != "uniform":
            raise ValueError("Task must use iid_uniform_gaussian_v1 with uniform arm means")
        means = np.asarray(self.means)
        if (means.ndim != 1 or len(means) < 2 or not np.all(np.isfinite(means)) or
                np.any((means < 0) | (means > 1))):
            raise ValueError("Task means must be finite and in [0, 1]")
        if not np.isfinite(self.reward_std) or self.reward_std <= 0:
            raise ValueError("reward_std must be finite and positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if value.get("sampler") != SAMPLER_VERSION or "reward_std" not in value:
            raise ValueError("Incompatible task reward specification; regenerate Gaussian tasks")
        return cls(**{**value, "means": tuple(value["means"])})


def sample_task(seed, distribution="uniform", num_arms=10, task_id=None, *, reward_std=0.3):
    if num_arms < 2:
        raise ValueError("num_arms must be >= 2")
    if distribution != "uniform":
        raise ValueError("Independent uniform arm means do not support odd/even bias")
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.0, 1.0, num_arms)
    return BanditTask(task_id or f"{distribution}-{seed}", tuple(means.tolist()),
                      distribution, int(seed), reward_std=float(reward_std))


def make_manifest(seed, count, distribution="uniform", num_arms=10,
                  namespace="train", *, reward_std=0.3):
    if count <= 0:
        raise ValueError("count must be positive")
    seeds = np.random.SeedSequence(seed).spawn(count)
    return [sample_task(int(s.generate_state(1, dtype=np.uint64)[0]), distribution,
                        num_arms, f"{namespace}-{i:06d}", reward_std=reward_std)
            for i, s in enumerate(seeds)]
