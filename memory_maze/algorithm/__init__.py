from .ppo import CNNPPO, HistoryRecorderCallback
from .dreamer_tbtt import DreamerTBTT

SOURCE_ALGORITHMS = {
    "ppo": CNNPPO,
    "dreamer_tbtt": DreamerTBTT,
}

__all__ = ["CNNPPO", "DreamerTBTT", "HistoryRecorderCallback", "SOURCE_ALGORITHMS"]
