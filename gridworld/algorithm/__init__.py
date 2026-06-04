from .ppo import PPOWrapper
from .utils import HistoryLoggerCallback

ALGORITHM = {
    'PPO': PPOWrapper,
}