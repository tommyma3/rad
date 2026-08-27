from .ppo import PPOOptimisticWrapper, PPOWrapper
from .utils import HistoryLoggerCallback

ALGORITHM = {
    'PPO': PPOWrapper,
    'PPOOptimistic': PPOOptimisticWrapper,
}
