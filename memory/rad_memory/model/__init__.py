"""Model registry for MiniGrid Memory AD and RAD."""

from .ad import AD
from .compressed_ad import RAD

MODEL = {"AD": AD, "RAD": RAD}

__all__ = ["AD", "RAD", "MODEL"]
