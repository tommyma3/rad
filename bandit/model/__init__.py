from .ad import AD
from .compressed_ad import RAD

MODEL = {"AD": AD, "RAD": RAD}


def make_model(config):
    return MODEL[config["model"]](config)
