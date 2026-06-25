from .darkroom import sample_darkroom, sample_darkroom_permuted, Darkroom, DarkroomPermuted, map_dark_states, map_dark_states_inverse
from .dktd import sample_dark_key_to_door, DarkKeyToDoor
from .optimistic import OptimisticExplorationWrapper


ENVIRONMENT = {
    'darkroom': Darkroom,
    'darkroompermuted': DarkroomPermuted,
    'dark_key_to_door': DarkKeyToDoor,
}


SAMPLE_ENVIRONMENT = {
    'darkroom': sample_darkroom,
    'darkroompermuted': sample_darkroom_permuted,
    'dark_key_to_door': sample_dark_key_to_door,
}


def make_env(config, optimistic_exploration=False, visit_counts=None, **kwargs):
    def _init():
        env = ENVIRONMENT[config['env']](config, **kwargs)
        if optimistic_exploration:
            env = OptimisticExplorationWrapper(env, config, visit_counts=visit_counts)
        return env
    return _init
