from enum import Enum, auto


class BrainLearningMode(Enum):
    """
    An enum for each of the brain modes:
        DEFAULT - normal, standard behaviour, where the outcome of neurons firing is the strengthening of the weight of
         the connection of a relevant pair of neurons.
        TRAINING - known-in-advance sets of neurons in OutputAreas are necessarily the winners of these areas. Hence,
            the weights of the connections to these neurons is strengthened.
        TESTING - the brain loses its plasticity: no weights are changed.
    """
    DEFAULT = auto()
    TRAINING = auto()
    TESTING = auto()
