from enum import Enum, auto


class BrainLearningMode(Enum):
    """
    An enum for each of the brain modes:
        DEFAULT - normal, standard behaviour, where the outcome of neurons firing is the strengthening of the weight of
         the connection of a relevant pair of neurons.
        FORCE_DESIRED_OUTPUT - Forces the output area's winners at the end of a projection to be the set of winners that
        is predefined as the 'desired_output' (an attribute of output area). Hence, the weights of the connections to
        the desired output neurons is strengthened. This mode can be used for training the brain.
        PLASTICITY_OFF - the brain loses its plasticity: no weights are changed. This mode can be used when trying to
        test the brain's behavior without affecting the weights of the existing connections.
    """
    DEFAULT = auto()
    FORCE_DESIRED_OUTPUT = auto()
    PLASTICITY_OFF = auto()
