from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import List, Union, Dict, Tuple, Optional

import matplotlib.pyplot as plt
from networkx import DiGraph, has_path, draw, draw_networkx_edge_labels, get_node_attributes, get_edge_attributes

from brain import Brain, Area, OutputArea
from learning.components.errors import MissingArea, SequenceRunNotInitializedOrInMidRun, NoPathException, \
    IllegalOutputAreasException, SequenceFinalizationError, MissingStimulus, InputStimuliMisused
from learning.components.input import InputStimuli


class IterationConfiguration:
    """
    An internal configuration used by the sequence to keep track of the current
    iteration.
    """
    def __init__(self, number_of_cycles: Union[int, float]):
        """
        Create a new configuration and initialize tracking attributes, using the
        following logic:
            A sequence can run for a variable (user defined) number
            of cycles. Each cycle contains a number of iterations, as defined
            before the sequence is finalized. Each iteration is defined to run a
            certain number of consecutive runs.

        So we get:
         - current_cycle: the current cycle number (out of the total cycles) for
            the current usage of the sequence.
         - current_iter: the current number of iteration within the cycle.
         - current_run: the current number of run out of the defined consecutive
            runs for this iteration.

        Note that we also define flag - is_in_mid_run - stating whether this
        configuration started to track a run of the sequence, to prevent from
        accidentally using the sequence starting from the middle of a cycle or
        iteration. After this flag is set, there is no way to unset it (a new
        configuration must be created, ensuring the sequence will start running
        from the beginning).
        """
        self.current_cycle = 0
        self.current_iter = 0
        self.current_run = -1
        self.number_of_cycles = number_of_cycles

        self.is_in_mid_run = False

