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


class Iteration:
    """
    A single iteration in a sequence, representing a certain number of
    consecutive projections of the same kind (same sources and targets).
    """
    def __init__(self, stimuli_to_areas: Dict[str, List[str]] = None, input_bits_to_areas: Dict[int, List[str]] = None,
                 areas_to_areas: Dict[str, List[str]] = None, consecutive_runs: int = 1):
        """
        Create a new iteration.
        :param stimuli_to_areas: a mapping between a stimulus and the areas/output areas it fires to.
        :param input_bits_to_areas: a mapping between a bit in the input and the areas it's stimuli fire to.
        :param areas_to_areas: a mapping between an area and the areas/output areas it fires to.
        :param consecutive_runs: the number of consecutive times this iteration is sent (for projection) before moving
            to the next iteration.
        """
        self.areas_to_areas = areas_to_areas or {}
        self.stimuli_to_areas = stimuli_to_areas or {}
        self.input_bits_to_areas = input_bits_to_areas or {}
        self.consecutive_runs = consecutive_runs

    @staticmethod
    def _to_bits(input_value: int, size: int) -> Tuple[int, ...]:
        return tuple(int(bit) for bit in bin(input_value)[2:].zfill(size))

    def format(self, input_stimuli: InputStimuli, input_value: int) -> dict:
        """
        Converting the Iteration object into project parameters, using the input definition
        (the InputStimuli object) and the current input value.
        :param input_stimuli: the InputStimuli object which defines the mapping between input bits and pairs of
        stimuli (one for each possible value of the bit).
        :param input_value: the input value as a base 10 integer (for example, for the input 101, use 5).
        """
        input_value = self._to_bits(input_value, len(input_stimuli))
        stimuli_to_area = defaultdict(list, deepcopy(self.stimuli_to_areas))
        for bit_index, area_names in self.input_bits_to_areas.items():
            if sorted(input_stimuli[bit_index].target_areas) != sorted(area_names):
                raise InputStimuliMisused(bit_index, input_stimuli[bit_index].target_areas, area_names)

            bit_value = input_value[bit_index]
            stimulus_name = input_stimuli[bit_index][bit_value]
            stimuli_to_area[stimulus_name] += area_names

        return dict(stim_to_area=dict(stimuli_to_area),
                    area_to_area=self.areas_to_areas)

