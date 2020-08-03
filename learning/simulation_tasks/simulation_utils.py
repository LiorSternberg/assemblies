from abc import abstractmethod
from math import log, ceil
from typing import Union, List, Callable

from brain import Brain
from learning.components.data_set.constructors import create_training_set_from_list, \
    create_explicit_mask_from_list, create_data_set_from_list
from learning.components.data_set.data_set import DataSet
from learning.components.input import InputStimuli
from learning.components.sequence import LearningSequence
from learning.simulation_tasks.strategy import Strategy
from non_lazy_brain import NonLazyBrain


class SimulationUtilsFactory:

    @staticmethod
    def init_utils(strategy: Strategy, input_size: int):
        if strategy == Strategy.Simple:
            return SimpleSimulationUtils(input_size)
        elif strategy == Strategy.Layered:
            return LayeredSimulationUtils(input_size)

        raise Exception('Unknown Strategy')


class SimulationUtils:

    def __init__(self, input_size: int):
        """
        :param input_size: the size of the input, by which to create the stimuli
        """
        self.input_size = input_size

    @abstractmethod
    def create_brain(self, n: int, k: int, p: float, beta: float) -> NonLazyBrain:
        """
        Creating a non-lazy brain with areas and stimuli according to the input size and the strategy
        """
        pass

    @abstractmethod
    def create_sequence(self, brain: Brain, input_stimuli: InputStimuli) -> LearningSequence:
        """
        Creating the learning sequence that was found most efficient for learning
        """
        pass

    @abstractmethod
    def create_input_stimuli(self, brain: Brain, k: int) -> InputStimuli:
        """
        Creating an input stimuli according to the input size and the strategy
        """
        pass

    def create_training_set(self, output_values_or_function: Union[List[int], Callable],
                            training_set_size_function: Callable,
                            noise: float) -> DataSet:
        """
        Creating a training set according to the given outputs (list of values or function)
        """
        output_values = self._get_output_values(output_values_or_function)
        full_mask = create_explicit_mask_from_list([1] * len(output_values))
        return create_training_set_from_list(data_set_return_values=output_values,
                                             mask=full_mask,
                                             training_set_length=training_set_size_function(self.input_size),
                                             noise_probability=noise)

    def create_test_set(self, output_values_or_function: Union[List[int], Callable]) -> DataSet:
        """
        Creating a test set according to the given outputs (list of values or function)
        """
        output_values = self._get_output_values(output_values_or_function)
        return create_data_set_from_list(output_values)

    def _get_output_values(self, output_values_or_function: Union[List[int], Callable]) -> List[int]:
        """
        Converting the outputs (list of output values or function) into outputs list
        """
        if isinstance(output_values_or_function, list):
            assert len(output_values_or_function) == 2 ** self.input_size, \
                f"input is of size {self.input_size}, hence the outputs must be of size {2 ** self.input_size}"
            outputs_list = output_values_or_function
        else:
            outputs_list = [output_values_or_function(i) for i in range(2 ** self.input_size)]
        return outputs_list

    @staticmethod
    def _name(i: int) -> str:
        """
        :param i: the index of the object
        :return: the object's name
        """
        return chr(ord('A') + i)


class SimpleSimulationUtils(SimulationUtils):
    """
    Utils for a simply-layered brain, as follows:

        Stimulus A   Stimulus B   Stimulus C   Stimulus D ...
                \         |         /           /
                        Area A
                          |
                        Output
    """

    def create_brain(self, n: int, k: int, p: float, beta: float) -> NonLazyBrain:
        brain = NonLazyBrain(p)

        brain.add_area('A', n, k, beta)
        brain.add_output_area('Output')

        return brain

    def create_input_stimuli(self, brain: Brain, k: int) -> InputStimuli:
        return InputStimuli(brain, k, *tuple(['A'] * self.input_size), verbose=False)

    def create_sequence(self, brain: Brain, input_stimuli: InputStimuli) -> LearningSequence:
        sequence = LearningSequence(brain, input_stimuli)

        # All stimuli fire to area 'A'
        input_bits = list(range(self.input_size))
        sequence.add_iteration(input_bits=input_bits)

        sequence.add_iteration(input_bits=input_bits, areas_to_areas={'A': ['A']}, consecutive_runs=2)

        sequence.add_iteration(areas_to_areas={'A': ['Output']})

        return sequence


class LayeredSimulationUtils(SimulationUtils):
    """
    Utils for a logarithmic-layered brain (according to the input size), as follows:

        Stimulus A   Stimulus B         Stimulus C   Stimulus D ...
                \      /                        \       /
                 Area A                          Area B
                    \                             /
                                Area C
                                  |
                                Output
    """

    @property
    def brain_layers(self):
        return ceil(log(self.input_size, 2)) + 1

    def create_brain(self, n: int, k: int, p: float, beta: float) -> NonLazyBrain:
        brain = NonLazyBrain(p)

        areas_count = 0
        for layer_index, area_layer in enumerate(range(self.brain_layers)):
            # For a input of size 4: first layer has 4 areas, second layer has 2 areas, third layer has one area
            areas_in_layer = ceil((2 ** (-layer_index)) * self.input_size)

            for i in range(areas_in_layer):
                area_name = self._name(areas_count)
                brain.add_area(area_name, n, k, beta)
                areas_count += 1

        brain.add_output_area('Output')
        return brain

    def create_input_stimuli(self, brain: Brain, k: int) -> InputStimuli:
        return InputStimuli(brain, k, *tuple(self._name(i) for i in range(self.input_size)), verbose=False)

    def create_sequence(self, brain: Brain, input_stimuli: InputStimuli) -> LearningSequence:
        sequence = LearningSequence(brain, input_stimuli)

        # The first input bit fires to the first area, the second fires to the second area, etc
        input_bits = list(range(self.input_size))
        sequence.add_iteration(input_bits=input_bits, areas_to_areas={})

        area_layers = self._split_to_area_layers(brain)

        # Every first-layer area (i.e. all areas connected directly to stimuli) fires at itself
        areas_to_areas = {area: [area] for area in area_layers[0]}
        sequence.add_iteration(input_bits=input_bits, areas_to_areas=areas_to_areas, consecutive_runs=2)

        for layer in range(self.brain_layers - 1):
            # Every 2 'consecutive' areas fire at an area of the next layer
            areas_to_areas = {area: [area_layers[layer + 1][idx // 2]] for idx, area in enumerate(area_layers[layer])}
            sequence.add_iteration(areas_to_areas=areas_to_areas)

            # In addition to the previous iteration, every area of the next layer fires at itself
            areas_to_areas = dict(areas_to_areas, **{area: [area] for area in area_layers[layer + 1]})
            sequence.add_iteration(areas_to_areas=areas_to_areas, consecutive_runs=2)

        # The last later fires at the output
        areas_to_areas = {area: ['Output'] for area in area_layers[-1]}
        sequence.add_iteration(areas_to_areas=areas_to_areas)

        return sequence

    def _split_to_area_layers(self, brain: Brain) -> List[List[str]]:
        """
        Splitting the given brain's area into different layers.
        For example, a brain of 3 layers would be split into:
            ['A', 'B', 'C', 'D']
                ['E', 'F']
                   ['G']
        """
        areas = sorted(list(brain.areas.keys()))

        layers = []
        for layer_index in range(self.brain_layers):
            areas_in_layer = ceil((2 ** (-layer_index)) * self.input_size)
            current_layer, areas = areas[:areas_in_layer], areas[areas_in_layer:]
            layers.append(current_layer)

        return layers
