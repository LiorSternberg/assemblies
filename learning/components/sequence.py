from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import List, Union, Dict, Tuple

import matplotlib.pyplot as plt
from networkx import DiGraph, has_path, draw, draw_networkx_edge_labels, get_node_attributes, get_edge_attributes

from brain import Brain, Area, OutputArea
from learning.components.errors import MissingArea, SequenceRunNotInitialized, NoPathException, \
    IllegalOutputAreasException, SequenceFinalizationError, MissingStimulus, InputStimuliMisused
from learning.components.input import InputStimuli


class LearningSequence:

    class Iteration:
        def __init__(self, stimuli_to_areas: Dict[str, List[str]] = None, input_bits_to_areas: Dict[int, List[str]] = None,
                     areas_to_areas: Dict[str, List[str]] = None, consecutive_runs: int = 1):
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

    class IterationConfiguration:
        def __init__(self, current_cycle: int, current_iter: int, current_run: int,
                     number_of_cycles: Union[int, float]):
            self.current_cycle = current_cycle
            self.current_iter = current_iter
            self.current_run = current_run
            self.number_of_cycles = number_of_cycles

            self.activated = False

    def __init__(self, brain: Brain):
        """
        :param brain: the brain object
        """
        self._brain = brain
        # Representing the given sequence as a graph, for connectivity checking
        self._connections_graph = DiGraph()

        self._iterations: List[LearningSequence.Iteration] = []
        self._configuration: Union[LearningSequence.IterationConfiguration, None] = None

        self._output_area = None
        self._finalized = False

    def __iter__(self):
        if self._configuration is None or self._configuration.activated:
            raise SequenceRunNotInitialized()
        return self

    def __next__(self):
        if self._configuration is None:
            raise SequenceRunNotInitialized()

        self._configuration.current_run += 1
        if self._configuration.current_run >= self._iterations[self._configuration.current_iter].consecutive_runs:
            # Moving to the next iteration
            self._configuration.current_run = 0
            self._configuration.current_iter += 1

            if self._configuration.current_iter >= len(self._iterations):
                # Moving to the next cycle
                self._configuration.current_cycle += 1
                self._configuration.current_iter = 0

                if self._configuration.current_cycle >= self._configuration.number_of_cycles:
                    # Number of cycles exceeded
                    raise StopIteration()

        current_iteration = self._iterations[self._configuration.current_iter]
        self._configuration.activated = True
        return current_iteration

    def finalize_sequence(self):
        """
        finalizing the sequence before initial running. The sequence cannot be edited after that
        """
        output_area = self._verify_single_output_area()
        self.output_area = self._brain.output_areas[output_area]
        self._verify_input_bits_are_connected_to_output()
        self._verify_stimuli_are_connected_to_output()

    def initialize_run(self, number_of_cycles=float('inf')):
        """
        Setting up the running of the sequence iterations
        :param number_of_cycles: the number of full cycles (of all defined iterations) that should be run consecutively
        """
        if not self._finalized:
            self.finalize_sequence()
        self._configuration = self.IterationConfiguration(current_cycle=0,
                                                          current_iter=0,
                                                          current_run=-1,
                                                          number_of_cycles=number_of_cycles)

    def _verify_stimulus(self, stimulus_name: str) -> None:
        """
        Raising an error if the given stimulus doesn't exist in the brain
        :param stimulus_name: the stimulus name
        :return: the stimulus object (or exception, on missing)
        """
        if stimulus_name not in self._brain.stimuli:
            raise MissingStimulus(stimulus_name)

    def _verify_and_get_area(self, area_name: str) -> Area:
        """
        :param area_name: the area name
        :return: the area/output area object (or exception, on missing)
        """
        if area_name not in chain(self._brain.areas, self._brain.output_areas):
            raise MissingArea(area_name)
        return self._brain.areas.get(area_name, self._brain.output_areas.get(area_name))

    def add_iteration(self, areas_to_areas: Dict[str, List[str]] = None, input_bits_to_areas: Dict[int, List[str]] = None,
                      stimuli_to_areas: Dict[str, List[str]] = None, consecutive_runs=1):
        """
        Adding an iteration to the learning sequence, consisting of firing stimuli/areas and fired-at areas/output areas
        :param stimuli_to_areas: a mapping between a stimulus and the areas/output areas it fires to
        :param input_bits_to_areas: a mapping between a bit in the input and the areas it's stimuli fire to
        :param areas_to_areas: a mapping between an area and the areas/output areas it fires to
        :param consecutive_runs: the number of consecutive times this iteration is sent (for projection) before moving
            to the next iteration
        """
        if self._finalized:
            raise SequenceFinalizationError()

        if stimuli_to_areas:
            for source_stimulus, target_areas in stimuli_to_areas.items():
                self._verify_stimulus(source_stimulus)

                stimulus_node = f'stimulus-{source_stimulus}'

                for target_area in target_areas:
                    area_type = 'output' if isinstance(self._verify_and_get_area(target_area), OutputArea) else 'area'
                    area_node = f'{area_type}-{target_area}'

                    self._add_edge(stimulus_node, area_node, consecutive_runs)

        if input_bits_to_areas:
            for source_input_bit, target_areas in input_bits_to_areas.items():
                input_bit_node = f'input-bit-{source_input_bit}'

                for target_area in target_areas:
                    area_type = 'output' if isinstance(self._verify_and_get_area(target_area), OutputArea) else 'area'
                    area_node = f'{area_type}-{target_area}'

                    self._add_edge(input_bit_node, area_node, consecutive_runs)

        if areas_to_areas:
            for source_area, target_areas in areas_to_areas.items():
                self._verify_and_get_area(source_area)

                source_area_node = f'area-{source_area}'

                for target_area in target_areas:
                    area_type = 'output' if isinstance(self._verify_and_get_area(target_area), OutputArea) else 'area'
                    target_area_node = f'{area_type}-{target_area}'

                    self._add_edge(source_area_node, target_area_node, consecutive_runs)

        new_iteration = self.Iteration(stimuli_to_areas=stimuli_to_areas,
                                       input_bits_to_areas=input_bits_to_areas,
                                       areas_to_areas=areas_to_areas,
                                       consecutive_runs=consecutive_runs)
        self._iterations.append(new_iteration)

    def _add_edge(self, source_node, target_node, weight):
        """
        Adding an edge to the sequence Graph
        """
        existing_horizontal_positions = max([position[1] for position in get_node_attributes(
            self._connections_graph, 'position').values()]) if self._connections_graph.nodes else 0
        horizontal_index = min(len(self._iterations), existing_horizontal_positions)

        if not self._connections_graph.has_node(source_node):
            vertical_index = len([position for position in get_node_attributes(
                self._connections_graph, 'position').values() if position[1] == horizontal_index])
            self._connections_graph.add_node(source_node, position=(vertical_index, horizontal_index))

        if not self._connections_graph.has_node(target_node):
            vertical_index = len([position for position in get_node_attributes(
                self._connections_graph, 'position').values() if position[1] == horizontal_index + 1])
            self._connections_graph.add_node(target_node, position=(vertical_index, horizontal_index + 1))

        if self._connections_graph.has_edge(source_node, target_node):
            # Adding to the previous edge weight
            weight += self._connections_graph.get_edge_data(source_node, target_node)['weight']

        self._connections_graph.add_edge(source_node, target_node, weight=weight)

    def _verify_stimuli_are_connected_to_output(self):
        """
        Checking that there is a directed path of projection between each stimulus and each output area.
        In case one doesn't exist, this function raises an exception
        """
        stimuli = [node for node in self._connections_graph.nodes if node.startswith('stimulus')]
        output_areas = [node for node in self._connections_graph.nodes if node.startswith('output')]

        for stimulus in stimuli:
            for area in output_areas:
                if not has_path(self._connections_graph, stimulus, area):
                    raise NoPathException(stimulus[9:], area[7:])

    def _verify_input_bits_are_connected_to_output(self):
        """
        Checking that there is a directed path of projection between each input bit and each output area.
        In case one doesn't exist, this function raises an exception
        """
        input_bits = [node for node in self._connections_graph.nodes if node.startswith('input-bit')]
        output_areas = [node for node in self._connections_graph.nodes if node.startswith('output')]

        for input_bit in input_bits:
            for area in output_areas:
                if not has_path(self._connections_graph, input_bit, area):
                    raise NoPathException(input_bit[10:], area[7:])

    def _verify_single_output_area(self):
        """
        Checking that there is a single output area in the sequence. In any other case (none or multiple output areas),
        this function raises an exception
        """
        output_areas = [node[7:] for node in self._connections_graph.nodes if node.startswith('output')]
        if len(output_areas) != 1:
            raise IllegalOutputAreasException(output_areas)
        return output_areas[0]

    def display(self):
        """
        Displaying the sequence graph
        """
        node_positions = get_node_attributes(self._connections_graph, 'position')
        edge_labels = get_edge_attributes(self._connections_graph, 'weight')
        draw(self._connections_graph, pos=node_positions, alpha=1, with_labels=True)
        draw_networkx_edge_labels(self._connections_graph, node_positions, edge_labels=edge_labels)
        plt.show()
