from itertools import chain
from typing import List, Union, Dict
from networkx import DiGraph, has_path, draw
import matplotlib.pyplot as plt

from brain import Brain, Area, OutputArea
from learning.errors import MissingStimulus, MissingArea, SequenceRunNotInitialized, NoPathException, \
    IllegalOutputAreasException, SequenceFinalizationError


class LearningSequence:

    class Iteration:
        def __init__(self, stimuli_to_areas: Dict[str, List[str]], areas_to_areas: Dict[str, List[str]],
                     consecutive_runs: int):
            self.stimuli_to_areas = stimuli_to_areas
            self.areas_to_areas = areas_to_areas
            self.consecutive_runs = consecutive_runs

        def format(self, active_stimuli: List[str]) -> dict:
            """
            Converting the Iteration object into project parameters, while filtering out firing non-active stimuli
            in the iteration)
            :param active_stimuli: the active stimuli when running the iteration
            """
            return dict(stim_to_area={stimulus: areas for stimulus, areas in self.stimuli_to_areas.items()
                                      if stimulus in active_stimuli}, area_to_area=self.areas_to_areas)

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

    def add_iteration(self, stimuli_to_areas: Dict[str, List[str]], areas_to_areas: Dict[str, List[str]],
                      consecutive_runs=1):
        """
        Adding an iteration to the learning sequence, consisting of firing stimuli/areas and fired-at areas/output areas
        :param stimuli_to_areas: a mapping between a stimulus and the areas/output areas it fires to
        :param areas_to_areas: a mapping between an area and the areas/output areas it fires to
        :param consecutive_runs: the number of consecutive times this iteration is sent (for projection) before moving
            to the next iteration
        """
        if self._finalized:
            raise SequenceFinalizationError()

        for source_stimulus, target_areas in stimuli_to_areas.items():
            self._verify_stimulus(source_stimulus)

            for target_area in target_areas:
                area_type = 'output' if isinstance(self._verify_and_get_area(target_area), OutputArea) else 'area'
                self._connections_graph.add_edge(f'stimulus-{source_stimulus}', f'{area_type}-{target_area}',
                                                 weight=consecutive_runs)

        for source_area, target_areas in areas_to_areas.items():
            self._verify_and_get_area(source_area)

            for target_area in target_areas:
                area_type = 'output' if isinstance(self._verify_and_get_area(target_area), OutputArea) else 'area'
                self._connections_graph.add_edge(f'area-{source_area}', f'{area_type}-{target_area}',
                                                 weight=consecutive_runs)

        new_iteration = self.Iteration(stimuli_to_areas=stimuli_to_areas,
                                       areas_to_areas=areas_to_areas,
                                       consecutive_runs=consecutive_runs)
        self._iterations.append(new_iteration)

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
        draw(self._connections_graph, alpha=0.5, with_labels=True)
        plt.show()
