from enum import Enum
from itertools import chain
from typing import List, Dict, Optional, Callable, Union

from brain import Brain, Area, OutputArea
from learning.components.errors import MissingArea, SequenceRunNotInitializedOrInMidRun, SequenceFinalizationError, \
    MissingStimulus
from learning.components.sequence_components.connections_graph import ConnectionsGraph
from learning.components.sequence_components.iteration import Iteration
from learning.components.sequence_components.iteration_configuration import IterationConfiguration


class SourceType(Enum):
    """
    Possible source types for the projections in a sequence.
    """
    STIMULUS = 'stimulus'
    INPUT_BIT = 'input-bit'
    AREA = 'area'


class LearningSequence:
    """
    The learning sequence represent a sequence of projections used for the
    learning process (either at the training or the test stage). it defines the
    order of projections, and the number of their repetitions.

    Note that the sequence is validated to ensure that there is only one
    output, and that each input is eventually connected to the output area.

    Also, after the sequence is created, it allows the user to generate a
    connections graph representing the projections in the sequence.
    """
    def __init__(self, brain: Brain):
        """
        Create a new empty sequence.

        Once created, you can add iterations to it, and eventually finalize it
        to indicate it is ready and should be validated (and that no new
        iterations will added).

        :param brain: the brain object
        """
        self._brain = brain

        # Representing the given sequence as a graph, for connectivity checking
        self._connections_graph = ConnectionsGraph()

        self._iterations: List[Iteration] = []
        self._configuration: Optional[IterationConfiguration] = None

        self._output_area: Optional[OutputArea] = None
        self._finalized: bool = False

    @property
    def number_of_iterations(self) -> int:
        """
        Returns the number of iterations currently in the sequence.
        """
        return len(self._iterations)

    @property
    def output_area(self) -> OutputArea:
        """
        Returns the output area of the sequence.
        """
        return self._output_area

    def __iter__(self):
        if self._configuration is None or self._configuration.is_in_mid_run:
            raise SequenceRunNotInitializedOrInMidRun()
        return self

    def __next__(self):
        if self._configuration is None:
            raise SequenceRunNotInitializedOrInMidRun()

        self._configuration.current_run += 1
        if self._configuration.current_run >= self._iterations[self._configuration.current_iter].consecutive_runs:
            # Moving to the next iteration
            self._configuration.current_run = 0
            self._configuration.current_iter += 1

            if self._configuration.current_iter >= self.number_of_iterations:
                # Moving to the next cycle
                self._configuration.current_cycle += 1
                self._configuration.current_iter = 0

                if self._configuration.current_cycle >= self._configuration.number_of_cycles:
                    # Number of cycles exceeded
                    self._configuration.is_in_mid_run = False
                    raise StopIteration()

        current_iteration = self._iterations[self._configuration.current_iter]

        # Indicate that the iteration started, and must exhausted or reset
        # (by initializing a new run) before starting another iteration.
        self._configuration.is_in_mid_run = True

        return current_iteration

    def finalize_sequence(self) -> None:
        """
        finalizing the sequence before initial running. The sequence cannot be edited after that
        """
        output_area = self._connections_graph.verify_single_output_area()
        self._output_area = self._brain.output_areas[output_area]
        self._connections_graph.verify_inputs_are_connected_to_output()

    def initialize_run(self, number_of_cycles=float('inf')) -> None:
        """
        Setting up the running of the sequence iterations
        :param number_of_cycles: the number of full cycles (of all defined iterations) that should be run consecutively
        """
        if not self._finalized:
            self.finalize_sequence()
        self._configuration = IterationConfiguration(number_of_cycles=number_of_cycles)

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

    def _validate_and_add_connections(self, source_type: SourceType,
                                      mapping: Dict[Union[str, int], List[str]],
                                      consecutive_runs: int,
                                      source_verification_method: Optional[Callable] = None) -> None:
        """
        Validate a source-to-targets mapping for the iteration's projections and add the relevant
        connections to the connections graph.
        :param source_type: one of the possible SourceTypes, indicating what kind source nodes
        should be created for the connections.
        :param mapping: The source to targets mapping for the iteration projections.
        :param consecutive_runs: Number of consecutive runs of the iteration, to be used as
        the weight of the connections.
        :param source_verification_method: An optional method to verify the source element before
        adding the connection to the connections graph (used for example to validate a source
        area actually exists.
        """
        for source, target_areas in mapping.items():
            if source_verification_method:
                source_verification_method(source)

            source_node = f'{source_type.value}-{source}'

            for target_area in target_areas:
                area_type = 'output' if isinstance(self._verify_and_get_area(target_area), OutputArea) else 'area'
                area_node = f'{area_type}-{target_area}'

                self._connections_graph.add_connection(source_node, area_node, consecutive_runs,
                                                       self.number_of_iterations)

    def add_iteration(self, areas_to_areas: Dict[str, List[str]] = None,
                      input_bits_to_areas: Dict[int, List[str]] = None,
                      stimuli_to_areas: Dict[str, List[str]] = None,
                      consecutive_runs: int = 1) -> None:
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
            self._validate_and_add_connections(SourceType.STIMULUS, stimuli_to_areas, consecutive_runs,
                                               self._verify_stimulus)

        if input_bits_to_areas:
            self._validate_and_add_connections(SourceType.INPUT_BIT, input_bits_to_areas, consecutive_runs)

        if areas_to_areas:
            self._validate_and_add_connections(SourceType.AREA, areas_to_areas, consecutive_runs,
                                               self._verify_and_get_area)

        new_iteration = Iteration(stimuli_to_areas=stimuli_to_areas,
                                  input_bits_to_areas=input_bits_to_areas,
                                  areas_to_areas=areas_to_areas,
                                  consecutive_runs=consecutive_runs)
        self._iterations.append(new_iteration)

    def display_connections_graph(self):
        self._connections_graph.display()
