from collections import namedtuple
from typing import List, Union

from brain import Brain, Stimulus, Area, OutputArea
from learning.errors import MissingStimulus, MissingArea, SequenceRunNotInitialized


class LearningSequence:

    Iteration = namedtuple('Iteration', ['source', 'target', 'consecutive_runs'])

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
        # A dummy output area, to symbolize a reference to the actual (yet to be defined) output area
        self.output_area = OutputArea('output')

        self._iterations: List[LearningSequence.Iteration] = []
        self._configuration: Union[LearningSequence.IterationConfiguration, None] = None

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
        return current_iteration.source, current_iteration.target

    def initialize_run(self, number_of_cycles=float('inf')):
        """
        Setting up the running of the sequence iterations
        :param number_of_cycles: the number of full cycles (of all defined iterations) that should be run consecutively
        """
        self._configuration = self.IterationConfiguration(current_cycle=0,
                                                          current_iter=0,
                                                          current_run=-1,
                                                          number_of_cycles=number_of_cycles)

    def _get_stimulus(self, stimulus_name: str) -> Stimulus:
        """
        :param stimulus_name: the stimulus name
        :return: the stimulus object (or exception, on missing)
        """
        if stimulus_name not in self._brain.stimuli:
            raise MissingStimulus(stimulus_name)
        stimulus = self._brain.stimuli[stimulus_name]
        # Setting a name attribute for future use (since the object doesn't have any)
        stimulus.name = stimulus_name
        return stimulus

    def _get_area(self, area_name: str) -> Area:
        """
        :param area_name: the area name
        :return: the area object (or exception, on missing)
        """
        if area_name not in self._brain.areas:
            raise MissingArea(area_name)
        return self._brain.areas[area_name]

    def add_stimulus_to_area_iteration(self, source_stimulus: str, target_area: str, consecutive_runs=1):
        """
        :param source_stimulus: the name of the source stimulus
        :param target_area: the name of the target area
        :param consecutive_runs: the number of times this step shall run consecutively (given its turn)
        """
        new_iteration = self.Iteration(
            source=self._get_stimulus(source_stimulus),
            target=self._get_area(target_area),
            consecutive_runs=consecutive_runs)
        self._iterations.append(new_iteration)

    def add_stimulus_to_output_iteration(self, source_stimulus: str, consecutive_runs=1):
        """
        :param source_stimulus: the name of the source stimulus
        :param consecutive_runs: the number of times this step shall run consecutively (given its turn)
        """
        new_iteration = self.Iteration(
            source=self._get_stimulus(source_stimulus),
            target=self.output_area,
            consecutive_runs=consecutive_runs)
        self._iterations.append(new_iteration)

    def add_area_to_area_iteration(self, source_area: str, target_area: str, consecutive_runs=1):
        """
        :param source_area: the name of the source area
        :param target_area: the name of the target area
        :param consecutive_runs: the number of times this step shall run consecutively (given its turn)
        """
        new_iteration = self.Iteration(
            source=self._get_area(source_area),
            target=self._get_area(target_area),
            consecutive_runs=consecutive_runs)
        self._iterations.append(new_iteration)

    def add_area_to_output_iteration(self, source_area: str, consecutive_runs=1):
        """
        :param source_area: the name of the source area
        :param consecutive_runs: the number of times this step shall run consecutively (given its turn)
        """
        new_iteration = self.Iteration(
            source=self._get_area(source_area),
            target=self.output_area,
            consecutive_runs=consecutive_runs)
        self._iterations.append(new_iteration)
