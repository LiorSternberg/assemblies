import math
from collections import namedtuple
from contextlib import contextmanager
from typing import List, Union

from brain import Brain, Stimulus, OutputArea, Area
from learning.data_set.lib.basic_types.data_set_base import DataSetBase
from learning.errors import DomainSizeMismatch, StimuliMismatch
from learning.learning_sequence import LearningSequence
from learning.learning_configurations import LearningConfigurations
from learning.learning_stages.learning_stages import BrainMode

TestResults = namedtuple('TestResults', ['accuracy',
                                         'true_positive',
                                         'false_negative'])


class LearningModel:

    def __init__(self, brain: Brain, domain_size: int, sequence: LearningSequence):
        """
        :param brain: the brain
        :param domain_size: the size of the model's domain (e.g. function: {0,1}^2 -> {0,1} is of domain size = 2)
        :param sequence: the sequence by which the model is projecting
        """
        self._brain = brain
        # Fixating the stimuli for a deterministic input<-->stimuli conversion
        self._stimuli = list(brain.stimuli.keys())

        self._domain_size = domain_size
        self._sequence = sequence

    @property
    def output_area(self) -> OutputArea:
        """
        :return: the output area, containing the model's results
        """
        return self._sequence.output_area

    def train_model(self, training_set: DataSetBase, number_of_sequence_cycles=None) -> None:
        """
        This function trains the model with the given training set
        :param training_set: the set by which to train the model
        :param number_of_sequence_cycles: the number of times the entire sequence should run while on training mode.
            If not given, the default value is taken from LearningConfigurations
        """
        if training_set.domain_size != self._domain_size:
            raise DomainSizeMismatch('Learning model', 'Training set', self._domain_size, training_set.domain_size)

        number_of_sequence_cycles = number_of_sequence_cycles or LearningConfigurations.NUMBER_OF_TRAINING_CYCLES

        for data_point in training_set:
            self._run_learning_projection(input_number=data_point.input,
                                          brain_mode=BrainMode.TRAINING,
                                          desired_output=data_point.output,
                                          number_of_sequence_cycles=number_of_sequence_cycles)

    def test_model(self, test_set: DataSetBase) -> TestResults:
        """
        Given a test set, this function runs the model on the data points' inputs - and compares it to the expected
        output. It later saves the percentage of the matching runs
        :param test_set: the set by which to test the model's accuracy
        :return: the model's accuracy
        """
        if test_set.domain_size != self._domain_size:
            raise DomainSizeMismatch('Learning model', 'Test set', self._domain_size, test_set.domain_size)

        true_positive = []
        false_negative = []
        for data_point in test_set:
            if self.run_model(data_point.input) == int(data_point.output):
                true_positive.append(data_point.input)
            else:
                false_negative.append(data_point.input)

        accuracy = round(len(true_positive) / (len(true_positive) + len(false_negative)), 2)
        # TODO change true negative / false negative labels, since it doesn't make sense
        return TestResults(accuracy=accuracy,
                           true_positive=true_positive,
                           false_negative=false_negative)

    def run_model(self, input_number: int) -> int:
        """
        This function runs the model with the given binary string and returns the result.
        It must be run after the model has finished its training process
        :param input_number: the input for the model to calculate
        :return: the result of the model to the given input
        """
        self._validate_input_number(input_number)

        self._run_learning_projection(input_number, brain_mode=BrainMode.TESTING)
        return self.output_area.winners[0]

    def _run_learning_projection(self, input_number: int, brain_mode: BrainMode, desired_output=None,
                                 number_of_sequence_cycles=1) -> None:
        """
        Running the unsupervised and supervised learning according to the configured sequence, i.e., setting up the
        connections between the areas of the brain (listed in the sequence), according to the activated stimuli
        (dictated by the given binary string)
        :param input_number: the input number, dictating which stimuli are activated
        :param brain_mode: the mode of the projecting (TESTING/TRAINING/DEFAULT)
        :param desired_output: the desired output, in case we are in training mode
        :param number_of_sequence_cycles: the number of times the entire sequence should run.
            Should be 1 for non-training.
        """
        active_stimuli = self._convert_input_to_stimuli(input_number)

        self._sequence.initialize_run(number_of_cycles=number_of_sequence_cycles)

        if desired_output:
            self.output_area.desired_output = [desired_output]

        for iteration in self._sequence:
            # Getting the projection parameters, after the removal of the nonactive stimuli
            projection_parameters = iteration.format(active_stimuli)
            with self._set_brain_mode(brain_mode=brain_mode):
                self._brain.project(**projection_parameters)

    def _convert_input_to_stimuli(self, input_number: int) -> List[str]:
        """
        Converting a binary string to a list of activated stimuli.
        For example: - given the stimuli [1,2,3,4], the binary string of "00" would convert to [1,3]
                     - given the stimuli [1,2,3,4], the binary string of "01" would convert to [1,4]
                     - given the stimuli [1,2,3,4], the binary string of "10" would convert to [2,3]
                     - given the stimuli [1,2,3,4], the binary string of "11" would convert to [2,4]
        :param input_number: the input number to be converted to a list of stimuli
        :return: the activated stimuli names
        """
        if len(self._brain.stimuli) != self._domain_size * 2:
            raise StimuliMismatch(self._domain_size * 2, len(self._brain.stimuli))

        self._validate_input_number(input_number)

        binary_string = str(bin(input_number))[2:].zfill(self._domain_size)
        active_stimuli = []
        for index, stimulus in enumerate(self._stimuli):
            relevant_char = binary_string[index // 2]
            if index % 2 == int(relevant_char):
                active_stimuli.append(stimulus)
        return active_stimuli

    def _validate_input_number(self, input_number: int) -> None:
        """
        Validating that the given number is in the model's domain
        :param: input_number: the number to validate
        """
        input_domain = math.ceil(math.log(input_number + 1, 2))
        if input_domain > self._domain_size:
            raise DomainSizeMismatch('Learning model', input_number, self._domain_size, input_domain)

    @contextmanager
    def _set_brain_mode(self, brain_mode: BrainMode) -> None:
        """
        Setting the brain to be of the given mode, and later returns its original mode
        """
        original_mode, self._brain.mode = self._brain.mode, brain_mode
        yield
        self._brain.mode = original_mode
