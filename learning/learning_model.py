import math
from collections import namedtuple
from contextlib import contextmanager
from typing import List, Union

from brain import Brain, Stimulus, OutputArea, Area
from learning.data_set.lib.basic_types.data_set_base import DataSetBase
from learning.errors import DomainSizeMismatch, StimuliMismatch, ModelInactivated
from learning.learning_architecture import LearningArchitecture
from learning.learning_configurations import LearningConfigurations
from learning.learning_stages.learning_stages import BrainMode

TestResults = namedtuple('TestResults', ['accuracy',
                                         'true_positive',
                                         'false_negative'])


class LearningModel:

    def __init__(self, brain: Brain, domain_size: int, architecture: LearningArchitecture):
        self._brain = brain
        # Fixating the stimuli for a deterministic input<-->stimuli conversion
        self._stimuli = list(brain.stimuli.keys())

        self._domain_size = domain_size
        self._architecture = architecture

        self._accuracy = None
        self._output_area = None
        self._inactivated = False

    def __del__(self):
        self.terminate()

    def terminate(self) -> None:
        """
        Terminating the model, i.e, deleting the model's output area
        """
        if self._inactivated:
            return

        if self._output_area:
            self._brain.remove_output_area(self._output_area.name)
        self._inactivated = True

    @property
    def output_area(self) -> OutputArea:
        """
        :return: the output area, containing the model's results
        """
        if self._inactivated:
            raise ModelInactivated()

        if not self._output_area:
            output_area_name = 'Output_{}'.format(id(self))
            if output_area_name in self._brain.output_areas:
                self._brain.remove_output_area(output_area_name)
            self._brain.add_output_area(output_area_name)
            self._output_area = self._brain.output_areas[output_area_name]
        return self._output_area

    def train_model(self, training_set: DataSetBase) -> None:
        """
        This function trains the model with the given training set
        :param training_set: the set by which to train the model
        """
        if self._inactivated:
            raise ModelInactivated()

        if training_set.domain_size != self._domain_size:
            raise DomainSizeMismatch('Learning model', 'Training set', self._domain_size, training_set.domain_size)

        for data_point in training_set:
            self._run_unsupervised_projection(data_point.input)
            self._run_supervised_projection(data_point.output)

    def test_model(self, test_set: DataSetBase) -> TestResults:
        """
        Given a test set, this function runs the model on the data points' inputs - and compares it to the expected
        output. It later saves the percentage of the matching runs
        :param test_set: the set by which to test the model's accuracy
        :return: the model's accuracy
        """
        if self._inactivated:
            raise ModelInactivated()

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
        if self._inactivated:
            raise ModelInactivated()

        self._validate_input_number(input_number)

        with self._set_training_mode(BrainMode.TESTING):
            self._run_unsupervised_projection(input_number)
            self._brain.project(stim_to_area={},
                                area_to_area={self._architecture.intermediate_area.name: [self.output_area.name]})
        return self.output_area.winners[0]

    def _run_unsupervised_projection(self, input_number: int) -> None:
        """
        Running the unsupervised learning according to the configured architecture, i.e., setting up the connections
        between the areas of the brain (listed in the architecture), according to the activated stimuli (dictated by
        the given binary string)
        :param input_number: the input number, dictating which stimuli are activated
        """
        active_stimuli = self._convert_input_to_stimuli(input_number)

        self._architecture.initialize_run(number_of_cycles=LearningConfigurations.NUMBER_OF_UNSUPERVISED_CYCLES)

        for source, target in self._architecture:
            # Only active stimuli are allowed to project
            if isinstance(source, Stimulus) and source.name not in active_stimuli:
                continue

            self._brain.project(**self._get_projection_parameters(source, target))

    def _run_supervised_projection(self, output: int) -> None:
        """
        Running the supervised learning, i.e., setting up the connections between the 'intermediate' area (the one
        containing the representation of the activated stimuli, obtained in the unsupervised learning) and the output
        area, while fixating the firing neuron in the output area to correspond with the given binary output
        :param output: the binary output
        """
        with self._set_training_mode(BrainMode.TRAINING):
            self.output_area.desired_output = [output]
            for iteration in range(LearningConfigurations.NUMBER_OF_SUPERVISED_CYCLES):
                self._brain.project(stim_to_area={},
                                    area_to_area={
                                        self._architecture.intermediate_area.name: [self.output_area.name]
                                    })

    @staticmethod
    def _get_projection_parameters(source: Union[Stimulus, Area], target: Area) -> dict:
        """
        Converting the source and target (given by the architecture) to parameters for brain.project
        :param source: the source stimulus/area
        :param target: the target area
        :return: the relevant parameters for projection
        """
        if isinstance(source, Stimulus):
            return dict(
                stim_to_area={source.name: [target.name]},
                area_to_area={}
            )
        return dict(
            stim_to_area={},
            area_to_area={source.name: [target.name]}
        )

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
    def _set_training_mode(self, brain_mode: BrainMode) -> None:
        """
        Setting the brain to be of the given mode, and later returns its original mode
        """
        original_mode, self._brain.mode = self._brain.mode, brain_mode
        yield
        self._brain.mode = original_mode
