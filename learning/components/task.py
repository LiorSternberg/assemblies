from brain import Brain
from learning.components.data_set.lib.training_set import TrainingSet
from learning.components.errors import ItemNotInitialized
from learning.components.input import InputStimuli
from learning.components.sequence import LearningSequence
from learning.components.model import LearningModel

# TODO: remove this class. I don't think we need to wrap LearningModel

class LearningTask:

    def __init__(self, brain: Brain):
        self.brain = brain

        self._sequence = None
        self._input_stimuli = None
        self._training_set = None

    @property
    def sequence(self):
        if not self._sequence:
            raise ItemNotInitialized('Learning sequence')
        return self._sequence

    @sequence.setter
    def sequence(self, sequence: LearningSequence):
        self._sequence = sequence

    @property
    def input_stimuli(self):
        if not self._input_stimuli:
            raise ItemNotInitialized('Input stimuli')
        return self._input_stimuli

    @input_stimuli.setter
    def input_stimuli(self, input_stimuli: InputStimuli):
        self._input_stimuli = input_stimuli

    @property
    def training_set(self):
        if not self._training_set:
            raise ItemNotInitialized('Training set')
        return self._training_set

    @training_set.setter
    def training_set(self, training_set: TrainingSet):
        self._training_set = training_set

    def create_model(self, number_of_sequence_cycles=None) -> LearningModel:
        """
        This function creates a learning model according to the configured preferences, and trains it
        :param number_of_sequence_cycles: the number of times the entire sequence should run while on training mode.
            If not given, the default value is taken from LearningConfigurations
        :return: the learning model
        """
        learning_model = LearningModel(brain=self.brain,
                                       sequence=self.sequence,
                                       input_stimuli=self.input_stimuli)
        learning_model.train_model(training_set=self.training_set, number_of_sequence_cycles=number_of_sequence_cycles)
        return learning_model
