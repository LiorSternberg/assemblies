from brain import Brain
from learning.data_set.lib.basic_types.partial_data_set import PartialDataSet
from learning.data_set.lib.training_set import TrainingSet
from learning.errors import ItemNotInitialized
from learning.learning_sequence import LearningSequence
from learning.learning_model import LearningModel


class Learning:

    def __init__(self, brain: Brain, domain_size: int):
        self.brain = brain
        self.domain_size = domain_size

        self._sequence = None
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
    def training_set(self):
        if not self._training_set:
            raise ItemNotInitialized('Training set')
        return self._training_set

    @training_set.setter
    def training_set(self, training_set: TrainingSet):
        self._training_set = training_set

    def create_model(self) -> LearningModel:
        """
        This function creates a learning model according to the configured preferences, and trains it
        :return: the learning model
        """
        learning_model = LearningModel(brain=self.brain,
                                       domain_size=self.domain_size,
                                       sequence=self.sequence)
        learning_model.train_model(training_set=self.training_set)
        return learning_model
