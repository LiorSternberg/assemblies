from unittest import TestCase

from learning.learning_sequence import LearningSequence
from tests.brain_test_utils import BrainTestUtils


class LearningTestBase(TestCase):

    def setUp(self) -> None:
        utils = BrainTestUtils(lazy=False)
        self.brain = utils.create_brain(number_of_areas=3, number_of_stimuli=4,
                                        area_size=100, winners_size=10, add_output_area=True)

        self.sequence = LearningSequence(self.brain)
        self.sequence.add_iteration(stimuli_to_areas={'A': ['A'], 'B': ['A'], 'C': ['B'], 'D': ['B']},
                                    areas_to_areas={})
        self.sequence.add_iteration(stimuli_to_areas={},
                                    areas_to_areas={'A': ['C'], 'B': ['C']})
        self.sequence.add_iteration(stimuli_to_areas={},
                                    areas_to_areas={'C': ['output']})

