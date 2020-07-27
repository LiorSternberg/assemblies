from unittest import TestCase

from learning.components.input import InputStimuli
from learning.components.sequence import LearningSequence
from tests.brain_test_utils import BrainTestUtils


class LearningComponentTestBase(TestCase):

    def setUp(self) -> None:
        utils = BrainTestUtils(lazy=False)
        self.brain = utils.create_brain(number_of_areas=3, number_of_stimuli=4,
                                        area_size=100, winners_size=10, add_output_area=True)

        self.input_stimuli = InputStimuli(self.brain, 10, 'A', 'B', verbose=False)
        self.sequence = LearningSequence(self.brain, self.input_stimuli)
        self.sequence.add_iteration(input_bits=[0, 1])
        self.sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})
        self.sequence.add_iteration(areas_to_areas={'C': ['output']})


