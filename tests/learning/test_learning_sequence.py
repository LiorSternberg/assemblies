from unittest import TestCase

from parameterized import parameterized

from learning.errors import SequenceRunNotInitialized
from learning.learning_sequence import LearningSequence
from tests import TestBrainUtils


class TestLearningSequence(TestCase):

    def setUp(self) -> None:
        self.utils = TestBrainUtils(lazy=False)
        self.brain = self.utils.create_brain(number_of_areas=5, number_of_stimuli=4)

    @parameterized.expand([
        ('one_cycle', 1),
        ('three_cycles', 3)
    ])
    def test_sequence_one_run_per_iteration(self, name, number_of_cycles):
        sequence = LearningSequence(self.brain, intermediate_area=self.utils.area4.name)
        sequence.add_stimulus_to_area_iteration(self.utils.stim0.name, self.utils.area0.name, 1)
        sequence.add_stimulus_to_area_iteration(self.utils.stim1.name, self.utils.area1.name, 1)
        sequence.add_area_to_area_iteration(self.utils.area0.name, self.utils.area2.name, 1)
        sequence.add_area_to_area_iteration(self.utils.area1.name, self.utils.area2.name, 1)

        expected_iterations = [
            (self.utils.stim0, self.utils.area0),
            (self.utils.stim1, self.utils.area1),
            (self.utils.area0, self.utils.area2),
            (self.utils.area1, self.utils.area2),
        ]
        expected_iterations = expected_iterations * number_of_cycles

        sequence.initialize_run(number_of_cycles=number_of_cycles)
        for idx, iteration in enumerate(sequence):
            self.assertEqual(expected_iterations[idx], iteration)

    def test_sequence_multiple_runs_per_iteration(self):
        sequence = LearningSequence(self.brain, intermediate_area=self.utils.area4.name)
        sequence.add_stimulus_to_area_iteration(self.utils.stim0.name, self.utils.area0.name, 2)
        sequence.add_stimulus_to_area_iteration(self.utils.stim1.name, self.utils.area1.name, 1)
        sequence.add_area_to_area_iteration(self.utils.area0.name, self.utils.area2.name, 3)
        sequence.add_area_to_area_iteration(self.utils.area1.name, self.utils.area2.name, 2)

        expected_iterations = [
            (self.utils.stim0, self.utils.area0),
            (self.utils.stim0, self.utils.area0),

            (self.utils.stim1, self.utils.area1),

            (self.utils.area0, self.utils.area2),
            (self.utils.area0, self.utils.area2),
            (self.utils.area0, self.utils.area2),

            (self.utils.area1, self.utils.area2),
            (self.utils.area1, self.utils.area2),
        ]

        sequence.initialize_run(number_of_cycles=1)
        for idx, iteration in enumerate(sequence):
            self.assertEqual(expected_iterations[idx], iteration)

    def test_sequence_not_initialized(self):
        sequence = LearningSequence(self.brain, intermediate_area=self.utils.area4.name)
        sequence.add_stimulus_to_area_iteration(self.utils.stim0.name, self.utils.area0.name, 3)

        # Iterating without initializing raises an error
        with self.assertRaises(SequenceRunNotInitialized):
            for iteration in sequence:
                self.assertIsNotNone(iteration)

        # Initializing and starting to iterate
        sequence.initialize_run(number_of_cycles=1)
        for iteration in sequence:
            break

        # Iterating again without re-initializing raises an error
        with self.assertRaises(SequenceRunNotInitialized):
            for iteration in sequence:
                self.assertIsNotNone(iteration)
