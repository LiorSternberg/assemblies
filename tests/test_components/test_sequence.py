from unittest import TestCase

from parameterized import parameterized

from learning.components.errors import SequenceRunNotInitialized, IllegalOutputAreasException, NoPathException, \
    InputStimuliMisused
from learning.components.input import InputStimuli
from learning.components.sequence import LearningSequence
from tests.brain_test_utils import BrainTestUtils


class TestLearningSequence(TestCase):

    def setUp(self) -> None:
        self.utils = BrainTestUtils(lazy=False)
        self.brain = self.utils.create_brain(number_of_areas=5, number_of_stimuli=4, add_output_area=True)

    @parameterized.expand([
        ('one_cycle', 1),
        ('three_cycles', 3)
    ])
    def test_sequence_one_run_per_iteration(self, name, number_of_cycles):
        input_stimuli = InputStimuli(self.brain, 100, 'A', 'B', verbose=False)
        sequence = LearningSequence(self.brain)
        sequence.add_iteration(input_bits_to_areas={0: ['A'], 1: ['B']})
        sequence.add_iteration(stimuli_to_areas={'A': ['A'], 'B': ['B']})
        sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})
        sequence.add_iteration(areas_to_areas={'C': ['output']})

        expected_iterations = [
            {
                'stim_to_area':
                    {
                        input_stimuli[0][0]: ['A'],
                        input_stimuli[1][0]: ['B']
                    },
                'area_to_area': {}
            },

            {
                'stim_to_area':
                    {
                        'A': ['A'],
                        'B': ['B']
                    },
                'area_to_area': {}
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'A': ['C'],
                        'B': ['C']
                    }
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'C': ['output'],
                    }
            },
        ]
        expected_iterations = expected_iterations * number_of_cycles

        sequence.initialize_run(number_of_cycles=number_of_cycles)
        for idx, iteration in enumerate(sequence):
            self.assertEqual(expected_iterations[idx], iteration.format(input_stimuli, 0))

    def test_sequence_multiple_consecutive_runs_per_iteration(self):
        input_stimuli = InputStimuli(self.brain, 100, 'A', 'B', verbose=False)
        sequence = LearningSequence(self.brain)
        sequence.add_iteration(input_bits_to_areas={0: ['A'], 1: ['B']}, consecutive_runs=2)
        sequence.add_iteration(stimuli_to_areas={'A': ['A'], 'B': ['B']}, consecutive_runs=2)
        sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']}, consecutive_runs=3)
        sequence.add_iteration(areas_to_areas={'C': ['output']}, consecutive_runs=1)

        expected_iterations = [
            {
                'stim_to_area':
                    {
                        input_stimuli[0][0]: ['A'],
                        input_stimuli[1][0]: ['B']
                    },
                'area_to_area': {}
            },
            {
                'stim_to_area':
                    {
                        input_stimuli[0][0]: ['A'],
                        input_stimuli[1][0]: ['B']
                    },
                'area_to_area': {}
            },

            {
                'stim_to_area':
                    {
                        'A': ['A'],
                        'B': ['B']
                    },
                'area_to_area': {}
            },
            {
                'stim_to_area':
                    {
                        'A': ['A'],
                        'B': ['B']
                    },
                'area_to_area': {}
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'A': ['C'],
                        'B': ['C']
                    }
            },
            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'A': ['C'],
                        'B': ['C']
                    }
            },
            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'A': ['C'],
                        'B': ['C']
                    }
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'C': ['output'],
                    }
            },
        ]

        sequence.initialize_run(number_of_cycles=1)
        for idx, iteration in enumerate(sequence):
            self.assertEqual(expected_iterations[idx], iteration.format(input_stimuli, 0))

    def test_sequence_with_only_stimuli(self):
        input_stimuli = InputStimuli(self.brain, 100, 'A', 'B', verbose=False)
        sequence = LearningSequence(self.brain)
        sequence.add_iteration(stimuli_to_areas={'A': ['A'], 'B': ['B']})
        sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})
        sequence.add_iteration(areas_to_areas={'C': ['output']})

        expected_iterations = [
            {
                'stim_to_area':
                    {
                        'A': ['A'],
                        'B': ['B']
                    },
                'area_to_area': {}
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'A': ['C'],
                        'B': ['C']
                    }
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'C': ['output'],
                    }
            },
        ]

        sequence.initialize_run(number_of_cycles=1)
        for idx, iteration in enumerate(sequence):
            self.assertEqual(expected_iterations[idx], iteration.format(input_stimuli, 0))

    def test_sequence_with_only_input_bits(self):
        input_stimuli = InputStimuli(self.brain, 100, 'A', 'B', verbose=False)
        sequence = LearningSequence(self.brain)
        sequence.add_iteration(input_bits_to_areas={0: ['A'], 1: ['B']})
        sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})
        sequence.add_iteration(areas_to_areas={'C': ['output']})

        expected_iterations = [
            {
                'stim_to_area':
                    {
                        input_stimuli[0][0]: ['A'],
                        input_stimuli[1][0]: ['B']
                    },
                'area_to_area': {}
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'A': ['C'],
                        'B': ['C']
                    }
            },

            {
                'stim_to_area': {},
                'area_to_area':
                    {
                        'C': ['output'],
                    }
            },
        ]

        sequence.initialize_run(number_of_cycles=1)
        for idx, iteration in enumerate(sequence):
            self.assertEqual(expected_iterations[idx], iteration.format(input_stimuli, 0))

    def test_sequence_with_bad_input_bits_mapping(self):
        input_stimuli = InputStimuli(self.brain, 100, 'A', 'B', verbose=False)
        sequence = LearningSequence(self.brain)
        sequence.add_iteration(input_bits_to_areas={0: ['B'], 1: ['B']})
        sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})
        sequence.add_iteration(areas_to_areas={'C': ['output']})
        sequence.initialize_run(number_of_cycles=1)

        for idx, iteration in enumerate(sequence):
            self.assertRaises(InputStimuliMisused, iteration.format, input_stimuli, 0)
            break

    def test_sequence_has_no_output_area(self):
        sequence = LearningSequence(self.brain)
        sequence.add_iteration(input_bits_to_areas={0: ['A'], 1: ['B']})
        sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})

        self.assertRaises(IllegalOutputAreasException, sequence.initialize_run, 1)

    def test_sequence_stimulus_has_no_path_to_output(self):
        sequence = LearningSequence(self.brain)
        # Input bit 0 has no path to the output area
        sequence.add_iteration(stimuli_to_areas={'A': ['A'], 'B': ['B']})
        sequence.add_iteration(areas_to_areas={'B': ['C']})
        sequence.add_iteration(areas_to_areas={'C': ['output']})

        self.assertRaises(NoPathException, sequence.initialize_run, 1)

    def test_sequence_input_bit_has_no_path_to_output(self):
        sequence = LearningSequence(self.brain)
        # Input bit 0 has no path to the output area
        sequence.add_iteration(input_bits_to_areas={0: ['A'], 1: ['B']})
        sequence.add_iteration(areas_to_areas={'B': ['C']})
        sequence.add_iteration(areas_to_areas={'C': ['output']})

        self.assertRaises(NoPathException, sequence.initialize_run, 1)

    def test_sequence_not_initialized(self):
        sequence = LearningSequence(self.brain)
        sequence.add_iteration(input_bits_to_areas={0: ['A'], 1: ['B']})
        sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})
        sequence.add_iteration(areas_to_areas={'C': ['output']})

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
