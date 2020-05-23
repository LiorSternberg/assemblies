from unittest import TestCase

from learning.errors import MissingArea, MissingStimulus
from non_lazy_brain import NonLazyBrain


class InputTests(TestCase):
    def setUp(self) -> None:
        n = 10000
        k = 100
        self.brain = NonLazyBrain(p=0.01)
        self.brain.add_area('A', n, k, beta=0.05)
        self.brain.add_area('B', n, k, beta=0.05)
        self.brain.add_area('C', n, k, beta=0.05)
        self.brain.add_stimulus('s0', k)
        self.brain.add_stimulus('s1', k)
        self.brain.add_stimulus('s2', k)
        self.brain.add_stimulus('s3', k)
        self.brain.add_output_area('Output')

    def test_input_bits_generates_list_of_input_bit_objects_from_single_areas(self):
        input_bits = InputBits('A', 'B', 'C')

        count = 0
        for input_bit in input_bits:
            count += 1
            self.assertIsInstance(input_bit, InputBit)
        self.assertEqual(3, count)

        self.assertEqual(3, len(input_bits))
        for i in range(len(input_bits)):
            self.assertIsInstance(input_bits[i], InputBit)

    def test_input_bits_generates_list_of_input_bit_objects_from_complex_areas(self):
        input_bits = InputBits('A', 'B', ['A', 'B'], 'C', ['A', 'C'], ['A', 'B', 'C'])

        count = 0
        for input_bit in input_bits:
            count += 1
            self.assertIsInstance(input_bit, InputBit)
        self.assertEqual(6, count)

        self.assertEqual(6, len(input_bits))
        for i in range(len(input_bits)):
            self.assertIsInstance(input_bits[i], InputBit)

    def test_input_bits_generates_list_of_input_bit_objects_from_areas_with_override(self):
        input_bits = InputBits('A', 'B', ['A', 'B'], override={0: ('s0', 's1')})

        count = 0
        for input_bit in input_bits:
            count += 1
            self.assertIsInstance(input_bit, InputBit)
        self.assertEqual(3, count)

        self.assertEqual(3, len(input_bits))
        for i in range(len(input_bits)):
            self.assertIsInstance(input_bits[i], InputBit)

    def test_input_bits_with_non_existent_area_raises(self):
        self.assertRaises(InputBits('A', 'B', 'Non-Existent'), MissingArea)

    def test_input_bits_with_non_existent_override_stimulus_raises(self):
        self.assertRaises(InputBits('A', 'B', override={1: ('s0', 's-non-existent')}), MissingStimulus)

    def test_input_bits_with_badly_formatted_override_stimulus_raises(self):
        # Must be a tuple of two, the first value representing a stimulus for 0, the second a stimulus for 1:
        self.assertRaises(InputBits('A', 'B', override={1: ('s0', 's1', 's2')}), ValueError)

        self.assertRaises(InputBits('A', 'B', override={1: ('s0',)}), ValueError)

        self.assertRaises(InputBits('A', 'B', override={1: {0: 's0', 1: 's1'}}), TypeError)

        self.assertRaises(InputBits('A', 'B', override={1: 's0'}), TypeError)

    def test_input_bits_with_different_areas_doesnt_use_the_same_stimuli(self):
        input_bits = InputBits('A', 'B', ['A', 'B'])

        # input_bits[0] is an input bit, which is of the form: InputBit(0 = stim0, 1 = stim1)
        # So we want to assure the two input bits are compiled of different stimuli:
        self.assertNotIn(input_bits[0][0], (input_bits[1][0], input_bits[2][0]))
        self.assertNotIn(input_bits[1][0], (input_bits[0][0], input_bits[2][0]))
        self.assertNotIn(input_bits[2][0], (input_bits[0][0], input_bits[1][0]))

        self.assertNotIn(input_bits[0][1], (input_bits[1][1], input_bits[2][1]))
        self.assertNotIn(input_bits[1][1], (input_bits[0][1], input_bits[2][1]))
        self.assertNotIn(input_bits[2][1], (input_bits[0][1], input_bits[1][1]))

    def test_input_bits_with_the_same_area_twice_doesnt_the_use_same_stimuli(self):
        input_bits = InputBits('A', 'A')

        # input_bits[0] is an input bit, which is of the form: InputBit(0 = stim0, 1 = stim1)
        # So we want to assure the two input bits are compiled of different stimuli:
        self.assertNotEqual(input_bits[0][0], input_bits[1][0])
        self.assertNotEqual(input_bits[0][1], input_bits[1][1])


