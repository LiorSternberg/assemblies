from unittest import TestCase

from learning.components.data_set import create_data_set_from_callable
from learning.components.data_set import DataSetValueError, InvalidFunctionError


class TestCallableDataSet(TestCase):
    def test_data_set_function_with_no_arguments_fails(self):
        self.assertRaises(InvalidFunctionError, create_data_set_from_callable, lambda: 1, 1)

    def test_data_set_function_with_2_arguments_fails(self):
        self.assertRaises(InvalidFunctionError, create_data_set_from_callable, lambda x, y: x + y, 2)

    def test_data_set_with_simple_callable_works(self):
        s = create_data_set_from_callable(lambda x: 1 - x, 1)
        self.assertEqual(1, s.domain_size)
        self.assertEqual(1, next(s).output)
        self.assertEqual(0, next(s).output)
        self.assertRaises(StopIteration, next, s)

    def test_data_set_with_non_boolean_values_fails(self):
        s = create_data_set_from_callable(lambda x: x + 1, 1)
        self.assertEqual(1, next(s).output)
        self.assertRaises(DataSetValueError, next, s)

    def test_data_set_with_function_of_domain_size_4_works(self):
        expected = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4)
        self.assertEqual(4, s.domain_size)
        for expected_value in expected:
            self.assertEqual(expected_value, next(s).output)
        self.assertRaises(StopIteration, next, s)

    def test_data_set_iterable_works(self):
        expected = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4)
        for i, data_point in enumerate(s):
            self.assertEqual(expected[i], data_point.output)

    def test_data_set_is_reusable(self):
        expected = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4)
        for i, data_point in enumerate(s):
            self.assertEqual(expected[i], data_point.output)

        # reuse
        reused = 0
        for i, data_point in enumerate(s):
            reused += 1
            self.assertEqual(expected[i], data_point.output)
        self.assertEqual(2 ** s.domain_size, reused)

    def test_data_set_with_full_noise_flips_all_results(self):
        expected_not_noisy = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4, noise_probability=1)
        for i, data_point in enumerate(s):
            self.assertEqual(expected_not_noisy[i] ^ 1, data_point.output)

    def test_data_set_with_noise_flips_some_results(self):
        expected_not_noisy = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4, noise_probability=0.5)
        count_flipped = sum(expected_not_noisy[i] ^ data_point.output for i, data_point in enumerate(s))

        # Note: This test can fail with extremely low probability.
        #       If it does, run again to verify it was one of those extreme cases.
        self.assertLess(0, count_flipped)
        self.assertGreater(len(expected_not_noisy), count_flipped)
