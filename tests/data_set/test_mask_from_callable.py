from unittest import TestCase

from learning.data_set.constructors import create_explicit_mask_from_callable
from learning.data_set.errors import MaskValueError, InvalidFunctionError


class TestMaskFromCallable(TestCase):
    def test_data_set_function_with_no_arguments_fails(self):
        self.assertRaises(InvalidFunctionError, create_explicit_mask_from_callable, lambda: 1)

    def test_data_set_function_with_2_arguments_fails(self):
        self.assertRaises(InvalidFunctionError, create_explicit_mask_from_callable, lambda x, y: x + y)

    def test_mask_with_non_boolean_values_fails(self):
        mask = create_explicit_mask_from_callable(lambda x: x + 1)
        self.assertRaises(MaskValueError, mask.in_training_set, 2)
        self.assertRaises(MaskValueError, mask.in_testing_set, 2)

    def test_simple_mask_returns_correct_results(self):
        mask = create_explicit_mask_from_callable(lambda x: x % 2)

        self.assertTrue(mask.in_testing_set(0))
        self.assertTrue(mask.in_training_set(1))
        self.assertTrue(mask.in_testing_set(2))
        self.assertTrue(mask.in_training_set(3))

        self.assertFalse(mask.in_training_set(0))
        self.assertFalse(mask.in_testing_set(1))
        self.assertFalse(mask.in_training_set(2))
        self.assertFalse(mask.in_testing_set(3))
