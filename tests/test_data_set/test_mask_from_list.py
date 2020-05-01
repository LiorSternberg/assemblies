from unittest import TestCase

from learning.data_set.constructors import create_explicit_mask_from_list
from learning.data_set.errors import MaskIndexError, MaskValueError


class TestMaskFromList(TestCase):
    def test_mask_with_non_boolean_values_fails(self):
        mask = create_explicit_mask_from_list([1, 0, 5, 1])
        self.assertRaises(MaskValueError, mask.in_training_set, 2)
        self.assertRaises(MaskValueError, mask.in_test_set, 2)

    def test_mask_with_non_existing_index_fails(self):
        mask = create_explicit_mask_from_list([1, 0, 1, 1])
        self.assertRaises(MaskIndexError, mask.in_test_set, 4)
        self.assertRaises(MaskIndexError, mask.in_test_set, 5)

    def test_simple_mask_returns_correct_results(self):
        mask = create_explicit_mask_from_list([1, 0, 1, 1])

        self.assertTrue(mask.in_training_set(0))
        self.assertTrue(mask.in_test_set(1))
        self.assertTrue(mask.in_training_set(2))
        self.assertTrue(mask.in_training_set(3))

        self.assertFalse(mask.in_test_set(0))
        self.assertFalse(mask.in_training_set(1))
        self.assertFalse(mask.in_test_set(2))
        self.assertFalse(mask.in_test_set(3))
