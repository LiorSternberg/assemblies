from unittest import TestCase

from learning.data_set.constructors import create_explicit_mask_from_list, create_test_set_from_callable, create_lazy_mask
from learning.data_set.data_point import DataPoint


class TestTestSet(TestCase):
    def test_test_set_returns_data_points(self):
        mask = create_explicit_mask_from_list([0, 0, 1, 1, 0, 1, 1, 1])
        s = create_test_set_from_callable(lambda x: x % 2, 3, mask)
        for data_point in s:
            self.assertIsInstance(data_point, DataPoint)

    def test_test_set_is_reusable(self):
        mask = create_explicit_mask_from_list([0, 0, 1, 1, 0, 1, 1, 1])
        s = create_test_set_from_callable(lambda x: x % 2, 3, mask)
        for data_point in s:
            self.assertIsInstance(data_point, DataPoint)

        # reuse
        for data_point in s:
            self.assertIsInstance(data_point, DataPoint)

    def test_test_set_length_is_correct(self):
        mask = create_explicit_mask_from_list([0, 0, 1, 1, 0, 1, 1, 1])
        s = create_test_set_from_callable(lambda x: x % 2, 3, mask)
        self.assertEqual(3, len([i for i in s]))

    def test_test_set_has_no_noise(self):
        def func(x):
            return x % 2

        mask = create_lazy_mask(0.3)
        s = create_test_set_from_callable(func, 10, mask)
        count_noisy = sum(data_point.output != func(data_point.input)
                          for data_point in s)

        self.assertEqual(0, count_noisy)

    def test_test_set_is_ordered(self):
        mask = create_lazy_mask(0.)
        s = create_test_set_from_callable(lambda x: x % 2, 8, mask)
        self.assertListEqual(list(range(2 ** 8)), [data_point.input for data_point in s])

    def test_test_set_is_partial(self):
        mask = create_explicit_mask_from_list([0, 0, 1, 1, 0, 1, 1, 1])
        s = create_test_set_from_callable(lambda x: x % 2, 3, mask)
        indices = {data_point.input for data_point in s}
        self.assertNotIn(2, indices)
        self.assertNotIn(3, indices)
        self.assertNotIn(5, indices)
        self.assertNotIn(6, indices)
        self.assertNotIn(7, indices)
