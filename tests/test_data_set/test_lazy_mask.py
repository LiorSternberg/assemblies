from math import sqrt
from random import shuffle
from unittest import TestCase

from learning.data_set.constructors import create_lazy_mask


class TestLazyMask(TestCase):
    def test_mask_returns_all_test_when_percentage_is_0(self):
        mask = create_lazy_mask(percentage=0)
        for index in range(32):
            self.assertTrue(mask.in_test_set(index))
            self.assertFalse(mask.in_training_set(index))

    def test_mask_returns_all_training_when_percentage_is_1(self):
        mask = create_lazy_mask(percentage=1)
        for index in range(32):
            self.assertFalse(mask.in_test_set(index))
            self.assertTrue(mask.in_training_set(index))

    def test_mask_is_returns_opposite_values_for_test_and_training(self):
        mask = create_lazy_mask(0.5)
        for check in range(10):
            for index in range(100):
                self.assertNotEqual(mask.in_test_set(index), mask.in_training_set(index))

    def test_mask_is_consistent_across_rounds(self):
        mask = create_lazy_mask(0.5)
        training_expected = [mask.in_training_set(i) for i in range(100)]
        test_expected = [mask.in_test_set(i) for i in range(100)]

        for check in range(10):
            indices = list(range(100))
            shuffle(indices)
            for index in indices:
                self.assertEqual(training_expected[index], mask.in_training_set(index))
                self.assertEqual(test_expected[index], mask.in_test_set(index))

    @staticmethod
    def _calc_deviation(n, p):
        # 3 standard deviations, should be fine for 99.7% of the time
        return 3 * sqrt(n * p * (1 - p)) / n

    def test_mask_is_split_correctly_50_50(self):
        mask = create_lazy_mask(0.5, seed=10)
        indices_in_training = [mask.in_training_set(i) for i in range(2 ** 14)]
        indices_in_test = [mask.in_test_set(i) for i in range(2 ** 14)]

        # Assert that the results are split approximately 50/50, allow a small deviation.
        self.assertAlmostEqual(0.5, sum(indices_in_training) / 2 ** 14, delta=self._calc_deviation(2 ** 14, 0.5))
        self.assertAlmostEqual(0.5, sum(indices_in_test) / 2 ** 14, delta=self._calc_deviation(2 ** 14, 0.5))

    def test_mask_is_split_correctly_30_70(self):
        mask = create_lazy_mask(0.3, seed=10)
        indices_in_training = [mask.in_training_set(i) for i in range(2 ** 14)]
        indices_in_test = [mask.in_test_set(i) for i in range(2 ** 14)]

        # Assert that the results are split approximately 30/70, allow a small deviation.
        self.assertAlmostEqual(0.3, sum(indices_in_training) / 2 ** 14, delta=self._calc_deviation(2 ** 14, 0.3))
        self.assertAlmostEqual(0.7, sum(indices_in_test) / 2 ** 14, delta=self._calc_deviation(2 ** 14, 0.7))
