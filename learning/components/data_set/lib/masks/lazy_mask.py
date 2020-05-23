import random

from learning.components.data_set.mask import Mask


class LazyMask(Mask):
    def __init__(self, percentage: float, seed: int = None) -> None:
        """
        Create a mask that contains the given percentage of "1"s in random
        indexes, based on an initial seed (optional). If a seed is not given,
        one will be chosen at random.
        :param percentage: the percentage of "1"s in the mask.
        :param seed: optional seed for the mask. All mask instances created with
        the same seed will be equivalent (return the same value for the same
        indices).
        """
        super().__init__()
        assert 0 <= percentage <= 1, "Percentage must be a number between 0 and 1."
        self._percentage = percentage
        self._base_seed = seed or random.randint(2*20, 2*30)
        self._random = random.Random()

    def in_training_set(self, index) -> bool:
        """
        Get the value of the mask for the training set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        self._random.seed(self._base_seed + index)
        return self._random.random() < self._percentage

    def in_test_set(self, index) -> bool:
        """
        Get the value of the mask for the test set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        return not self.in_training_set(index)
