import random

from learning.data_set.data_point import DataPoint
from learning.data_set.data_set import DataSet
from learning.data_set.lib.basic_types.partial_data_set import PartialDataSet
from learning.data_set.mask import Mask


class TrainingSet(PartialDataSet):
    """
    TrainingSet is the partial data set representing the set used for the
    training phase.
    A training set can contain noise, and is not ordered. The length of the
    training set determines how many data points the iterator will output
    overall. On each iteration a data point from the partial data set dedicated
    to the training phase will be outputted at random, so repetitions are likely,
    and are expected to occur if the training set is set to be long enough.
    """
    def __init__(self, base_data_set: DataSet, mask: Mask, length: int = None,
                 noise_probability: float = 0.) -> None:
        super().__init__(base_data_set, mask, noise_probability)
        self._count_left = length
        self._range_max = 2 ** self.domain_size - 1
        self._random = random.Random()

    def _next(self) -> DataPoint:
        if self._count_left == 0:
            self.reset()
            raise StopIteration()

        mask_value = 0
        index = 0
        while not mask_value:
            index = self._random.randint(0, self._range_max)
            mask_value = self._mask.in_training_set(index)

        self._count_left -= 1
        return self._base_data_set[index]
