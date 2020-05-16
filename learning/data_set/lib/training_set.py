import random

from learning.data_set.data_point import DataPoint
from learning.data_set.lib.basic_types.data_set_base import DataSetBase
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
    def __init__(self, base_data_set: DataSetBase, mask: Mask, length: int = None,
                 noise_probability: float = 0.) -> None:
        super().__init__(base_data_set, mask, noise_probability)
        assert type(length) == int, f'TrainingSet length must be an integer (got {length} of type {type(length)})'
        self._length = length
        self._random = random.Random()
        self._shuffled_indices = self._get_training_indices()
        self._inner_index = -1

    def _get_training_indices(self):
        return [index for index in range(2 ** self.domain_size) if self._mask.in_training_set(index)]

    def _increment_inner_index(self):
        self._inner_index = (self._inner_index + 1) % (len(self._shuffled_indices))
        if self._inner_index == 0:
            self._random.shuffle(self._shuffled_indices)

    def _next(self) -> DataPoint:
        self._value += 1
        if self._value == self._length:
            self.reset()
            raise StopIteration()

        self._increment_inner_index()
        return self._base_data_set[self._shuffled_indices[self._inner_index]]
