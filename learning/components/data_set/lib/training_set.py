import random

from learning.components.data_set.data_point import DataPoint
from learning.components.data_set.lib.basic_types.data_set_base import DataSetBase
from learning.components.data_set.lib.basic_types.partial_data_set import PartialDataSet
from learning.components.data_set.mask import Mask


class TrainingSet(PartialDataSet):
    """
    TrainingSet is the partial data set representing the set used for the
    training phase.
    A training set can contain noise, and is not ordered (but shuffled, see
    elaboration below). The length of the training set determines how many data
    points the iterator will output overall.

    Shuffle process:
    At first, the list of data points used for training is shuffled
    (conceptually, in practice only their indices are shuffled, but the result
    is the same). Then, each data point in the shuffled list is outputted by
    the iterator, one by one. Once all the data points in the shuffled list are
    outputted, the data points are shuffled again (to generate a new random
    inner order for the data set), and are outputted once more, and so on.
    Overall, the number of times the data points will be shuffled depends on
    the given length of the training set, and the number of data points the
    base data set contains, so that:
    overall length =
        (number data points in the base data set after the mask is applied)
        * (times the data points are shuffled)

    Note that this means that repetitions are guaranteed if the training set
    length is set to be longer than the portion of the base data set covered by
    the mask.

    Motivation for the shuffled order:
    Using the base data set in the original order again and again is not a good
    enough solution as it causes a "long streak" of the same value for a
    certain input bit (consider a data set with inputs of length 6, then
    ordered they will be 000000, 000001, 000010, 000011, ... This way, the
    first 2^5 inputs will all have '0' as the first bit of the input. This
    means that that input value (as input bits are considered separate in the
    learning process) will be strengthened too aggressively, causing a
    noticeable sway in results, and failing the learning process.
    On the other hand, random choice (with replacement) of data points from the
    base data set is not good enough either, as it can cause too long of a
    streak of *not choosing* a certain data point, which leads to that data
    point not being learned properly.
    As of the point in time in which this documentation has been written, the
    shuffled order produces the best results for the learning process.
    """
    def __init__(self, base_data_set: DataSetBase, mask: Mask, length: int = None,
                 noise_probability: float = 0.) -> None:
        super().__init__(base_data_set, mask, noise_probability)
        assert type(length) == int, f'TrainingSet length must be an integer (got {length} of type {type(length)})'
        assert length > 0, f'TrainingSet length must be a positive number (got {length})'
        self._length = length
        self._random = random.Random()
        self._shuffled_indices = self._get_training_indices()
        self._inner_index = -1

    def _get_training_indices(self):
        return [index for index in range(2 ** self.input_size) if self._mask.in_training_set(index)]

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
