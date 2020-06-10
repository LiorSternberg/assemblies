from learning.components.data_set.data_point import DataPoint
from learning.components.data_set.data_set import DataSet
from learning.components.data_set.lib.basic_types.partial_data_set import PartialDataSet
from learning.components.data_set.mask import Mask


class TestSet(PartialDataSet):
    """
    TestSet is the partial data set representing the set used for the
    test phase.
    A test set does not contain noise (by definition), and is ordered.
    Iterating over the test set will output data points from the portion of
    the data dedicated to testing.
    """
    def __init__(self, base_data_set: DataSet, mask: Mask) -> None:
        super().__init__(base_data_set, mask, noise_probability=0.)

    def _next(self) -> DataPoint:
        if self._value == 2 ** self._base_data_set.input_size - 1:
            self.reset()
            raise StopIteration()

        self._value += 1
        data_point = next(self._base_data_set)
        mask_value = self._mask.in_test_set(self._value)

        while not mask_value:
            self._value += 1
            data_point = next(self._base_data_set)
            mask_value = self._mask.in_test_set(self._value)

        return data_point
