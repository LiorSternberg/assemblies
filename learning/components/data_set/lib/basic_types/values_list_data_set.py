from math import log

from learning.components.data_set.data_point import DataPoint
from learning.components.data_set.errors import DataSetSizeError
from learning.components.data_set.lib.basic_types.indexed_data_set import IndexedDataSet
from learning.components.data_set.lib.data_point import DataPointImpl


class ValuesListDataSet(IndexedDataSet):
    """
    An iterator defining the data_set set for a brain, based on a list of output
    values of binary function. For example, given binary function such as
    f(x) =  x (identity) or f(x, y) = (x + y) % 2, the list of values should be
    [0, 1] and [0, 1, 1, 0] (respectively).
    """
    def __init__(self, return_values, noise_probability=0.) -> None:
        super().__init__(noise_probability=noise_probability)
        self._return_values = return_values
        self._domain_size = self._get_domain_size(return_values)

    @staticmethod
    def _get_domain_size(return_values):
        domain_size = log(len(return_values), 2)
        if not domain_size.is_integer():
            raise DataSetSizeError(len(return_values))

        return int(domain_size)

    @property
    def domain_size(self):
        return self._domain_size

    def _next(self) -> DataPoint:
        if self._value == 2 ** self._domain_size - 1:
            self.reset()
            raise StopIteration()

        self._value += 1
        return DataPointImpl(self._value, self._return_values[self._value])

    def _get_item(self, item):
        return DataPointImpl(item, self._return_values[item])
