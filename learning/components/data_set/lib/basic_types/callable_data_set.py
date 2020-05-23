from inspect import signature

from learning.components.data_set.data_point import DataPoint
from learning.components.data_set.errors import InvalidFunctionError
from learning.components.data_set.lib.basic_types.indexed_data_set import IndexedDataSet
from learning.components.data_set.lib.data_point import DataPointImpl


class CallableDataSet(IndexedDataSet):
    """
    An iterator defining the data_set set for a brain, based on an appropriate
    Callable. For example, given binary functions such as f(x) =  x (identity)
    and f(x, y) = (x + y) % 2, appropriate Callables for example are
    lambda x: x and lambda x: bin(x)[2:].count('1') % 2 (respectively).
    Importantly, the inputs of the Callable are considered to be integers from
    the interval [0, 2^domain_size-1].
    """
    def __init__(self, function, domain_size, noise_probability=0.) -> None:
        super().__init__(noise_probability=noise_probability)
        self._validate_function(function)
        self._function = function
        self._domain_size = domain_size

    @staticmethod
    def _validate_function(function):
        sig = signature(function)
        if len(sig.parameters) != 1:
            raise InvalidFunctionError(len(sig.parameters))

    @property
    def domain_size(self):
        return self._domain_size

    def _next(self) -> DataPoint:
        if self._value == 2 ** self._domain_size - 1:
            self.reset()
            raise StopIteration()

        self._value += 1
        return DataPointImpl(self._value, self._function(self._value))

    def _get_item(self, item) -> DataPoint:
        return DataPointImpl(item, self._function(item))
