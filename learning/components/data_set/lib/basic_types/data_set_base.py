from abc import ABCMeta, abstractmethod

import numpy as np

from learning.components.data_set.data_point import DataPoint
from learning.components.data_set.data_set import DataSet
from learning.components.data_set.errors import DataSetValueError
from learning.components.data_set.lib.data_point import DataPointImpl


class DataSetBase(DataSet, metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the data_set set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    This class contains any and all shared logic between different types of DataSets.
    """
    def __init__(self, noise_probability=0.) -> None:
        super().__init__()
        self._noise_probability = noise_probability
        self._value = -1

    def reset(self):
        self._value = -1

    @abstractmethod
    def _next(self) -> DataPoint:
        """
        Get the next element in the data set, if any.
        :return: int
        """
        pass

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> DataPoint:
        data_point = self._next()
        return self._process_output_values(data_point)

    def _process_output_values(self, data_point: DataPoint):
        if data_point.output not in (0, 1):
            raise DataSetValueError(data_point.input, data_point.output)

        if self._noise_probability and np.random.binomial(1, self._noise_probability):
            return DataPointImpl(data_point.input, data_point.output ^ 1)
        return data_point

    def set_noise_probability(self, noise_probability):
        self._noise_probability = noise_probability

