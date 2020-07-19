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
        """
        Base for creating a new DataSet of any kind.

        :param noise_probability: The probability in which the data set outputs a
        'noisy' result (bit flip). For example, with noise_probability=1 the data set
        will always flips the output bit, and for noise_probability=0.5 it is
        expected to flip half of the outputs. Note that noise is probabilistic, so
        Going over the same noisy data set multiple time will most likely generate
        different results (different outputs might be flipped).
        """
        super().__init__()
        self._noise_probability = noise_probability
        self._value = -1

    def reset(self):
        """
        Base reset method for the DataSet's iterator.
        :return:
        """
        self._value = -1

    @abstractmethod
    def _next(self) -> DataPoint:
        """
        Get the next element in the data set, if any.
        :return: data point.
        """
        pass

    def __iter__(self):
        self.reset()
        return self

    def __next__(self) -> DataPoint:
        data_point = self._next()
        return self._process_output_values(data_point)

    def _process_output_values(self, data_point: DataPoint):
        """
        Validate output, and add noise (with noise probability) before
        outputting a data point.
        """
        if data_point.output not in (0, 1):
            raise DataSetValueError(data_point.input, data_point.output)

        if self._noise_probability and np.random.binomial(1, self._noise_probability):
            return DataPointImpl(data_point.input, data_point.output ^ 1)
        return data_point

