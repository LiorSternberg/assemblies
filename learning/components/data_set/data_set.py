from abc import ABCMeta, abstractmethod
from collections import namedtuple

from learning.components.data_set.data_point import DataPoint


DataSets = namedtuple('DataSets', ['training_set', 'test_set'])


class DataSet(metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the data set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    """
    @abstractmethod
    def set_noise_probability(self, noise_probability):
        """
        Set the noise probability of the data set to a new value.
        """
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self) -> DataPoint:
        pass

    @property
    @abstractmethod
    def domain_size(self):
        """
        Get the domain size (i.e., the number of bits required to represent
        an item from the domain of the data set's function's).
        :return: int
        """
        pass


