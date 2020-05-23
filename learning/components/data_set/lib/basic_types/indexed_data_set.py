from abc import ABCMeta, abstractmethod

from learning.components.data_set.data_point import DataPoint
from learning.components.data_set.lib.basic_types.data_set_base import DataSetBase


class IndexedDataSet(DataSetBase, metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the data_set set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    This class contains any and all shared logic between different types of DataSets.
    """
    @abstractmethod
    def _get_item(self, item) -> DataPoint:
        """
        Get the element in the <item> index in the data_set.
        :return: int
        """
        pass

    def __getitem__(self, item) -> DataPoint:
        if not 0 <= item < 2 ** self.domain_size:
            raise IndexError(f"Item of index {item} is out of range (choose an"
                             f"index between 0 and {2 ** self.domain_size})")

        return self._process_output_values(self._get_item(item))
