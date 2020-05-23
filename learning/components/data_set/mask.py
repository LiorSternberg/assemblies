from abc import ABCMeta, abstractmethod


class Mask(metaclass=ABCMeta):
    """
    Mask is an object used to split a data set into a training set and a test
    set.
    """
    @abstractmethod
    def in_training_set(self, index) -> bool:
        """
        Get the value of the mask for the training set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        pass

    @abstractmethod
    def in_test_set(self, index) -> bool:
        """
        Get the value of the mask for the test set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        pass
