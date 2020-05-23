from inspect import signature
from typing import Callable

from learning.components.data_set.errors import InvalidFunctionError, MaskValueError
from learning.components.data_set.mask import Mask


class ExplicitCallableMask(Mask):
    def __init__(self, function: Callable[[int], int]) -> None:
        """
        Create a mask based entirely of the given function return values.
        """
        super().__init__()
        self._validate_function(function)
        self._function = function

    @staticmethod
    def _validate_function(function):
        sig = signature(function)
        if len(sig.parameters) != 1:
            raise InvalidFunctionError(len(sig.parameters))

    @staticmethod
    def _validate_mask_value(index, mask_value):
        if mask_value not in (0, 1):
            raise MaskValueError(index, mask_value)

    def in_training_set(self, index) -> bool:
        """
        Get the value of the mask for the training set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        mask_value = self._function(index)
        self._validate_mask_value(index, mask_value)
        return bool(mask_value)

    def in_test_set(self, index) -> bool:
        """
        Get the value of the mask for the test set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        return not self.in_training_set(index)
