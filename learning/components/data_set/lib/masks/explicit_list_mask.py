from typing import List

from learning.components.data_set.errors import MaskIndexError, MaskValueError
from learning.components.data_set.mask import Mask


class ExplicitListMask(Mask):
    def __init__(self, mask_values: List[int]) -> None:
        """
        Create a mask based entirely of the given mask values.
        """
        super().__init__()
        self._mask_values = mask_values

    def _validate_mask_index(self, index):
        if index >= len(self._mask_values):
            raise MaskIndexError(index, len(self._mask_values))

    @staticmethod
    def _validate_mask_value(index, mask_value):
        if mask_value not in (0, 1):
            raise MaskValueError(index, mask_value)

    def in_training_set(self, index) -> bool:
        """
        Get the value of the mask for the training set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        self._validate_mask_index(index)
        mask_value = self._mask_values[index]
        self._validate_mask_value(index, mask_value)
        return bool(mask_value)

    def in_test_set(self, index) -> bool:
        """
        Get the value of the mask for the test set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        return not self.in_training_set(index)
