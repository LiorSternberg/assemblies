from typing import Callable, List

from learning.components.data_set.data_set import DataSet, DataSets
from learning.components.data_set.lib.basic_types.callable_data_set import CallableDataSet as _CallableDataSet
from learning.components.data_set.lib.basic_types.values_list_data_set import ValuesListDataSet as _ValuesListDataSet
from learning.components.data_set.lib.masks.explicit_callable_mask import ExplicitCallableMask as _ExplicitCallableMask
from learning.components.data_set.lib.masks.explicit_list_mask import ExplicitListMask as _ExplicitListMask
from learning.components.data_set.lib.masks.lazy_mask import LazyMask as _LazyMask
from learning.components.data_set.lib.test_set import TestSet as _TestSet
from learning.components.data_set.lib.training_set import TrainingSet as _TrainingSet
from learning.components.data_set.mask import Mask


def create_data_set_from_callable(
        function: Callable[[int], int],
        domain_size: int,
        noise_probability: float = 0.) -> DataSet:
    """
    Create a base data set (used to create a training / test set) from a
    function (a python callable). Note that the function should get one argument,
    an integer between 0 and 2 ** domain_size, and return 0 or 1.
    :param function: The boolean function to generate a data set from.
    :param domain_size: The size of the function's domain.
    :param noise_probability: The probability in which the data set outputs a
    'noisy' result (bit flip). For example, with noise_probability=1 the data set
    will always flips the output bit, and for noise_probability=0.5 it is
    expected to flip half of the outputs. Note that noise is probabilistic, so
    Going over the same noisy data set multiple time will most likely generate
    different results (different outputs might be flipped).
    :return: The data set representing these parameters.

    Usage example:

    To create a data set with inputs of size 4 (such as 0000, 0001, ..., 1111),
    and with noise probability 0.5 (expected half the bits are flipped), and
    set the output function to be 0 if the input is even, and 1 if the input is
    odd, one can run the following:

    >>> data_set = create_data_set_from_callable(lambda x: x % 2, 4, noise_probability=0.5)

    The returned data set is iterable, so to get the next data point one can
    simply loop over it, like this:

    >>> for data_point in data_set:
    ...  print(data_point.input, data_point.output)
    """
    return _CallableDataSet(function, domain_size, noise_probability)


def create_data_set_from_list(
        return_values: List[int],
        noise_probability: float = 0.) -> DataSet:
    """
    Create a base data set (used to create a training / test set) from an
    explicit list of return values. Note that the list should be of length that
    is a power of two (to represent a full function), and contain only 0s and 1s.
    :param return_values: The return values of the function represented in the
    data set.
    :param noise_probability: The probability in which the data set outputs a
    'noisy' result (bit flip). For example, with noise_probability=1 the data set
    will always flips the output bit, and for noise_probability=0.5 it is
    expected to flip half of the outputs. Note that noise is probabilistic, so
    Going over the same noisy data set multiple time will most likely generate
    different results (different outputs might be flipped).
    :return: The data set representing these parameters.

    Usage example:

    To create a data set with inputs of size 4 (such as 0000, 0001, ..., 1111),
    and with noise probability 0.5 (expected half the bits are flipped), and
    set the output function to be 0 if the input is even, and 1 if the input is
    odd, one can run the following:

    >>> return_values = [
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1
    ... ]
    >>> data_set = create_data_set_from_list(return_values, noise_probability=0.5)

    The returned data set is iterable, so to get the next data point one can
    simply loop over it, like this:

    >>> for data_point in data_set:
    ...  print(data_point.input, data_point.output)
    """
    return _ValuesListDataSet(return_values, noise_probability)


def create_lazy_mask(percentage: float, seed: int = None) -> Mask:
    """
    Create a random mask that covers <percentage> of the indexes. The generated
    mask is lazy (so the entire mask is never saved in memory, and is calculated
    fo a given index at runtime). Note that different seeds will generate
    different lazy masks, but if you do wish to recreate the same mask you
    simply need to set the seed.
    :param percentage: Which percentage of the mask is set to 1.
    :param seed: Seeds the random element of the mask. Masks based on the same
    seed will perform exactly the same.
    :return: The mask object, used to split a data set into a training set and a
    test set.

    Usage example:

    To create a lazy mask that covers 60% of the indices, one can run:
    >>> mask = create_lazy_mask(0.6)
    """
    return _LazyMask(percentage, seed)


def create_explicit_mask_from_list(mask_values: List[int]) -> Mask:
    """
    Create a mask that covers the indexes that are 1s in the given list. Note
    the given mask should cover all indices of the data set it is meant to be
    applied to, and that all values should be 0 or 1.
    :param mask_values: The mask values, as a list of 0s and 1s.
    :return: The mask object, used to split a data set into a training set and a
    test set.

    Usage example:

    To create an explicit mask that covers the first 4 indices, and does not
    cover the last 4 indices, one can run:
    >>> mask = create_explicit_mask_from_list([1, 1, 1, 1, 0, 0, 0, 0])

    And then:
    >>> mask.in_training_set(0)
    Returns True.

    >>> mask.in_training_set(4)
    Returns False.
    """
    return _ExplicitListMask(mask_values)


def create_explicit_mask_from_callable(function: Callable[[int], int]) -> Mask:
    """
    Create a mask that covers the indexes that to which the given function returns 1.
    Note the given mask should cover all indices of the data set it is meant to be
    applied to, and that all values should be 0 or 1.
    :param function: The boolean function to use as the mask.
    :return: The mask object, used to split a data set into a training set and a
    test set.

    Usage example:

    To create an explicit mask that covers the first 4 indices, and does not
    cover the last 4 indices, one can run:
    >>> mask = create_explicit_mask_from_callable(lambda x: x < 4)

    And then:
    >>> mask.in_training_set(0)
    Returns True.
    >>> mask.in_test_set(0)
    Returns False.

    >>> mask.in_training_set(4)
    Returns False.
    >>> mask.in_test_set(4)
    Returns True.
    """
    return _ExplicitCallableMask(function)


def create_training_and_test_sets_from_callable(
        data_set_function: Callable[[int], int],
        domain_size: int,
        mask: Mask,
        training_set_length: int,
        noise_probability: float = 0.) -> DataSets:
    """
    Simplified way to create matching training and test sets from a function
    (a python callable). Note that the function should get one argument,
    an integer between 0 and 2 ** domain_size, and return 0 or 1.
    :param data_set_function: The boolean function to generate a data set from.
    :param domain_size: The size of the function's domain.
    :param mask: The mask object used to split the data set into a training set
    and a test set. Covered indices will belong to the training set, and the
    rest to test set.
    :param training_set_length: How long should the training set iterator be.
    Note that a training set is created by randomly choosing data points from
    the portion of the data that belongs to the training set, so repetitions are
    likely if the training set is made to be long enough.
    :param noise_probability: The probability in which the data set outputs a
    'noisy' result (bit flip). Only applies to the training set. The test set
    is never noisy.
    :return: The data sets representing these parameters.

    Usage example:

    To create a training set and a matching test set with inputs of size 4
    (such as 0000, 0001, ..., 1111), and with noise probability 0.5 (expected
    half the bits are flipped), and set the output function to be 0 if the
    input is even, and 1 if the input is odd, and use the first 4 inputs for
    the training set of length 100, and the rest for the test set, one can run
    the following:

    >>> mask = create_explicit_mask_from_callable(lambda x: x < 4)
    >>> data_sets = create_training_and_test_sets_from_callable(
    ...     lambda x: x % 2, 4, mask, 100, noise_probability=0.5)

    This returns a tuple of training and test sets:
    >>> test_set = data_sets.test_set
    >>> training_set = data_sets.training_set
    """
    base_data_set = create_data_set_from_callable(data_set_function, domain_size, noise_probability)
    return DataSets(training_set=_TrainingSet(base_data_set, mask, training_set_length, noise_probability),
                    test_set=_TestSet(base_data_set, mask))


def create_training_and_test_sets_from_list(
        data_set_return_values: List[int],
        mask: Mask,
        training_set_length: int,
        noise_probability: float = 0.) -> DataSets:
    """
    Simplified way to create matching training and test sets from a list of
    return values representing a boolean function. Note that the list should be
    of length that is a power of two (to represent a full function), and contain
    only 0s and 1s.
    :param data_set_return_values: The return values of the function represented
    in the data set.
    :param mask: The mask object used to split the data set into a training set
    and a test set. Covered indices will belong to the training set, and the
    rest to test set.
    :param training_set_length: How long should the training set iterator be.
    Note that a training set is created by randomly choosing data points from
    the portion of the data that belongs to the training set, so repetitions are
    likely if the training set is made to be long enough.
    :param noise_probability: The probability in which the data set outputs a
    'noisy' result (bit flip). Only applies to the training set. The test set
    is never noisy.
    :return: The data sets representing these parameters.

        Usage example:

    To create a training set and a matching test set with inputs of size 4
    (such as 0000, 0001, ..., 1111), and with noise probability 0.5 (expected
    half the bits are flipped), and set the output function to be 0 if the
    input is even, and 1 if the input is odd, and use the first 4 inputs for
    the training set of length 100, and the rest for the test set, one can run
    the following:

    >>> return_values = [
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1
    ... ]
    >>> mask = create_explicit_mask_from_callable(lambda x: x < 4)
    >>> data_sets = create_training_and_test_sets_from_list(
    ...     return_values, mask, 100, noise_probability=0.5)

    This returns a tuple of training and test sets:
    >>> test_set = data_sets.test_set
    >>> training_set = data_sets.training_set
    """
    base_data_set = create_data_set_from_list(data_set_return_values, noise_probability)
    return DataSets(training_set=_TrainingSet(base_data_set, mask, training_set_length, noise_probability),
                    test_set=_TestSet(base_data_set, mask))


def create_training_set_from_callable(
        data_set_function: Callable[[int], int],
        domain_size: int,
        mask: Mask,
        training_set_length: int,
        noise_probability: float = 0.) -> DataSet:
    """
    Simplified way to create a training set from a function (a python callable).
    Note that the function should get one argument, an integer between 0 and
    2 ** domain_size, and return 0 or 1.
    :param data_set_function: The boolean function to generate a data set from.
    :param domain_size: The size of the function's domain.
    :param mask: The mask object used to split the data set into a training set
    and a test set. Only covered indices will belong to the training set.
    :param training_set_length: How long should the training set iterator be.
    Note that a training set is created by randomly choosing data points from
    the portion of the data that belongs to the training set, so repetitions are
    likely if the training set is made to be long enough.
    :param noise_probability: The probability in which the data set outputs a
    'noisy' result (bit flip). Only applies to the training set. The test set
    is never noisy.
    :return: The data sets representing these parameters.

    Usage example:

    To create a training set of length 100 with inputs of size 4 (such as 0000,
    0001, ..., 1111), that only contains the first 10 inputs, and with noise
    probability 0.5 (expected half the bits are flipped), and set the output
    function to be 0 if the input is even, and 1 if the input is odd, one can
    run the following:

    >>> create_explicit_mask_from_callable(lambda x: x < 10)
    >>> training_set = create_training_set_from_callable(
    ...     lambda x: x % 2, 4, mask, 100, noise_probability=0.5)

    The returned training set is iterable, so to get the next data point one can
    simply loop over it, like this:

    >>> for data_point in training_set:
    ...  print(data_point.input, data_point.output)
    """

    base_data_set = create_data_set_from_callable(data_set_function, domain_size, noise_probability)
    return _TrainingSet(base_data_set, mask, training_set_length, noise_probability)


def create_test_set_from_callable(
        data_set_function: Callable[[int], int],
        domain_size: int,
        mask: Mask) -> DataSet:
    """
    Simplified way to create a test set from a function (a python callable).
    Note that the function should get one argument, an integer between 0 and
    2 ** domain_size, and return 0 or 1.
    :param data_set_function: The boolean function to generate a data set from.
    :param domain_size: The size of the function's domain.
    :param mask: The mask object used to split the data set into a training set
    and a test set. Only uncovered indices will belong to the test set.
    :return: The data sets representing these parameters.

    Usage example:

    To create a test set with inputs of size 4 (such as 0000, 0001, ..., 1111),
    that only contains the first 10 inputs, and set the output function to be 0
    if the input is even, and 1 if the input is odd, one can run the following:

    >>> create_explicit_mask_from_callable(lambda x: x < 10)
    >>> test_set = create_test_set_from_callable(lambda x: x % 2, 4, mask)

    The returned test set is iterable, so to get the next data point one can
    simply loop over it, like this:

    >>> for data_point in test_set:
    ...  print(data_point.input, data_point.output)
    """
    base_data_set = create_data_set_from_callable(data_set_function, domain_size)
    return _TestSet(base_data_set, mask)


def create_training_set_from_list(
        data_set_return_values: List[int],
        mask: Mask,
        training_set_length: int,
        noise_probability: float = 0.) -> DataSet:
    """
    Simplified way to create a training set from a list of return values
    representing a boolean function. Note that the list should be of length that
    is a power of two (to represent a full function), and contain only 0s and 1s.
    :param data_set_return_values: The return values of the function represented
    in the data set.
    :param mask: The mask object used to split the data set into a training set
    and a test set. Only covered indices will belong to the training set.
    :param training_set_length: How long should the training set iterator be.
    Note that a training set is created by randomly choosing data points from
    the portion of the data that belongs to the training set, so repetitions are
    likely if the training set is made to be long enough.
    :param noise_probability: The probability in which the data set outputs a
    'noisy' result (bit flip). Only applies to the training set. The test set
    is never noisy.
    :return: The data sets representing these parameters.

    Usage example:

    To create a training set of length 100 with inputs of size 4 (such as 0000,
    0001, ..., 1111), that only contains the first 10 inputs, and with noise
    probability 0.5 (expected half the bits are flipped), and set the output
    function to be 0 if the input is even, and 1 if the input is odd, one can
    run the following:

    >>> create_explicit_mask_from_callable(lambda x: x < 10)
    >>> return_values = [
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1
    ... ]
    >>> training_set = create_training_set_from_list(
    ...     return_values, mask, 100, noise_probability=0.5)

    The returned training set is iterable, so to get the next data point one can
    simply loop over it, like this:

    >>> for data_point in training_set:
    ...  print(data_point.input, data_point.output)
"""
    base_data_set = create_data_set_from_list(data_set_return_values, noise_probability)
    return _TrainingSet(base_data_set, mask, training_set_length, noise_probability)


def create_test_set_from_list(
        data_set_return_values: List[int],
        mask: Mask) -> DataSet:
    """
    Simplified way to create a test set from a list of return values
    representing a boolean function. Note that the list should be of length that
    is a power of two (to represent a full function), and contain only 0s and 1s.
    :param data_set_return_values: The return values of the function represented
    in the data set.
    :param mask: The mask object used to split the data set into a training set
    and a test set. Only uncovered indices will belong to the test set.
    :return: The data sets representing these parameters.

    Usage example:

    To create a test set with inputs of size 4 (such as 0000, 0001, ..., 1111),
    that only contains the first 10 inputs, and set the output function to be 0
    if the input is even, and 1 if the input is odd, one can run the following:

    >>> create_explicit_mask_from_callable(lambda x: x < 10)
    >>> return_values = [
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1,
    ...     0, 1, 0, 1
    ... ]
    >>> test_set = create_test_set_from_list(return_values, mask)

    The returned test set is iterable, so to get the next data point one can
    simply loop over it, like this:

    >>> for data_point in test_set:
    ...  print(data_point.input, data_point.output)

    """
    base_data_set = create_data_set_from_list(data_set_return_values)
    return _TestSet(base_data_set, mask)
