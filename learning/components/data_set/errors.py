class DataSetValueError(Exception):
    def __init__(self, input_value, output_value, *args: object) -> None:
        super().__init__(*args)
        self._input_value = input_value
        self._output_value = output_value

    def __str__(self) -> str:
        return f"Received an invalid value for a boolean function data set: " \
               f"input {self._input_value} returned value " \
               f"{self._output_value}, which is not a boolean value."


class DataSetSizeError(Exception):
    def __init__(self, length, *args: object) -> None:
        super().__init__(*args)
        self._length = length

    def __str__(self) -> str:
        return f"Return values list must be of proper length (a power of 2), " \
               f"got list of length {self._length}"


class MaskValueError(Exception):
    def __init__(self, index, value, *args: object) -> None:
        super().__init__(*args)
        self._index = index
        self._value = value

    def __str__(self) -> str:
        return f"Received an invalid value for a boolean mask: " \
               f"at index {self._index} returned value " \
               f"{self._value}, which is not a boolean value."


class MaskIndexError(Exception):
    def __init__(self, index, length, *args: object) -> None:
        super().__init__(*args)
        self._index = index
        self._length = length

    def __str__(self) -> str:
        return f"Mask index out of range: can't access mask at index " \
               f"{self._index}, mask only contains value for indices 0 to " \
               f"{self._length - 1}"


class InvalidFunctionError(Exception):
    def __init__(self, num_of_params, *args: object) -> None:
        super().__init__(*args)
        self._num_of_params = num_of_params

    def __str__(self) -> str:
        return f"Function must receive exactly one argument! Given function " \
               f"expects {self._num_of_params} arguments."


