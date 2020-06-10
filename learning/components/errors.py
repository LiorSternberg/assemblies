from typing import List


class MissingItem(Exception):
    def __init__(self, item_name: str) -> None:
        self._item_name = item_name


class MissingStimulus(MissingItem):
    def __str__(self) -> str:
        return f"Stimulus of name {self._item_name} doesn't exist in the configured brain"


class MissingArea(MissingItem):
    def __str__(self) -> str:
        return f"Area of name {self._item_name} doesn't exist in the configured brain"


class ItemNotInitialized(Exception):
    def __init__(self, item_name):
        self._item_name = item_name

    def __str__(self) -> str:
        return f"{self._item_name} must be initialized first"


class SequenceRunNotInitialized(Exception):
    def __str__(self) -> str:
        return f"The learning sequence instance must be reset before starting to iterate over it"


class InputSizeMismatch(Exception):
    def __init__(self, expected_object, actual_object, expected_size: int, actual_size: int) -> None:
        self._expected_object = expected_object
        self._actual_object = actual_object
        self._expected_size = expected_size
        self._actual_size = actual_size

    def __str__(self) -> str:
        return f"The input size of {self._actual_object} is expected to be the same as the input size of " \
               f"{self._expected_object} ({self._expected_size}), but instead it's of size {self._actual_size}"


class InputStimuliAndSequenceMismatch(Exception):
    def __init__(self, input_expected_length, input_bit_index) -> None:
        self._input_expected_length = input_expected_length
        self._input_bit_index = input_bit_index

    def __str__(self) -> str:
        return f"Can't use a sequence which includes input bit {self._input_bit_index} when the input stimuli object " \
               f"is defined for inputs of length {self._input_expected_length}."


class InputStimuliMisused(Exception):
    def __init__(self, input_bit_index, expected_areas, actual_areas) -> None:
        self._input_bit_index = input_bit_index
        self._expected_areas = expected_areas
        self._actual_areas = actual_areas

    def __str__(self) -> str:
        return f"Input stimuli of bit {self._input_bit_index} was defined to fire to areas {self._expected_areas}. " \
               f"It cannot be used fire to a different set of areas (attempted firing to {self._actual_areas})."


class SequenceFinalizationError(Exception):

    def __str__(self) -> str:
        return "Sequence has already been finalized"


class NoPathException(Exception):
    def __init__(self, stimulus, output_area):
        self._stimulus = stimulus
        self._output_area = output_area

    def __str__(self) -> str:
        return f"A projection path between stimulus {self._stimulus} and output area {self._output_area} doesn't exist"


class IllegalOutputAreasException(Exception):
    def __init__(self, output_areas: List[str]):
        self._output_areas = output_areas

    def __str__(self) -> str:
        if len(self._output_areas) == 0:
            return "An output area must be part of the sequence"
        return f"Found {len(self._output_areas)} output areas ({','.join(self._output_areas)}), while there can " \
               f"only be one"


class MaxAttemptsToGenerateStimuliReached(Exception):
    def __str__(self) -> str:
        return "Failed auto-generating stimuli for the input, there are too many stimuli with the same name format " \
               "as the auto-generated stimulus name format (couldn't find an available name which is not already in " \
               "use). Try removing un-used stimuli or raising the max attempts limit."
