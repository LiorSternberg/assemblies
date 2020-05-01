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


class ValuesMismatch(Exception):
    def __init__(self, expected_value, actual_value):
        self._expected_value = expected_value
        self._actual_value = actual_value


class DomainSizeMismatch(ValuesMismatch):
    def __init__(self, expected_object, actual_object, expected_size: int, actual_size: int) -> None:
        super().__init__(expected_size, actual_size)
        self._expected_object = expected_object
        self._actual_object = actual_object

    def __str__(self) -> str:
        return f"The domain size of {self._actual_object} is expected to be the same as the domain size of " \
               f"{self._expected_object} (i.e. {self._expected_value}), but instead it's of size {self._actual_value}"


class StimuliMismatch(ValuesMismatch):
    def __init__(self, expected_stimuli, actual_stimuli) -> None:
        super().__init__(expected_stimuli, actual_stimuli)
        
    def __str__(self) -> str:
        return f"Number of stimuli should be {self._expected_value}. Instead, it's {self._actual_value}"


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
