from pprint import pformat
from typing import Union, List, Dict, Tuple

from brain import Brain
from learning.components.errors import MissingArea, MissingStimulus, MaxAttemptsToGenerateStimuliReached

AUTO_GENERATED_STIMULUS_NAME_FORMAT = "__s_{area_name}_{input_bit_value}{postfix}"
AUTO_GENERATED_STIMULUS_NAME_POSTFIX = "_({})"

# This is the limits for the number of times we try to auto-generate a stimulus
# name that doesn't already exist in the brain. Feel free to change it.
MAX_ATTEMPTS = 100


class InputBitStimuli:
    """
    An object representing a pair of stimuli which match a specific bit in the input,
    and which are meant to fire into a certain set of brain areas.
    """
    def __init__(self, stimulus_for_0: str, stimulus_for_1: str, target_areas: List[str]) -> None:
        super().__init__()
        self._stimulus_for_0 = stimulus_for_0
        self._stimulus_for_1 = stimulus_for_1
        self._target_areas = target_areas

    @property
    def stimulus_for_0(self) -> str:
        return self._stimulus_for_0

    @property
    def stimulus_for_1(self) -> str:
        return self._stimulus_for_1

    @property
    def target_areas(self) -> List[str]:
        return self._target_areas

    def __getitem__(self, item) -> str:
        if item not in (0, 1):
            raise IndexError(
                f"Stimulus bit only supports binary inputs (of base 2), so "
                f"possible input values for a single bit can be 0 or 1 "
                f"({item} is out of range)"
            )

        if item == 0:
            return self.stimulus_for_0
        else:  # item == 1
            return self.stimulus_for_1

    def __repr__(self) -> str:
        return f"<InputBitStimuli(0: {self.stimulus_for_0}, 1: {self.stimulus_for_1}, target: {self.target_areas})>"


class InputStimuli:
    """
    An object representing a the mapping between input bits and their pair of stimuli.
    Each pair of stimuli is also associated with a set of brain areas that these stimuli can fire
    to (and only to this set).
    The definition of a new InputStimuli creates auto-generated stimuli where needed, or uses
    the provided override stimuli.
    Note that is a limit of how many auto-generated stimuli the InputStimuli will try to create
    in a given brain before reaching a max attempt limit if all the names it attempted to use were
    already in use. This limit can be changed if needed.
    """
    def __init__(self, brain: Brain, stimulus_k: int, *area_names: Union[str, List[str]],
                 override: Dict[int, Tuple[str, str]] = None, verbose=True) -> None:
        """
        :param brain: the brain to create the input stimuli in.
        :param stimulus_k: the number of firing neurons of an auto-generated input stimulus.
        :param area_names: the target area names to associate the stimuli pairs to. can be a string (the name of a
        single brain area) or a list of names.
        :param override: an optional override dictionary of bits to their existing stimuli pair to use instead of an
        auto-generated pair of stimuli.
        :param verbose: if True, prints the created InputStimuli object. True by default.

        Examples:
            input_stimuli = InputStimuli(brain, k, 'A', 'B', 'C')

            <InputStimuli(length=3, bits_mapping_to_stimuli={
                0: (__s_A_0, __s_A_1) -> ['A'],
                1: (__s_B_0, __s_B_1) -> ['B'],
                2: (__s_C_0, __s_C_1) -> ['C']
            })>

            input_stimuli = InputStimuli(brain, k, 'A', 'B', ['A', 'B'], override={1: ('s0', 's1')})

            <InputStimuli(length=3, bits_mapping_to_stimuli={
                0: (__s_A_0, __s_A_1) -> ['A'],
                1: (s0, s1) -> ['B'],
                2: (__s_A_B_0, __s_A_B_1) -> ['A', 'B']
            })>

            input_stimuli = InputStimuli(brain, k, 'A', 'B', ['A', 'B'], 'C', ['A', 'C'], ['A', 'B', 'C'])

            <InputStimuli(length=6, bits_mapping_to_stimuli={
                0: (__s_A_0, __s_A_1) -> ['A'],
                1: (__s_B_0, __s_B_1) -> ['B'],
                2: (__s_A_B_0, __s_A_B_1) -> ['A', 'B'],
                3: (__s_C_0, __s_C_1) -> ['C'],
                4: (__s_A_C_0, __s_A_C_1) -> ['A', 'C'],
                5: (__s_A_B_C_0, __s_A_B_C_1) -> ['A', 'B', 'C']
            })>
        """
        super().__init__()
        self._input_bits: List[InputBitStimuli] = self._generate_input_bits(brain, stimulus_k, area_names, override)
        if verbose:
            print(self)

    def __len__(self) -> int:
        return len(self._input_bits)

    def __getitem__(self, item):
        return self._input_bits[item]

    def __repr__(self) -> str:
        bits_mapping_to_stimuli = ',\n'.join(f"\t{i}: ({input_bit.stimulus_for_0}, {input_bit.stimulus_for_1})"
                                             f" -> {input_bit.target_areas}"
                                             for i, input_bit in enumerate(self._input_bits))
        return f"<InputStimuli(length={len(self)}, " \
               f"bits_mapping_to_stimuli={{\n{bits_mapping_to_stimuli}\n}})>" \

    @staticmethod
    def _validate_area_names_item(brain, area_names_item) -> None:
        if isinstance(area_names_item, str):
            if area_names_item not in brain.areas:
                raise MissingArea(area_names_item)

        elif isinstance(area_names_item, list) and all(isinstance(area_name, str) for area_name in area_names_item):
            for area_name in area_names_item:
                if area_name not in brain.areas:
                    raise MissingArea(area_name)

        else:
            raise TypeError(f"Area name must be a string or a list of strings, "
                            f"got {type(area_names_item).__name__} instead.")

    @staticmethod
    def _validate_override_input_bit(brain, override_input_bit) -> None:
        if not isinstance(override_input_bit, tuple):
            raise TypeError(f"Override input bit (pair of stimuli) must be a tuple, "
                            f"got {type(override_input_bit).__name__} instead.")

        if len(override_input_bit) != 2:
            raise ValueError(f"Override input bit must have exactly 2 stimuli names, "
                             f"the first representing the stimulus for input bit = 0, "
                             f"and the second for input bit = 1. "
                             f"Got {len(override_input_bit)} items instead.")

        for stimulus_name in override_input_bit:
            if stimulus_name not in brain.stimuli:
                raise MissingStimulus(stimulus_name)

    @staticmethod
    def _format_stimulus_name(area_name, input_bit_value, attempt_counter):
        if attempt_counter == 1:
            postfix = ''
        else:
            postfix = AUTO_GENERATED_STIMULUS_NAME_POSTFIX.format(attempt_counter)

        return AUTO_GENERATED_STIMULUS_NAME_FORMAT.format(
            area_name=area_name, input_bit_value=input_bit_value, postfix=postfix)

    def _get_available_stimulus_name(self, brain: Brain, area_name: Union[str, List[str]], input_bit_value: int) -> str:
        assert input_bit_value in (0, 1)

        if isinstance(area_name, list):
            area_name = '_'.join(area_name)

        attempt_counter = 1
        stimulus_name = self._format_stimulus_name(area_name, input_bit_value, attempt_counter)
        while stimulus_name in brain.stimuli:
            if attempt_counter == MAX_ATTEMPTS:
                raise MaxAttemptsToGenerateStimuliReached()

            attempt_counter += 1
            stimulus_name = self._format_stimulus_name(area_name, input_bit_value, attempt_counter)

        return stimulus_name

    def _generate_input_bits(self, brain, stimulus_k, area_names, override) -> List[InputBitStimuli]:
        input_bits = []

        for bit, area_names_item in enumerate(area_names):
            self._validate_area_names_item(brain, area_names_item)

            if override and bit in override:
                self._validate_override_input_bit(brain, override[bit])
                input_bits.append(InputBitStimuli(override[bit][0], override[bit][1], list(area_names_item)))

            else:
                stimulus_0 = self._get_available_stimulus_name(brain, area_names_item, input_bit_value=0)
                stimulus_1 = self._get_available_stimulus_name(brain, area_names_item, input_bit_value=1)
                brain.add_stimulus(name=stimulus_0, k=stimulus_k)
                brain.add_stimulus(name=stimulus_1, k=stimulus_k)
                input_bits.append(InputBitStimuli(stimulus_0, stimulus_1, list(area_names_item)))

        return input_bits
