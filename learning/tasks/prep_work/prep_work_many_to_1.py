from collections import defaultdict
from contextlib import contextmanager
from functools import reduce
from tqdm import tqdm
from tabulate import tabulate

from learning.data_set.constructors import create_training_set_from_list, \
    create_explicit_mask_from_list, create_data_set_from_callable
from learning.brain_modes import BrainLearningMode
from non_lazy_brain import NonLazyBrain


HEADERS = ['input', 'same winners (as last fire for the same input)', 'intersection', 'output winners']


class PrepWork:
    def __init__(self, dimension, func_to_learn, training_set_size_func) -> None:
        super().__init__()
        self._dimension = dimension
        self._func_to_learn = func_to_learn
        self.training_set_size_func = training_set_size_func

        self._brain = self._construct_brain(dimension)
        self._outputs_list = [func_to_learn(i) for i in range(2**dimension)]
        self._data_set = create_data_set_from_callable(function=func_to_learn,
                                                       domain_size=dimension,
                                                       noise_probability=0)
        self._training_set = self._create_training_set(self._outputs_list, dimension, training_set_size_func)
        self._training_results = [HEADERS]
        self._test_results = [HEADERS]
        self._intersections = []
        self._winners = defaultdict(list)
        self._test_output_winners = {}

    @staticmethod
    def _construct_brain(dimension) -> NonLazyBrain:
        n = 10000
        k = 100
        brain = NonLazyBrain(p=0.01)
        brain.add_area('A', n, k, beta=0.05)
        for bit in range(dimension):
            brain.add_stimulus(f's{bit}_0', k)
            brain.add_stimulus(f's{bit}_1', k)
        brain.add_output_area('Output')
        return brain

    def _split_to_bits(self, input_value):
        return tuple(int(bit) for bit in self._binary(input_value))

    def _binary(self, input_value):
        return bin(input_value)[2:].zfill(self._data_set.domain_size)

    def _fire(self, brain: NonLazyBrain, input_value, brain_mode):
        stim_to_area = {f's{bit}_{value}': ['A']
                        for bit, value in enumerate(self._split_to_bits(input_value))}

        brain.project(stim_to_area=stim_to_area,
                      area_to_area={})

        for iteration in range(2):
            brain.project(stim_to_area=stim_to_area,
                          area_to_area={'A': ['A']})

        with self._set_learning_mode(brain, brain_mode):
            brain.project(stim_to_area={}, area_to_area={'A': ['Output']})

    def _train(self, brain: NonLazyBrain, input_value, output_value):
        brain.output_areas['Output'].desired_output = [output_value]
        self._fire(brain, input_value, BrainLearningMode.FORCE_DESIRED_OUTPUT)

    def _test(self, brain: NonLazyBrain, input_value):
        self._fire(brain, input_value, BrainLearningMode.PLASTICITY_OFF)

    @contextmanager
    def _set_learning_mode(self, brain, brain_mode):
        brain.learning_mode = brain_mode
        yield
        brain.learning_mode = BrainLearningMode.DEFAULT

    @staticmethod
    def _create_training_set(outputs, dimension, training_set_size_function):
        full_mask = create_explicit_mask_from_list([1] * len(outputs))
        return create_training_set_from_list(outputs, full_mask, training_set_size_function(dimension))

    @staticmethod
    def _get_last_set(list_of_sets):
        return list_of_sets[-1] if list_of_sets else set()

    def _update_data(self, data_table, input_value, same_winners, intersection, output):
        row = [self._binary(input_value), str(len(same_winners)), str(len(intersection)), str(output)]
        data_table.append(row)

    def _calculate_intersection(self, input_value, cur_winners):
        intersection = set(cur_winners)
        for i in range(len(self._outputs_list)):
            if i == input_value:
                continue
            intersection &= self._get_last_set(self._winners[i])
        self._intersections.append(intersection)
        return intersection

    def _calculate_winners(self, input_value, area_name):
        cur_winners = set(self._brain.areas[area_name].winners)
        prev_winners = self._get_last_set(self._winners[input_value])
        same_winners = cur_winners & prev_winners
        self._winners[input_value].append(cur_winners)
        return cur_winners, same_winners

    def _calculate_winners_and_intersection(self, input_value, area_name):
        cur_winners, same_winners = self._calculate_winners(input_value, area_name)
        intersection = self._calculate_intersection(input_value, cur_winners)
        return same_winners, intersection

    def _calculate_accuracy(self):
        return sum(len(self._test_output_winners[i]) == 1 and
                   self._test_output_winners[i][-1] == self._outputs_list[i]
                   for i in range(len(self._outputs_list))) \
               / len(self._outputs_list)

    def run(self):
        # Training:
        print('------------ Training ------------')
        for data_point in tqdm(self._training_set):
            self._train(self._brain, data_point.input, data_point.output)
            same_winners, intersection = self._calculate_winners_and_intersection(data_point.input, 'A')
            self._update_data(self._training_results, data_point.input, same_winners, intersection, self._brain.output_areas['Output'].winners)

        # Test:
        print('------------ Testing ------------')
        for data_point in tqdm(self._data_set):
            self._test(self._brain, data_point.input)
            self._test_output_winners[data_point.input] = self._brain.output_areas['Output'].winners
            same_winners, intersection = self._calculate_winners_and_intersection(data_point.input, 'A')
            self._update_data(self._test_results, data_point.input, same_winners, intersection, self._brain.output_areas['Output'].winners)

        print("Training:")
        print('-' * 91)
        print(tabulate(self._training_results, headers="firstrow", numalign='left', stralign='left'))
        print('-' * 91)
        print("Test:")
        print('-' * 91)
        print(tabulate(self._test_results, headers="firstrow", numalign='left', stralign='left'))
        print('-' * 91)
        accuracy = self._calculate_accuracy()
        print("Accuracy:", accuracy)
        return accuracy


def parity_func(x):
    return x % 2


def xor_bits_func(x):
    return reduce(lambda d1, d2: d1 ^ d2, [int(d) for d in list(bin(x)[2:])])


def log_training_set_funcs_generator(constants=(1, 5, 10, 20, 30)):
    for constant in constants:
        print(f'------ Yielding func: int({constant}*log(2^dim)) = int({constant}*dim) ------')
        yield lambda dimension: int(constant * dimension)


def fourth_root_training_set_funcs_generator(constants=(1, 5, 10, 20, 30)):
    for constant in constants:
        print(f'------ Yielding func: int({constant} * ((2**dim)**(1/4))) ------')
        yield lambda dimension: int(constant * ((2**dimension)**(1/4)))


def eighth_root_training_set_funcs_generator(constants=(10, 20, 30, 40, 50)):
    for constant in constants:
        print(f'------ Yielding func: int({constant} * ((2**dim)**(1/8))) ------')
        yield lambda dimension: int(constant * ((2**dimension)**(1/8)))


FUNCS_TO_LEARN = [(parity_func, 'parity'),
                   (xor_bits_func, 'xor')]
DIMS = list(range(1, 7))
NUM_RUNS = 6
TRAINING_SETS_SIZE_FUNC_GENERATORS = [log_training_set_funcs_generator,
                                      fourth_root_training_set_funcs_generator,
                                      eighth_root_training_set_funcs_generator]

if __name__ == '__main__':
    for dim in DIMS:
        print(f"Learning function of dimension {dim}...")

        for func_to_learn, func_description in FUNCS_TO_LEARN:
            print(f'--- Learning function: {func_description} ---')

            for training_set_size_func_generator in TRAINING_SETS_SIZE_FUNC_GENERATORS:
                for training_set_size_func in training_set_size_func_generator():

                    accuracy_list = []
                    for run in range(NUM_RUNS):
                        print(f"--------- Run #{run + 1 } out of {NUM_RUNS} ---------")

                        prep = PrepWork(dimension=dim,
                                        func_to_learn=func_to_learn,
                                        training_set_size_func=training_set_size_func)
                        accuracy_list.append(prep.run())
                        print('=' * 91)
                        print('\n')

                    print('=' * 91)
                    print(f'Actual accuracy results: {accuracy_list}')
                    print("Average Accuracy:", round(sum(accuracy_list) / len(accuracy_list), 3))
                    print('=' * 91)
                    print('\n')
