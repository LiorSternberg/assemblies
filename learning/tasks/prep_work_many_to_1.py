from collections import defaultdict
from contextlib import contextmanager
from tabulate import tabulate

from learning.data_set.constructors import create_training_set_from_list, \
    create_explicit_mask_from_list, create_data_set_from_list
from learning.learning_stages.learning_stages import BrainLearningMode
from non_lazy_brain import NonLazyBrain


HEADERS = ['input', 'same winners (as last fire for the same input)', 'intersection', 'output winners']


class PrepWork:
    def __init__(self, dimension) -> None:
        super().__init__()
        self._brain = self._construct_brain(dimension)
        self._outputs_list = [0, 1] * (2 ** (dimension - 1))
        self._data_set = create_data_set_from_list(self._outputs_list)
        self._training_set = self._create_training_set(self._outputs_list, dimension)
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
        self._fire(brain, input_value, BrainLearningMode.TRAINING)

    def _test(self, brain: NonLazyBrain, input_value):
        self._fire(brain, input_value, BrainLearningMode.TESTING)

    @contextmanager
    def _set_learning_mode(self, brain, brain_mode):
        brain.learning_mode = brain_mode
        yield
        brain.learning_mode = BrainLearningMode.DEFAULT

    @staticmethod
    def _create_training_set(outputs, dimension):
        full_mask = create_explicit_mask_from_list([1] * len(outputs))
        return create_training_set_from_list(outputs, full_mask, 40 * (dimension ** 0.5))

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
        for data_point in self._training_set:
            self._train(self._brain, data_point.input, data_point.output)
            same_winners, intersection = self._calculate_winners_and_intersection(data_point.input, 'A')
            self._update_data(self._training_results, data_point.input, same_winners, intersection, self._brain.output_areas['Output'].winners)

        # Test:
        for data_point in self._data_set:
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


if __name__ == '__main__':
    for dim in range(3, 10):
        accuracy_list = []
        print(f"Learning function of dimension {dim}...")
        for run in range(6):
            prep = PrepWork(dimension=dim)
            accuracy_list.append(prep.run())
            print('=' * 91)
            print('\n')

        print('=' * 91)
        print("Average Accuracy:", round(sum(accuracy_list) / len(accuracy_list), 3))
        print('=' * 91)
        print('\n')
