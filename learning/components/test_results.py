from enum import Enum, auto
from typing import List

from learning.components.data_set.data_point import DataPoint


# TODO: Handle case of non-binary return values (can throw non-implemented error)

class ResultType(Enum):
    FALSE_NEGATIVE = auto()
    TRUE_NEGATIVE = auto()
    FALSE_POSITIVE = auto()
    TRUE_POSITIVE = auto()


class _DataPointResult:
    """
    Internal class representing a data point and the result the model predicted
    for it.
    """
    def __init__(self, data_point: DataPoint, model_prediction_result: int) -> None:
        """
        :param data_point: The data point.
        :param model_prediction_result: The predicted result outputted by the
        model.
        """
        super().__init__()
        self.data_point = data_point
        self.model_prediction_result = model_prediction_result

    def get_result_type(self) -> ResultType:
        """

        We will consider the actual model prediction result a an indicator
        whether the output is negative / positive, and the data point's
        (expected) output to decide if it's a true/false result (meaning the
        prediction is correct of not).

        :return: The result type.
        """
        if self.model_prediction_result == 0:
            if self.data_point.output == 0:
                return ResultType.TRUE_NEGATIVE

            else:  # self.data_point.output == 1:
                return ResultType.FALSE_NEGATIVE

        else:  # self.model_prediction_result == 1:
            if self.data_point.output == 0:
                return ResultType.FALSE_POSITIVE

            else:  # self.data_point.output == 1:
                return ResultType.TRUE_POSITIVE


class TestResults:
    """
    A class representing the results of testing a model against a certain data
    set.
    The results allow you to query how many data points were categorized as a
    certain ResultType (such as true positive), and also exposes the accuracy,
    precision, and recall of the model (as calculated based on the results).
    """
    def __init__(self) -> None:
        """
        Create an empty TestResult object.
        """
        super().__init__()
        self.results: List[_DataPointResult] = []

    def add_result(self, data_point: DataPoint, predicted_result) -> None:
        """
        Add a test result.
        :param data_point: The data point that was tested.
        :param predicted_result: The predicted result outputted by the model.
        """
        self.results.append(_DataPointResult(data_point, predicted_result))

    def count_filtered_results(self, *result_types) -> int:
        return len(list(filter(lambda result: result.get_result_type() in result_types, self.results)))

    @property
    def accuracy(self) -> float:
        correct = self.count_filtered_results(ResultType.TRUE_NEGATIVE, ResultType.TRUE_POSITIVE)
        return round(correct / len(self.results), 2)

    @property
    def precision(self) -> float:
        true_positives = self.count_filtered_results(ResultType.TRUE_POSITIVE)
        predicted_positives = self.count_filtered_results(ResultType.FALSE_POSITIVE, ResultType.TRUE_POSITIVE)
        return round(true_positives / predicted_positives, 2)

    @property
    def recall(self) -> float:
        true_positives = self.count_filtered_results(ResultType.TRUE_POSITIVE)
        actual_positives = self.count_filtered_results(ResultType.FALSE_NEGATIVE, ResultType.TRUE_POSITIVE)
        return round(true_positives / actual_positives, 2)
