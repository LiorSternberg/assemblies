from abc import abstractmethod, ABCMeta


class DataPoint(metaclass=ABCMeta):
    """
    A data point contains the information of (input, output).
    A data set is made of data points.
    """
    @property
    @abstractmethod
    def input(self):
        """
        :return: The data point's input value
        """
        pass

    @property
    @abstractmethod
    def output(self):
        """
        :return: The data point's output value
        """
        pass

