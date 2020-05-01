import unittest

from tests import TestBrainUtils


class TestLazyBrain(unittest.TestCase):

    def setUp(self) -> None:
        self.utils = TestBrainUtils(lazy=True)
