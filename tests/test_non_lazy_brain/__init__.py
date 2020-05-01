import unittest

from tests import TestBrainUtils


class TestNonLazyBrain(unittest.TestCase):

    def setUp(self) -> None:
        self.utils = TestBrainUtils(lazy=False)
