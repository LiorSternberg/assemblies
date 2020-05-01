import unittest

from tests.brain_test_utils import BrainTestUtils


class NonLazyBrainTestBase(unittest.TestCase):

    def setUp(self) -> None:
        self.utils = BrainTestUtils(lazy=False)
