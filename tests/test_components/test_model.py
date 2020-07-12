from learning.components.data_set.constructors import create_data_set_from_list
from learning.components.errors import InputSizeMismatch

from learning.components.model import LearningModel
from tests.test_components.learning_component_test_base import LearningComponentTestBase


class TestLearningModel(LearningComponentTestBase):

    def test_run_model_input_allowed_range(self):
        model = LearningModel(brain=self.brain, sequence=self.sequence, input_stimuli=self.input_stimuli)
        self.assertIn(model.run_model(0), [0, 1])
        self.assertIn(model.run_model(1), [0, 1])
        self.assertIn(model.run_model(2), [0, 1])
        self.assertIn(model.run_model(3), [0, 1])
        self.assertRaises(InputSizeMismatch, model.run_model, 4)

    def test_run_model_consistency(self):
        model = LearningModel(brain=self.brain, sequence=self.sequence, input_stimuli=self.input_stimuli)

        result_00 = model.run_model(0)
        result_11 = model.run_model(3)

        result_00_2 = model.run_model(0)
        result_11_2 = model.run_model(3)

        self.assertEqual(result_11, result_11_2)
        self.assertEqual(result_00, result_00_2)

