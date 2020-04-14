from learning.data_set.constructors import create_data_set_from_list
from learning.errors import DomainSizeMismatch

from learning.learning_model import LearningModel
from tests.learning import TestLearningBase, modify_configurations


class TestLearningModel(TestLearningBase):

    @modify_configurations(1)
    def test_run_model_sanity(self):
        model = LearningModel(brain=self.brain, domain_size=2, sequence=self.sequence)
        self.assertIn(model.run_model(0), [0, 1])
        self.assertIn(model.run_model(1), [0, 1])
        self.assertIn(model.run_model(2), [0, 1])
        self.assertIn(model.run_model(3), [0, 1])
        self.assertRaises(DomainSizeMismatch, model.run_model, 4)

    @modify_configurations(10)
    def test_run_model_consistency(self):
        model = LearningModel(brain=self.brain, domain_size=2, sequence=self.sequence)

        result_00 = model.run_model(0)
        result_11 = model.run_model(3)

        result_00_2 = model.run_model(0)
        result_11_2 = model.run_model(3)

        self.assertEqual(result_11, result_11_2)
        self.assertEqual(result_00, result_00_2)

    @modify_configurations(50)
    def test_train_model_sanity(self):
        model = LearningModel(brain=self.brain, domain_size=2, sequence=self.sequence)

        training_set = create_data_set_from_list([0, 1, 0, 1])

        model.train_model(training_set)
        test_results = model.test_model(training_set)
        self.assertEqual(1, test_results.accuracy)
        self.assertEqual([], test_results.false_negative)
        self.assertEqual([0, 1, 2, 3], test_results.true_positive)

    def test_convert_input_to_stimuli(self):
        model = LearningModel(brain=self.brain, domain_size=2, sequence=self.sequence)

        result_00 = model._convert_input_to_stimuli(0)
        result_01 = model._convert_input_to_stimuli(1)
        result_10 = model._convert_input_to_stimuli(2)
        result_11 = model._convert_input_to_stimuli(3)

        self.assertEqual(['A', 'C'], result_00)
        self.assertEqual(['A', 'D'], result_01)
        self.assertEqual(['B', 'C'], result_10)
        self.assertEqual(['B', 'D'], result_11)

    def test_model_termination(self):
        model = LearningModel(brain=self.brain, domain_size=2, sequence=self.sequence)

        output_area_name = model.output_area.name
        self.assertIsNotNone(self.brain.output_areas.get(output_area_name))

        model.terminate()
        self.assertIsNone(self.brain.output_areas.get(output_area_name))

        training_set = create_data_set_from_list([0, 1, 0, 1])
        self.assertRaises(Exception, model.train_model, training_set)
