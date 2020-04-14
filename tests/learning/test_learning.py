from learning.data_set.constructors import create_data_set_from_list
from learning.learning import Learning
from tests.learning import TestLearningBase, modify_configurations


class TestLearning(TestLearningBase):

    @modify_configurations(50)
    def test_learning_sanity(self):
        learning = Learning(brain=self.brain, domain_size=2)

        data_set = create_data_set_from_list([0, 1, 0, 1])

        learning.sequence = self.sequence
        learning.training_set = data_set

        model = learning.create_model()
        test_results = model.test_model(data_set)

        self.assertEqual(1, test_results.accuracy)

    @modify_configurations(1)
    def test_learning_with_two_separated_models(self):
        learning = Learning(brain=self.brain, domain_size=2)

        data_set = create_data_set_from_list([0, 1, 0, 1])

        learning.sequence = self.sequence
        learning.training_set = data_set

        model1 = learning.create_model()
        model2 = learning.create_model()

        self.assertNotEqual(model1.output_area.name, model2.output_area.name)

    @modify_configurations(50)
    def test_learning_with_two_separated_models_sanity(self):
        learning = Learning(brain=self.brain, domain_size=2)
        learning.sequence = self.sequence

        data_set1 = create_data_set_from_list([0, 1, 0, 1])
        learning.training_set = data_set1
        model1 = learning.create_model()

        data_set2 = create_data_set_from_list([1, 0, 1, 0])
        learning.training_set = data_set2
        model2 = learning.create_model()

        # Models have been trained with opposite data, so we expect different results
        self.assertNotEqual(model1.run_model(0), model2.run_model(0))
