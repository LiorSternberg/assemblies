from brain import OutputArea
from learning.brain_modes import BrainLearningMode
from tests.test_lazy_brain.lazy_brain_test_base import LazyBrainTestBase
from utils import get_matrix_max, get_matrix_min


class TestProject(LazyBrainTestBase):

    def test_project_from_area_to_itself(self):
        brain = self.utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1)

        area = self.utils.area0
        winners_before_projection = area.winners
        connectome_before_projection = brain.connectomes[area.name][area.name]
        self.assertEqual(self.utils.winners_size, len(winners_before_projection))
        self.assertEqual(1, get_matrix_max(connectome_before_projection))
        self.assertEqual(0, get_matrix_min(connectome_before_projection))

        brain.project(area_to_area={area.name: [area.name]}, stim_to_area={})
        connectome_after_projection = brain.connectomes[area.name][area.name]
        self.assertEqual(self.utils.winners_size, len(area.winners))
        self.assertNotEqual(connectome_after_projection, area.winners)
        self.assertAlmostEqual((1 + self.utils.beta) * 1, get_matrix_max(connectome_after_projection))
        self.assertEqual(0, get_matrix_min(connectome_after_projection))

    def test_project_from_area_to_another_area(self):
        brain = self.utils.create_and_stimulate_brain(number_of_areas=2, number_of_stimulated_areas=1)

        source_area = self.utils.area0
        target_area = self.utils.area1

        self.assertEqual([], target_area.winners)

        brain.project(area_to_area={source_area.name: [target_area.name]}, stim_to_area={})
        connectome_after_projection = brain.connectomes[source_area.name][target_area.name]
        self.assertEqual(source_area.k, len(source_area.winners))
        self.assertAlmostEqual((1 + target_area.beta) * 1, get_matrix_max(connectome_after_projection))
        self.assertEqual(0, get_matrix_min(connectome_after_projection))

    def test_project_from_area_to_output_area(self):
        brain = self.utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1,
                                                      add_output_area=True)

        origin_area = self.utils.area0
        output_area = self.utils.output_area

        self.assertEqual([], output_area.winners)

        brain.project(area_to_area={origin_area.name: [output_area.name]}, stim_to_area={})
        connectome_after_projection = brain.output_connectomes[origin_area.name][output_area.name]
        self.assertEqual(origin_area.k, len(origin_area.winners))
        self.assertEqual(output_area.k, len(output_area.winners))
        self.assertAlmostEqual((1 + output_area.beta) * 1, get_matrix_max(connectome_after_projection))
        self.assertEqual(0, get_matrix_min(connectome_after_projection))

    def test_project_from_area_to_output_area_with_size_2(self):
        # Setting up desired OutputArea size
        with self.utils.change_output_area_settings(n=2, k=1):
            OutputArea.n = 2
            OutputArea.k = 1

            brain = self.utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1,
                                                          add_output_area=True)

            origin_area = self.utils.area0
            output_area = self.utils.output_area

            self.assertEqual([], output_area.winners)

            brain.project(area_to_area={origin_area.name: [output_area.name]}, stim_to_area={})
            connectome_after_projection = brain.output_connectomes[origin_area.name][output_area.name]
            self.assertEqual(origin_area.k, len(origin_area.winners))
            self.assertEqual(1, len(output_area.winners))
            self.assertAlmostEqual((1 + output_area.beta) * 1, get_matrix_max(connectome_after_projection))
            self.assertEqual(0, get_matrix_min(connectome_after_projection))

    def test_betas_have_no_effect_only_testing_mode(self):
        brain = self.utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1,
                                                      add_output_area=True)
        origin_area = self.utils.area0
        output_area = self.utils.output_area

        brain.learning_mode = BrainLearningMode.PLASTICITY_OFF
        brain.project(area_to_area={origin_area.name: [output_area.name]}, stim_to_area={})
        connectome_after_projection = brain.output_connectomes[origin_area.name][output_area.name]
        self.assertAlmostEqual(1, get_matrix_max(connectome_after_projection))

    def test_output_in_training_mode_is_predetermined_indeed(self):
        with self.utils.change_output_area_settings(n=2, k=1):
            OutputArea.n = 2
            OutputArea.k = 1

            brain = self.utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1,
                                                          add_output_area=True)
            origin_area = self.utils.area0
            output_area = self.utils.output_area

            brain.learning_mode = BrainLearningMode.FORCE_DESIRED_OUTPUT
            output_area.desired_output = [1]
            brain.project(area_to_area={origin_area.name: [output_area.name]}, stim_to_area={})
            connectome_after_projection = brain.output_connectomes[origin_area.name][output_area.name]
            self.assertAlmostEqual(1.0, get_matrix_max(connectome_after_projection[:,0]))
            self.assertAlmostEqual(1.05, get_matrix_max(connectome_after_projection[:,1]))

    def test_project_from_stimulus_to_output_area(self):
        brain = self.utils.create_brain(number_of_areas=0, add_output_area=True)

        output_area = self.utils.output_area

        self.assertEqual([], output_area.winners)

        stimulus_name = 'stimulus'
        brain.add_stimulus(stimulus_name, k=self.utils.area_size)

        connectome_before_projection = brain.output_stimuli_connectomes[stimulus_name][output_area.name]
        self.assertEqual(0, get_matrix_max(connectome_before_projection))
        self.assertEqual(0, get_matrix_min(connectome_before_projection))

        brain.project(area_to_area={}, stim_to_area={stimulus_name: [output_area.name]})

        connectome_after_projection = brain.output_stimuli_connectomes[stimulus_name][output_area.name]
        self.assertEqual(output_area.k, len(output_area.winners))
        self.assertEqual(0, get_matrix_max(connectome_after_projection))
        self.assertEqual(0, get_matrix_min(connectome_after_projection))
