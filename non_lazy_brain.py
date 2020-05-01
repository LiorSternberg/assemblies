from brain import Brain, Stimulus, Area, OutputArea
import logging
from typing import List, Dict
import numpy as np
import heapq
from numpy.core._multiarray_umath import ndarray

from learning.learning_stages.learning_stages import BrainLearningMode


class NonLazyBrain(Brain):
    """ Represents a simulated brain, with it's different areas, stimuli, and all the synapse weights.
        The connectomes are fully generated when adding a stimulus / area.
    """

    def add_stimulus(self, name: str, k: int) -> None:
        """ Initialize a random stimulus with 'k' neurons firing.
        This stimulus can later be applied to different areas of the brain,
        also updating its outgoing connectomes in the process.

        Connectomes to all areas is initialized as an empty numpy array.
        For every target area, which are all existing areas, set the plasticity coefficient,
        beta, to equal that area's beta.

        :param name: Name used to refer to stimulus
        :param k: Number of neurons in the stimulus
        """
        self.stimuli[name]: Stimulus = Stimulus(k)
        self.conectomes_init_stimulus(self.stimuli[name], name)

    def add_output_area(self, name: str) -> None:
        assert name not in self.areas, "Can't create an output area in this name, an area with this name already exists!"
        self.output_areas[name] = OutputArea(name)

        self.conectomes_init_output_area(self.output_areas[name], OutputArea.beta)

    def add_area(self, name: str, n: int, k: int, beta: float) -> None:
        """Add an area to this brain, randomly connected to all other areas and stimulus.

        Initialize each synapse weight to have a value of 0 or 1 with probability 'p'.
        Initialize incoming and outgoing connectomes as empty arrays.
        Initialize incoming betas as 'beta'.
        Initialize outgoing betas as the target area.beta

        :param name: Name of area
        :param n: Number of neurons in the new area
        :param k: Number of winners in the new area
        :param beta: plasticity parameter of connectomes coming INTO this area.
                The plastiity parameter of connectomes FROM this area INTO other areas are decided by
                the betas of those other areas.
        """
        assert name not in self.output_areas, "Can't create an area in this name, an output area with this name already exists!"

        self.areas[name] = Area(name, n, k, beta)

        # This should be replaced by conectomes_init_area(self, self.areas[name], beta).
        # (From here to the end of the function).
        self.conectomes_init_area(self.areas[name], beta)

    def conectomes_init_area(self, area: Area, beta: float):
        # self.connectomes: Dict[str, Dict[str, ndarray]] = {}
        # self.connectomes[area.name][other_area] = neurons: ndarray (of size (area.n, other_area.n))
        # ndarray[i][j] = weight of connectome from neuron i (in area) to neuron j (in other area)
        name = area.name
        for stim_name, stim_connectomes in self.stimuli_connectomes.items():
            stimulus: Stimulus = self.stimuli[stim_name]
            stim_connectomes[name] = np.random.binomial(1, self.p, (stimulus.k, area.n)).astype(dtype='f')
            self.areas[name].stimulus_beta[stim_name] = beta

        new_connectomes: Dict[str, ndarray] = {}
        for other_area_name, other_area in self.areas.items():
            new_connectomes[other_area_name] = np.random.binomial(1, self.p, (area.n, other_area.n)).astype(dtype='f')
            if other_area is not area:
                self.connectomes[other_area_name][name] = np.random.binomial(1, self.p, (other_area.n, area.n)).astype(
                    dtype='f')
            other_area.area_beta[name] = other_area.beta
            area.area_beta[other_area_name] = beta
        self.connectomes[name] = new_connectomes

        # Each output_area[area_beta] gets the beta of the output_area (not the area,
        # since the betas direction is defined to be the 'to beta').
        for output_area_name, output_area in self.output_areas.items():
            self.output_connectomes[name][output_area_name] = np.random.binomial(1, self.p, (area.n, output_area.n)).astype(dtype='f')
            output_area.area_beta[name] = output_area.beta

    def conectomes_init_output_area(self, area: Area, beta: float):
        # self.connectomes: Dict[str, Dict[str, ndarray]] = {}
        # self.connectomes[area.name][other_area] = neurons: ndarray (of size (area.n, other_area.n))
        # ndarray[i][j] = weight of connectome from neuron i (in area) to neuron j (in other area)
        name = area.name
        for stim_name, stim_connectomes in self.stimuli_connectomes.items():
            stimulus: Stimulus = self.stimuli[stim_name]
            stim_connectomes[name] = np.random.binomial(1, self.p, (stimulus.k, area.n)).astype(dtype='f')
            self.output_areas[name].stimulus_beta[stim_name] = beta

        for other_area_name, other_area in self.areas.items():
            self.output_connectomes[other_area_name][name] = np.random.binomial(1, self.p, (other_area.n, area.n)).astype(dtype='f')
            area.area_beta[other_area_name] = beta

    def conectomes_init_stimulus(self, stimulus: Stimulus, name: str):
        # self.stimuli_connectomes: Dict[str, Dict[str, ndarray]] = {}
        # self.connectomes[self.stimuli[name]][other_area] = neurons: ndarray (of size (stimuli.k, other_area.n))
        # ndarray[i][j] = weight of connectome from neuron i (in stimulus) to neuron j (in other area)
        new_connectomes: Dict[str, ndarray] = {}
        for area_name, area in self.areas.items():
            new_connectomes[area_name] = np.random.binomial(1, self.p, (stimulus.k, area.n)).astype(dtype='f')
            self.areas[area_name].stimulus_beta[name] = self.areas[area_name].beta
        self.stimuli_connectomes[name] = new_connectomes

        new_connectomes: Dict[str, ndarray] = {}
        for area_name, area in self.output_areas.items():
            new_connectomes[area_name] = np.random.binomial(1, self.p, (stimulus.k, area.n)).astype(dtype='f')
            self.output_areas[area_name].stimulus_beta[name] = self.output_areas[area_name].beta
        self.output_stimuli_connectomes[name] = new_connectomes

    def project_into_calculate_inputs(self, area: Area, from_stimuli: List[str], from_areas: List[str]) -> List[float]:
        """ Calculates the total input for each neuron from other given areas' winners and given stimuli.
        Said total inputs list is saved in prev_winner_inputs
        The parameters are the same as the project_into method parameters.
        """

        prev_winner_inputs: List[float] = np.zeros(area.n)

        if from_areas:
            for from_area in from_areas:
                area_connectomes = self.get_area_connectomes(from_area, area.name)
                for winner in self.areas[from_area].winners:
                    prev_winner_inputs += area_connectomes[winner]

        if from_stimuli:
            prev_winner_inputs += sum([
                np.dot(
                    np.ones(
                        self.stimuli[stim].k
                    ),
                    self.get_stimulus_connectomes(stim, area.name)
                )
                for stim in from_stimuli
            ])

        logging.debug(f'prev_winner_inputs: {prev_winner_inputs}')
        return prev_winner_inputs

    def project_into_calculate_winners(self, area: Area, inputs) -> int:
        """
        find k neurons with maximal inputs to be the new winners
        update area._new_winners, area.support and area._new_support_size
        :return: number of winners that weren't in area.support before
        """
        if self.learning_mode != BrainLearningMode.TRAINING or not isinstance(area, OutputArea):
            area._new_winners = heapq.nlargest(area.k, list(range(len(inputs))), inputs.__getitem__)
        else:
            area._new_winners = area.desired_output
        num_first_winners: int = 0
        for winner in area._new_winners:
            if not area.support[winner]:
                num_first_winners += 1
            area.support[winner] = 1
        area._new_support_size = num_first_winners + area.support_size
        logging.debug(f'new_winners: {area._new_winners}')
        return num_first_winners

    def project_into_update_connectomes(self, area: Area, from_stimuli: List[str], from_areas: List[str]) -> None:
        # connectome for each stim->area
        # for i in new_winners, stimulus_inputs[i] *= (1+beta)
        for stim in from_stimuli:
            beta = area.stimulus_beta[stim]
            for i in area._new_winners:
                for j in range(self.stimuli[stim].k):
                    self.get_stimulus_connectomes(stim, area.name)[j][i] *= (1 + beta)
            logging.debug(f'stimulus {stim} now looks like: {self.get_stimulus_connectomes(stim, area.name)}')

        # connectome for each in_area->area
        # for each i in _new_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
        for from_area in from_areas:
            from_area_winners = self.areas[from_area]._new_winners
            beta = area.area_beta[from_area]
            # connectomes of winners are now stronger
            for i in area._new_winners:
                for j in from_area_winners:
                    self.get_area_connectomes(from_area, area.name)[j][i] *= (1 + beta)
            logging.debug(f'Connectome of {from_area} to {area.name} is now {self.get_area_connectomes(from_area, area.name)}')

    def project_into(self, area: Area, from_stimuli: List[str], from_areas: List[str]) -> int:
        """Project multiple stimuli and area assemblies into area 'area' at the same time.
        :param area: The area projected into
        :param from_stimuli: The stimuli that we will be applying
        :param from_areas: List of separate areas whose assemblies we will projected into this area
        :return: Returns the number of area neurons that were winners for the first time during this projection
        """
        inputs = self.project_into_calculate_inputs(area, from_stimuli, from_areas)
        num_first_winners = self.project_into_calculate_winners(area, inputs)
        if self.learning_mode != BrainLearningMode.TESTING:
            self.project_into_update_connectomes(area, from_stimuli, from_areas)
        return num_first_winners
