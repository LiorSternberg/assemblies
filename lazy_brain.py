from brain import Brain, Stimulus, Area, OutputArea
import logging
from typing import List, Dict, Union
import numpy as np
import heapq

from numpy.core._multiarray_umath import ndarray
from scipy.stats import binom
from scipy.stats import truncnorm
import math
import random

from learning.brain_modes import BrainLearningMode


class LazyBrain(Brain):
    """ Represents a simulated brain where the the connectomes are generated lazily, i.e. generated only when needed.

    The brain updates by selecting a subgraph of stimuli and areas, and activating only those connections.

    """

    def __init__(self, p: float):
        super().__init__(p)

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

        # This should be replaced by conectomes_init_stimulus(self, self.areas[name], name).
        # (From here to the end of the function).
        new_connectomes: Dict[str, ndarray] = {}
        for key in self.areas:
            new_connectomes[key] = np.empty((0, 0))
            self.areas[key].stimulus_beta[name] = self.areas[key].beta
        self.stimuli_connectomes[name] = new_connectomes

        new_connectomes: Dict[str, ndarray] = {}
        for area_name, area in self.output_areas.items():
            new_connectomes[area_name] = np.random.binomial(1, self.p, (area.n)).astype(dtype='f')
            self.output_areas[area_name].stimulus_beta[name] = self.output_areas[area_name].beta
        self.output_stimuli_connectomes[name] = new_connectomes

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
                The plasticity parameter of connectomes FROM this area INTO other areas are decided by
                the betas of those other areas.
        """
        self.areas[name] = Area(name, n, k, beta)

        # This should be replaced by conectomes_init_area(self, self.areas[name], beta).
        # (From here to the end of the function).
        for stim_name, stim_connectomes in self.stimuli_connectomes.items():
            stim_connectomes[name] = np.empty(0)  # TODO: Should this be np.empty((0,0))?
            self.areas[name].stimulus_beta[stim_name] = beta

        new_connectomes: Dict[str, ndarray] = {}
        for key in self.areas:
            new_connectomes[key] = np.empty((0, 0))
            if key != name:
                self.connectomes[key][name] = np.empty((0, 0))
            self.areas[key].area_beta[name] = self.areas[key].beta
            self.areas[name].area_beta[key] = beta
        self.connectomes[name] = new_connectomes

        # Each output_area[area_beta] gets the beta of the output_area (not the area,
        # since the betas direction is defined to be the 'to beta').
        for output_area_name, output_area in self.output_areas.items():
            self.output_connectomes[name][output_area_name] = \
                np.random.binomial(1, self.p, (area.n, output_area.n)).astype(dtype='f')
            output_area.area_beta[name] = output_area.beta

    def conectomes_init_output_area(self, area: Area, beta: float):
        name = area.name
        for stim_name, stim_connectomes in self.stimuli_connectomes.items():
            stimulus: Stimulus = self.stimuli[stim_name]
            stim_connectomes[name] = np.random.binomial(1, self.p, (stimulus.k, area.n)).astype(dtype='f')
            self.output_areas[name].stimulus_beta[stim_name] = beta

        for other_area_name, other_area in self.areas.items():
            self.output_connectomes[other_area_name][name] = np.random.binomial(1, self.p, (other_area.n, area.n)).astype(dtype='f')
            area.area_beta[other_area_name] = beta

    def project_into(self, area: Union[Area, OutputArea], from_stimuli: List[str], from_areas: List[str]) -> int:
        """Project multiple stimuli and area assemblies into area 'area' at the same time.

        :param area: The area projected into
        :param from_stimuli: The stimuli that we will be applying
        :param from_areas: List of separate areas whose assemblies we will projected into this area
        :return: Returns the number of area neurons that were winners for the first time during this projection
        """
        # projecting everything in from stim_in[area] and area_in[area]
        # calculate: inputs to self.connectomes[area] (previous winners)
        # calculate: potential new winners, Binomial(sum of in sizes, k-top)
        # k top of previous winners and potential new winners
        # if new winners > 0, redo connectome and intra_connectomes
        # have to wait to replace new_winners
        # TODO Handle case of projecting from an area without previous winners.
        # TODO: there is a bug when adding a new stimulus later on.
        # TODO: Stimulus is updating to somehow represent >100 neurons.
        logging.info(f'Projecting {",".join(from_stimuli)} and {",".join(from_areas)} into area.name')

        def calc_prev_winners_input() -> List[float]:
            """
            Creates a list of size support_size
            prev_winners_input[i] := sum of all incoming weights into neuron #i (0 <= i < support_size),
            which can be coming from both stimuli and areas
            :return: prev_winner_inputs: List[float]
            """
            prev_winner_inputs: List[float] = [0.] * area.support_size
            for stim in from_stimuli:
                stim_inputs = self.get_stimulus_connectomes(stim, area.name)
                for i in range(area.support_size):
                    prev_winner_inputs[i] += stim_inputs[i]
            for from_area in from_areas:
                connectome = self.get_area_connectomes(from_area, area.name)
                for w in self.areas[from_area].winners:
                    for i in range(area.support_size):
                        prev_winner_inputs[i] += connectome[w][i]
            logging.debug(f'prev_winner_inputs: {prev_winner_inputs}')
            return prev_winner_inputs

        def calculate_input_sizes() -> List[int]:
            """
            input_sizes := a list containing all stimuli sizes, followed by all incoming areas winner counts
            [[[indexed in the same way as from_areas. TODO: does it makes sense?]]]]
            :return: input_sizes: List[int]
            """
            input_sizes: List[int] = [self.stimuli[stim].k for stim in from_stimuli]
            input_sizes += [self.areas[from_area].k for from_area in from_areas]
            logging.debug(f'total_k = {sum(input_sizes)} and input_sizes = {input_sizes}')
            return input_sizes

        def calc_potential_new_winners(total_k: int) -> List[float]:
            """
            Calculate list of potential new winners
            We take a normal distribution centered around p * (incoming count) and truncated at
            [the probability that the number of neurons that aren't going to fire in the area will be lower than
            p * (number of neurons in the area that never fired)]
            and [incoming count] and sample [new winner count] of them.
            we return the samples rounded to the nearest integer as the list `potential_new_winners`
            :param total_k: sum of the number of winner in each upstream stimulus/area.
            :return: List of potential new winners (represented as a list of integers = input sizes)
            """
            # effective_n := Number of neurons that never fired in the area
            effective_n = area.n - area.support_size
            if not effective_n:
                return []
            # Threshold for inputs that are above (n-k)/n percentile. alpha is the smallest number such that:
            # Pr(Bin(total_k,self.p) <= alpha) >= (effective_n-area.k)/effective_n
            # A.k.a the probability that the number of neurons that aren't going to fire in the area will be lower than
            # p * (number of neurons in the area that never fired)
            alpha = binom.ppf((float(effective_n - area.k) / effective_n), total_k, self.p)
            logging.debug(f'Alpha = {alpha}')
            # Std(Binomial(n,p)) := Sqrt(n * p * (1-p))
            std = math.sqrt(total_k * self.p * (1.0 - self.p))
            mu = total_k * self.p
            a = float(alpha - mu) / std
            b = float(total_k - mu) / std  # note that b>=a and corresponds to the maximum value of Bin(total_k,self.p)
            # potential_new_winners := area.k samples of the normal distribution truncated in the range [a,b] and
            # translated by mu, all divided by std
            potential_new_winners = truncnorm.rvs(a, b, scale=std, loc=mu, size=area.k)
            for i in range(area.k):
                potential_new_winners[i] = float(round(potential_new_winners[i]))
            logging.debug(f'potential_new_winners: {potential_new_winners}')
            return potential_new_winners.tolist()

        def calc_new_winners(prev_winner_inputs: List[float], potential_new_winners: List[float]) -> List[float]:
            """
            find area.k maximal values in both - these are the new winners.
            find the ones that are winners for the first time.
            update area._new_winners and area._new_support_size
            :param prev_winner_inputs:
            :param potential_new_winners:
            :return: list of inputs of the new winners that weren't winners before (represented as a list of floats)
            """
            # take max among prev_winner_inputs, potential_new_winners
            # get num_first_winners (think something small)
            # can generate area._new_winners, note the new indices
            both = prev_winner_inputs + potential_new_winners
            if self.learning_mode != BrainLearningMode.FORCE_DESIRED_OUTPUT or not isinstance(area, OutputArea):
                new_winner_indices = heapq.nlargest(area.k, list(range(len(both))), both.__getitem__)
            else:
                new_winner_indices = area.desired_output  # For purposes of learning
            num_first_winners = 0
            first_winner_inputs = []
            for i in range(area.k):
                if new_winner_indices[i] >= area.support_size:  # winner for the first time
                    # index in potential_new_winners - a new assembly neuron
                    first_winner_inputs.append(potential_new_winners[new_winner_indices[i] - area.support_size])
                    new_winner_indices[i] = area.support_size + num_first_winners
                    num_first_winners += 1
            area._new_winners = new_winner_indices  # Note that from here on 'new_winner_indices' is not in use.
            area._new_support_size = area.support_size + num_first_winners
            logging.debug(f'new_winners: {area._new_winners}')
            return first_winner_inputs

        def calculate_first_winner_to_inputs(num_first_winners: int, input_sizes: List[int]) -> Dict[int, ndarray]:
            """
            Calculates first_winner_to_inputs
            first_winner_to_inputs := for each first winner i, first_winner_to_inputs[i] is a list of the number
            of inputs from each stimuli / area, randomly generated
            :param num_first_winners: the number of first winners.
            :param input_sizes: a list containing all stimuli sizes, followed by all incoming areas winner counts
            :returns: first_winner_to_inputs
            """
            # for i in num_first_winners
            # generate where input came from
            # 	1) can sample input from array of size total_k, use ranges
            # 	2) can use stars/stripes method: if m total inputs, sample (m-1) out of total_k
            first_winner_to_inputs: Dict[int, ndarray] = {}
            for i in range(num_first_winners):
                # first_winner_inputs[i] - how many fired into first winner # i
                # we randomize the indices that fired
                input_indices = random.sample(range(0, total_k), int(first_winner_inputs[i]))
                # inputs := a randomized array of the input size from each stimuli / area
                inputs: ndarray = np.zeros(len(input_sizes))
                total_so_far = 0
                for j in range(len(input_sizes)):
                    # divide the random indices to the different inputs, each input receives an amount of
                    # input indices proportional to its size ("on average")
                    inputs[j] = sum([(total_so_far <= w < (total_so_far + input_sizes[j])) for w in input_indices])
                    total_so_far += input_sizes[j]
                first_winner_to_inputs[i] = inputs
                logging.debug(f'for first_winner #{i} with input {first_winner_inputs[i]} split as so: {inputs}')
            return first_winner_to_inputs

        def calculate_new_stim_area_connectomes(num_first_winners: int,
                                                first_winner_to_inputs: Dict[int, ndarray]) -> None:
            """
            connectome for each stim->area
            add num_first_winners cells, sampled input * (1+beta)
            for i in repeat_winners, stimulus_inputs[i] *= (1+beta)
            :param num_first_winners: number of new neurons that won (these connectomes were not generated yet)
            :param first_winner_to_inputs: a list of the number of inputs from each stimuli / area
            :return: none
            """
            nonlocal input_index
            for stim in from_stimuli:
                if num_first_winners > 0 and area.name in self.areas:
                    # resize connectomes stim->area to the new support size
                    self.stimuli_connectomes[stim][area.name] = np.resize(self.stimuli_connectomes[stim][area.name],
                                                                          area.support_size + num_first_winners)
                # connectomes["first winner"] = how many fired from stim to this first winner
                for i in range(num_first_winners):
                    self.get_stimulus_connectomes(stim, area.name)[area.support_size + i] = \
                        first_winner_to_inputs[i][input_index]
                beta = area.stimulus_beta[stim]

                if self.learning_mode != BrainLearningMode.PLASTICITY_OFF:  # the connectomes should not update when we wish the look at a frozen "snapshot" of the brain (e.g., to assess whether it had succeeded in learning something)
                    # connectomes of winners are now stronger
                    for i in area._new_winners:
                        self.get_stimulus_connectomes(stim, area.name)[i] *= (1 + beta)
                logging.debug(f'stimulus {stim} now looks like: {self.get_stimulus_connectomes(stim, area.name)}')
                input_index += 1

        def calculate_new_from_area_area_connectomes(num_first_winners: int,
                                                     first_winner_to_inputs: Dict[int, ndarray]) -> None:
            """
            connectome for each in_area->area
            add num_first_winners columns
            for each i in num_first_winners, fill in (1+beta) for chosen neurons
            for each i in repeat_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
            :param num_first_winners: number of new neurons that won (these connectomes were not generated yet)
            :param first_winner_to_inputs: a list of the number of inputs from each stimuli / area
            :return: none
            """
            nonlocal input_index
            for from_area in from_areas:
                from_area_support = self.areas[from_area].support_size
                from_area_winners = self.areas[from_area].winners
                if area.name in self.areas:
                    # add num_first_winners columns to the connectomes
                    self.connectomes[from_area][area.name] = np.pad(self.connectomes[from_area][area.name],
                                                                    ((0, 0), (0, num_first_winners)),
                                                                    'constant', constant_values=0)
                for i in range(num_first_winners):
                    # total_in - how many fired from from_area to this first winner (i)
                    total_in = first_winner_to_inputs[i][input_index]
                    # randomize which winners in from_area fired to i
                    sample_indices = random.sample(from_area_winners, int(total_in))
                    for j in range(from_area_support):
                        # j that fired has connectome with weight 1 (in prob 1)
                        if j in sample_indices:
                            self.get_area_connectomes(from_area, area.name)[j][area.support_size + i] = 1
                        # j that is not winner has connectome weight 1 in prob p
                        if j not in from_area_winners:
                            self.get_area_connectomes(from_area, area.name)[j][area.support_size + i] = \
                                np.random.binomial(1, self.p)
                        # j that is a winner and did not fire has connectome 0 (since otherwise, it would fire)

                beta = area.area_beta[from_area]
                if self.learning_mode != BrainLearningMode.PLASTICITY_OFF:
                    # connectomes of winners are now stronger
                    for i in area._new_winners:
                        for j in from_area_winners:
                            self.get_area_connectomes(from_area, area.name)[j][i] *= (1.0 + beta)
                logging.debug(f'Connectome of {from_area} to {area.name} is now '
                              f'{self.get_area_connectomes(from_area, area.name)}')
                input_index += 1

        def calculate_new_all_area_area_connectomes(num_first_winners: int) -> None:
            """
            expand connectomes from other areas that did not fire into area
            also expand connectome for area->other_area
            :param num_first_winners: number of new neurons that won (these connectomes were not generated yet)
            :return: none
            """
            for other_area in self.areas:
                # expand the other_area->area connectomes for areas that did not fire
                if other_area not in from_areas:
                    # add num_first_winners columns to self.connectomes[other_area][name]
                    if area.name not in self.output_areas:
                        self.connectomes[other_area][area.name] = np.pad(self.connectomes[other_area][area.name],
                                                                         ((0, 0), (0, num_first_winners)), 'constant',
                                                                         constant_values=0)
                    for j in range(self.areas[other_area].support_size):
                        for i in range(area.support_size, area._new_support_size):
                            # for all new neurons in support, add connectome from other_area with weight 1 in prob p
                            self.get_area_connectomes(other_area, area.name)[j][i] = np.random.binomial(1, self.p)

                if area.name not in self.output_areas:
                    # expand the area->other_area connectomes for all areas
                    # for all new neurons in support, add connectomes to other areas with weight 1 in prob p
                    self.connectomes[area.name][other_area] = np.pad(self.connectomes[area.name][other_area],
                                                                     ((0, num_first_winners), (0, 0)), 'constant',
                                                                     constant_values=0)
                    columns = len(self.connectomes[area.name][other_area][0])
                    for i in range(area.support_size, area._new_support_size):
                        for j in range(columns):
                            self.connectomes[area.name][other_area][i][j] = np.random.binomial(1, self.p)
                    logging.debug(f'Connectome of {area.name} to {other_area} is now: '
                                  f'{self.connectomes[area.name][other_area]}')

        prev_winner_inputs: List[float] = calc_prev_winners_input()
        input_sizes = calculate_input_sizes()
        total_k = sum(input_sizes)
        potential_new_winners = calc_potential_new_winners(total_k)
        first_winner_inputs = calc_new_winners(prev_winner_inputs, potential_new_winners)
        num_first_winners = len(first_winner_inputs)
        first_winner_to_inputs: Dict[int, ndarray] = calculate_first_winner_to_inputs(num_first_winners, input_sizes)
        input_index = 0
        calculate_new_stim_area_connectomes(num_first_winners, first_winner_to_inputs)
        calculate_new_from_area_area_connectomes(num_first_winners, first_winner_to_inputs)
        calculate_new_all_area_area_connectomes(num_first_winners)

        return num_first_winners
