""" Configurable brain assembly model for simulations and research.
Author: Daniel Mitropolsky, 2018

This module contains classes to represent different elements of a brain simulation:
	- Area - Represents an individual area of the brain, with the relevant parameters.
	- Connectomes are the connections between neurons. They have weights, which are initialized randomly but
		due to plasticity they can updated every time some neuron fires. These weights are represented by numpy arrays.
		The ones that are not random, because they were influenced by previous projections, are referred to as the 'support'.
	- Winners in a given 'round' are the specific neurons that fired in that round.
		In any specific area, these will be the 'k' neurons with the highest value flown into them.
		These are also the only neurons whose connectome weights get updated.
	- Stimulus - Represents a random stimulus that can be applied to any part of the brain.
		When a stimulus is created it is initialized randomly, but when applied multiple times this will change.
		This is equivalent to k neurons from an unknown part of the brain firing and their (initially, random)
		connectomes decide how this stimulus affects a given area of the brain.
	- Brain - A class representing a simulated brain, with it's different areas, stimulus, and all the connectome weights.
		A brain is initialized as a random graph, and it is maintained in a 'sparse' representation,
		meaning that all neurons that have their original, random connectome weights (0 or 1) are not saved explicitly.
	- Assembly - TODO define and express in code
"""
import logging
from typing import List, Mapping, Tuple, Dict, Any
import numpy as np
import heapq
from collections import defaultdict

from numpy.core._multiarray_umath import ndarray
from scipy.stats import binom
from scipy.stats import truncnorm
import math
import random


class Stimulus:
	""" Represents a random stimulus that can be applied to any part of the brain.
	That is, a specific set of k neurons that fire together that do not reside in
	any of the brain areas. These k neurons can be though of as representing a
	specific input stimulus.
	A stimulus can be connected to any areas and each pair of stimulus neuron and
	an area neuron initially have a synapse with probability p (Brain.p).

	The data for the synaptic weights is found in the downstream areas (the areas
	that this stimulus goes into).

	Attributes:
		k: number of neurons that fire
	"""
	def __init__(self, k: int):
		self.k = k


class Area:
	"""Represents an individual area of the brain.

	The list of neurons that are firing is given by 'winners'. It is updated through application of 'Brain.project',
	where the set of '_new_winners' is calculated and only updated once all brain areas settle on their new winners.
	The winners are the 'k' neurons with the highest value going in each round.

	Initially, most computation are represented implicitly. Winners are represented explicitly, and the changes in
	their incoming synapses are maintained in 'Brain.connectomes' and 'Brain.stimuli_connectomes'. The explicit neurons
	are represented by indices starting with 0 up to 'support_size'-1.

	Since it is initialized randomly, all the programmer needs to provide for initialization is the number 'n' of neurons,
	number 'k' of winners in any given round (meaning the k neurons with heights values will fire),
	and the parameter 'beta' of plasticity controlling connectome weight updates.

	TODO: remove '_new_winners'.
	TODO: remove 'name'. We prefer to use variable names to refer to areas.

	Attributes:
		n: number of neurons in this brain area
		k: number of winners in each round
		beta: plasticity parameter for self-connections
		stimulus_beta: plasticity parameters for connections from each incoming stimulus
		area_beta: plasticity parameters for connections from each incoming area
		support_size: The number of neurons that are represented explicitly (= total number of previous winners)
		winners: List of current winners. That is, 'k' top neurons from previous round.
		_new_support_size: the size of the support for the new update. Should be 'support_size' + 'num_first_winners'.
		_new_winners: During the projection process, a new set of winners is formed. The winners are only
			updated when the projection ends, so that the newly computed winners won't affect computation
		num_first_winners: should be equal to 'len(_new_winners)'
	"""

	def __init__(self, name: str, n: int, k: int, beta: float = 0.05):
		self.name = name
		self.n = n
		self.k = k
		self.beta = beta
		self.stimulus_beta: Dict[str, float] = {}
		self.area_beta: Dict[str, float] = {}
		self.support_size: int = 0
		self.winners: List[int] = []
		self._new_support_size: int = 0
		self._new_winners: List[int] = []
		self.num_first_winners: int = -1

	def update_winners(self) -> None:
		""" This function updates the list of winners for this area after a projection step.

			TODO: redesign this so that the list of new winners is not saved in area.
		"""
		self.winners = self._new_winners
		self.support_size = self._new_support_size


class Brain:
	"""Represents a simulated brain, with it's different areas, stimuli, and all the synapse weights.

	The brain updates by selecting a subgraph of stimuli and areas, and activating only those connections.


	Attributes:
		areas: A mapping from area names to Area objects representing them.
		stimuli: A mapping from stimulus names to Stimulus objects representing them.
		stimuli_connectomes: Maps each pair of (stimulus,area) to the ndarray representing the synaptic weights among
			stimulus neurons and neurons in the support of area.
		connectomes: Maps each pair of areas to the ndarray representing the synaptic weights among neurons in
			the support.
		p: Probability of connectome (edge) existing between two neurons (vertices)
	"""

	def __init__(self, p: float):
		self.areas: Dict[str, Area] = {}
		self.stimuli: Dict[str, Stimulus] = {}
		self.stimuli_connectomes: Dict[str, Dict[str, ndarray]] = {}
		self.connectomes: Dict[str, Dict[str, ndarray]] = {}
		self.p: float = p

	def add_stimulus(self, name: str, k: int) -> None:
		""" Initialize a random stimulus with 'k' neurons firing.
		This stimulus can later be applied to different areas of the brain,
		also updating its outgoing connectomes in the process.

		Connectomes to all areas is initialized as an empty numpy array.
		For every target area, which are all existing areas, set the plasticity coefficient, beta, to equal that area's beta.

		:param name: Name used to refer to stimulus
		:param k: Number of neurons in the stimulus
		"""
		self.stimuli[name]: Stimulus = Stimulus(k)
		new_connectomes: Dict[str, ndarray] = {}
		for key in self.areas:
			new_connectomes[key] = np.empty((0, 0))
			self.areas[key].stimulus_beta[name] = self.areas[key].beta
		self.stimuli_connectomes[name] = new_connectomes

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

	def project(self, stim_to_area: Mapping[str, List[str]],
					area_to_area: Mapping[str, List[str]]) -> None:
		""" Project is the basic operation where some stimuli and some areas are activated,
		with only specified connections between them active.

		:param stim_to_area: Dictionary that matches to each stimuli applied a list of areas to project into.
			Example: {"stim1":["A"], "stim2":["C","A"]}
		:param area_to_area: Dictionary that matches for each area a list of areas to project into.
			Note that an area can also be projected into itself.
			Example: {"A":["A","B"],"C":["C","A"]}
		"""
		stim_in: defaultdict[str, List[str]] = defaultdict(lambda: [])
		area_in: defaultdict[str, List[str]] = defaultdict(lambda: [])

		# Validate stim_area, area_area well defined
		# Set stim_in to be the Dictionary that matches for every area the list of input stimuli.
		# Set areas_in to be the Dictionary that matches for every area the list of input areas.
		for stim, areas in stim_to_area.items():
			if stim not in self.stimuli:
				raise IndexError(stim + " not in brain.stimuli")
			for area in areas:
				if area not in self.areas:
					raise IndexError(area + " not in brain.areas")
				stim_in[area].append(stim)
		for from_area, to_areas in area_to_area.items():
			if from_area not in self.areas:
				raise IndexError(from_area + " not in brain.areas")
			for to_area in to_areas:
				if to_area not in self.areas:
					raise IndexError(to_area + " not in brain.areas")
				area_in[to_area].append(from_area)

		# to_update is the set of all areas that receive input
		to_update = set().union(list(stim_in.keys()), list(area_in.keys()))

		for area in to_update:
			num_first_winners = self.project_into(self.areas[area], stim_in[area], area_in[area])
			self.areas[area].num_first_winners = num_first_winners

		# once done everything, for each area in to_update: area.update_winners()
		for area in to_update:
			self.areas[area].update_winners()

	def project_into(self, area: Area, from_stimuli: List[str], from_areas: List[str]) -> int:
		"""Project multiple stimuli and area assemblies into area 'area' at the same time.

		:param area: The area projected into
		:param from_stimuli: The stimuli that we will be applying
		:param from_areas: List of separate areas whose assemblies we will project into this area
		:return: Returns the number of area neurons that were winners for the first time during this projection
		"""
		# projecting everything in from stim_in[area] and area_in[area]
		# calculate: inputs to self.connectomes[area] (previous winners)
		# calculate: potential new winners, Binomial(sum of in sizes, k-top)
		# k top of previous winners and potential new winners
		# if new winners > 0, redo connectome and intra_connectomes
		# have to wait to replace new_winners
		# TODO Add more documentation to this function which does most of the work
		# TODO Handle case of projecting from an area without previous winners.
		# TODO: there is a bug when adding a new stimulus later on.
		# TODO: Stimulus is updating to somehow represent >100 neurons.
		logging.info(("Projecting " + ",".join(from_stimuli) + " and " + ",".join(from_areas) + " into " + area.name))

		name: str = area.name
		prev_winner_inputs: List[float] = [0.] * area.support_size
		for stim in from_stimuli:
			stim_inputs = self.stimuli_connectomes[stim][name]
			for i in range(area.support_size):
				prev_winner_inputs[i] += stim_inputs[i]
		for from_area in from_areas:
			connectome = self.connectomes[from_area][name]
			for w in self.areas[from_area].winners:
				for i in range(area.support_size):
					prev_winner_inputs[i] += connectome[w][i]

		logging.debug("prev_winner_inputs: %s" % prev_winner_inputs)

		# simulate area.k potential new winners
		total_k: int = 0
		input_sizes: List[int] = []  	# list of the number of winners in each upstream stimulus/area,
										# indexed in the same way as from_areas. TODO: does it makes sense?
		for stim in from_stimuli:
			total_k += self.stimuli[stim].k
			input_sizes.append(self.stimuli[stim].k)
		for from_area in from_areas:
			# if self.areas[from_area].support_size < self.areas[from_area].k:
			#	raise ValueError("Area " + from_area + "does not have enough support.")
			effective_k = len(self.areas[from_area].winners)
			total_k += effective_k
			input_sizes.append(effective_k)

		logging.debug("total_k = " + str(total_k) + " and input_sizes = " + str(input_sizes))

		effective_n = area.n - area.support_size
		# Threshold for inputs that are above (n-k)/n percentile. alpha is the smallest number such that:
		# 							Pr(Bin(total_k,self.p) <= alpha) >= (effective_n-area.k)/effective_n
		alpha = binom.ppf((float(effective_n-area.k)/effective_n), total_k, self.p)
		logging.debug(("Alpha = " + str(alpha)))
		# use normal approximation, between alpha and total_k, round to integer
		# create k potential_new_winners
		std = math.sqrt(total_k * self.p * (1.0-self.p))
		mu = total_k * self.p
		a = float(alpha - mu) / std
		b = float(total_k - mu) / std  # note that b>=a and corresponds to the maximum value of Bin(total_k,self.p)
		potential_new_winners = truncnorm.rvs(a, b, scale=std, loc=mu, size=area.k)
		for i in range(area.k):
			potential_new_winners[i] = round(potential_new_winners[i])
		potential_new_winners = potential_new_winners.tolist()

		logging.debug("potential_new_winners: %s" % potential_new_winners)

		# take max among prev_winner_inputs, potential_new_winners
		# get num_first_winners (think something small)
		# can generate area.new_winners, note the new indices
		both = prev_winner_inputs + potential_new_winners
		new_winner_indices = heapq.nlargest(area.k, list(range(len(both))), both.__getitem__)
		num_first_winners = 0
		first_winner_inputs = []
		for i in range(area.k):
			if new_winner_indices[i] >= area.support_size:  # index in potential_new_winners - a new assembly neuron
				first_winner_inputs.append(potential_new_winners[new_winner_indices[i] - area.support_size])
				new_winner_indices[i] = area.support_size + num_first_winners
				num_first_winners += 1
		area._new_winners = new_winner_indices # Note that from here on 'new_winner_indices' is not in use.
		area._new_support_size = area.support_size + num_first_winners

		logging.debug("new_winners: %s" % area._new_winners)

		# for i in num_first_winners
		# generate where input came from
		# 	1) can sample input from array of size total_k, use ranges
		# 	2) can use stars/stripes method: if m total inputs, sample (m-1) out of total_k
		first_winner_to_inputs: Dict[int, ndarray] = {}
		for i in range(num_first_winners):
			input_indices = random.sample(range(0, total_k), int(first_winner_inputs[i]))
			inputs: ndarray = np.zeros(len(input_sizes))
			total_so_far = 0
			for j in range(len(input_sizes)):
				# inputs[j] is the randomly generated number of connections from the j'th input to area i.
				inputs[j] = sum([((total_so_far + input_sizes[j]) > w >= total_so_far) for w in input_indices])
				total_so_far += input_sizes[j]
			first_winner_to_inputs[i] = inputs
			logging.debug("for first_winner #%d with input %s split as so: %s" % (i, first_winner_inputs[i], inputs))

		m = 0
		# connectome for each stim->area
			# add num_first_winners cells, sampled input * (1+beta)
			# for i in repeat_winners, stimulus_inputs[i] *= (1+beta)
		for stim in from_stimuli:
			if num_first_winners > 0:
				self.stimuli_connectomes[stim][name] = np.resize(self.stimuli_connectomes[stim][name],
																area.support_size + num_first_winners)
			for i in range(num_first_winners):
				self.stimuli_connectomes[stim][name][area.support_size + i] = first_winner_to_inputs[i][m]
			stim_to_area_beta = area.stimulus_beta[stim]
			for i in area._new_winners:
				self.stimuli_connectomes[stim][name][i] *= (1+stim_to_area_beta)
			logging.debug("stimulus %s now looks like: %s" % (stim, self.stimuli_connectomes[stim][name]))
			m += 1

		# connectome for each in_area->area
			# add num_first_winners columns
			# for each i in num_first_winners, fill in (1+beta) for chosen neurons
			# for each i in repeat_winners, for j in in_area.winners, connectome[j][i] *= (1+beta)
		for from_area in from_areas:
			from_area_w = self.areas[from_area].support_size
			from_area_winners = self.areas[from_area].winners
			self.connectomes[from_area][name] = np.pad(self.connectomes[from_area][name],
														((0,0), (0,num_first_winners)),
														'constant', constant_values=0)
			for i in range(num_first_winners):
				total_in = first_winner_to_inputs[i][m]
				sample_indices = random.sample(from_area_winners, int(total_in))
				for j in range(from_area_w):
					if j in sample_indices:
						self.connectomes[from_area][name][j][area.support_size + i] = 1
					if j not in from_area_winners:
						self.connectomes[from_area][name][j][area.support_size + i] = np.random.binomial(1, self.p)
			area_to_area_beta = area.area_beta[from_area]
			for i in area._new_winners:
				for j in from_area_winners:
					self.connectomes[from_area][name][j][i] *= (1.0 +area_to_area_beta)
			logging.debug("Connectome of %s to %s is now %s" % (from_area, name, self.connectomes[from_area][name]))
			m += 1

		# expand connectomes from other areas that did not fire into area
		# also expand connectome for area->other_area
		for other_area in self.areas:
			if other_area not in from_areas:
				self.connectomes[other_area][name] = np.pad(self.connectomes[other_area][name],
					((0,0),(0,num_first_winners)), 'constant', constant_values=0)
				for j in range(self.areas[other_area].support_size):
					for i in range(area.support_size, area._new_support_size):
						self.connectomes[other_area][name][j][i] = np.random.binomial(1, self.p)
			# add num_first_winners rows, all bernoulli with probability p
			self.connectomes[name][other_area] = np.pad(self.connectomes[name][other_area],
				((0, num_first_winners),(0, 0)), 'constant', constant_values=0)
			columns = len(self.connectomes[name][other_area][0])
			for i in range(area.support_size, area._new_support_size):
				for j in range(columns):
					self.connectomes[name][other_area][i][j] = np.random.binomial(1, self.p)
			logging.debug("Connectome of %s to %s is now: %s" % (name, other_area, self.connectomes[name][other_area]))

		return num_first_winners








