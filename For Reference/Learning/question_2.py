from brain import Brain

P = 0.01
BETA = 0.05
N = 10**7
K = 10**4


def get_neuron_status():
    brain = Brain(P)
    brain.add_area(name='A', n=10**7, k=K, beta=BETA)
    brain.add_area(name='B', n=10**7, k=K, beta=BETA)
    brain.add_area(name='C', n=10**7, k=K, beta=BETA)

    output_area = brain.areas['A']
    stimulus_name = 'Stimulus'
    brain.add_stimulus(name=stimulus_name, k=K)
    brain.project({stimulus_name: ['A']}, {})


get_neuron_status()