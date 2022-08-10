import scipy as sp
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler

from GA_Manager import backend

def main():
	events = np.load('10k_dijet.npy', requires_grad=False)
	scaler = MinMaxScaler(feature_range=(0, sp.pi))
	events = scaler.fit_transform(events)

	rng_seed = None
	rng = np.random.default_rng(seed=rng_seed)
	config = {
    		'backend_type': 'high',
    		'n_qubits': 3,
    		'n_moments': 4,
    		'gates_arr': ['I', 'RX', 'RY', 'RZ', 'CNOT'],
    		'gates_probs': [0.125, 0.125, 0.125, 0.125, 0.5],
    		'pop_size': 30,
    		'n_winners': 15,
    		'n_mutations': 2,
    		'n_steps': 15,
    		'latent_qubits': 1,
    		'n_shots': 500,
    		'seed': rng_seed,
    		'events': events,
	}

	ga = backend(config)
	ga.evolve()

if __name__ == '__main__':
        main()
