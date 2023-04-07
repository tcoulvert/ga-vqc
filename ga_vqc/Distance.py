import numpy as np
from sklearn.manifold import TSNE

def euclidian_distances(ansatz_comp, population):
    vector_comp = create_vector(ansatz_comp)
    distances = []
    for ansatz in population:
        vector = create_vector(ansatz)
        distances.append(
            np.power(
                np.absolute(
                    vector_comp - vector
                ), 
                0.5
            )
        )
    
    return distances

def tsne(population, rng_seed):
    vectors = []
    for ansatz in population:
        vectors.append(create_vector(ansatz))
    
    t_sne = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        n_iter=250,
        random_state=rng_seed,
    )

    S_t_sne = t_sne.fit_transform(vectors)

    return S_t_sne

def create_vector(ansatz):
    vector = []
    ### single-qubit gates ###
    for moment in ansatz.n_moments:
        for qubit in ansatz.n_qubits:
            if ansatz[moment][qubit] == 'I':
                vector.append(0)
            if ansatz[moment][qubit] == 'U3':
                vector.append(1)

    print(f"Length of vector after 1-qubit gates: {len(vector)}")

    for moment in ansatz.n_moments:
        # [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
        two_qubit_pairs = [0, 0, 0, 0, 0, 0]
        for qubit in ansatz.n_qubits:
            if ansatz[moment][qubit].find('_') > 0:
                if ansatz[moment][qubit][-3] == 'C':
                    two_qubit_pairs[2*qubit] = 1
                elif ansatz[moment][qubit][-3] == 'T':
                    two_qubit_pairs[2*qubit + 1] = 1
                break
        vector.extend(two_qubit_pairs)
    
    print(f"Length of vector after 2-qubit gates: {len(vector)}")

    return vector