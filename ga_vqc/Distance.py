import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def euclidean_distances(ansatz_comp, population):
    vector_comp = np.array(ansatz_comp.vector)
    distances = []
    for ansatz in population:
        vector = np.array(ansatz.vector)
        distances.append(
            np.sum(
                np.power(
                    vector_comp - vector,
                    2
                )
            )**0.5
        )
    
    return distances

def tsne(population, perplexity=2, rng_seed=None):
    """
    TODO: switch from PCA to TruncatedSVD (b/c works better on sparse data)
    """
    vectors = np.array([np.array(ansatz.vector) for ansatz in population])
    
    if vectors.shape[1] > 100:
        tSVD = TruncatedSVD(
            n_components=100,
            n_iter=7,
            random_state=rng_seed,
        )
        vectors = tSVD.fit_transform(vectors)
        
    t_sne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        n_iter=250,
        random_state=rng_seed,
    )

    S_t_sne = t_sne.fit_transform(vectors)

    return S_t_sne

# def create_vector(ansatz, max_moments, return_type='numpy'):
#     """
#     TODO: Change pad from affecting circuit to happeneing automatically here
#     """

#     vector = []

#     ### single-qubit gates ###
#     for moment in range(max_moments):
#         for qubit in range(ansatz.n_qubits):
#             one_qubit_states = []
#             for _ in range(
#                 ansatz.genepool.n_gates(
#                     search_param={'n_qubits': 1}
#                 )
#             ):
#                 one_qubit_states.extend([0])

#             if moment >= ansatz.n_moments or ansatz.genepool.n_qubits(ansatz[moment][qubit]) != 1:
#                 vector.extend([i for i in one_qubit_states])
#                 continue
#             one_qubit_states[ansatz.genepool.index_of(ansatz[moment][qubit])] = 1 # Assumes 'I' always in index 0, and cannot NOT include 'I'
#             vector.extend([i for i in one_qubit_states])

#     ### 2-qubit gates ###
#     for moment in range(max_moments):
#         # [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1), ('I', 'I')]
#         two_qubit_states = []
#         for _ in range(
#             ansatz.genepool.n_gates(
#                 search_param={'n_qubits': 2}
#             )
#         ):
#             two_qubit_states.extend([0 for __ in range(np.math.factorial(ansatz.n_qubits) + 1)])
#             two_qubit_states[-1] = 1

#         for qubit in range(ansatz.n_qubits):
#             if moment >= ansatz.n_moments:
#                 break
#             if ansatz[moment][qubit].find('_') > 0: # Doesn't work for passing more than 1 2-qubit gate, and only works for 'control'/'target' gates
#                 two_qubit_states[-1] = 0
#                 if ansatz[moment][qubit][-3] == 'C':
#                     two_qubit_states[2*qubit] = 1
#                 elif ansatz[moment][qubit][-3] == 'T':
#                     two_qubit_states[2*qubit + 1] = 1
#                 break
#         vector.extend(two_qubit_states)

#     ### 3+ qubit gates ###
#     # TO DO

#     if return_type == 'numpy':
#         return np.array(vector)
#     elif return_type == 'list':
#         return vector