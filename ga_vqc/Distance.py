from difflib import SequenceMatcher

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def euclidean_distances(ansatz_A, empty_circuit_ansatz, population):
    vector_A = np.array(empty_circuit_ansatz.vector)
    baseline_distances = []
    for ansatz_B in population:
        vector_B = np.array(ansatz_B.vector)
        baseline_distances.append(
            np.sum(
                np.power(
                    vector_A - vector_B,
                    2
                )
            )**0.5
        )
    
    vector_A = np.array(ansatz_A.vector)
    distances = []
    for ansatz_B in population:
        vector_B = np.array(ansatz_B.vector)
        distances.append(
            np.sum(
                np.power(
                    vector_A - vector_B,
                    2
                )
            )**0.5
        )
    
    return np.array(distances) / np.array(baseline_distances)

def string_distances(ansatz_A, empty_circuit_diagram, population):
    s = SequenceMatcher(isjunk=lambda x: x in ' ')

    # By adding in distance to circuit A in denom it makes distance matrix symmetric
    #   -> doubled distances so that diagonal of matrix remains all 1s
    s.set_seq2(empty_circuit_diagram)
    s.set_seq1(ansatz_A.diagram)
    A_baseline_distance = s.ratio()

    B_baseline_distances = []
    for ansatz_B in population:
        s.set_seq1(ansatz_B.diagram)
        B_baseline_distances.append(s.ratio() + A_baseline_distance)

    s.set_seq2(ansatz_A.diagram)
    doubled_distances = []
    for ansatz_B in population:
        s.set_seq1(ansatz_B.diagram)
        doubled_distances.append(2 * s.ratio())

    # return np.array(doubled_distances) / np.array(B_baseline_distances)
    return  np.array(B_baseline_distances) / np.array(doubled_distances)


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
