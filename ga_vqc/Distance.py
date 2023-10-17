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