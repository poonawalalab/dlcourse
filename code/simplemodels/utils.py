import numpy as np
from sklearn.datasets import load_iris

def load_two_species(species_pair=(0, 1), *, remap=True):
    """
	ChatGPT code gen 7/6/25
    Return X, y containing only the two requested Iris species.

    Parameters
    ----------
    species_pair : tuple[int, int]
        The two class labels to keep (choose from 0, 1, 2).
    remap : bool, default True
        If True, relabel the kept classes to 0 and 1 (order follows species_pair).

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4)
    y : np.ndarray, shape (n_samples,)
    """
    if len(species_pair) != 2:
        raise ValueError("species_pair must contain exactly two integers (e.g. (0, 2)).")
    if not all(label in {0, 1, 2} for label in species_pair):
        raise ValueError("Labels must be chosen from 0, 1, 2.")

    iris = load_iris()
    X_all, y_all = iris.data, iris.target

    # Boolean mask: keep rows whose label is in species_pair
    mask = np.isin(y_all, species_pair)
    X, y = X_all[mask], y_all[mask]

    if remap:
        # Map the first chosen label → 0, the second → 1
        label_map = {species_pair[0]: 0, species_pair[1]: 1}
        y = np.vectorize(label_map.get)(y)

    return X, y

