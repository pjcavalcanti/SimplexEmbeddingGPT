import numpy as np
from itertools import product


def GellmannBasisElement(i, j, d):
    """
    Constructs one element of a generalised Gell-Mann basis for a given dimension.
    This function generates a single Gell-Mann matrix for the specified indices and dimension.

    Args:
    i (int): The row index for the element in the Gell-Mann matrix.
    j (int): The column index for the element in the Gell-Mann matrix.
    d (int): The dimension of the SU(n) group, determining the size of the Gell-Mann matrix.

    Returns:
    np.ndarray: The specified Gell-Mann matrix element for dimension d.

    Copyright:
    © Copyright 2014, Jonathan Gross. Revision 449580a1.
    """
    if i > j:  # symmetric elements
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = 1
        L[j - 1][i - 1] = 1

    elif i < j:  # antisymmetric elements
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = -1.0j
        L[j - 1][i - 1] = 1.0j

    elif i == j and i < d:  # diagonal elements
        L = np.sqrt(2 / (i * (i + 1))) * np.diag(
            [1 if n <= i else (-i if n == (i + 1) else 0) for n in range(1, d + 1)]
        )
    else:  # identity
        L = np.eye(d)

    return np.array(L / np.sqrt((L @ L).trace()))


def GelmannBasis(d):
    """
    Constructs the complete Gell-Mann basis for a given dimension.

    The Gell-Mann matrices form a basis for the Lie algebra su(n) of the special unitary group SU(n).
    This function generates all the Gell-Mann matrices for the specified dimension.

    Args:
    d (int): The dimension of the SU(n) group, determining the size and number of the Gell-Mann matrices.

    Returns:
    list: A list of np.ndarray, each being a Gell-Mann matrix element for dimension d.

    Note:
    This function relies on the GellmannBasisElement function to generate individual basis elements.
    """
    return [
        GellmannBasisElement(i, j, d) for i, j in product(range(1, d + 1), repeat=2)
    ]


def fromListOfMatrixToListOfVectors(states, effects):
    """
    Converts inputs from density operator/POVM representation to GPT representation.

    This function uses the Gell-Mann basis to represent quantum states and effects (measurements)
    as column real vectors, as is in the Generalized Probabilistic Theories (GPT) framework.

    Args:
    states (list of np.ndarray): List of density matrices representing quantum states.
    effects (list of np.ndarray): List of POVM elements representing quantum effects.

    Returns:
    tuple: A tuple containing four elements, all numpy arrays:
        - Transformed states in GPT representation.
        - Transformed effects in GPT representation.
        - The identity element in GPT representation.
        - The maximally mixed state in GPT representation.

    Note:
    The function uses the GelmannBasis function to construct the necessary Gell-Mann matrices for the transformation.

    Copyright:
    © Copyright 2023 Mathew Weiss.
    """
    d = states[0].shape[0]
    basis = GelmannBasis(d)
    to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
    return (
        np.array([to_gellmann(v) for v in states]).T.real,
        np.array([to_gellmann(v) for v in effects]).T.real,
        to_gellmann(np.eye(d)).real,
        to_gellmann(np.eye(d) / d).real,
    )
