import numpy as np
from itertools import product


def GellmannBasisElement(i, j, d):
    # Constructs one element of a Gell-Mann basis for dimension d
    # ©Copyright 2014, Jonathan Gross. Revision 449580a1.
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


def GelmannBasis(d):  # Constructs gell-mann basis for dimension d
    return [
        GellmannBasisElement(i, j, d) for i, j in product(range(1, d + 1), repeat=2)
    ]


def fromListOfMatrixToListOfVectors(states, effects):
    # Converts inputs from density operator/POVM representation to
    # GPT representation. ©Copyright 2023 Mathew Weiss
    d = states[0].shape[0]
    basis = GelmannBasis(d)
    to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
    return (
        np.array([to_gellmann(v) for v in states]).T.real,
        np.array([to_gellmann(v) for v in effects]).T.real,
        to_gellmann(np.eye(d)).real,
        to_gellmann(np.eye(d) / d).real,
    )
