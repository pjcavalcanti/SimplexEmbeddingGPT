import numpy as np
from preprocessing import fromListOfMatrixToListOfVectors


def quantum():
    Zup, Zdown = np.array([1, 0]), np.array([0, 1])
    Xup, Xdown = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
    E1 = np.array([[2 + np.sqrt(2), np.sqrt(2)], [np.sqrt(2), 2 - np.sqrt(2)]]) / 4
    E2 = np.array([[2 - np.sqrt(2), -np.sqrt(2)], [-np.sqrt(2), 2 + np.sqrt(2)]]) / 4
    E3 = np.array([[2 + np.sqrt(2), -np.sqrt(2)], [-np.sqrt(2), 2 - np.sqrt(2)]]) / 4
    E4 = np.array([[2 - np.sqrt(2), np.sqrt(2)], [np.sqrt(2), 2 + np.sqrt(2)]]) / 4
    s = [np.outer(s, s.conj()) for s in [Zup, Zdown, Xup, Xdown]]
    e = [E1, E2, E3, E4]
    return fromListOfMatrixToListOfVectors(s, e)
