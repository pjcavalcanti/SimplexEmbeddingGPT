import scipy as sc
import numpy as np


def gellmann_basis(d):
    def Ld(l):
        M = np.zeros((d,d))
        for j in range(l + 1):
            M += E[j][j]
        return M - (l + 1) * E[l + 1][l + 1]
    
    E =[[np.zeros((d,d)) for k in range(d)] for j in range(d)]
    for j in range(d):
        for k in range(d):
            E[j][k][j,k] = 1
    
    Lsym = []
    Lasym = []
    for j in range(d):
        for k in range(j + 1, d):
            Lsym.append(E[j][k] + E[k][j])
            Lasym.append(-1j*(E[j][k] - E[k][j]))
    Ldiag = [Ld(l) for l in range(d-1)]
    
    return np.array([e / np.sqrt(np.trace(e @ e)) for e in Lsym + Lasym + Ldiag])
       

def tinyfier(X):
    U, D, V = np.linalg.svd(X)
    rank = np.isclose(D, 0).argmax([0])
    if rank == 0:
        rank = D.shape[0]
    return U[:,:rank].T

np.set_printoptions(formatter={'complexfloat': lambda x: '{:.2f}'.format(x)})


d = 3

A = gellmann_basis(d)

for i in range(d**2 - 1):
    print("Ours:\n")
    print(A[i])
    print("\n------------------------------------------\n")
    