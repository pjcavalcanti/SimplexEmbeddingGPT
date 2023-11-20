import scipy as sc
import numpy as np

def gellmann_basis_cara(d):
    def h_helper(d,k):
        if k == 1:
            return np.eye(d)
        if k > 1 and k < d:
            return sc.linalg.block_diag(h_helper(d-1, k), [0])
        if k == d:
            return np.sqrt(2/(d*(d+1)))*sc.linalg.block_diag(np.eye(d-1), [1-d])

    E = [[np.zeros((d,d)) for k in range(d)] for j in range(d)]
    for j in range(d):
        for k in range(d):
            E[j][k][j,k] = 1
    F = []
    for j in range(d):
        for k in range(d):
            if k < j:
                F.append(E[k][j] + E[j][k])
            elif k > j:
                F.append(-1j*(E[j][k] - E[k][j]))
    F.extend([h_helper(d, k) for k in range(1,d+1)])
    return np.array([f/np.sqrt((f@f).trace()) for f in F])

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
B = gellmann_basis_cara(d)

for i in range(d**2 - 1):
    print("Ours:\n")
    print(A[i])
    # print("\n")
    # print("Cara:\n")
    # print(B[i])
    print("\n------------------------------------------\n")
    