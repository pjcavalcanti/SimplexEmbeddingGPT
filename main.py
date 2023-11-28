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


#============================ 28/11/2023 ===================================#
from scipy.linalg import lu
from scipy.sparse import csr_matrix
import numpy as np
from itertools import product

#======================================================================#
#=============== A-ASSISTING MATHEMATICAL TOOLS =======================#
#======================================================================#

#.................A1. Gell-Mann basis.............................#

def gellmann_element(j,k,d):
    #Constructs one element of a Gell-Mann basis for dimension d
    #©Copyright 2014, Jonathan Gross. Revision 449580a1.
    if j>k:#symmetric elements
        L=np.zeros((d,d), dtype=np.complex128)
        L[j-1][k-1]=1
        L[k-1][j-1]=1
    elif j<k: #antisymmetric elements
        L=np.zeros((d,d), dtype=np.complex128)
        L[j-1][k-1]=-1.j
        L[k-1][j-1]=1.j
    elif j==k and j<d: #diagonal elements
        L=np.sqrt(2/(j*(j+1)))*np.diag([1 if n<=j
                                      else (-j if n==(j+1)
                                      else 0)
                                      for n in range(1,d+1)])
    else: #identity
        L=np.eye(d)
        
    return np.array(L/np.sqrt((L@L).trace()))

def gellmann_basis(d):#Constructs gell-mann basis for dimension d
    return [gellmann_element(j,k,d) for j, k in product(range(1,d+1), repeat=2)]

#.............A2. Reduced Row Echelon Form (RREF)................#

def rref(B, tol=1e-8, debug=False):
    #Provides the reduced row echelon form for matrix B
    #©Copyright 2017 Stelios Sfakianakis <https://gist.github.com/sgsfak/77a1c08ac8a9b0af77393b24e44c9547>
  A = B.copy()
  rows, cols = A.shape
  r = 0
  pivots_pos = []
  row_exchanges = np.arange(rows)
  for c in range(cols):
    if debug: print("Now at row", r, "and col", c, "with matrix:"); print(A)

    ## Find the pivot row:
    pivot = np.argmax (np.abs (A[r:rows,c])) + r
    m = np.abs(A[pivot, c])
    if debug: print("Found pivot", m, "in row", pivot)
    if m <= tol:
      ## Skip column c, making sure the approximately zero terms are
      ## actually zero.
      A[r:rows, c] = np.zeros(rows-r)
      if debug: print("All elements at and below (", r, ",", c, ") are zero.. moving on..")
    else:
      ## keep track of bound variables
      pivots_pos.append((r,c))

      if pivot != r:
        ## Swap current row and pivot row
        A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
        row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]
        
        if debug: print("Swap row", r, "with row", pivot, "Now:"); print(A)

      ## Normalize pivot row
      A[r, c:cols] = A[r, c:cols] / A[r, c];

      ## Eliminate the current column
      v = A[r, c:cols]
      ## Above (before row r):
      if r > 0:
        ridx_above = np.arange(r)
        A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
        if debug: print("Elimination above performed:"); print(A)
      ## Below (after row r):
      if r < rows-1:
        ridx_below = np.arange(r+1,rows)
        A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
        if debug: print("Elimination below performed:"); print(A)
      r += 1
    ## Check if done
    if r == rows:
      break;
  return (A, pivots_pos, row_exchanges)

#================================================================#
#===================B-PREPROCESSING INPUTS=======================#
#================================================================#

#..........B1. From Matrix inputs to vector inputs...............#

def hilbert_to_gpt(states, effects):
    #Converts inputs from density operator/POVM representation to
    #GPT representation. ©Copyright 2023 Mathew Weiss
    d = states[0].shape[0]
    basis = gellmann_basis(d)
    to_gellmann = lambda O: np.array([(O@b).trace() for b in basis[::-1]])
    return np.array([to_gellmann(o) for o in states]).T.real,\
           np.array([to_gellmann(o) for o in effects]).real,\
                     to_gellmann(np.eye(d)).real,\
                     to_gellmann(np.eye(d)/d).real

#...........B2. Characterising the accessible fragment...........#

def tinyfier(X):
    #Uses RREF of the matrix of states/effects to construct the
    #inclusion map (RREF without the kernel) and projection (pseudo
    #inverse of the inclusion), and the accessible fragment.
    REF = rref(X.T)[0] #RREF of the input, the transpose keeps 
                       #things similar to the original code
    P, L, U = lu(X)    #Performs lower-upper decomposition of X
    r = len(np.unique(csr_matrix(U).indptr))-1 
    #r counts how many row pointers in U (upper-triangular component)
    #are different, and picks the largest one. -1 fixes counting issues
    Inc = REF[:r,:]
    Proj = np.linalg.pinv(Inc).T
    return Inc, Proj, Proj@X
                     

#================================================================#
#========================= D-EXAMPLES ===========================#
#================================================================#
def quantum():
    Zup, Zdown = np.array([1,0]), np.array([0,1])
    Xup, Xdown = np.array([1,1])/np.sqrt(2), np.array([1,-1])/np.sqrt(2)
    #Yup, Ydown = np.array([1, 1.j])/np.sqrt(2), np.array([1, -1.j])/np.sqrt(2)
    s = [np.outer(s, s.conj()) for s in [Zup, Zdown, Xup, Xdown]]
    return hilbert_to_gpt(s, s)

    
