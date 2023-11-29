from scipy.linalg import lu
from scipy.sparse import csr_matrix
import numpy as np
from itertools import product
import cvxpy as cp
import cdd

#======================================================================#
#=============== A-ASSISTING MATHEMATICAL TOOLS =========================#
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

#.............3. Converting real matrix to integer...............#
    

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
           np.array([to_gellmann(o) for o in effects]).T.real,\
                     to_gellmann(np.eye(d)).T.real,\
                     to_gellmann(np.eye(d)/d).T.real

#...........B2. Characterising the accessible fragment...........#

def accessibleGPT(X):
    #Uses RREF of the matrix of states/effects to construct the
    #inclusion map (RREF without the kernel) and projection (pseudo
    #inverse of the inclusion), and the accessible fragment.
    REF = rref(X.T)[0] #RREF of the input, the transpose keeps 
                       #things similar to the original code
    P, L, U = lu(X)    #Performs lower-upper decomposition of X
    r = len(np.unique(csr_matrix(U).indptr))-1 
    #r counts how many row pointers in U (upper-triangular component)
    #are different, and picks the largest one. -1 fixes counting issues
    Inc = REF[:r,:].T
    Proj = np.linalg.pinv(Inc)
    return Inc, Proj, Proj@X

#.......... B3. Characterising state cone facets.................#

def dualize_states(S):
        C = cdd.Matrix(S.T, number_type="float")
        C.rep_type = cdd.RepType.GENERATOR
        return np.array(cdd.Polyhedron(C).get_inequalities())

def dualize_effects(E):
        C = cdd.Matrix(E.T, number_type="float")
        C.rep_type = cdd.RepType.INEQUALITY
        return np.array(cdd.Polyhedron(C).get_generators())
    

#================================================================#
#======================== C- MAIN CODE ==========================#
#================================================================#

def simplicial_embedding(H_S, H_E, Id, D):
    p_, Phi_ = cp.Variable(nonneg=True),\
               cp.Variable(shape=(H_E.shape[0], H_S.shape[0]), nonneg=True)
    problem = cp.Problem(cp.Minimize(p_),\
               [p_*D + (1-p_)*Id - H_E.T @ Phi_ @ H_S == 0])
    problem.solve()
    return p_.value, Phi_.value

def find_embedding(states, effects, unit, mms):
    IncS, ProjS, SA = accessibleGPT(states)
    IncE, ProjE, EA = accessibleGPT(effects)
    u = ProjE @ unit
    mu = ProjS @ mms
    
    HS = dualize_states(SA)
    HE = dualize_effects(EA).T
    Id = accessibleGPT(effects)[0].T @ accessibleGPT(states)[0]
    D = np.outer(u,mu)
    p, Phi = simplicial_embedding(HS, HE, Id, D)

   # tauE = accessibleGPT(effects)[0].T @ H_E @ Phi
   # tauS = H_S @ accessibleGPT(states)[0]
   # sigmaS = np.array([tauS[i,:]*(u @ tauE[:,i]) for i in range(tauS.shape[0])\
   #               if not np.isclose((u @ tauE[:,i]),0)])
   # sigmaE = np.array([tauE[:,i]/(u @ tauE[:,i]) for i in range(tauE.shape[1])\
   #               if not np.isclose((u @ tauE[:,i]),0)]).T

   # PE, PS = effects @ sigmaE, sigmaS @ states
    return p, Phi#PE, PS
                     

#================================================================#
#========================= D-EXAMPLES ===========================#
#================================================================#
def quantum():
    Zup, Zdown = np.array([1,0]), np.array([0,1])
    Xup, Xdown = np.array([1,1])/np.sqrt(2), np.array([1,-1])/np.sqrt(2)
 #   ZupR, ZdownR = np.array([np.sqrt(3),1])/4, np.array([1,-np.sqrt(3)])/4
  #  XupR, XdownR = np.array([np.sqrt(3),-1])/4, np.array([1,np.sqrt(3)])/4
    E1 = np.array([[2+np.sqrt(2), np.sqrt(2)],
                  [np.sqrt(2), 2-np.sqrt(2)]])/4
    E2 = np.array([[2-np.sqrt(2), -np.sqrt(2)],
                  [-np.sqrt(2), 2+np.sqrt(2)]])/4
    E3 = np.array([[2+np.sqrt(2), -np.sqrt(2)],
                  [-np.sqrt(2), 2-np.sqrt(2)]])/4
    E4 = np.array([[2-np.sqrt(2), np.sqrt(2)],
                  [np.sqrt(2), 2+np.sqrt(2)]])/4
    #Yup, Ydown = np.array([1, 1.j])/np.sqrt(2), np.array([1, -1.j])/np.sqrt(2)
    s = [np.outer(s, s.conj()) for s in [Zup, Zdown, Xup, Xdown]]
    e = [E1, E2, E3, E4]
    return hilbert_to_gpt(s, e)
          


    

