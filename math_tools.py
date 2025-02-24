import numpy as np
import cdd


def rref(B, tol=1e-8, debug=False):
    """
    Provides the Reduced Row Echelon Form (RREF) of a given matrix B.
    
    Args: 
    B (np.ndarray): The matrix to be transformed into RREF. Should be a 2D array.
    tol (float, optional): A tolerance value to identify negligible elements. Defaults to 1e-8.
    debug (bool, optional): If set to True, additional debugging information is printed. Defaults to False.

    Returns:
    np.ndarray: The RREF of the input matrix B.
    
    Copyright:
    © Copyright 2017 Stelios Sfakianakis. Original code available at 
    <https://gist.github.com/sgsfak/77a1c08ac8a9b0af77393b24e44c9547>    
    """
    
    A = B.copy()
    rows, cols = A.shape
    r = 0
    pivots_pos = []
    row_exchanges = np.arange(rows)
    for c in range(cols):
        if debug:
            print("Now at row", r, "and col", c, "with matrix:")
            print(A)

        ## Find the pivot row:
        pivot = np.argmax(np.abs(A[r:rows, c])) + r
        m = np.abs(A[pivot, c])
        if debug:
            print("Found pivot", m, "in row", pivot)
        if m <= tol:
            ## Skip column c, making sure the approximately zero terms are
            ## actually zero.
            A[r:rows, c] = np.zeros(rows - r)
            if debug:
                print(
                    "All elements at and below (", r, ",", c, ") are zero.. moving on.."
                )
        else:
            ## keep track of bound variables
            pivots_pos.append((r, c))

            if pivot != r:
                ## Swap current row and pivot row
                A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
                row_exchanges[[pivot, r]] = row_exchanges[[r, pivot]]

                if debug:
                    print("Swap row", r, "with row", pivot, "Now:")
                    print(A)

            ## Normalize pivot row
            A[r, c:cols] = A[r, c:cols] / A[r, c]

            ## Eliminate the current column
            v = A[r, c:cols]
            ## Above (before row r):
            if r > 0:
                ridx_above = np.arange(r)
                A[ridx_above, c:cols] = (
                    A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
                )
                if debug:
                    print("Elimination above performed:")
                    print(A)
            ## Below (after row r):
            if r < rows - 1:
                ridx_below = np.arange(r + 1, rows)
                A[ridx_below, c:cols] = (
                    A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
                )
                if debug:
                    print("Elimination below performed:")
                    print(A)
            r += 1
        ## Check if done
        if r == rows:
            break
    return A

def FindStateConeFacets(S):
    """
    Computes the facets of the positive cone defined by the input set of states S of an accessible GPT fragment.

    Args:
    - S (np.ndarray): A numpy array representing the states in the accessible GPT fragment.
        shape = (dimension of the space spanned by states in the accessible GPT fragment, number of states)

    Returns:
    np.ndarray:
        An array listing the facets of the positive cone defined by the input matrix S.
        shape = (number of facets, dimension of the space spanned by states in the accessible GPT fragment)
    """
    
    C = cdd.matrix_from_array(S.T)
    C.rep_type = cdd.RepType.GENERATOR
    H_S=cdd.copy_inequalities(cdd.polyhedron_from_matrix(C))
    H_S=np.array(H_S.array)
    H_S[np.abs(H_S) < 1e-6] = 0
    return H_S


def FindEffectConeFacets(E):
    """
    Computes the facets of the positive cone defined by the input set of effects E of an accessible GPT fragment.

    Args:
    - E (np.ndarray): A numpy array representing the states in the accessible GPT fragment.
        shape = (dimension of the space spanned by effects in the accessible GPT fragment, number of effects)

    Returns:
    np.ndarray:
        An array listing the facets of the positive cone defined by the input matrix E.
        shape = (dimension of the space spanned by effects in the accessible GPT fragment, number of facets)
    """
    
    C = cdd.matrix_from_array(E.T)
    C.rep_type = cdd.RepType.INEQUALITY
    return np.array(cdd.copy_generators(cdd.polyhedron_from_matrix(C)).array).T
