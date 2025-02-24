from scipy.linalg import lu
from scipy.sparse import csr_matrix
import numpy as np
import cvxpy as cp

from math_tools import rref, FindStateConeFacets, FindEffectConeFacets


def DefineAccessibleGPTFragment(statesOrEffects):
    """
    Constructs the accessible Generalized Probabilistic Theory (GPT) fragment from a given set of states or effects.
    Uses the Reduced Row Echelon Form (RREF) of the matrix of states/effects to construct the inclusion map (RREF without the kernel)
    and projection (pseudo inverse of the inclusion), and the set of states/effects represented in the accessible fragment.

    Args:
    statesOrEffects (np.ndarray): A numpy array representing states or effects. Should be a 2D array.
        shape = (gpt fragment dimension, number of states or effects)

    Returns:
    tuple: A tuple containing:
        - inclusionMap (np.ndarray): The inclusion map matrix derived from the Reduced Row Echelon Form (RREF).
            shape = (GPT fragment dimension, accessible fragment dimension)

        - projectionMap (np.ndarray): The projection map matrix, which is the pseudo-inverse of the inclusion map.
            shape = (accessible fragment dimension, GPT fragment dimension)

        - accessibleFragment (np.ndarray): The accessible fragment, computed as the projection of statesOrEffects.
            shape = (accessible fragment dimension, number of states or effects)
    """
    REF = rref(statesOrEffects.T)
    
    P, L, U = lu(statesOrEffects)
    r = len(np.unique(csr_matrix(U).indptr)) - 1
    
    inclusionMap = REF[:r, :].T
    projectionMap = np.linalg.pinv(inclusionMap.T@inclusionMap)@inclusionMap.T

    return inclusionMap, projectionMap, projectionMap @ statesOrEffects


def SimplicialConeEmbedding(H_S, H_E, accessibleFragmentBornRule, depolarizingMap):
    """
    Solves Linear Program 2 from the paper by testing whether a simplicial cone embedding exists
    given the GPT's state and effect cone facets, the bilinear form giving the Born rule in
    the accessible fragment, and its depolarizing map.

    Args:
    H_S (np.ndarray): A numpy array representing the state cone facets.
        shape = (number of state cone facets, dimension of states in the GPT accessible fragment)
        
    H_E (np.ndarray): A numpy array representing the effect cone facets.
        shape = (dimension of effects in the GPT accessible fragment, number of effect cone facets)
        
    accessibleFragmentBornRule (np.ndarray): The bilinear form giving the Born rule in the accessible fragment.
        shape = (dimension of states in the GPT accessible fragment, dimension of effects in the GPT accessible fragment)
        
    depolarizingMap (np.ndarray): The depolarizing map, typically the outer product of accessible fragment unit and MMS.
        shape = (dimension of effects in the GPT accessible fragment, dimension of states in the GPT accessible fragment)

    Returns:
    tuple: A tuple containing:
        - robustness (float): The minimum amount of noise such that the depolarizing map causes a simplicial cone embedding to exist. 
        - sigma (np.ndarray): The sigma matrix obtained from the optimization problem.
    """
    robustness, sigma = cp.Variable(nonneg=True), cp.Variable(
        shape=(H_E.shape[1], H_S.shape[0]), nonneg=True
    )

    problem = cp.Problem(
        cp.Minimize(robustness),
        [
            robustness * depolarizingMap
            + (1 - robustness) * accessibleFragmentBornRule
            - H_E @ sigma @ H_S
            == 0
        ],
    )

    problem.solve()
    return robustness.value, sigma.value


def SimplexEmbedding(states, effects, unit, mms, debug=False):
    """
    Constructs a noncontextual ontological model for the (possibly depolarized) GPT fragment.
    Tests whether a simplex embedding exists given the GPT's states, effects, unit, and maximally mixed state.
    

    Args:
    states (np.ndarray): A numpy array of states.
    effects (np.ndarray): A numpy array of effects.
    unit (np.ndarray): The unit effect.
    mms (np.ndarray): The maximally mixed state.
    debug (bool, optional): Flag to print debug information. Default is False.

    Returns:
    tuple: A tuple containing:
        - robustness (float): The robustness value.
        - ResponseFunction (np.ndarray): The response function matrix.
        - EpistemicStates (np.ndarray): The epistemic states matrix.
    """

    inclusion_S, projection_S, accessibleFragmentStates = DefineAccessibleGPTFragment(
        states
    )
    inclusion_E, projection_E, accessibleFragmentEffects = DefineAccessibleGPTFragment(
        effects
    )
    accessibleFragmentUnit = projection_E @ unit
    # MMS: Maximally mixed state
    accessibleFragmentMMS = projection_S @ mms

    H_S = FindStateConeFacets(accessibleFragmentStates)
    H_E = FindEffectConeFacets(accessibleFragmentEffects)

    # Bilinear form giving the Born rule in the accessible fragment:
    accessibleFragmentBornRule = inclusion_E.T @ inclusion_S
    depolarizingMap = np.outer(accessibleFragmentUnit, accessibleFragmentMMS)
    robustness, sigma = SimplicialConeEmbedding(
        H_S, H_E, accessibleFragmentBornRule, depolarizingMap
    )

    # Trivial factorization of the matrix sigma:
    alpha = H_S
    beta = H_E @ sigma

    tau_S = np.array(
        [
            alpha[i, :] * (accessibleFragmentUnit @ beta[:, i])
            for i in range(alpha.shape[0])
            if not np.isclose((accessibleFragmentUnit @ beta[:, i]), 0)
        ]
    )
    tau_E = np.array(
        [
            beta[:, i] / (accessibleFragmentUnit @ beta[:, i])
            for i in range(beta.shape[1])
            if not np.isclose((accessibleFragmentUnit @ beta[:, i]), 0)
        ]
    ).T

    ResponseFunction = tau_E.T @ inclusion_E.T @ effects
    EpistemicStates = tau_S @ inclusion_S.T @ states

    if debug:
        print(
            f"""
            
{inclusion_S.shape = },
{inclusion_E.shape = },
{H_S.shape = }
{H_E.shape = },
{accessibleFragmentBornRule.shape = },

{effects.shape = },
{inclusion_E.shape = },
{tau_E.shape = },

{tau_S.shape = },
{inclusion_S.shape = },
{states.shape = },

{robustness = },
{ResponseFunction.shape = },
{EpistemicStates.shape = },

Tau_E =
{tau_E},
Tau_S =
{tau_S},

effects =
{effects},
states =
{states},

Inc_E =
{inclusion_E},
Inc_S =
{inclusion_S},

{ResponseFunction},

{EpistemicStates},
"""
        )

    return robustness, ResponseFunction, EpistemicStates


if __name__ == "__main__":
    from examples import example1

    np.set_printoptions(precision=2, suppress=True)

    states, effects, unit, mms = example1()
    SimplexEmbedding(states, effects, unit, mms, debug=True)
