import numpy as np
from preprocessing import fromListOfMatrixToListOfVectors


def example1():
    #Example 1 in the main text: four pure states living in the real hemisphere of the Bloch sphere and their respective projectors.
    Zup, Zdown = np.array([1,0]), np.array([0,1])
    Xup, Xdown = np.array([1,1])/np.sqrt(2), np.array([1,-1])/np.sqrt(2)
    s = [np.outer(s, s.conj()) for s in [Zup, Zdown, Xup, Xdown]]
    return fromListOfMatrixToListOfVectors(s,s)

def example2():
    #Example 2 in the main text: four quantum eigenstates of a 4-level system, and 4 tilted projectors that span a smaller space than the dual of the states.
    s = 1/2*np.array([[1, 1, -1, 1],
                  [1, 1, 1, -1],
                  [1, -1, 1, 1],
                  [1, -1, -1, -1]])
    e = np.array([[1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [1, -1, 0, 0],
                  [1, 0, -1, 0]])
    u = np.array([2, 0, 0, 0])
    mms = np.array([0.5, 0, 0, 0])
    return s.T,e.T,u,mms

def example3():
    #Example 3 in the main text: the same quantum states from example 1, but with effects rotated by pi/4.
    Zup, Zdown = np.array([1,0]), np.array([0,1])
    Xup, Xdown = np.array([1,1])/np.sqrt(2), np.array([1,-1])/np.sqrt(2)

    E1 = np.array([[2+np.sqrt(2), np.sqrt(2)],
                  [np.sqrt(2), 2-np.sqrt(2)]])/4
    E2 = np.array([[2-np.sqrt(2), -np.sqrt(2)],
                  [-np.sqrt(2), 2+np.sqrt(2)]])/4
    E3 = np.array([[2+np.sqrt(2), -np.sqrt(2)],
                  [-np.sqrt(2), 2-np.sqrt(2)]])/4
    E4 = np.array([[2-np.sqrt(2), np.sqrt(2)],
                  [np.sqrt(2), 2+np.sqrt(2)]])/4
    s = [np.outer(s, s.conj()) for s in [Zup, Zdown, Xup, Xdown]]
    e = [E1, E2, E3, E4]
    return fromListOfMatrixToListOfVectors(s, e)
          
def example4():
    #Example 4 in the main text: gbit states and effects.
    s = np.array([[1, 1, 0],
                 [1, 0, 1],
                 [1, -1, 0],
                 [1, 0, -1]])
    e = 0.5*np.array([[1,-1,-1],
                      [1, 1,-1],
                      [1, 1, 1],
                      [1,-1, 1]])
    u = np.array([1, 0, 0])
    mms = np.array([1, 0, 0])
    return s.T, e.T, u, mms
