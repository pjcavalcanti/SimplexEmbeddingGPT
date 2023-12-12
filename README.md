# Simplex Embedding for Generalised Probabilistic Theory (GPT) Fragments

This repository contains a Python implementation that finds a simplex embedding for a (possibly depolarised) generalised probabilistic theory (GPT) fragment, which was originally developed in <https://arxiv.org/abs/2204.11905> as a Linear Program implemented in Mathematica.

## Table of Contents
- Introduction
- Installation
- Troubleshooting
- Usage
- Acknowledgements
- Contributing
- License

## Introduction

This code has been developed for any user interested in investigating the existence of noncontextual ontological models for prepare-and-measure scenarios, either described by quantum theory or another GPT, as well as quantifying the resourcefulness of contextuality in tasks that have an equivalent implementation as prepare-and-measure scenarios. This is a translation of the original code, developed by Elie Wolfe in Mathematica, into Python. 

The main functionality in this repository checks for the existence of a simplex embedding for a GPT fragment. In particular, the main function:
1. constructs the inclusion and projection maps taking the states and effects from the GPT fragment to its accessible GPT fragment and back;
2. characterises the positive cones of states and effects in the accessible GPT fragment by enumerating its facets with the `pycddlib`; 
3. computes the minimal depolarising noise necessary for the existence of a simplicial-cone embedding; 
4. renormalises the simplicial-cone embedding, deriving a simplicial embedding; 
5. constructs the epistemic states and response functions of the corresponding noncontextual ontological model (respectivelly, the embedded states and effects).


## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repository.git
```

Install dependencies:

```bash
pip install numpy scipy cvxpy itertools pycddlib
```

Make sure to have the required dependencies installed before running the code.

## Troubleshooting
Installing `pycddlib` in Anaconda

If you are using Anaconda and encounter issues when installing `pycddlib`, you may need to install additional compilers. This is particularly relevant if you are on a Linux system. To resolve this, install `gcc_linux-64` and `gxx_linux-64` using the following commands:

```bash
conda install gcc_linux-64 gxx_linux-64
```

This step ensures that the necessary C++ compilers are available in your Anaconda environment, allowing for the successful installation and functioning of `pycddlib`.


## Usage

The main functionality of the code is encapsulated in the `SimplexEmbedding` function found in `simplexEmbedding.py`. This will take a set of states, a set of effects, a unit effect and a maximally mixed state **(necessarily in vector form)**, find the accessible GPT fragment representation for the states and effects with the `DefineAccessibleGPTFragment` function, characterise its cone facets, find the minimal amount of noise `r` necessary for a simplicial-cone embedding with the function `SimplicialConeEmbedding`, and compute a simplex embedding from the result and the respective sets of embedded states and effects, `μ` and `ξ`. The function outputs an array `(r,μ,ξ)`.

### Example Usage 

```python
from simplexEmbedding import SimplexEmbedding
import examples

states, effects, unit, mms = example1()
result = SimplexEmbedding(states, effects, unit, mms)
print("Result: Robustness of contextuality (r) = {result[0]}, Epistemic States (μ) = {result[1]}, Response Functions (ξ) = {result[2]}")
```

In this example, the printed output should be an array with a number `r`, which in the case of example 1 is equal to 0; a list `μ` of 4-dimensional vectors representing the epistemic states, and a list `ξ` of 4-dimensional vectors representing the response functions. 

## Files

`preprocessing.py`: contains functions to construct the Gell-Mann orthonormal basis of hermitian operators for a Hilbert space of any dimension, and convert states and inputs from the matricial representation to vector representation. Example usage can be found in examples.py for a quantum type input. **Calling the function `fromListOfMatrixToListOfVectors` is necessary whenever the input states and effects are represented by density operators and POVMs**. The main function of this repository will not run if the input states and effects are not in vector form.

`math_tools.py`: provides mathematical accessories necessary for the main linear program, such as a function `rref` finding the Reduced Row Echelon Form of a matrix in order to determine the dimension of the space spanned by the sets of states and effects in their accessible GPT fragment representation. The functions characterising the positive cone of states and effects are also specified, employing tools imported from the `pycddlib` library.

`simplexEmbedding.py`: provides the main functionality of the repository, as explained above, and auxiliary functions needed for the main computation.

`examples.py`: Provides example data for testing the main functions. These are the 4 examples explored in <https://arxiv.org/abs/2204.11905>.

## Acknowledgements

This project was funded by the Program for Young Leaders of UG Research Groups - IDUB junG of the University of Gdańsk. Some functions in this repository are inspired on the work of Jonathan Gross, Stelios Sfakianakis, and Mathew Weiss, who we hereby acknowledge.

## Contributing

If you'd like to contribute to the project, feel free to submit issues or pull requests. Any optimisation in the execution of the linear program is encouraged, particularly concerning inputs of greater dimension.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
