# bloch4spin — Generalized Bloch toolkit for spin-j systems

`bloch4spin` is a small NumPy/SciPy/SymPy toolkit for finite spin systems using the generalized Bloch (COBITO) representation. It supports operator conversion, Hamiltonian and Lindblad evolution, Kraus/POVM/projective measurements, and sequential joint probabilities.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer
- SciPy 1.11 or newer
- SymPy 1.10 or newer

## Installation

Install from a local checkout:

```bash
python -m pip install .
```

## Quick start

```python
import numpy as np
from bloch4spin import (
    BlochBasis,
    GeneralizedBlochEvolutionMatrix,
    GeneralizedBlochObservable,
    GeneralizedBlochState,
    run,
)

# Use one explicit basis throughout the calculation.
basis = BlochBasis(2)

# Spin-1/2 Hamiltonian H = J_y and its Bloch-space generator.
H = np.array([[0, -0.5j], [0.5j, 0]])
L = GeneralizedBlochEvolutionMatrix.from_Hamiltonian(H, basis=basis)

# Maximally mixed initial state.
state = GeneralizedBlochState.from_matrix(np.eye(2) / 2, basis=basis)

# Projective J_z measurements.
projectors = [np.diag([1, 0]), np.diag([0, 1])]
observables = [
    GeneralizedBlochObservable.from_projector(P, basis=basis)
    for P in projectors
]

# joint[i, j]: probability of outcome i at t=0.7 and outcome j at t=1.9.
joint = run(L, state, [(0.7, observables), (1.9, observables)])
print(joint)
```

A `BlochBasis` owns its dimension and caches, so independent dimensions can coexist. Initialization constructs the COBITO basis; tensor-product coefficients are computed only when needed. Code that performs many tensor products can call `basis.precompute_products()` once.

Pass the same explicit `basis` to every state, evolution matrix, and measurement in a calculation. Objects from different basis instances are not mixed implicitly.

## Measurement normalization

- `normalized`: check for a complete, trace-preserving measurement;
- `renormalized`: normalize the retained outcome probabilities, as in explicit postselection;
- `unnormalized`: return raw outcome probabilities.

## Further reading

- [Mathematical and implementation guide](https://github.com/PolarisMegrez/bloch4spin/blob/main/user_guide_en.md)
- [中文说明](https://github.com/PolarisMegrez/bloch4spin/blob/main/user_guide_ch.md)

Licensed under the MIT License.
