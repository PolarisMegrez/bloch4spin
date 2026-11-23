# bloch4spin — Generalized Bloch toolkit for spin-j systems

`bloch4spin` is a lightweight NumPy/SciPy/SymPy toolkit for finite spin systems using the generalized Bloch (COBITO) representation. It supports operator conversion, Hamiltonian and Lindblad evolution, quantum measurements, and sequential joint-probability calculations.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer
- SciPy 1.11 or newer
- SymPy 1.10 or newer

## Installation

Install the current release from PyPI:

```bash
python -m pip install bloch4spin
```

Alternatively, install from source:

```bash
git clone https://github.com/PolarisMegrez/bloch4spin.git
cd bloch4spin
python -m pip install .
```

For development, use `python -m pip install -e .` from the repository root.

## Quick start

The following example computes the joint probabilities for two projective measurements of a spin-1/2 system evolving under $H=J_y$:

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

A `BlochBasis` owns its dimension, operator basis, and caches. Pass the same basis instance to every state, evolution matrix, and measurement in a calculation; objects from different basis instances are not mixed implicitly. Tensor-product coefficients are computed only when needed, and `basis.precompute_products()` can be used to warm this cache explicitly.

## Measurement normalization

Three modes are available for outcome probabilities:

- `normalized`: check for a complete, trace-preserving measurement;
- `renormalized`: normalize the retained outcome probabilities, as in explicit postselection;
- `unnormalized`: return raw outcome probabilities.

These modes control probability handling and completeness checks. For every nonzero-probability outcome, `apply_measurement` returns a separately normalized conditional state.

## Documentation

- [Mathematical and implementation guide](https://github.com/PolarisMegrez/bloch4spin/blob/main/user_guide_en.md)
- [中文说明](https://github.com/PolarisMegrez/bloch4spin/blob/main/user_guide_ch.md)
- [PyPI project page](https://pypi.org/project/bloch4spin/)

## License and AI Assistance

Released under the [MIT License](https://github.com/PolarisMegrez/bloch4spin/blob/main/LICENSE).

AI coding assistants were used to support software development and documentation. The author reviewed and approved the final release and takes responsibility for its contents.
