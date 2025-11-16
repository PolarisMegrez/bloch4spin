# bloch4spin — Generalized Bloch representation for spin-j systems

`bloch4spin` provides a numerically stable implementation of the generalized Bloch (COBITO) representation for finite spin-j systems (Hilbert space dimension `d = 2j + 1`). It focuses on:

- Construction of the COBITO orthonormal basis and Bloch vector operations
- Unitary evolution induced by a Hamiltonian in Bloch coordinates (Liouvillian form)
- Measurement superoperators derived from Kraus / POVM elements / projectors
- An experiment engine composing evolution and measurements to yield joint probability tensors

The guide is structured as: Mathematical Foundations → API Overview → Quick Start → Implementation Notes → References.

## Installation

Requirements: Python >= 3.9; NumPy and SymPy are required at runtime.

Install editable for development:

```powershell
pip install -e .
```

## Mathematical Foundations

This section systematizes irreducible tensor operators (ITO) under the Condon–Shortley convention and builds the Bloch coordinate framework.

1) COBITO definition and orthonormality

- Definition (matrix form):
  $T_q^{(k)} = \sum_{m_1,m_2} (-1)^{j-m_2}\,\sqrt{2k+1}\begin{pmatrix} j & k & j \\ m_1 & q & -m_2 \end{pmatrix} |j m_2\rangle\langle j m_1|$.
- Orthonormality (Hilbert–Schmidt inner product):
  $\mathrm{Tr}\big[T_{q_1}^{(k_1)\dagger} T_{q_2}^{(k_2)}\big] = \delta_{k_1 k_2}\,\delta_{q_1 q_2}$ with fixed $T_0^{(0)} = I/\sqrt{d}$.
- Phase & conjugation: $T_{-q}^{(k)} = (-1)^q (T_q^{(k)})^{\dagger}$; in our convention matrices can be chosen real.

2) Indexing and operator expansion

- Linear index: $(k,q) \mapsto n = k(k+1)+q$, yielding a Bloch vector of length `d^2`; inverse via smallest k with $k(k+2)\ge n$ and $q=n-k(k+1)$.
- Expansion: coefficients $r_{kq} = \mathrm{Tr}(A\,T_q^{(k)})$ give $A = \sum_{k,q} r_{kq} T_q^{(k)}$.

3) Tensor product and commutator structure

- Product: $T_{q_1}^{(k_1)} T_{q_2}^{(k_2)} = \sum_{K,Q} c_{k_1 k_2 K}^{q_1 q_2 Q} T_{Q}^{(K)}$ with coefficients from Wigner–3j/6j and Clebsch–Gordan (CG) combinations.
- Commutator (Schwegler 1965):
  $$[T_{q_1}^{(k_1)}, T_{q_2}^{(k_2)}] = \sum_{K,Q} f_{k_1 k_2 K}^{q_1 q_2 Q} T_{Q}^{(K)},$$
  $$f_{k_1 k_2 K}^{q_1 q_2 Q} = \sqrt{(2k_1+1)(2k_2+1)}\,(-1)^{2j+K}\begin{Bmatrix} k_1 & k_2 & K \\ j & j & j \end{Bmatrix}\big(1-(-1)^{k_1+k_2-K}\big)\langle k_1 q_1; k_2 q_2 | K Q \rangle,$$
  with $Q=q_1+q_2$, $\{\cdot\}$ the Wigner–6j symbol and $\langle\cdot|\cdot\rangle$ a CG coefficient.

4) Angular momentum commutators (examples)

- $[J_{z}, T_{q}^{(k)}]=q\,T_{q}^{(k)}$, $[J_{\pm}, T_{q}^{(k)}]=\sqrt{(k\mp q)(k\pm q+1)}\,T_{q\pm1}^{(k)}$ → $\sum_{i=x,y,z}[J_i,[J_i,T_q^{(k)}]]=k(k+1)T_q^{(k)}$.

5) Liouville equation in Bloch form

- Starting from $\dot\rho=-i[H,\rho]$ we obtain $\dot r = L r$ with
  $$(L_H)_{ab} = -i\,\mathrm{Tr}\big(T_a^{\dagger}[H, T_b]\big) = -i \sum_c h_c\, f_{ab}{}^{c},$$
  where $h_c$ are Bloch coefficients of $H$.

6) Measurement superoperators & probability extraction

- Kraus map: $\rho\mapsto K\rho K^{\dagger}$ becomes $r' = L_K r$ with row a of $L_K$ equal to $\overline{\mathrm{Bloch}(K^{\dagger} T_a K)}$.
- Rank-1 projector: if $P=|\psi\rangle\langle\psi|$, then $\rho'=(\langle\psi|\rho|\psi\rangle)P$ and $L = r_P r_P^{\dagger}$.
- Probability: using $T_0^{(0)}=I/\sqrt{d}$, $p=\mathrm{Tr}(\rho')=\sqrt{d}\,r'_0$ (component 0 of the unnormalized post-measurement vector).

## API Overview

### Module: `bloch4spin.basis`

- `bloch_init(d: int)` / `bloch_dim()` / `bloch_basis((k,q))`: initialize and query the COBITO basis.
- `GeneralizedBlochVector(data)`: 1D complex vector (length `d^2`) with arithmetic ops;
	`from_matrix(A)` and `to_matrix()` convert between operators and Bloch vectors.
- `basis_product((k1,q1),(k2,q2))`: Bloch vector for $T_{q_1}^{(k_1)} T_{q_2}^{(k_2)}$.
- `structure_const((k1,q1),(k2,q2))`: Bloch vector of the commutator coefficients.
- `bloch_tensor_product(u,v)`: Bloch-space product of two operators; `bloch_hermitian_transpose(r)`.

Notes vs prompt.md:
- We derive commutators via tensor products; no global dense structure-constant table is stored.
- Negative q matrices are synthesized from $T_{-q}^{(k)} = (-1)^q (T_q^{(k)})^T$; COBITO matrices are real in this convention.

### Module: `bloch4spin.evolution`

- `GeneralizedBlochHamiltonian(GeneralizedBlochVector)`: Hermiticity validated via COBITO adjoint.
- `GeneralizedBlochEvolutionMatrix(data)`: stores L; `from_Hamiltonian(H)` accepts ndarray or Bloch vector;
	constructs $(L_H)_{ab} = -i\,\mathrm{Tr}(T_a^{\dagger}[H, T_b])$ using `structure_const`.
- `GeneralizedBlochState(GeneralizedBlochVector)`: Hermiticity validated; `evolve(L,t)` applies $e^{Lt}$; `normalization()` enforces $r_{00}=1/\sqrt{d}$.

### Module: `bloch4spin.observable`

- `GeneralizedBlochObservable.from_Kraus(K)`: builds $L_K$ with rows $\overline{\mathrm{Bloch}(K^{\dagger} T_a K)}$.
- `GeneralizedBlochObservable.from_POVM_elem(M)`: uses $K=\sqrt{M}$ (via eigh) then delegates to `from_Kraus`.
- `GeneralizedBlochObservable.from_projector(P)`: rank-1 shortcut $L=r_P r_P^{\dagger}$.
- `apply_measurement([L_i], state, norm_mode=...)`: returns a list of `(probability, post_state)` with modes:
	- `"normalized"`: checks completeness (sum of `L_i[0,0]` equals 1)
	- `"renormalized"`: no check; renormalizes probabilities
	- `"unnormalized"`: no check; raw probabilities

### Module: `bloch4spin.engine`

- `run(L, init_state, schedule, norm_mode="normalized", t0=0.0) -> ndarray`:
	Simulates interleaved evolution and measurement. `schedule` is a list of
	`(time, [observables_for_that_step])`. Returns an ndarray of joint probabilities
	with shape `(M1, M2, ..., MK)` where `Mr` is the number of outcomes at step r.

## Quick Start

Physical context (spin-1/2 two-time measurement):

- System: spin $j=1/2$ (`d=2`), initial state $\rho_0=I/d$ (maximally mixed).
- Evolution: unitary rotation generated by $J_y$ (Larmor precession in a y-directed field).
- Measurements: projective measurements of $J_z$ at two times; first is uniform, second conditioned by transition matrix $T=|U|^2$, $U=e^{-i J_y \Delta t}$.
- Goal: obtain a joint probability tensor of shape `2×2` from `engine.run`.

```python
import numpy as np
from bloch4spin.basis import bloch_init
from bloch4spin.evolution import GeneralizedBlochEvolutionMatrix, GeneralizedBlochState
from bloch4spin.observable import GeneralizedBlochObservable
from bloch4spin.engine import run

# choose dimension (spin-1/2 -> d=2)
d = 2
bloch_init(d)

# build J_y in the |m_z> basis
def spin_ops(d: int):
    j = (d - 1) / 2.0
    Jp = np.zeros((d,d), complex); Jm = np.zeros((d,d), complex)
    for i in range(1, d):
        amp = np.sqrt(i * (d - i))
        Jp[i-1, i] = amp
    for i in range(0, d-1):
        amp = np.sqrt((i+1) * (d - (i+1)))
        Jm[i+1, i] = amp
    Jy = (Jp - Jm) / (2j)  # 2j is 2*i
    m_vals = np.array([j - i for i in range(d)], float)
    Jz = np.diag(m_vals)
    return Jy, Jz

Jy, Jz = spin_ops(d)
L = GeneralizedBlochEvolutionMatrix.from_Hamiltonian(Jy)

# initial state: maximally mixed
rho0 = np.eye(d) / d
r0 = GeneralizedBlochState.from_matrix(rho0)

# J_z projective measurement
Ps = [np.diag([1.0 if k==i else 0.0 for k in range(d)]) for i in range(d)]
obs = [GeneralizedBlochObservable.from_projector(P) for P in Ps]

# two measurement times
schedule = [(0.7, obs), (1.9, obs)]
joint = run(L, r0, schedule)
print(joint)
```

## Implementation Notes and Differences vs prompt.md

- Structure constants are produced on demand by tensor-product expansion; we do not keep a dense global `Dict` of full commutator tables.
- `GeneralizedBlochObservableSet` is not a separate type; each observable is a superoperator. Measurement sets are simple Python lists passed to `apply_measurement` / `engine.run`.
- The engine returns a dense ndarray for joint probabilities (ordered axes by step). A sparse-dictionary representation is feasible if needed and can be added as an option.
- Numerical tolerances are local to call sites (e.g., PSD checks use a small `atol`).

## References

This guide embeds key ITO formulas directly (no need to consult a separate `ITO.md`).

## References

- Wigner symbol computation: `sympy.physics.wigner` (`wigner_3j`, `wigner_6j`).
- H. Schwegler, “On commutation relations of tensor operators,” 1965. DOI: https://doi.org/10.1007/BF02814299.
- Eugene P. Wigner, “Group Theory and Its Application to the Quantum Mechanics of Atomic Spectra,” 1959, publisher & city: <publisher placeholder>, <city placeholder>.
