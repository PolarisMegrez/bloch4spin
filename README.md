# bloch4spin — Generalized Bloch toolkit for spin-j systems

bloch4spin provides a generalized Bloch representation for finite spin systems (`d = 2j + 1`) using irreducible tensor operators (ITO). The ITO-based Bloch vector is typically sparse, allowing fast Hamiltonian-to-Liouvillian conversion, efficient time evolution, and lightweight measurement operations.

**Authors:** <Author Name 1>, <Author Name 2> (<Affiliation Placeholder>)

- Real ITO basis matrices for arbitrary spin dimension.
- Direct operator ↔ Bloch vector conversion (`T_0^{(0)} = I/\sqrt{d}` keeps trace in component 0).
- Automatic Liouvillian construction from a Hamiltonian (unitary dynamics in Bloch space).
- Flexible measurement superoperators: Kraus, POVM element (via matrix square root), or projector.
- Experiment engine for time-ordered evolution + branching measurements producing joint outcome tensors.
- Exact probability extraction: `p = sqrt(d) * r'_0` for each branch.

## Installation

**Requirements: Python >= 3.9, Numpy and SymPy.**

Install from source (editable for development):

```powershell
git clone https://github.com/PolarisMegrez/bloch4spin.git
cd bloch4spin
pip install -e .
```

Or install directly from a local checkout without editable mode:

```powershell
pip install .
```

## Notes

- Probability: $p=\sqrt{d}\,r'_0$ (trace stored in first Bloch component).
- Structure constants generated on demand (tensor-product expansion) for memory efficiency.
- Full mathematical background and examples: `user_guide_en.md`, `user_guide_ch.md`; demo notebooks show validated spin-1/2 and spin-1 cases.
