"""
bloch4spin: Canonical Orthonormal Basis of Irreducible Tensor Operators (COBITO)
--------------------------------------------------------------------------------
Implements the Canonical Orthonormal Basis of Irreducible Tensor Operators
(COBITO) ``T_q^{(k)}`` for Hilbert space dimension ``d = 2j + 1`` under the
Condon–Shortley convention. Provides construction of the real matrix basis,
Bloch-vector wrappers, tensor products in Bloch coordinates, and commutator
structure constants derived from tensor products.

Definitions
-----------
The basis elements are

.. math::
     T_q^{(k)} = \sum_{m_1,m_2} (-1)^{j-m_2}\,\sqrt{2k+1}
     \begin{pmatrix} j & k & j \\ m_1 & q & -m_2 \end{pmatrix}
     |j m_2\rangle\langle j m_1|,

and obey orthonormality with respect to the Hilbert–Schmidt inner product. With
the chosen normalization, ``T_0^{(0)} = I/\sqrt{d}``.

Tensor products expand as

.. math::
     T_{q_1}^{(k_1)} T_{q_2}^{(k_2)} = \sum_{k_3,q_3}
     c_{k_1 k_2 k_3}^{q_1 q_2 q_3} T_{q_3}^{(k_3)},

with coefficients expressed using Wigner 3j/6j symbols. Commutators follow from
``[A,B] = AB - BA`` and are computed via these product coefficients.

Public API
----------
- ``bloch_init``, ``bloch_dim``, ``bloch_basis``
- ``basis_product``, ``structure_const``
- ``GeneralizedBlochVector``, ``bloch_tensor_product``, ``bloch_inner_product``
- ``bloch_hermitian_transpose``

Notes
-----
- Normalization: ``T_{0}^{(0)} = I/\sqrt{d}`` ensures orthonormality.
- Indexing: Linear index ``n = k(k+1) + q`` with inverse recovered by minimal
    ``k`` such that ``k(k+2) \ge n``.
- Phases: Reversed factor ordering in tensor products contributes a phase
    ``(-1)^{k_1+k_2+k_3}`` from 3j symmetry.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable
import numpy as np
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

__all__ = [
    "GeneralizedBlochVector",
    "bloch_init",
    "bloch_dim",
    "bloch_basis",
    "basis_product",
    "structure_const",
    "bloch_inner_product",
    "bloch_tensor_product",
    "bloch_hermitian_transpose",
]

_bloch_dim: int | None = None
"""Current Hilbert space dimension ``d`` (set by :func:`bloch_init`)."""

_bloch_basis: Dict[Tuple[int, int], np.ndarray] = {}
"""Cache of COBITO basis matrices keyed by ``(k, q)``.

For ``k > 0``, only nonnegative ``q`` are stored. Negative ``q`` are generated
on demand via the relation ``T_{-q}^{(k)} = (-1)^q (T_q^{(k)})^T`` under the
adopted convention where COBITO are real matrices.
"""

_bloch_product_coeff: Dict[Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray] = {}
"""Compact tensor product decomposition coefficients for COBITO basis ``((k1,q1),(k2,q2))``.

For each pair with ``index(k1,q1) < index(k2,q2)``, stores a contiguous vector
over allowed ``k3`` in ``[max(|k1-k2|, |q1+q2|), min(k1+k2, d-1)]``. Full vectors
are expanded by :func:`get_product_coeff` when needed.
"""

# --------- Indexing helpers ---------
def _ensure_initialized() -> None:
    """Ensure :func:`bloch_init` has been called.

    Raises
    ------
    RuntimeError
        If the module has not been initialized.
    """
    if _bloch_dim is None:
        raise RuntimeError("Bloch basis not initialized. Call bloch_init(d) first.")

def _idx_from_kq(k: int, q: int) -> int:
    """Map ``(k, q)`` to linear index ``n = k*(k+1) + q``."""
    return k * (k + 1) + q

def _kq_from_idx(n: int) -> Tuple[int, int]:
    """Inverse mapping from linear index ``n`` to ``(k, q)``.

    Finds the smallest nonnegative integer ``k`` such that ``k*(k+2) >= n``;
    then returns ``q = n - k*(k+1)``.
    """
    k = 0
    while k * (k + 2) < n:
        k += 1
    q = n - k * (k + 1)
    return k, q

# --------- COBITO construction ---------
def _generate_tensor_basis(d: int, k: int, q: int) -> np.ndarray:
    """Construct real COBITO matrix ``T_q^(k)`` for dimension ``d``.

    Parameters
    ----------
    d : int
        Hilbert space dimension, with ``d = 2j + 1``.
    k, q : int
        Tensor rank and component, with ``0 <= k <= d-1`` and ``-k <= q <= k``.

    Returns
    -------
    numpy.ndarray
        The real matrix representation of ``T_q^(k)``.
    """
    ITO = np.zeros((d, d), dtype=float)
    pref = np.sqrt(2 * k + 1)
    jR = Rational(d - 1, 2)
    if q >= 0:
        m1_start, m1_end = q, d
    else:
        m1_start, m1_end = 0, d + q
    for m1_idx in range(m1_start, m1_end):
        m1 = jR - m1_idx                # m descends by 1 each index step
        m2 = m1 + q                     # from selection rule m1 + q - m2 = 0
        m2_idx = m1_idx - q
        three_j = float(wigner_3j(jR, k, jR, -m2, q, m1))
        phase = (-1) ** (int(jR - m2))
        ITO[m2_idx, m1_idx] += phase * pref * three_j
    return ITO                          # real matrix

def _derive_tensor_product_coeff(d: int, k1: int, q1: int, k2: int, q2: int, k3: int, q3: int) -> float:
    """Return a single tensor-product coefficient.

    Coefficient ``c_{k1 k2 k3}^{q1 q2 q3}`` in the expansion
    ``T_{q1}^{(k1)} T_{q2}^{(k2)} = \sum_{k3,q3} c_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.

    Notes
    -----
    Condon–Shortley convention:

    ``c = sqrt((2k1+1)(2k2+1)(2k3+1)) * (-1)^(2j+q3) * {k1 k2 k3; j j j} * (k3 k2 k1; -q3 q2 q1)``,
    with ``j=(d-1)/2``, where ``{...}`` is the Wigner 6j and ``(...)`` the Wigner 3j symbol.
    """
    pref = np.sqrt((2 * k1 + 1) * (2 * k2 + 1) * (2 * k3 + 1))
    phase = (-1) ** (d - 1 + q3)
    jR = Rational(d - 1, 2)
    six_j = float(wigner_6j(k1, k2, k3, jR, jR, jR))
    three_j = float(wigner_3j(k3, k2, k1, -q3, q2, q1))
    return float(pref * phase * six_j * three_j)

def _init_tensor_basis(d: int) -> None:
    """Populate basis cache for dimension ``d``."""
    global _bloch_basis
    _bloch_basis.clear()
    # Build and store only q>=0 for k>0
    for k in range(1, d):               # since k runs 0...2j (integer)
        for q in range(0, k + 1):
            _bloch_basis[(k, q)] = _generate_tensor_basis(d, k, q)

def _init_tensor_product_coeff(d: int) -> None:
    """Populate product coefficient cache (optimized bounds version).

    Constrains ``q2`` so the linear index ordering ``n(k1,q1) <= n(k2,q2)`` holds,
    avoiding post-filtering and cutting roughly half the pair iterations for large ``d``.

    See Also
    --------
    _init_tensor_product_coeff_old : Simpler reference implementation (slower).
    """
    global  _bloch_product_coeff
    _bloch_product_coeff.clear()
    # Precompute product coefficients for ordered pairs; store compact k3-ranges.
    for k1 in range(1, d):
        for q1 in range(-k1, k1 + 1):
            for k2 in range(1, d):
                q2min = max(-k2, 1 - d - q1, k1 * (k1 + 1) + q1 - k2 * (k2 + 1))  # n(k1, q1) <= n(k2, q2)
                q2max = min(k2, d - 1 - q1)
                for q2 in range(q2min, q2max + 1):
                    q3 = q1 + q2
                    k3_min, k3_max = max(abs(k1 - k2), abs(q3)), min(k1 + k2, d - 1)
                    # allocate vector over k3 range; we will map later when expanding
                    coeffs = []
                    for k3 in range(k3_min, k3_max + 1):
                        c_val = _derive_tensor_product_coeff(d, k1, q1, k2, q2, k3, q3)
                        coeffs.append(c_val)
                    _bloch_product_coeff[((k1, q1), (k2, q2))] = np.array(coeffs, dtype=float)

def _init_tensor_product_coeff_old(d: int) -> None:
    """Legacy product coefficient population (reference / slower).

    Enumerates full ``q2`` range then discards pairs violating ordering. Retained
    only for clarity and comparison; not invoked by ``bloch_init``.
    """
    global  _bloch_product_coeff
    _bloch_product_coeff.clear()
    # Precompute commutator coefficients for ordered pairs; store compact k3-ranges.
    for k1 in range(1, d):
        for q1 in range(-k1, k1 + 1):
            n1 = _idx_from_kq(k1, q1)
            for k2 in range(1, d):
                for q2 in range(-k2, k2 + 1):
                    n2 = _idx_from_kq(k2, q2)
                    if n1 > n2:
                        continue
                    q3 = q1 + q2
                    k3_min, k3_max = max(abs(k1 - k2), abs(q3)), min(k1 + k2, d - 1)
                    if abs(q3) > d - 1:
                        continue
                    # allocate vector over k3 range; we will map later when expanding
                    coeffs = []
                    for k3 in range(k3_min, k3_max + 1):
                        c_val = _derive_tensor_product_coeff(d, k1, q1, k2, q2, k3, q3)
                        coeffs.append(c_val)
                    _bloch_product_coeff[((k1, q1), (k2, q2))] = np.array(coeffs, dtype=float)

# --------- Core vector class and operations ---------
@dataclass
class GeneralizedBlochVector:
    """Bloch-space vector wrapper over a 1D complex array of length ``d**2``.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional complex array of length ``d**2``.

    Notes
    -----
    - ``data`` is a mutable view; direct mutation can improve performance but use with care.
    - Supports elementwise arithmetic and scalar broadcasting via operator overloads.
    - Use :meth:`to_matrix` / :meth:`from_matrix` to convert between operators and Bloch vectors.
    """
    data: np.ndarray  # shape ((2j+1)^2,), complex

    # Prefer our arithmetic when mixed with NumPy arrays
    __array_priority__ = 1000

    def __post_init__(self) -> None:
        """Validate shape and dtype of the underlying array."""
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        expected = d * d
        if self.data.shape != (expected,):
            raise ValueError(f"Expected vector of length {expected}, got {self.data.shape}.")
        if self.data.dtype != complex:
            self.data = self.data.astype(complex)

    # ---- NumPy interop ----
    def __array__(self, dtype=None) -> np.ndarray:
        """Return the underlying data as a NumPy array (optionally cast)."""
        arr = self.data
        return arr.astype(dtype) if dtype is not None else arr

    # ---- Helpers ----
    @staticmethod
    def _is_scalar(x) -> bool:
        """Return True if ``x`` is a scalar (Python or NumPy scalar)."""
        return np.isscalar(x) or isinstance(x, (np.generic,))

    @staticmethod
    def _as_1d_array(x) -> np.ndarray | None:
        """Coerce ``x`` to a 1D array of length ``d**2`` or return ``None``."""
        try:
            arr = np.asarray(x)
        except Exception:
            return None
        if arr.ndim != 1:
            return None
        d = _bloch_dim  # type: ignore
        if arr.shape[0] != d * d:
            return None
        return arr

    @classmethod
    def _wrap_result(cls, arr: np.ndarray) -> "GeneralizedBlochVector":
        """Wrap a 1D complex array, preserving subclass type (validates length)."""
        return cls(arr)

    def copy(self) -> "GeneralizedBlochVector":
        """Return a deep copy, preserving the dynamic subclass type."""
        return self.__class__(self.data.copy())

    def __repr__(self) -> str:
        """Compact string representation with logical shape and dtype."""
        d = _bloch_dim if _bloch_dim is not None else 0
        return f"GeneralizedBlochVector(shape=({d*d if d else 0},), dtype=complex)"

    @staticmethod
    def zeros() -> "GeneralizedBlochVector":
        """Return a zero Bloch vector for the current dimension ``d``."""
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        return GeneralizedBlochVector(np.zeros((d * d,), dtype=complex))

    @staticmethod
    def from_matrix(mat: np.ndarray) -> "GeneralizedBlochVector":
        """Convert an operator matrix to its Bloch vector.

        Parameters
        ----------
        mat : numpy.ndarray
            Square matrix of shape ``(d, d)``.

        Returns
        -------
        GeneralizedBlochVector
            Bloch coefficients ``r_{kq} = Tr(mat @ T_q^(k))``.

        Raises
        ------
        ValueError
            If the matrix shape does not match current ``d``.
        """
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        if mat.shape != (d, d):
            raise ValueError("Density/operator matrix has incompatible shape.")
        r = np.zeros((d * d,), dtype=complex)
        for k in range(0, d):
            for q in range(-k, k + 1):
                T = bloch_basis((k, q))
                r[_idx_from_kq(k, q)] = np.trace(mat @ T)
        return GeneralizedBlochVector(r)

    def to_matrix(self) -> np.ndarray:
        """Reconstruct the operator matrix from Bloch coefficients.

        Returns
        -------
        numpy.ndarray
            Matrix ``A = \sum_{k,q} r_{kq} T_{q}^{(k)}`` of shape ``(d, d)``.
        """
        _ensure_initialized()
        d = _bloch_dim   # type: ignore
        mat = np.zeros((d, d), dtype=complex)
        for k in range(0, d):
            for q in range(-k, k + 1):
                mat += self.data[_idx_from_kq(k, q)] * bloch_basis((k, q))
        return mat

    # ---- Arithmetic operators ----
    def __add__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(self.data + other.data)
        if self._is_scalar(other):
            return self._wrap_result(self.data + other)
        arr = self._as_1d_array(other)
        if arr is not None:
            return self._wrap_result(self.data + arr)
        return NotImplemented

    def __radd__(self, other):
        # commutative
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(self.data - other.data)
        if self._is_scalar(other):
            return self._wrap_result(self.data - other)
        arr = self._as_1d_array(other)
        if arr is not None:
            return self._wrap_result(self.data - arr)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            return self._wrap_result(other.data - self.data)
        if self._is_scalar(other):
            return self._wrap_result(other - self.data)
        arr = self._as_1d_array(other)
        if arr is not None:
            return self._wrap_result(arr - self.data)
        return NotImplemented

    def __mul__(self, other):
        # Scalar multiplication
        if self._is_scalar(other):
            return self._wrap_result(self.data * other)
        # Bloch-space operator product with another vector or array-like (1D, length d^2)
        if isinstance(other, GeneralizedBlochVector):
            return bloch_tensor_product(self, other)
        arr = self._as_1d_array(other)
        if arr is not None:
            return bloch_tensor_product(self, GeneralizedBlochVector(arr.astype(complex)))
        return NotImplemented

    def __rmul__(self, other):
        # Scalar multiplication (commutative)
        if self._is_scalar(other):
            return self._wrap_result(other * self.data)
        # Bloch-space operator product for array-like on the left
        arr = self._as_1d_array(other)
        if arr is not None:
            return bloch_tensor_product(GeneralizedBlochVector(arr.astype(complex)), self)
        return NotImplemented

    def __truediv__(self, other):
        # Only scalar division is supported
        if self._is_scalar(other):
            return self._wrap_result(self.data / other)
        return NotImplemented

    def __neg__(self):
        return self._wrap_result(-self.data)

    # ---- In-place operators (mutating) ----
    def __iadd__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self.data += other.data
            return self
        if self._is_scalar(other):
            self.data += other
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            self.data += arr
            return self
        return NotImplemented

    def __isub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self.data -= other.data
            return self
        if self._is_scalar(other):
            self.data -= other
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            self.data -= arr
            return self
        return NotImplemented

    def __imul__(self, other):
        if self._is_scalar(other):
            self.data *= other
            return self
        if isinstance(other, GeneralizedBlochVector):
            prod = bloch_tensor_product(self, other)
            self.data = prod.data
            return self
        arr = self._as_1d_array(other)
        if arr is not None:
            prod = bloch_tensor_product(self, GeneralizedBlochVector(arr.astype(complex)))
            self.data = prod.data
            return self
        return NotImplemented

    def __itruediv__(self, other):
        # Only scalar division is supported
        if self._is_scalar(other):
            self.data /= other
            return self
        return NotImplemented

# --------- Public API ---------
def bloch_init(d: int) -> None:
    """Initialize global caches for dimension ``d = 2j + 1``.

    Parameters
    ----------
    d : int
        Hilbert space dimension. Must be a positive integer.

    Raises
    ------
    ValueError
        If ``d <= 0`` or not an integer.
    """
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer.")
    global _bloch_dim
    # If we've already initialized for this dimension and caches exist, skip
    if _bloch_dim == d and _bloch_basis and _bloch_product_coeff:
        return
    _bloch_dim = d
    _init_tensor_basis(d)
    _init_tensor_product_coeff(d)

def bloch_dim() -> int:
    """Return the current Hilbert space dimension ``d`` set by ``bloch_init``.

    Returns
    -------
    int
        The dimension ``d``.

    Raises
    ------
    RuntimeError
        If the module has not been initialized via :func:`bloch_init`.
    """
    _ensure_initialized()
    return int(_bloch_dim)  # type: ignore

def bloch_basis(kq: Tuple[int, int]) -> np.ndarray:
    """Return COBITO basis matrix ``T_q^(k)``.

    Parameters
    ----------
    kq : tuple of int
        Pair ``(k, q)`` with ``0 <= k <= d-1`` and ``-k <= q <= k``.

    Returns
    -------
    numpy.ndarray
        The real matrix ``T_q^(k)`` of shape ``(d, d)``.

    Raises
    ------
    ValueError
        If ``(k, q)`` is invalid for the current dimension.

    Notes
    -----
    For ``q < 0``, the matrix is synthesized by ``T_{-q}^{(k)} = (-1)^q (T_q^{(k)})^T``.
    """
    _ensure_initialized()
    k, q = kq
    d = _bloch_dim  # type: ignore
    if k < 0 or k > (d - 1) or q < -k or q > k:
        raise ValueError("Invalid (k,q) for current spin j.")
    if k == 0:
        return np.eye(d, dtype=float) / np.sqrt(d)
    if q >= 0:
        return _bloch_basis[(k, q)]
    # q < 0: derive via transpose relation
    T_pos = _bloch_basis[(k, -q)]
    phase = (-1) ** (q)
    return phase * T_pos.T

def basis_product(kq1: Tuple[int, int], kq2: Tuple[int, int]) -> "GeneralizedBlochVector":
    """Return Bloch vector for the product ``T_{q1}^{(k1)} T_{q2}^{(k2)}``.

    Implements the expansion
    ``T_{q1}^{(k1)} T_{q2}^{(k2)} = \sum_{k3,q3} c_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.

    Parameters
    ----------
    kq1, kq2 : tuple(int, int)
        Basis labels ``(k, q)``.

    Returns
    -------
    GeneralizedBlochVector
        Length ``d**2`` vector with nonzeros at allowed ``(k3,q3)``.

    Raises
    ------
    ValueError
        If any of ``(k1,q1)`` or ``(k2,q2)`` is invalid for current ``d``.

    Notes
    -----
    - Identity: ``T_{00} = I/\sqrt{d}`` so products with ``T_{00}`` scale by ``1/\sqrt{d}``.
    - Ordering: Reversed order applies phase ``(-1)^{k1+k2+k3}`` from 3j symmetry.
    """
    _ensure_initialized()
    d = _bloch_dim   # type: ignore
    k1, q1 = kq1
    k2, q2 = kq2
    if k1 < 0 or k1 > (d - 1) or q1 < -k1 or q1 > k1:
        raise ValueError("Invalid (k1,q1) for current spin j.")
    if k2 < 0 or k2 > (d - 1) or q2 < -k2 or q2 > k2:
        raise ValueError("Invalid (k2,q2) for current spin j.")
    vec = np.zeros((d * d,), dtype=complex)

    # Handle identity cases: T00 = I/sqrt(d) so T00*T = T*T00 = (1/sqrt(d)) T
    if k1 == 0 or k2 == 0:
        n = _idx_from_kq(k1 + k2, q1 + q2)
        vec[n] = 1.0 / np.sqrt(d)
        return GeneralizedBlochVector(vec)
    
    q3 = q1 + q2
    n1, n2 = _idx_from_kq(k1, q1), _idx_from_kq(k2, q2)
    # Determine sign/order
    if n1 > n2:
        coeffs = _bloch_product_coeff.get(((k2, q2), (k1, q1)))
    else:
        coeffs = _bloch_product_coeff.get(((k1, q1), (k2, q2)))
    if coeffs is None:
        return GeneralizedBlochVector.zeros()
    
    # Expand to full vector; only k3 in allowed range, q3=q1+q2
    k3_min, k3_max = max(abs(k1 - k2), abs(q3)), min(k1 + k2, d - 1)
    for idx, k3 in enumerate(range(k3_min, k3_max + 1)):
        n = _idx_from_kq(k3, q3)
        if n1 > n2:
            sign = (-1) ** (k1 + k2 + k3)
        else:
            sign = 1.0
        vec[n] = sign * coeffs[idx]
    return GeneralizedBlochVector(vec)

def structure_const(kq1: Tuple[int, int], kq2: Tuple[int, int]) -> "GeneralizedBlochVector":
    """Return Bloch vector of commutator structure constants.

    Parameters
    ----------
    kq1, kq2 : tuple(int, int)
        Basis labels ``(k, q)`` for the two factors in the commutator.

    Returns
    -------
    GeneralizedBlochVector
        Vector of coefficients ``f_{k_1 k_2 k_3}^{q_1 q_2 q_3}`` such that
        ``[T_{q1}^{(k1)}, T_{q2}^{(k2)}] = \sum_{k3,q3} f_{k1 k2 k3}^{q1 q2 q3} T_{q3}^{(k3)}``.

    Notes
    -----
    Implemented via tensor products: ``[A,B] = AB - BA``. Each product is
    expanded using cached coefficients from :func:`basis_product`.
    """
    _ensure_initialized()
    return basis_product(kq1, kq2) - basis_product(kq2, kq1)

def bloch_inner_product(u: GeneralizedBlochVector, v: GeneralizedBlochVector) -> complex:
    """Complex inner product ``<u, v>`` between two Bloch vectors.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors of length ``d**2``.

    Returns
    -------
    complex
        The value ``np.vdot(u.data, v.data)``.
    """
    return np.vdot(u.data, v.data)

def bloch_tensor_product(u: GeneralizedBlochVector,
                     v: GeneralizedBlochVector,
                     tol: float = 0.0) -> GeneralizedBlochVector:
    """Compute operator product ``A * B`` in Bloch space.

    Parameters
    ----------
    u, v : GeneralizedBlochVector
        Bloch-space vectors representing operators ``A`` and ``B``.
    tol : float, optional
        Components with absolute value ``<= tol`` are treated as zero.

    Returns
    -------
    GeneralizedBlochVector
        Bloch vector of the product ``A*B``.

    Notes
    -----
    Accumulates over nonzero coordinates only. Rough complexity ``O(p*q)`` for
    ``p,q`` nonzero terms. Coefficients drawn from :func:`basis_product` cache.
    """
    _ensure_initialized()

    idx_r = np.flatnonzero(np.abs(u.data) > tol)   # non-zero indices
    idx_s = np.flatnonzero(np.abs(v.data) > tol)   # non-zero indices
    if idx_r.size == 0 or idx_s.size == 0:
        return GeneralizedBlochVector.zeros()

    ru = u.data[idx_r]
    sv = v.data[idx_s]

    kq_r = [_kq_from_idx(int(n)) for n in idx_r]
    kq_s = [_kq_from_idx(int(n)) for n in idx_s]

    # Cache product coefficient vectors to avoid repeated expansion
    bloch_vec_cache: dict[tuple[int,int,int,int], np.ndarray] = {}
    bloch_vecs = []
    weights = []

    for i1, (k1, q1) in enumerate(kq_r):
        r = ru[i1]
        for i2, (k2, q2) in enumerate(kq_s):
            s = sv[i2]
            key = (k1, q1, k2, q2)
            # Get or compute the bloch vector
            bloch_vec = bloch_vec_cache.get(key)
            if bloch_vec is None:
                bloch_vec = basis_product((k1, q1), (k2, q2)).data
                bloch_vec_cache[key] = bloch_vec
            # Skip zero vectors
            if np.any(bloch_vec):
                bloch_vecs.append(bloch_vec)
                weights.append(r * s)

    if not weights:  # all commutators zero
        return GeneralizedBlochVector.zeros()

    coeff_mat = np.vstack(bloch_vecs)      # shape: (P, D)
    weights_vec = np.asarray(weights)      # shape: (P,)
    out = weights_vec @ coeff_mat          # (D,)

    return GeneralizedBlochVector(out)

def bloch_hermitian_transpose(r: GeneralizedBlochVector) -> GeneralizedBlochVector:
    """Hermitian transpose in Bloch space.

    Parameters
    ----------
    r : GeneralizedBlochVector
        Bloch vector with components ``r_{kq}``.

    Returns
    -------
    GeneralizedBlochVector
        The vector ``s`` with components ``s_{kq} = (-1)^q conj(r_{k,-q})``.
    """
    _ensure_initialized()
    d = _bloch_dim  # type: ignore
    s = np.zeros((d * d,), dtype=complex)
    # q = 0 terms
    for k in range(0, d):
        s[_idx_from_kq(k, 0)] = np.conj(r.data[_idx_from_kq(k, 0)])
    # q != 0 terms: fill positive and negative q simultaneously
    for k in range(0, d):
        for q in range(1, k + 1):
            phase = (-1) ** q
            s[_idx_from_kq(k, q)] = phase * np.conj(r.data[_idx_from_kq(k, -q)])
            s[_idx_from_kq(k, -q)] = phase * np.conj(r.data[_idx_from_kq(k, q)])
    return GeneralizedBlochVector(s)