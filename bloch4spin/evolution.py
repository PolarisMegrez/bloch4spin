"""Hamiltonians, Liouvillians, and states in the COBITO representation."""

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock

import numpy as np
from scipy.linalg import expm as dense_expm
from scipy.sparse import csr_matrix, issparse, isspmatrix_csr
from scipy.sparse.linalg import expm_multiply

from .basis import (
    BlochBasis,
    GeneralizedBlochVector,
    _resolve_basis,
    bloch_hermitian_transpose,
)

__all__ = [
    "GeneralizedBlochHamiltonian",
    "GeneralizedBlochEvolutionMatrix",
    "GeneralizedBlochState",
]


@dataclass
class GeneralizedBlochHamiltonian(GeneralizedBlochVector):
    def __post_init__(self):
        super().__post_init__()
        self._validate_hermitian()

    def _validate_hermitian(self):
        adjoint = bloch_hermitian_transpose(self).data
        left = self.data.toarray() if issparse(self.data) else self.data
        right = adjoint.toarray() if issparse(adjoint) else adjoint
        if not np.allclose(left, right, rtol=1e-10, atol=1e-12):
            raise ValueError("Hamiltonian must be Hermitian")


@dataclass
class GeneralizedBlochEvolutionMatrix:
    data: csr_matrix
    basis: BlochBasis | None = None
    _dense: np.ndarray | None = field(default=None, init=False, repr=False)
    _expm_cache: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _expm_cache_maxsize: int = field(default=512, init=False, repr=False)
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cache_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = Lock()

    def __post_init__(self):
        self.basis = _resolve_basis(self.basis)
        array = self.data
        if not isspmatrix_csr(array):
            array = csr_matrix(np.asarray(array, dtype=np.complex128))
        expected = self.basis.d**2
        if array.shape != (expected, expected):
            raise ValueError("Evolution matrix has incompatible shape for this basis")
        self.data = array.astype(np.complex128)

    def _wrap_result(self, data):
        return self.__class__(data, basis=self.basis)

    def _check_basis(self, other):
        if not isinstance(other, GeneralizedBlochEvolutionMatrix) or other.basis is not self.basis:
            raise ValueError("Evolution matrices belong to different bases")

    def _invalidate_cache(self):
        self._dense = None
        self._expm_cache.clear()

    def copy(self):
        return self.__class__(self.data.copy(), basis=self.basis)

    def __array__(self, dtype=None):
        array = self.data.toarray()
        return array.astype(dtype) if dtype is not None else array

    def __add__(self, other):
        self._check_basis(other)
        return self._wrap_result(self.data + other.data)

    def __sub__(self, other):
        self._check_basis(other)
        return self._wrap_result(self.data - other.data)

    def __mul__(self, scalar):
        return self._wrap_result(self.data * scalar) if np.isscalar(scalar) else NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self._wrap_result(self.data / scalar) if np.isscalar(scalar) else NotImplemented

    def __neg__(self):
        return self._wrap_result(-self.data)

    def __iadd__(self, other):
        self._check_basis(other)
        self.data = (self.data + other.data).tocsr()
        self._invalidate_cache()
        return self

    def __isub__(self, other):
        self._check_basis(other)
        self.data = (self.data - other.data).tocsr()
        self._invalidate_cache()
        return self

    def __imul__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        self.data = (self.data * scalar).tocsr()
        self._invalidate_cache()
        return self

    def __itruediv__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        self.data = (self.data / scalar).tocsr()
        self._invalidate_cache()
        return self

    @classmethod
    def from_Hamiltonian(cls, Hamiltonian, basis=None):
        if isinstance(Hamiltonian, GeneralizedBlochVector):
            basis = Hamiltonian.basis
            H = Hamiltonian.to_matrix()
        else:
            basis = _resolve_basis(basis)
            H = np.asarray(Hamiltonian)
        H = GeneralizedBlochHamiltonian.from_matrix(H, basis=basis).to_matrix()
        matrices = basis.matrices
        commutators = (
            np.einsum("ij,ajk->aik", H, matrices)
            - np.einsum("aik,kj->aij", matrices, H)
        )
        data = -1j * np.einsum("aji,bji->ab", matrices, commutators)
        return cls(csr_matrix(data), basis=basis)

    @classmethod
    def from_Lindblad(cls, jump, basis=None):
        if isinstance(jump, GeneralizedBlochVector):
            basis = jump.basis
            K = jump.to_matrix()
        else:
            basis = _resolve_basis(basis)
            K = np.asarray(jump)
        matrices = basis.matrices
        Kd = K.conj().T
        A = Kd @ K
        jumps = np.einsum("ij,ajk,kl->ail", Kd, matrices, K, optimize=True)
        transformed = jumps - 0.5 * (
            np.einsum("ij,ajk->aik", A, matrices)
            + np.einsum("aik,kj->aij", matrices, A)
        )
        data = np.conj(np.einsum("aji,bji->ab", transformed, matrices))
        return cls(csr_matrix(data), basis=basis)

    def _get_dense(self):
        if self._dense is None:
            self._dense = self.data.toarray()
        return self._dense

    def _dense_expm_cached(self, time):
        key = float(time)
        with self._cache_lock:
            if key in self._expm_cache:
                self._expm_cache.move_to_end(key)
                return self._expm_cache[key]
        matrix = dense_expm(self._get_dense() * key)
        with self._cache_lock:
            self._expm_cache[key] = matrix
            if len(self._expm_cache) > self._expm_cache_maxsize:
                self._expm_cache.popitem(last=False)
        return matrix


@dataclass
class GeneralizedBlochState(GeneralizedBlochVector):
    def __post_init__(self):
        super().__post_init__()
        adjoint = bloch_hermitian_transpose(self).data
        left = self.data.toarray() if issparse(self.data) else self.data
        right = adjoint.toarray() if issparse(adjoint) else adjoint
        if not np.allclose(left, right, rtol=1e-10, atol=1e-10):
            raise ValueError("Density matrix must be Hermitian")

    def normalization(self):
        d = self.basis.d
        target = 1 / np.sqrt(d)
        if issparse(self.data):
            r00 = self.data[0, :].toarray().flatten()
            scale = np.zeros_like(r00)
            mask = r00 != 0
            scale[mask] = target / r00[mask]
            from scipy.sparse import diags
            self.data = self.data @ diags(scale)
            return scale
        if self.data.ndim == 1:
            r00 = self.data[0]
            if r00 == 0:
                raise ValueError("Cannot normalize a state with zero trace")
            scale = target / r00
            self.data *= scale
            return scale
        r00 = self.data[0, :]
        scale = np.zeros_like(r00)
        mask = r00 != 0
        scale[mask] = target / r00[mask]
        self.data *= scale[np.newaxis, :]
        return scale

    def evolve(self, evolution, time):
        if not isinstance(evolution, GeneralizedBlochEvolutionMatrix):
            raise TypeError("L must be a GeneralizedBlochEvolutionMatrix")
        if evolution.basis is not self.basis:
            raise ValueError("State and evolution belong to different bases")
        time = float(time)
        if evolution.data.shape[0] <= 16:
            self.data = evolution._dense_expm_cached(time) @ self.data
        else:
            self.data = expm_multiply(evolution.data * time, self.data)
        adjoint = bloch_hermitian_transpose(self)
        self.data = 0.5 * (self.data + adjoint.data)
