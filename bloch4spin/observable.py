"""Measurement superoperators in the COBITO representation."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse, isspmatrix_csr

from .basis import BlochBasis, GeneralizedBlochVector, _resolve_basis

__all__ = ["GeneralizedBlochObservable", "apply_measurement"]


@dataclass
class GeneralizedBlochObservable:
    data: csr_matrix
    basis: BlochBasis | None = None

    def __post_init__(self):
        self.basis = _resolve_basis(self.basis)
        array = self.data
        if not isspmatrix_csr(array):
            array = csr_matrix(np.asarray(array, dtype=np.complex128))
        expected = self.basis.d**2
        if array.shape != (expected, expected):
            raise ValueError("Observable superoperator has incompatible shape")
        self.data = array.astype(np.complex128)

    @classmethod
    def from_Kraus(cls, K, basis=None):
        if isinstance(K, GeneralizedBlochVector):
            basis = K.basis
            matrix = K.to_matrix()
        else:
            basis = _resolve_basis(basis)
            matrix = np.asarray(K)
        matrices = basis.matrices
        transformed = np.einsum(
            "ij,ajk,kl->ail", matrix.conj().T, matrices, matrix, optimize=True
        )
        data = np.conj(np.einsum("aji,bji->ab", transformed, matrices))
        return cls(csr_matrix(data), basis=basis)

    @classmethod
    def from_POVM_elem(cls, M, basis=None, atol=1e-12):
        if isinstance(M, GeneralizedBlochVector):
            basis = M.basis
            matrix = M.to_matrix()
        else:
            basis = _resolve_basis(basis)
            matrix = np.asarray(M)
        if not np.allclose(matrix, matrix.conj().T, atol=atol):
            raise ValueError("POVM element must be Hermitian")
        eigenvalues, vectors = np.linalg.eigh(matrix)
        if np.any(eigenvalues < -atol):
            raise ValueError("POVM element must be positive semidefinite")
        K = (vectors * np.sqrt(np.clip(eigenvalues, 0, None))) @ vectors.conj().T
        return cls.from_Kraus(K, basis=basis)

    @classmethod
    def from_projector(cls, P, basis=None):
        if isinstance(P, GeneralizedBlochVector):
            basis = P.basis
            matrix = P.to_matrix()
        else:
            basis = _resolve_basis(basis)
            matrix = np.asarray(P)
        if np.linalg.matrix_rank(matrix, tol=1e-10) != 1:
            raise ValueError("from_projector requires a rank-one projector; use from_Kraus instead")
        projector = GeneralizedBlochVector.from_matrix(matrix, basis=basis)
        vector = projector.data
        if issparse(vector):
            vector = vector.toarray().flatten()
        else:
            vector = vector.flatten()
        return cls(csr_matrix(np.outer(vector, np.conj(vector))), basis=basis)


def apply_measurement(observables, state, norm_mode="normalized", clip_negative=True, atol=1e-12):
    from .evolution import GeneralizedBlochState

    if norm_mode not in ("normalized", "renormalized", "unnormalized"):
        raise ValueError("norm_mode must be normalized, renormalized, or unnormalized")
    basis = state.basis
    d = basis.d
    if any(observable.basis is not basis for observable in observables):
        raise ValueError("Measurement and state belong to different bases")

    if norm_mode == "normalized":
        probability_row = np.zeros(d * d, dtype=complex)
        for observable in observables:
            probability_row += observable.data.getrow(0).toarray().ravel()
        expected = np.zeros(d * d, dtype=complex)
        expected[0] = 1
        if not np.allclose(probability_row, expected, atol=10 * atol):
            raise ValueError("Observable set is not trace preserving")

    input_data = state.data
    sqrt_d = np.sqrt(d)
    probabilities, states = [], []
    for observable in observables:
        updated = observable.data @ input_data
        r00 = updated[0, :]
        if issparse(r00):
            r00 = r00.toarray().flatten()
        probability = (sqrt_d * r00).real
        if probability.ndim == 0:
            if clip_negative and -atol < probability < 0:
                probability = 0.0
        else:
            probability = np.asarray(probability).flatten()
            if clip_negative:
                probability[(probability < 0) & (probability > -atol)] = 0.0
        post = GeneralizedBlochState(updated, basis=basis)
        if np.any(probability > 0):
            post.normalization()
        probabilities.append(probability)
        states.append(post)

    if norm_mode == "renormalized":
        total = sum(probabilities)
        if np.isscalar(total):
            if total > 0:
                probabilities = [p / total for p in probabilities]
        else:
            scale = np.zeros_like(total)
            mask = total > 0
            scale[mask] = 1 / total[mask]
            probabilities = [p * scale for p in probabilities]
    return list(zip(probabilities, states))
