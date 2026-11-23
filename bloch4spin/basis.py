"""COBITO bases and Bloch-vector operations for finite spin systems."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import (
    csc_matrix,
    coo_matrix,
    csc_matrix as sparse_csc,
    hstack as sparse_hstack,
    issparse,
    spmatrix,
)
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j

__all__ = [
    "BlochBasis",
    "GeneralizedBlochVector",
    "bloch_commutator",
    "bloch_hermitian_transpose",
    "bloch_inner_product",
    "bloch_tensor_product",
    "structure_const",
]


def _idx_from_kq(k, q):
    return k * (k + 1) + q


def _kq_from_idx(n):
    k = 0
    while k * (k + 2) < n:
        k += 1
    return k, n - k * (k + 1)


class BlochBasis:
    """COBITO basis with dimension-local, lazily populated coefficient caches."""

    def __init__(self, d: int):
        if not isinstance(d, int) or d <= 0:
            raise ValueError("d must be a positive integer")
        self.d = d
        self.j = Rational(d - 1, 2)
        self._basis = {}
        self._matrices = None
        self._products = {}
        self._w3 = {}
        self._w6 = {}
        for k in range(1, d):
            for q in range(k + 1):
                self._basis[(k, q)] = self._generate_matrix(k, q)

    def _generate_matrix(self, k, q):
        d = self.d
        matrix = np.zeros((d, d), dtype=float)
        prefactor = np.sqrt(2 * k + 1)
        m1_start, m1_end = (q, d) if q >= 0 else (0, d + q)
        for m1_idx in range(m1_start, m1_end):
            m1 = self.j - m1_idx
            m2 = m1 + q
            three_j = float(wigner_3j(self.j, k, self.j, -m2, q, m1))
            phase = (-1) ** (m1_idx - q)
            matrix[m1_idx - q, m1_idx] += phase * prefactor * three_j
        return matrix

    def validate_kq(self, kq):
        k, q = kq
        if k < 0 or k >= self.d or q < -k or q > k:
            raise ValueError(f"Invalid (k,q) for dimension {self.d}")

    def matrix(self, kq):
        self.validate_kq(kq)
        k, q = kq
        if k == 0:
            return np.eye(self.d) / np.sqrt(self.d)
        if q >= 0:
            return self._basis[(k, q)]
        return (-1) ** q * self._basis[(k, -q)].T

    @property
    def matrices(self):
        if self._matrices is None:
            self._matrices = np.asarray([
                self.matrix(_kq_from_idx(index)) for index in range(self.d**2)
            ])
        return self._matrices

    def from_matrix(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.shape != (self.d, self.d):
            raise ValueError("Matrix has incompatible shape for this basis")
        coefficients = np.zeros((self.d**2, 1), dtype=complex)
        for k in range(self.d):
            for q in range(-k, k + 1):
                coefficients[_idx_from_kq(k, q), 0] = np.trace(
                    self.matrix((k, q)).T @ matrix
                )
        return GeneralizedBlochVector(coefficients, basis=self)

    def to_matrix(self, vector):
        if vector.basis is not self:
            raise ValueError("Vector belongs to a different basis")
        matrix = np.zeros((self.d, self.d), dtype=complex)
        data = vector.data.toarray() if issparse(vector.data) else vector.data
        for k in range(self.d):
            for q in range(-k, k + 1):
                value = data[_idx_from_kq(k, q)]
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                matrix += value * self.matrix((k, q))
        return matrix

    def _wigner_3j(self, key):
        if key not in self._w3:
            self._w3[key] = float(wigner_3j(*key))
        return self._w3[key]

    def _wigner_6j(self, key):
        if key not in self._w6:
            self._w6[key] = float(wigner_6j(*key))
        return self._w6[key]

    def _coefficient(self, k1, q1, k2, q2, k3):
        q3 = q1 + q2
        prefactor = np.sqrt((2 * k1 + 1) * (2 * k2 + 1) * (2 * k3 + 1))
        phase = (-1) ** (self.d - 1 + q3)
        angular = [k3, k2, k1]
        projections = [-q3, q2, q1]
        order = sorted(range(3), key=lambda i: angular[i], reverse=True)
        angular = [angular[i] for i in order]
        projections = [projections[i] for i in order]
        swaps = 0
        for i in range(3):
            for j in range(i + 1, 3):
                if order[i] > order[j]:
                    swaps += 1
        three_j = self._wigner_3j(tuple(angular + projections))
        six_j = self._wigner_6j(tuple(angular + [self.j, self.j, self.j]))
        if swaps % 2:
            three_j *= (-1) ** (k1 + k2 + k3)
        return float(prefactor * phase * six_j * three_j)

    def product(self, kq1, kq2):
        self.validate_kq(kq1)
        self.validate_kq(kq2)
        k1, q1 = kq1
        k2, q2 = kq2
        if k1 == 0 or k2 == 0:
            index = _idx_from_kq(k1 + k2, q1 + q2)
            vector = coo_matrix(
                ([1 / np.sqrt(self.d)], ([index], [0])),
                shape=(self.d**2, 1), dtype=complex,
            ).tocsc()
            return GeneralizedBlochVector(vector, basis=self)

        n1, n2 = _idx_from_kq(k1, q1), _idx_from_kq(k2, q2)
        swapped = n1 > n2
        ordered = (kq2, kq1) if swapped else (kq1, kq2)
        if ordered not in self._products:
            (a1, b1), (a2, b2) = ordered
            q3 = q1 + q2
            k3_min = max(abs(k1 - k2), abs(q3))
            k3_max = min(k1 + k2, self.d - 1)
            self._products[ordered] = {
                k3: self._coefficient(a1, b1, a2, b2, k3)
                for k3 in range(k3_min, k3_max + 1)
            }

        rows, values = [], []
        q3 = q1 + q2
        for k3, coefficient in self._products[ordered].items():
            value = coefficient
            if swapped:
                value *= (-1) ** (k1 + k2 + k3)
            if value != 0:
                rows.append(_idx_from_kq(k3, q3))
                values.append(value)
        vector = coo_matrix(
            (values, (rows, np.zeros(len(rows), dtype=int))),
            shape=(self.d**2, 1), dtype=complex,
        ).tocsc()
        return GeneralizedBlochVector(vector, basis=self)

    def precompute_products(self):
        for k1 in range(1, self.d):
            for q1 in range(-k1, k1 + 1):
                for k2 in range(1, self.d):
                    for q2 in range(-k2, k2 + 1):
                        if _idx_from_kq(k1, q1) <= _idx_from_kq(k2, q2):
                            self.product((k1, q1), (k2, q2))


def _resolve_basis(basis):
    if not isinstance(basis, BlochBasis):
        raise TypeError("An explicit BlochBasis instance is required")
    return basis


@dataclass
class GeneralizedBlochVector:
    data: np.ndarray | spmatrix
    basis: BlochBasis

    def __post_init__(self):
        self.basis = _resolve_basis(self.basis)
        expected = self.basis.d**2
        if issparse(self.data):
            if self.data.shape[0] != expected:
                raise ValueError(f"Expected {expected} rows, received {self.data.shape}")
            if not np.issubdtype(self.data.dtype, np.complexfloating):
                self.data = self.data.astype(complex)
        else:
            if isinstance(self.data, np.matrix):
                self.data = np.asarray(self.data)
            self.data = np.asarray(self.data)
            if self.data.ndim == 1:
                if self.data.shape != (expected,):
                    raise ValueError(f"Expected vector of length {expected}")
                self.data = self.data.reshape(-1, 1)
            elif self.data.ndim != 2 or self.data.shape[0] != expected:
                raise ValueError("Bloch data must have shape (d^2, n)")
            if self.data.dtype != complex:
                self.data = self.data.astype(complex)

    def _wrap_result(self, data):
        return self.__class__(data, basis=self.basis)

    def _check_basis(self, other):
        if not isinstance(other, GeneralizedBlochVector) or other.basis is not self.basis:
            raise ValueError("Bloch vectors belong to different bases")

    @staticmethod
    def _is_scalar(value):
        return np.isscalar(value) or isinstance(value, np.generic)

    def _as_1d_array(self, value):
        try:
            array = np.asarray(value)
        except Exception:
            return None
        if array.ndim != 1 or array.shape[0] != self.basis.d**2:
            return None
        return array.reshape(-1, 1)

    def __array__(self, dtype=None):
        array = self.data.toarray() if issparse(self.data) else self.data
        return array.astype(dtype) if dtype is not None else array

    def __repr__(self):
        shape = self.data.shape
        return f"GeneralizedBlochVector(d={self.basis.d}, shape={shape})"

    @classmethod
    def from_matrix(cls, matrix, basis):
        basis = _resolve_basis(basis)
        vector = basis.from_matrix(matrix)
        return cls(vector.data, basis=basis)

    def to_matrix(self):
        return self.basis.to_matrix(self)

    @classmethod
    def zeros(cls, batch_size=1, *, basis):
        basis = _resolve_basis(basis)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return cls(csc_matrix((basis.d**2, batch_size), dtype=complex), basis=basis)

    def copy(self):
        return self.__class__(self.data.copy(), basis=self.basis)

    def __add__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self._check_basis(other)
            return self._wrap_result(self.data + other.data)
        if self._is_scalar(other):
            if other == 0:
                return self.copy()
            return self._wrap_result(self.data + other)
        array = self._as_1d_array(other)
        if array is not None:
            return self._wrap_result(self.data + array)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self._check_basis(other)
            return self._wrap_result(self.data - other.data)
        if self._is_scalar(other):
            return self.copy() if other == 0 else self._wrap_result(self.data - other)
        array = self._as_1d_array(other)
        if array is not None:
            return self._wrap_result(self.data - array)
        return NotImplemented

    def __rsub__(self, other):
        if self._is_scalar(other):
            return self._wrap_result(other - self.data)
        array = self._as_1d_array(other)
        if array is not None:
            return self._wrap_result(array - self.data)
        return NotImplemented

    def __mul__(self, other):
        if self._is_scalar(other):
            return self._wrap_result(self.data * other)
        if isinstance(other, GeneralizedBlochVector):
            return bloch_tensor_product(self, other)
        array = self._as_1d_array(other)
        if array is not None:
            return bloch_tensor_product(
                self, GeneralizedBlochVector(array.astype(complex), basis=self.basis)
            )
        return NotImplemented

    def __rmul__(self, other):
        if self._is_scalar(other):
            return self._wrap_result(other * self.data)
        array = self._as_1d_array(other)
        if array is not None:
            return bloch_tensor_product(
                GeneralizedBlochVector(array.astype(complex), basis=self.basis), self
            )
        return NotImplemented

    def __truediv__(self, other):
        return self._wrap_result(self.data / other) if self._is_scalar(other) else NotImplemented

    def __neg__(self):
        return self._wrap_result(-self.data)

    def __iadd__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self._check_basis(other)
            self.data = self.data + other.data
        elif self._is_scalar(other):
            if other == 0:
                return self
            self.data = self.data + other
        else:
            array = self._as_1d_array(other)
            if array is None:
                return NotImplemented
            self.data = self.data + array
        return self

    def __isub__(self, other):
        if isinstance(other, GeneralizedBlochVector):
            self._check_basis(other)
            self.data = self.data - other.data
        elif self._is_scalar(other):
            if other == 0:
                return self
            self.data = self.data - other
        else:
            array = self._as_1d_array(other)
            if array is None:
                return NotImplemented
            self.data = self.data - array
        return self

    def __imul__(self, other):
        if self._is_scalar(other):
            self.data = self.data * other
        else:
            product = self * other
            if product is NotImplemented:
                return NotImplemented
            self.data = product.data
        return self

    def __itruediv__(self, other):
        if self._is_scalar(other):
            self.data = self.data / other
            return self
        return NotImplemented


def structure_const(kq1, kq2, basis):
    basis = _resolve_basis(basis)
    return basis.product(kq1, kq2) - basis.product(kq2, kq1)


def bloch_inner_product(u, v):
    if u.basis is not v.basis:
        raise ValueError("Bloch vectors belong to different bases")
    d1, d2 = u.data, v.data
    sparse_result = issparse(d1) or issparse(d2)
    if sparse_result:
        if not issparse(d1):
            d1 = d1[:, np.newaxis] if d1.ndim == 1 else d1
        if not issparse(d2):
            d2 = d2[:, np.newaxis] if d2.ndim == 1 else d2
        product = d1.conj().multiply(d2)
        result = np.sum(product, axis=0)
        if issparse(result):
            result = result.toarray()
        result = np.asarray(result).flatten()
    else:
        if d1.ndim == 1 and d2.ndim == 1:
            return np.vdot(d1, d2)
        if d1.ndim == 1:
            d1 = d1[:, np.newaxis]
        if d2.ndim == 1:
            d2 = d2[:, np.newaxis]
        result = np.sum(np.conj(d1) * d2, axis=0)
    return result.item() if np.size(result) == 1 else result


def _nonzero_components(vector, tol):
    if issparse(vector.data):
        coo = vector.data.tocoo()
        mask = np.abs(coo.data) > tol
        return coo.row[mask], coo.data[mask]
    flat = np.abs(vector.data).ravel()
    indices = np.flatnonzero(flat > tol)
    return indices, vector.data.ravel()[indices]


def _weighted_vector(vectors, weights, basis):
    if not weights:
        return GeneralizedBlochVector.zeros(basis=basis)
    if any(issparse(vector) for vector in vectors):
        coefficients = sparse_hstack(vectors)
        weight_matrix = sparse_csc(np.asarray(weights)[:, np.newaxis])
        data = coefficients @ weight_matrix
    else:
        coefficients = np.hstack([np.asarray(v).reshape(basis.d**2, 1) for v in vectors])
        data = coefficients @ np.asarray(weights)
    return GeneralizedBlochVector(data, basis=basis)


def bloch_tensor_product(u, v, tol=0.0):
    if u.basis is not v.basis:
        raise ValueError("Bloch vectors belong to different bases")
    if u.data.shape[1] > 1 or v.data.shape[1] > 1:
        raise NotImplementedError("Batched tensor products are not supported")
    left_indices, left_values = _nonzero_components(u, tol)
    right_indices, right_values = _nonzero_components(v, tol)
    vectors, weights, cache = [], [], {}
    for i, left_index in enumerate(left_indices):
        kq_left = _kq_from_idx(int(left_index))
        for j, right_index in enumerate(right_indices):
            kq_right = _kq_from_idx(int(right_index))
            key = (kq_left, kq_right)
            if key not in cache:
                cache[key] = u.basis.product(kq_left, kq_right).data
            product = cache[key]
            if not issparse(product) or product.nnz:
                vectors.append(product)
                weights.append(left_values[i] * right_values[j])
    return _weighted_vector(vectors, weights, u.basis)


def bloch_commutator(u, v, tol=0.0):
    return bloch_tensor_product(u, v, tol) - bloch_tensor_product(v, u, tol)


def bloch_hermitian_transpose(vector):
    basis = vector.basis
    d = basis.d
    if issparse(vector.data):
        permutation = np.zeros(d * d, dtype=int)
        phases = np.zeros(d * d, dtype=complex)
        for k in range(d):
            index0 = _idx_from_kq(k, 0)
            permutation[index0] = index0
            phases[index0] = 1
            for q in range(1, k + 1):
                positive, negative = _idx_from_kq(k, q), _idx_from_kq(k, -q)
                permutation[positive], permutation[negative] = negative, positive
                phases[positive] = phases[negative] = (-1) ** q
        from scipy.sparse import diags
        data = diags(phases) @ vector.data[permutation, :].conj()
        return GeneralizedBlochVector(data, basis=basis)

    source = vector.data
    result = np.zeros_like(source)
    for k in range(d):
        index0 = _idx_from_kq(k, 0)
        result[index0] = np.conj(source[index0])
        for q in range(1, k + 1):
            positive, negative = _idx_from_kq(k, q), _idx_from_kq(k, -q)
            result[positive] = (-1) ** q * np.conj(source[negative])
            result[negative] = (-1) ** q * np.conj(source[positive])
    return GeneralizedBlochVector(result, basis=basis)
