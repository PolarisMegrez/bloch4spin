# bloch4spin — Mathematical and implementation guide

For installation and a minimal executable example, see [README.md](README.md).

## 1. COBITO basis

For a spin-$j$ system with Hilbert-space dimension $d=2j+1$, the canonical orthonormal basis of irreducible tensor operators is written as

$$
T_q^{(k)} = \sum_{m_1,m_2} (-1)^{j-m_2}\sqrt{2k+1}
\begin{pmatrix} j & k & j \\ -m_2 & q & m_1 \end{pmatrix}
|j,m_2\rangle\langle j,m_1|.
$$

The matrices satisfy

$$
\operatorname{Tr}\!\left[(T_q^{(k)})^\dagger T_{q'}^{(k')}\right]
=\delta_{kk'}\delta_{qq'},
\qquad
T_0^{(0)}=\frac{I}{\sqrt d}.
$$

Negative-$q$ matrices are obtained from

$$
T_{-q}^{(k)}=(-1)^q(T_q^{(k)})^\dagger.
$$

## 2. Bloch coordinates

The linear index is $n=k(k+1)+q$, where $0\le k<d$ and $-k\le q\le k$, so that $0\le n<d^2$. An operator $A$ is expanded as

$$
A=\sum_{k,q}r_{kq}T_q^{(k)},\qquad
r_{kq}=\operatorname{Tr}\!\left[(T_q^{(k)})^\dagger A\right].
$$

Below, a single index $a$ denotes the linear index $(k,q)$, with the shorthand $T_a=T_q^{(k)}$ and $r_a=r_{kq}$.

A `BlochBasis` owns these matrices and their dimension-local caches. The basis is constructed at initialization. Tensor-product coefficients and Wigner symbols are populated lazily; `precompute_products()` explicitly warms the product cache.

## 3. Operator products and commutators

Products of basis elements are expanded as

$$
T_{q_1}^{(k_1)}T_{q_2}^{(k_2)}
=\sum_{k_3,q_3}C_{k_1q_1,k_2q_2}^{k_3q_3}T_{q_3}^{(k_3)},
$$

with $q_3=q_1+q_2$. The coefficients are obtained from Wigner 3j and 6j symbols under the Condon–Shortley convention. Bloch-space tensor products and commutators use only the coefficients required by the requested operation.

## 4. Hamiltonian and Lindblad evolution

For a Hamiltonian $H$, the Bloch vector obeys $\dot r=L_Hr$, where

$$
(L_H)_{ab}=-i\operatorname{Tr}\!\left(T_a^\dagger[H,T_b]\right).
$$

The implementation evaluates these commutators directly from the stored COBITO matrices rather than constructing every structure constant in advance.

For a jump operator $K$, the dissipator is

$$
\mathcal D_K[\rho]=K\rho K^\dagger-\frac12\{K^\dagger K,\rho\}.
$$

Since $r_a=\operatorname{Tr}(T_a^\dagger\rho)$, the Bloch vector obeys $\dot r=L_Kr$, where

$$
(L_K)_{ab}=\operatorname{Tr}\!\left(T_a^\dagger\mathcal D_K[T_b]\right)
=\operatorname{Tr}\!\left[T_a^\dagger\left(
KT_bK^\dagger-\frac12K^\dagger KT_b-\frac12T_bK^\dagger K
\right)\right].
$$

The implementation uses the equivalent Hilbert–Schmidt adjoint map. Define

$$
\mathcal D_K^\dagger[A]=K^\dagger AK-\frac12\{K^\dagger K,A\},
$$

so that

$$
(L_K)_{ab}=\operatorname{Tr}\!\left[
\left(\mathcal D_K^\dagger[T_a]\right)^\dagger T_b\right].
$$

The code evaluates $\mathcal D_K^\dagger[T_a]$ for all $a$ in a batch and then takes inner products with every $T_b$, without storing a complete structure-constant table. This also explains why the basis matrices are transformed as $K^\dagger T_aK$ in the code, whereas the Schrödinger-picture jump term is $K\rho K^\dagger$.

For a master equation with multiple dissipative channels,

$$
\dot\rho=-i[H,\rho]+\sum_\alpha\gamma_\alpha\mathcal D_{K_\alpha}[\rho],
\qquad
L=L_H+\sum_\alpha\gamma_\alpha L_{K_\alpha}.
$$

`from_Lindblad(K)` constructs one $L_K$ without an additional rate factor; the rate can either be absorbed into $K$ or included explicitly in the sum.

`GeneralizedBlochState.evolve(L,t)` applies $e^{Lt}$ in place. Repeated propagation times are cached per evolution-matrix object; modifying an evolution matrix in place invalidates those caches.

## 5. Measurements

A single Kraus operator defines the unnormalized update $\rho\mapsto\rho'=K\rho K^\dagger$. In Bloch coordinates, $r'=M_Kr$, where

$$
(M_K)_{ab}
=\operatorname{Tr}\!\left(T_a^\dagger KT_bK^\dagger\right)
=\operatorname{Tr}\!\left[\left(K^\dagger T_aK\right)^\dagger T_b\right].
$$

`from_Kraus(K)` uses the second expression to construct $M_K$ in a batch. `from_projector(P)` uses the equivalent outer-product shortcut for a rank-one projector; higher-rank projectors should be passed to `from_Kraus(P)`.

For a POVM effect $E$, the corresponding outcome probability is $p=\operatorname{Tr}(E\rho)$. `from_POVM_elem(E)` sets $K=\sqrt E$, giving the Lüders measurement update. An effect alone does not uniquely determine the conditional post-measurement state; for another update compatible with the same effect, pass the actual Kraus operator to `from_Kraus`.

Since $T_0^{(0)}=I/\sqrt d$, the outcome probability associated with an unnormalized post-measurement vector $r'$ is

$$
p=\operatorname{Tr}(\rho')=\sqrt d\,r'_0.
$$

A complete measurement satisfies $\sum_iK_i^\dagger K_i=I$, so its outcome probabilities sum to one. The measurement modes are:

- `normalized`: check that the measurement is complete and trace preserving;
- `renormalized`: normalize the retained outcome probabilities to sum to one;
- `unnormalized`: retain the raw outcome probabilities.

These modes only control the handling of outcome probabilities and the completeness check. For every nonzero-probability outcome, `apply_measurement` returns a separately normalized conditional state; `unnormalized` therefore does not mean that the returned conditional state is unnormalized.

The engine applies evolution and measurement steps in time order, batches active branches, and returns a dense joint-probability tensor. Its $i$th axis indexes the outcomes of the $i$th measurement.

## 6. References

- SymPy Wigner-symbol functions: `sympy.physics.wigner`.
- H. Schwegler, “On commutation relations of tensor operators,” 1965. DOI: https://doi.org/10.1007/BF02814299.
- E. P. Wigner, *Group Theory and Its Application to the Quantum Mechanics of Atomic Spectra*, 1959.
- D. A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii, *Quantum Theory of Angular Momentum*.
