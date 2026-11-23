# bloch4spin — 数学与实现说明

安装方法和最小可运行示例见 [README.md](README.md)。

## 1. COBITO 基

对于自旋 $j$ 系统，希尔伯特空间维数为 $d=2j+1$。不可约张量算符的规范正交基写为

$$
T_q^{(k)} = \sum_{m_1,m_2}(-1)^{j-m_2}\sqrt{2k+1}
\begin{pmatrix}j&k&j\\-m_2&q&m_1\end{pmatrix}
|j,m_2\rangle\langle j,m_1|.
$$

其满足

$$
\operatorname{Tr}\!\left[(T_q^{(k)})^\dagger T_{q'}^{(k')}\right]
=\delta_{kk'}\delta_{qq'},\qquad T_0^{(0)}=\frac I{\sqrt d}.
$$

负 $q$ 基元由厄米共轭关系生成：

$$
T_{-q}^{(k)}=(-1)^q(T_q^{(k)})^\dagger.
$$

## 2. 广义布洛赫坐标

线性索引为 $n=k(k+1)+q$，其中 $0\le k<d$、$-k\le q\le k$，故 $0\le n<d^2$。任意算符 $A$ 展开为

$$
A=\sum_{k,q}r_{kq}T_q^{(k)},\qquad
r_{kq}=\operatorname{Tr}\!\left[(T_q^{(k)})^\dagger A\right].
$$

下文用单个下标 $a$ 表示线性索引 $(k,q)$，并简写为 $T_a=T_q^{(k)}$、$r_a=r_{kq}$。

`BlochBasis` 持有维数、COBITO 基和相应缓存。初始化只构造基；张量积系数和 Wigner 符号在需要时计算。大量张量积操作前可显式调用 `precompute_products()` 预热缓存。

## 3. 基元乘积与交换子

两个基元的乘积展开为

$$
T_{q_1}^{(k_1)}T_{q_2}^{(k_2)}
=\sum_{k_3,q_3}C_{k_1q_1,k_2q_2}^{k_3q_3}T_{q_3}^{(k_3)},
\qquad q_3=q_1+q_2.
$$

展开系数采用 Condon–Shortley 约定，由 Wigner 3j 和 6j 符号得到。张量积与交换子只计算当前操作需要的系数。

## 4. Hamiltonian 与 Lindblad 演化

Hamiltonian $H$ 对应的布洛赫方程为 $\dot r=L_Hr$，其中

$$
(L_H)_{ab}=-i\operatorname{Tr}\!\left(T_a^\dagger[H,T_b]\right).
$$

实现直接利用已有 COBITO 矩阵批量计算交换子，不预先构造完整结构常数表。

对于跳跃算符 $K$，耗散项为

$$
\mathcal D_K[\rho]=K\rho K^\dagger-\frac12\{K^\dagger K,\rho\}.
$$

由 $r_a=\operatorname{Tr}(T_a^\dagger\rho)$ 可得 $\dot r=L_Kr$，其中

$$
(L_K)_{ab}=\operatorname{Tr}\!\left(T_a^\dagger\mathcal D_K[T_b]\right)
=\operatorname{Tr}\!\left[T_a^\dagger\left(
KT_bK^\dagger-\frac12K^\dagger KT_b-\frac12T_bK^\dagger K
\right)\right].
$$

代码采用与之等价的 Hilbert–Schmidt 伴随映射构造。定义

$$
\mathcal D_K^\dagger[A]=K^\dagger AK-\frac12\{K^\dagger K,A\},
$$

则

$$
(L_K)_{ab}=\operatorname{Tr}\!\left[
\left(\mathcal D_K^\dagger[T_a]\right)^\dagger T_b\right].
$$

实现对所有 $a$ 批量计算 $\mathcal D_K^\dagger[T_a]$，再与所有 $T_b$ 作内积，因而无需预先存储完整的结构常数表。这也解释了代码中的基矩阵变换为何写成 $K^\dagger T_aK$，而 Schrödinger 图景主方程中的跃迁项写成 $K\rho K^\dagger$。

对于包含多个耗散通道的主方程

$$
\dot\rho=-i[H,\rho]+\sum_\alpha\gamma_\alpha\mathcal D_{K_\alpha}[\rho],
\qquad
L=L_H+\sum_\alpha\gamma_\alpha L_{K_\alpha}.
$$

`from_Lindblad(K)` 构造单个未额外乘速率的 $L_K$；速率可以吸收进 $K$，也可以在求和时显式乘入。

`GeneralizedBlochState.evolve(L,t)` 原地应用 $e^{Lt}$。每个演化矩阵对象缓存重复使用的时间指数；原地修改矩阵会使相关缓存失效。

## 5. 测量

单个 Kraus 算符定义未归一化更新 $\rho\mapsto\rho'=K\rho K^\dagger$。在布洛赫坐标中，$r'=M_Kr$，其中

$$
(M_K)_{ab}
=\operatorname{Tr}\!\left(T_a^\dagger KT_bK^\dagger\right)
=\operatorname{Tr}\!\left[\left(K^\dagger T_aK\right)^\dagger T_b\right].
$$

`from_Kraus(K)` 利用第二种表达式批量构造 $M_K$。`from_projector(P)` 对秩一投影使用等价的外积捷径；秩更高的投影应使用 `from_Kraus(P)`。

对于 POVM 效应 $E$，相应结果的概率为 $p=\operatorname{Tr}(E\rho)$。`from_POVM_elem(E)` 取 $K=\sqrt E$，即采用 Lüders 型测量更新。效应本身并不唯一确定条件后态；若需要与同一效应相容的其他测量更新，应将实际 Kraus 算符传给 `from_Kraus`。

由 $T_0^{(0)}=I/\sqrt d$，未归一化后态 $r'$ 对应的结果概率为

$$
p=\operatorname{Tr}(\rho')=\sqrt d\,r'_0.
$$

完备测量满足 $\sum_iK_i^\dagger K_i=I$，因而所有结果的概率之和为 1。三种测量模式为：

- `normalized`：检查测量是否完备且保持迹；
- `renormalized`：将保留结果的概率总和归一化为 1；
- `unnormalized`：保留原始结果概率。

这些模式仅控制结果概率的处理方式及完备性检查。对于概率非零的结果，`apply_measurement` 返回的条件后态始终分别归一化；因此，`unnormalized` 不表示返回未归一化的条件后态。

引擎按时间顺序执行演化和测量，批量处理活跃分支，并返回稠密联合概率张量；张量的第 $i$ 个轴对应第 $i$ 次测量的结果索引。

## 6. 参考资料

- SymPy 的 Wigner 符号函数：`sympy.physics.wigner`。
- H. Schwegler，*On commutation relations of tensor operators*，1965。DOI: https://doi.org/10.1007/BF02814299。
- E. P. Wigner，*Group Theory and Its Application to the Quantum Mechanics of Atomic Spectra*，1959。
- D. A. Varshalovich、A. N. Moskalev、V. K. Khersonskii，*Quantum Theory of Angular Momentum*。
