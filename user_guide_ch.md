# bloch4spin — 自旋系统的广义布洛赫表示工具包

`bloch4spin` 是一个针对有限维自旋系统的数值计算库，基于不可约张量算符（Irreducible Tensor Operators, ITO）构建广义布洛赫表示（COBITO basis）。

本项目提供以下功能：
- 算符展开
- 哈密顿量生成的时间演化
- Kraus/POVM/投影测量
- “演化→测量”流水线模拟

本说明文档依次介绍数学基础、API 概览、物理示例、设计与实现要点以及参考资料。

---

## 1. 数学基础

### 1.1 基本符号约定

* 希尔伯特空间维度：$d = 2j + 1$；
* Wigner $3j$、$6j$ 系数的定义与 `SymPy` 保持一致；
* 算符空间配备 Frobenius 内积 $\langle A,B\rangle=\text{Tr}(A^{\dagger}B)$。

### 1.2 不可约张量算符（ITO）基

对于自旋 $j$ 的希尔伯特空间，ITO 定义为：

$$
T_q^{(k)} = \sum_{m_{1},m_{2}} (-1)^{j-m_{2}}\sqrt{2k+1}
\begin{pmatrix} j & k & j \\ -m_{2} & q & m_{1} \end{pmatrix}
|j,m_{2}\rangle\langle j,m_{1}|.
$$

上述规范使得我们选择的 ITO 具有性质：

* **正交归一性**：
  $$
  \mathrm{Tr}\bigl[(T_{q_{1}}^{(k_{1})})^\dagger T_{q_{2}}^{(k_{2})}\bigr]
  = \delta_{k_{1}k_{2}}\delta_{q_{1}q_{2}},
  $$

* **厄米共轭关系**：
  $$
  (T_q^{(k)})^\dagger = (-1)^q T_{-q}^{(k)},
  $$

* **标量项**：
  $$
  T^{(0)}_0 = I/\sqrt{d}.
  $$

### 1.3 广义布洛赫矢量表示

任意算符 $A$ 都可在 ITO 基上展开：

$$
A = \sum_{k,q} r_{kq} T_q^{(k)},
\qquad
r_{kq} = \mathrm{Tr}(T_q^{(k)\dagger}A).
$$

经线性索引映射 $(k,q) \to n$ 可将系数 $r_{kq}$ 组成长度 $(2j+1)^2$ 的广义布洛赫向量，我们采取的映射为

$$
n = k(k+1) + q, \qquad (0\le k\le 2j,\,-k\le q\le k).
$$

### 1.4 张量乘法与耦合系数

两个 ITO 的乘积可以展开为广义布洛赫表示：

$$
T^{(k_{1})}_{q_{1}} T^{(k_{2})}_{q_{2}} = \sum_{k_{3},q_{3}} C^{k_{3}q_{3}}_{k_{1}q_{1};\,k_{2}q_{2}} T^{(k_{3})}_{q_{3}}.
$$

其中，分量系数为

$$
C^{k_{3}q_{3}}_{k_{1}q_{1};\,k_{2}q_{2}} =
(-1)^{2j+q_{3}}
\sqrt{(2k_{1}+1)(2k_{2}+1)(2k_{3}+1)}
\begin{Bmatrix} k_{1} & k_{2} & k_{3} \\ j & j & j \end{Bmatrix}
\begin{pmatrix} k_{3} & k_{2} & k_{1} \\ -q_{3} & q_{2} & q_{1} \end{pmatrix}.
$$

上述系数非零的条件为：

* **三角形规则**：$|k_{1}-k_{2}| \le k_{3} \le k_{1}+k_{2},$
* **磁量子数守恒**：$q_{3} = q_{1} + q_{2}.$

利用其结构的稀疏性可以大幅提升数值性能。

### 1.5 李维尔演化

量子李维尔方程：

$$
\dot{\rho} = -i[H,\rho]
$$

在布洛赫坐标中：

$$
\dot{r} = L_H r,
$$

其中

$$
(L_H)_{ab} = -i\mathrm{Tr}(T_a^\dagger [H,T_b]).
$$

### 1.6 测量的超算符形式

单个 Kraus 元对应映射：

$$
\rho \mapsto K_i \rho K_i^\dagger,
\qquad
r' = L_i r.
$$

支持三种测量模式：

* **normalized**：POVM 完备、概率和为 1。
* **renormalized**：原始概率归一化，适用于只能看到相对比率的实验。
* **unnormalized**：返回实际概率，不归一化。

## 2. API 总览

以下接口为本库的核心功能；完整参数解释可见对应 docstring。

### 2.1 `bloch4spin.basis`

* `bloch_init(d)`：初始化基与缓存。
* `GeneralizedBlochVector`：广义布洛赫向量容器。
* `from_matrix(A)` / `to_matrix()`：与算符矩阵互转。
* `basis_product((k1,q1),(k2,q2))`：ITO 基元乘法的展开。
* `bloch_tensor_product(u,v)`：任意算符乘积的布洛赫向量形式。

### 2.2 `bloch4spin.evolution`

* `GeneralizedBlochEvolutionMatrix.from_Hamiltonian(H)`：构造李维尔演化矩阵。
* `GeneralizedBlochState`：定义量子态与演化操作。
* `evolve(L,t)`：计算 (e^{Lt}) 并作用于状态。

### 2.3 `bloch4spin.observable`

* `from_Kraus(K)`：从 Kraus 集构建测量描述。
* `from_POVM_elem(M)`：从 POVM 元构建测量。
* `from_projector(P)`：构建 rank-1 投影测量。
* `apply_measurement(obs, state, mode)`：返回概率与后态。

### 2.4 `bloch4spin.engine`

* `run(L, init_state, schedule, norm_mode)`：执行“演化→测量”流水线。

## 3. 示例：自旋 $1/2$ 系统的两时刻测量

考虑自旋 $j=1/2$（维度 $d=2$）的量子系统，其初态选择为完全混态 $\rho=I/2$。系统的动力学演化由哈密顿量$H=J_{y}$ 生成，而测量沿 $z$ 方向进行，采用 rank-1 投影测量。以下示例代码的目标是计算该系统在两个不同时间点 $(t_{1}=0.7,\,t_{2}=1.9)$ 的联合概率分布。

### 示例代码

```python
bloch_init(2)
Jy = np.array([[0, -0.5j], [0.5j, 0]])
L = GeneralizedBlochEvolutionMatrix.from_Hamiltonian(Jy)

r0 = GeneralizedBlochState.from_matrix(np.eye(2) / 2)

P0 = np.array([[1, 0], [0, 0]])
P1 = np.array([[0, 0], [0, 1]])
obs = [GeneralizedBlochObservable.from_projector(P0),
       GeneralizedBlochObservable.from_projector(P1)]

schedule = [(0.7, obs), (1.9, obs)]

joint = run(L, r0, schedule)
```

## 4. 参考资料

* H. Schwegler, *On commutation relations of tensor operators*.
* E. P. Wigner, *Group Theory and Its Applications to the Quantum Mechanics of Atomic Spectra*.
* Varshalovich, Moskalev, Khersonskii, *Quantum Theory of Angular Momentum*.
* SymPy 文档：`wigner_{3}j`, `wigner_6j`。
