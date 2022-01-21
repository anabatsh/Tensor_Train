$
A[i_1, i_2, \dots, i_d] = \; G_1[i_1] \; G_2[i_2] \dots G_d[i_d]
$

$
\left\{ \begin{array}{l}
G_1[i_1] - [r_0 \times r_1] \\
G_2[i_2] - [r_1 \times r_2] \\
\dots \\
G_d[i_d] - [r_{d-1} \times r_d] \\
\end{array} \right.
\Rightarrow \;\;
G_k[i_k] - [r_{k-1} \times r_k]
$

$
\left\{ \begin{array}{l}
G_1 - [r_0 \times n_1 \times r_1] \\
G_2 - [r_1 \times n_2 \times r_2] \\
\dots \\
G_d - [r_{d-1} \times n_d \times r_d] \\
\end{array} \right.
\Rightarrow \;\;
G_k - [r_{k-1} \times n_k \times r_k]
$

$
A[i, j] = F[i] \; G[j]
$

$G_1, \; G_2 \; \dots, \; G_d$

$r_0, \; r_1, \; \dots, \; r_d$

$r_0 = r_d = 1$

$
n = \max{(n_1, \dots, n_d)} \\
r = \max{(r_0, \dots, r_d)} 
$

$O(n^d)$

$O(d r^2 n)$


$
W \in \mathbb{R}^{N \times M} \\
x \in \mathbb{R}^{M} \\
y = Wx \in \mathbb{R}^{N}
$

$
y[i] = \sum\limits_{j=1}^{M}W[i, j] \; x[j]
$


$
N = \prod\limits_{i=1}^{d} n_{i}, \quad
M = \prod\limits_{i=1}^{d} m_{i}
$

$
y[i_1, \dots, i_d] = 
\sum\limits_{j_1=1}^{m_1} \dots \sum\limits_{j_d=1}^{m_d} 
W[i_1, \dots, i_d \;, \; j_1, \dots, j_d] \; 
x[j_1, \dots, j_d]
$

$
W[i_1, \dots, i_d \;, \; j_1, \dots, j_d] = 
W[(i_1, j_1), \dots, (i_d, j_d)] = 
G_1[i_1, j_1] \dots G_d[i_d, j_d]
$

$
y[i_1] \dots y[i_d] = 
\sum\limits_{j_1, \dots, j_d}
G_1[i_1, j_1] \dots G_d[i_d, j_d] \; 
x[j_1] \dots x[j_d]
$

