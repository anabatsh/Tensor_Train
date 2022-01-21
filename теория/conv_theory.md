- $i$ - индексы каналов входа ($0 \leqslant i \leqslant C_{in}$)
- $j$ - индексы каналов выхода ($0 \leqslant j \leqslant C_{out}$)
- $i_k, j_k$ - индексы окна свертки ($0 \leqslant i_k \leqslant k_H$, $0 \leqslant j_k \leqslant k_W$)

Каждый элемент свертки $K^{C_{in} \times C_{out} \times k_H \times k_W}$ это $K[i, j, i_k, j_k]$

Действие свертки на входные данные:
$$
y[j, \hat{h}, \hat{w}] = 
\sum_{i=1}^{C_{in}}\sum_{i_k, j_k}X[i, \hat{h} + i_k, \hat{w} + j_k]K[i, j, i_k, j_k]
$$

---

Поскольку $C_{in}, C_{out}$ может быть довольно большим, имеет смысл представить свертку в ТТ-формате. Для этого разложим 

$$
C_{in} = c^{in}_{0} \prod_{i=1}^d c^{in}_{i}, \quad
C_{out} = c^{out}_{0} \prod_{i=1}^d c^{out}_{i} \\
$$

Тогда: 
$$
K[i, j, i_k, j_k] = 
K[
\; \overset{i}{\underset{\text{итерация по } C_{in}}{(i_0, i_1, \dots, i_d)}} \;, 
\; \overset{j}{\underset{\text{итерация по } C_{out}}{(j_0, j_1, \dots, j_d)}} \;, 
i_k, j_k]
$$

Перегруппируем индексы:
$$
K[(i_0, j_0), \; (i_1, j_1), \dots, (i_d, j_d), \; i_k, j_k]
$$

Сгруппируем $i_0, j_0$ с $i_k, j_k$:
$$
K[(i_0, j_0, i_k, j_k), \; (i_1, j_1), \dots, (i_d, j_d)]
$$

Теперь представим данный тензор в ТТ-формате:
$$
K_0[i_0, j_0, i_k, j_k] K_1[i_1, j_1] \dots K_d[i_d, j_d]
$$

Будем использовать первое ядро как свертку, а остальные ядра - как полносвязные слои.

---

Теперь посмотрим, как действует наша свертка:
$$
\hat{X}[j_0, (i_1, \dots, i_d), \hat{h}, \hat{w}] = 
\sum_{i_0=1}^{c^{in}_0}\sum_{i_k, j_k}
X[\underset{i}{i_0, (i_1, \dots, i_d)}, \hat{h} + i_k, \hat{w} + j_k]
K_0[i_0, j_0, i_k, j_k]
$$

$$
y[j_0, (j_1, \dots, j_d), \hat{h}, \hat{w}] = 
\sum_{i_1, \dots, i_d}
\hat{X}[j_0, (i_1, \dots, i_d), \hat{h}, \hat{w}] \;
K_1[i_1, j_1] \dots K_d[i_d, j_d]
$$

Перепишем в одно выражение:
$$
y[j_0, (j_1, \dots, j_d), \hat{h}, \hat{w}] = 
\sum_{i_1, \dots, i_d}
\left[
\sum_{i_0=1}^{c^{in}_0}\sum_{i_k, j_k}
X[\underset{i}{i_0, (i_1, \dots, i_d)}, \hat{h} + i_k, \hat{w} + j_k]
K_0[i_0, j_0, i_k, j_k]
\right]
K_1[i_1, j_1] \dots K_d[i_d, j_d]
$$

$$
y[j, \hat{h}, \hat{w}] = 
\sum_{i_0, i_1, \dots, i_d}\sum_{i_k, j_k}
X[i, \hat{h} + i_k, \hat{w} + j_k] \;
K_0[i_0, j_0, i_k, j_k] \; K_1[i_1, j_1] \dots K_d[i_d, j_d]
$$

---

Итого:

Обычная свертка:
$$
y[j, \hat{h}, \hat{w}] = 
\sum_{i=1}^{C_{in}}\sum_{i_k, j_k}X[i, \hat{h} + i_k, \hat{w} + j_k]K[i, j, i_k, j_k]
$$

Свертка с ТТ-разложением:
$$
y[j, \hat{h}, \hat{w}] = 
\sum_{i_0, i_1, \dots, i_d}\sum_{i_k, j_k}
X[i, \hat{h} + i_k, \hat{w} + j_k] \;
K_0[i_0, j_0, i_k, j_k] \; K_1[i_1, j_1] \dots K_d[i_d, j_d]
$$