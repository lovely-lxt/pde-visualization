# -*- coding: utf-8 -*-
"""第一步：用"你已经认识的方法"解一维 Poisson 方程  −u'' = f。

方程：  −u'' = 1,   u(0) = u(1) = 0
精确解： u(x) = x(1−x)/2   （抛物线）

这一步不引入任何新概念——直接写出三对角方程组 AU = F 并求解。
同一个方程组其实有三个身份（note0 §2.4）：

  1. 中心差分格式：
       −(u_{i−1} − 2u_i + u_{i+1}) / h² = f(x_i)
  2. 均匀网格上的 P1 有限元（note0 (2.11)）：
       刚度矩阵 A = tridiag(−1, 2, −1) / h，载荷向量 F_i = h·f(x_i)
  3. heat_1d.py 里热方程空间离散用的正是同一个算子
       u_{i+1} − 2u_i + u_{i−1}（乘上 r 而已）

预告（note0 Theorem 3.3 节点精确性）：
节点上的数值解与精确解完全相等——图上看，蓝线会精确穿过橙线。
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无界面环境，直接渲染到文件
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------- 画布风格（与 heat_1d.py / wave_1d.py 保持一致） ----------
plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.titlecolor": "#0b0b0b",
})

# ---------- 网格 ----------
N = 5                       # 内部节点数。故意取很小：折线和抛物线的差别一眼可见
x = np.linspace(0, 1, N + 2)   # N+2 个点，含 x_0 = 0 和 x_{N+1} = 1 两个边界
h = x[1] - x[0]                # 网格步长

# ---------- 构造方程组 AU = F ----------
# A = tridiag(−1, 2, −1) / h：
#   对角线是 2/h，上下两条副对角线是 −1/h（note0 式 (2.9)、(2.11)）
A = (np.diag(2.0 * np.ones(N))
     - np.diag(np.ones(N - 1), 1)     # 上副对角线
     - np.diag(np.ones(N - 1), -1)) / h   # 下副对角线
F = np.ones(N) * h            # f = 1 时，F_i = h（note0 式 (2.10) 梯形积分）

u_inner = np.linalg.solve(A, F)          # 解线性方程组，就是"组装求解"的最后一步
u = np.concatenate([[0.0], u_inner, [0.0]])   # 补上两个 Dirichlet 边界值 u(0)=u(1)=0

# ---------- 与精确解对比（节点精确性检验） ----------
u_nodes_exact = 0.5 * x * (1 - x)
err = np.max(np.abs(u_inner - u_nodes_exact[1:-1]))
print(f"N = {N}, h = {h:.4f}")
print(f"节点最大误差 |u_numerical − u_exact| = {err:.2e}   ← 机器精度，节点上完全重合")

# ---------- 画图 ----------
xs = np.linspace(0, 1, 500)          # 画精确解的细网格
u_exact = 0.5 * xs * (1 - xs)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 0.16)
ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.grid(True, which="both")
ax.tick_params(direction="out")
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# 精确解：橙线（调色板第 2 槽位，与蓝线 CVD 可区分，已通过校验）
ax.plot(xs, u_exact, color="#eb6834", linewidth=2, label="精确解  u = x(1−x)/2")
# 数值解：蓝线 + 空心圆节点标记（白底 = 画布色，线穿过节点）
ax.plot(x, u, color="#2a78d6", linewidth=2, marker="o", markersize=6,
        markerfacecolor="#fcfcfb", markeredgecolor="#2a78d6", markeredgewidth=1.6,
        label=f"数值解（N = {N} 个节点）")
ax.fill_between(x, 0, u, color="#cde2fb", alpha=0.55)

# 直接标注：指出"节点精确性"正在发生的位置
ax.annotate("节点上数值解 = 精确解",
            xy=(x[3], u[3]), xytext=(0.34, 0.115),
            color="#52514e", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#898781", lw=1.2))

ax.set_title("一维 Poisson 方程  $−u'' = 1$,  $u(0) = u(1) = 0$   （第一步：三对角方程组）")
ax.legend(loc="upper center", framealpha=1, edgecolor="#e1e0d9")

fig.savefig(OUT_DIR / "poisson_1d_step1.png", bbox_inches="tight")
print(f"完成: {OUT_DIR / 'poisson_1d_step1.png'}")
