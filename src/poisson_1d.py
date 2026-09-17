# -*- coding: utf-8 -*-
"""第二步：真正的有限元管线 —— hat 基函数 + 逐单元组装。

第一步直接手写了三对角矩阵；这一步走"远路"，但远路才是能推广到
二维、变系数、非均匀网格的通用框架（note0 Algorithm 1）：

  ① 定义 hat 函数（草帽函数）ϕᵢ —— 分片线性基函数
  ② 局部刚度阵 (1/h)·[[1,−1],[−1,1]] —— 每个单元上算一次
  ③ 组装：局部阵按端点编号"加进"全局阵（含边界的 (N+2)×(N+2)）
  ④ 边界处理：删掉边界自由度（Dirichlet 是"本质"条件，直接进试探空间）
  ⑤ 载荷向量 Fᵢ = ∫fϕᵢ dx —— 用积分算出来，不再照抄 h
  ⑥ 求解 —— 结果必须与第一步完全一样（殊途同归）

产出：
  output/poisson_1d_step2_hat.png   草帽全家福：什么是"局部支撑"
  output/poisson_1d_step2.png       有限元解 = 加权 hat 函数之和
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无界面环境，直接渲染到文件
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "output"  # 脚本在 src\ 里，output\ 在仓库根
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
N = 5                       # 内部节点数
x = np.linspace(0, 1, N + 2)   # N+2 个节点：x_0=0, ..., x_{N+1}=1
h = x[1] - x[0]                # 网格步长
xs = np.linspace(0, 1, 20001)  # 画图/积分用的细网格

# ============================================================
# ① hat 函数（草帽函数）
# ============================================================
def hat(xx, xi, hh):
    """以 xi 为中心、相邻节点间距 hh 的草帽函数（note0 式 (2.3) 的紧凑写法）。

    分片线性：在 [xi−h, xi] 上从 0 升到 1，在 [xi, xi+h] 上从 1 降到 0，
    其余位置为 0。节点性质：ϕᵢ(xⱼ) = δᵢⱼ。
    """
    return np.clip(1.0 - np.abs(xx - xi) / hh, 0.0, None)

# ---------- 图 1：草帽全家福 ----------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
for i in range(1, N + 1):
    ax.plot(xs, hat(xs, x[i], h), color="#2a78d6", alpha=0.18, linewidth=1.5)
ax.plot(xs, hat(xs, x[3], h), color="#2a78d6", linewidth=2.5)   # 高亮 ϕ₃
ax.scatter(x, np.zeros_like(x), s=18, color="#898781", zorder=3)  # 网格节点
# δ 性质：ϕ₃ 在自己节点上取 1
ax.annotate("$\\phi_3(x_3)$ = 1", xy=(x[3], 1.0), xytext=(x[3] - 0.02, 1.10),
            ha="center", color="#52514e", fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#898781", lw=1.2))
# 局部支撑：只有相邻两个单元上非零
ax.annotate("", xy=(x[2], -0.10), xytext=(x[4], -0.10),
            arrowprops=dict(arrowstyle="<->", color="#eb6834", lw=1.4))
ax.text((x[2] + x[4]) / 2, -0.17, "$\\phi_3$ 的支撑 = 相邻两个单元\n（其余位置恒为 0 → 局部支撑，所以矩阵稀疏）",
        ha="center", va="top", color="#52514e", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(-0.45, 1.30)
ax.set_xlabel("x")
ax.set_ylabel("$\\phi_i(x)$")
ax.grid(True, which="both")
ax.tick_params(direction="out")
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
ax.set_title("hat 函数（草帽函数）：P1 有限元的基函数（N = 5）")
fig.savefig(OUT_DIR / "poisson_1d_step2_hat.png", bbox_inches="tight")
print(f"完成: {OUT_DIR / 'poisson_1d_step2_hat.png'}")

# ============================================================
# ② 局部刚度阵
# ============================================================
# 单元 K_k = [x_{k−1}, x_k] 上只有两个基函数非零（note0 的"重心坐标"）：
#   λ1(x) = (x_k − x)/h_k（从 1 线性降到 0），λ2(x) = (x − x_{k−1})/h_k（从 0 升到 1）
# 导数：λ1′ = −1/h，λ2′ = +1/h
# 局部刚度阵 (A_k)_{pq} = ∫ λ_p′ λ_q′ dx = (±1/h)·(±1/h)·h = ±1/h
# 即 A_k = (1/h)·[[1,−1],[−1,1]]（note0 式 (2.7)）—— 这就是第一步那个矩阵的"细胞"
A_local = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
print("\n=== 局部刚度阵 A_k = (1/h)·[[1,−1],[−1,1]]（每个单元都一样） ===")
print(A_local)

# ============================================================
# ③ 组装（note0 Algorithm 1）
# ============================================================
n_nodes = N + 2                  # 全部节点 x_0 ... x_{N+1}
A_global = np.zeros((n_nodes, n_nodes))
print("\n=== 组装过程：每个单元的局部阵按端点编号加进全局阵 ===")
for k in range(1, N + 2):        # 单元 K_k = [x_{k−1}, x_k]，共 N+1 个
    idx = [k - 1, k]             # 该单元两个端点的全局编号
    A_global[np.ix_(idx, idx)] += A_local
    print(f"单元 K{k} = [x{k-1}, x{k}] → 自由度 {tuple(idx)}：局部阵加在 A[{idx[0]}:{idx[1]+1}, {idx[0]}:{idx[1]+1}]")

print("\n组装完的 (N+2)×(N+2) 全局阵（含边界自由度 0 和 6）：")
print(np.round(A_global, 3))

# 边界处理：u(0)=u(1)=0 已知，删掉第 0 和 N+1 行/列（Algorithm 1 第 7 行）
# 这就是"Dirichlet 是本质边界条件"在代码里的样子：边界值不进方程组
A = A_global[1:-1, 1:-1]

# ============================================================
# ④ 载荷向量：Fᵢ = ∫₀¹ f(x)·ϕᵢ(x) dx
# ============================================================
# f ≡ 1 时精确值就是 h（hat 的面积 = 两个小三角形 h/2 + h/2）。
# 这里用数值积分算——对任意 f 都管用，为第三步（任意载荷）铺路。
f = lambda xx: np.ones_like(xx)
F = np.array([np.trapezoid(f(xs) * hat(xs, x[i], h), xs) for i in range(1, N + 1)])
print(f"\n=== 载荷向量 Fᵢ = ∫fϕᵢ dx（数值积分） ===")
print(f"F = {np.round(F, 6)}   ← 每个都等于 h = {h:.4f}：第一步'照抄'的 h 找到了来历")

# ============================================================
# ⑤ 求解，并与第一步对比
# ============================================================
u_inner = np.linalg.solve(A, F)
u = np.concatenate([[0.0], u_inner, [0.0]])   # 补上边界值

# 第一步的三对角矩阵与结果（回顾用，不参与本步计算）
A_step1 = (np.diag(2.0 * np.ones(N))
           - np.diag(np.ones(N - 1), 1)
           - np.diag(np.ones(N - 1), -1)) / h
F_step1 = np.ones(N) * h
u_step1 = np.linalg.solve(A_step1, F_step1)

print("\n=== 与第一步对比（殊途同归） ===")
print("组装出的 A == 第一步手写的三对角阵:", np.allclose(A, A_step1))
print("解出的 u == 第一步的结果:", np.allclose(u_inner, u_step1))
u_nodes_exact = 0.5 * x * (1 - x)
print(f"节点最大误差 |u − u_exact| = {np.max(np.abs(u - u_nodes_exact)):.2e}（节点精确性）")

# ---------- 图 2：有限元解 = 加权 hat 函数之和 ----------
u_exact = 0.5 * xs * (1 - xs)
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
# 浅蓝：每个"分量" uᵢ·ϕᵢ —— 解被拆成基函数的加权和
for i in range(1, N + 1):
    ax.plot(xs, u[i] * hat(xs, x[i], h), color="#2a78d6", alpha=0.16, linewidth=1.5)
# 橙：精确解
ax.plot(xs, u_exact, color="#eb6834", linewidth=2, label="精确解  u = x(1−x)/2")
# 深蓝：有限元解（浅蓝分量之和）
ax.plot(x, u, color="#2a78d6", linewidth=2, marker="o", markersize=6,
        markerfacecolor="#fcfcfb", markeredgecolor="#2a78d6", markeredgewidth=1.6,
        label="有限元解  $u_h = \\sum_i u_i\\phi_i$")
ax.fill_between(x, 0, u, color="#cde2fb", alpha=0.55)
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 0.16)
ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.grid(True, which="both")
ax.tick_params(direction="out")
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
ax.set_title("第二步：有限元解 = 加权 hat 函数之和（与第一步同一条折线）")
ax.legend(loc="upper center", framealpha=1, edgecolor="#e1e0d9")
fig.savefig(OUT_DIR / "poisson_1d_step2.png", bbox_inches="tight")
print(f"完成: {OUT_DIR / 'poisson_1d_step2.png'}")
