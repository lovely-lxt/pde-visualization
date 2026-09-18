# -*- coding: utf-8 -*-
"""一维 Poisson 方程 −u″ = f 的 P1 有限元学习脚本（note0 配套，三步走）。

  第一步（回顾） 三对角方程组：差分 = 有限元 = 热方程算子，三个身份
  第二步        真正的有限元管线：hat 基函数 + 逐单元组装（Algorithm 1）
  第三步        任意载荷与收敛阶：f = π²sin(πx)，log-log 图验证 L² 二阶 / H¹ 一阶

产出：
  output/poisson_1d_step2_hat.png   草帽全家福：什么是"局部支撑"
  output/poisson_1d_step2.png       有限元解 = 加权 hat 函数之和
  output/poisson_1d_step3.png       左：N=15 解 vs 精确解；右：log-log 收敛阶
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

# ============================================================
# 第三步：任意载荷与收敛阶
# ============================================================
# 前两步的载荷都是 f ≡ 1。这一步换成任意函数 f(x) = π²·sin(πx)——
# 挑它的原因：−u″ = f、u(0) = u(1) = 0 有手算可得的精确解 u = sin(πx)
# （代进去即验证），有了精确解才能量误差、谈收敛阶。
#
# 理论（note0 误差估计）：H¹ 半范误差 O(h)、L² 误差 O(h²)。
# log-log 图上直线的斜率就是收敛阶：预期 H¹ 斜率 −1、L² 斜率 −2。


def assemble_poisson(N, f):
    """完整 FEM 管线浓缩版（第二步 ③④⑤ 合体）：解 −u″=f, u(0)=u(1)=0。

    参数
    ----
    N : 内部节点数（N+1 个等长单元）
    f : 右端载荷函数 f(x)

    返回
    ----
    (u, x) —— 节点值向量（含边界 0）、节点坐标向量
    """
    x = np.linspace(0, 1, N + 2)
    h = x[1] - x[0]
    # 组装刚度阵：每个单元的 A_k = (1/h)·[[1,−1],[−1,1]] 按端点编号加进全局阵
    A_global = np.zeros((N + 2, N + 2))
    A_local = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
    for k in range(1, N + 2):
        idx = [k - 1, k]
        A_global[np.ix_(idx, idx)] += A_local
    A = A_global[1:-1, 1:-1]     # 本质边界条件：删掉边界自由度
    # 载荷：Fᵢ = ∫fϕᵢ dx，细网格 xs 上复合梯形（积分误差必须远小于 FEM 误差）
    F = np.array([np.trapezoid(f(xs) * hat(xs, x[i], h), xs) for i in range(1, N + 1)])
    u_inner = np.linalg.solve(A, F)
    return np.concatenate([[0.0], u_inner, [0.0]]), x


def hat_prime(xx, xi, hh):
    """hat 函数的导数（右连续约定）：在 [xi−h, xi) 上 = +1/h，在 [xi, xi+h) 上 = −1/h，其余 0。

    端点用 ≥ 闭住，是为了让细网格在节点 xᵢ 上采样到"右极限"(uᵢ₊₁−uᵢ)/h
    而不是 0——u_h′ 在节点上有跳跃，梯形法若在节点上采到 0 会污染收敛阶
    （踩过坑，见学习记录第三步日志）。
    """
    d = np.zeros_like(xx)
    d[(xx >= xi - hh) & (xx < xi)] = 1.0 / hh
    d[(xx >= xi) & (xx < xi + hh)] = -1.0 / hh
    return d


def evaluate_uh(xx, u, x):
    """在任意点集 xx 上重建 u_h = Σuᵢϕᵢ 和 u_h′ = Σuᵢϕᵢ′（折线 + 阶梯导数）。"""
    h = x[1] - x[0]
    uh = np.zeros_like(xx)
    duh = np.zeros_like(xx)
    for ui, xi in zip(u, x):
        uh += ui * hat(xx, xi, h)
        duh += ui * hat_prime(xx, xi, h)
    return uh, duh


# ---------- 收敛性实验：h → 0，误差怎么变小 ----------
f = lambda xx: np.pi ** 2 * np.sin(np.pi * xx)   # 任意载荷
u_exact = lambda xx: np.sin(np.pi * xx)          # 对应精确解

print("\n=== 收敛性实验：L² 误差应 ~ h²（二阶），H¹ 误差应 ~ h（一阶） ===")
N_list = [3, 7, 15, 31, 63]          # h = 1/4, 1/8, 1/16, 1/32, 1/64
hs, errs_L2, errs_H1 = [], [], []
for N in N_list:
    u, x = assemble_poisson(N, f)
    uh, duh = evaluate_uh(xs, u, x)
    h = x[1] - x[0]
    err_L2 = np.sqrt(np.trapezoid((uh - u_exact(xs)) ** 2, xs))              # ‖u_h − u‖_{L²}
    err_H1 = np.sqrt(np.trapezoid((duh - np.pi * np.cos(np.pi * xs)) ** 2, xs))  # |u_h − u|_{H¹}
    hs.append(h)
    errs_L2.append(err_L2)
    errs_H1.append(err_H1)
    print(f"N = {N:>2}（h = {h:.4f}）:  L² = {err_L2:.3e}   H¹ = {err_H1:.3e}")

# 最小二乘拟合 log(误差) ~ slope·log(h)：斜率就是收敛阶
slope_L2 = np.polyfit(np.log(hs), np.log(errs_L2), 1)[0]
slope_H1 = np.polyfit(np.log(hs), np.log(errs_H1), 1)[0]
print(f"\n拟合收敛阶：L² ≈ {slope_L2:.2f}（理论 −2，二阶）   H¹ ≈ {slope_H1:.2f}（理论 −1，一阶）")

# ---------- 图：左 = 解的形状，右 = log-log 收敛阶 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=110)

# 左图：N = 15 时折线已肉眼贴住精确解
u15, x15 = assemble_poisson(15, f)
uh15, _ = evaluate_uh(xs, u15, x15)
ax1.plot(xs, u_exact(xs), color="#eb6834", linewidth=2.4, label="精确解  u = sin(πx)")
ax1.plot(x15, u15, color="#2a78d6", linewidth=1.8, marker="o", markersize=5,
         markerfacecolor="#fcfcfb", markeredgecolor="#2a78d6", markeredgewidth=1.4,
         label="有限元解  u_h（N = 15）")
ax1.set_xlabel("x")
ax1.set_ylabel("u(x)")
ax1.set_title("N = 15：折线已经肉眼贴住精确解")
ax1.legend(loc="upper right", framealpha=1, edgecolor="#e1e0d9")
ax1.grid(True, which="both")
ax1.tick_params(direction="out")
for spine in ax1.spines.values():
    spine.set_linewidth(0.8)

# 右图：log-log 收敛图 —— 直线斜率 = 收敛阶
ax2.loglog(hs, errs_L2, "o-", color="#2a78d6", linewidth=1.8, markersize=7,
           label=f"L² 误差（斜率 {slope_L2:.2f}）")
ax2.loglog(hs, errs_H1, "s-", color="#eb6834", linewidth=1.8, markersize=6,
           label=f"H¹ 误差（斜率 {slope_H1:.2f}）")
# 参考线：斜率 −2 和 −1，从第一个数据点出发，便于肉眼对斜率
ax2.loglog(hs, errs_L2[0] * (np.array(hs) / hs[0]) ** 2, "--", color="#a8b9d4",
           linewidth=1.4, label="参考：∝ h²")
ax2.loglog(hs, errs_H1[0] * (np.array(hs) / hs[0]) ** 1, "--", color="#e8b59b",
           linewidth=1.4, label="参考：∝ h")
ax2.set_xlabel("h（网格步长）")
ax2.set_ylabel("误差")
ax2.set_title("log-log 收敛图：L² 二阶、H¹ 一阶")
ax2.legend(loc="lower left", framealpha=1, edgecolor="#e1e0d9")
ax2.grid(True, which="both")
ax2.tick_params(direction="out")
for spine in ax2.spines.values():
    spine.set_linewidth(0.8)

fig.suptitle("第三步：任意载荷 f = π² sin(πx) 与收敛阶验证", color="#0b0b0b", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_DIR / "poisson_1d_step3.png", bbox_inches="tight")
print(f"完成: {OUT_DIR / 'poisson_1d_step3.png'}")
