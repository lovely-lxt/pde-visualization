# -*- coding: utf-8 -*-
"""二维热传导方程 u_t = a^2 * (u_xx + u_yy) 的可视化（热图动画）。

显式有限差分格式 (FTCS):
    u_{i,j}^{n+1} = u_{i,j}^n + r * (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j})^n
稳定性条件（二维）: r = a^2*dt/dx^2 <= 1/4。

初始条件: 三个不同强度的高斯热斑，边界条件: 四周 u = 0。
色带: 单一蓝色 hue 从浅到深（浅色 = 接近零，深色 = 高温），不用彩虹色。
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "output"  # 脚本在 src\ 里，output\ 在仓库根
OUT_DIR.mkdir(exist_ok=True)

# ---------- 画布风格 ----------
plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#898781",
    "ytick.color": "#898781",
})

# ---------- 单 hue 顺序色带（浅→深 = 数值低→高） ----------
blue_ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#1c5cab", "#104281", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("pde_blue", blue_ramp)

# ---------- 网格与参数 ----------
L = 1.0
nx = ny = 81
x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
dx = x[1] - x[0]

a2 = 1.0
r = 0.2           # 二维稳定性条件 r <= 0.25
dt = r * dx * dx / a2
T = 0.03          # 总模拟时间
nsteps = int(T / dt)
frame_step = 25

X, Y = np.meshgrid(x, y, indexing="ij")

# ---------- 初始条件：三个高斯热斑 ----------
u0 = (np.exp(-((X - 0.35) ** 2 + (Y - 0.65) ** 2) / 0.004)
      + 0.7 * np.exp(-((X - 0.65) ** 2 + (Y - 0.35) ** 2) / 0.004)
      + 0.5 * np.exp(-((X - 0.50) ** 2 + (Y - 0.50) ** 2) / 0.010))
u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0  # Dirichlet 边界

# ---------- 显式差分迭代 ----------
snapshots = [u0.copy()]
u = u0.copy()
for step in range(1, nsteps + 1):
    u_new = u.copy()
    u_new[1:-1, 1:-1] = (u[1:-1, 1:-1]
                         + r * (u[2:, 1:-1] + u[:-2, 1:-1]
                                + u[1:-1, 2:] + u[1:-1, :-2]
                                - 4 * u[1:-1, 1:-1]))
    u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
    u = u_new
    if step % frame_step == 0:
        snapshots.append(u.copy())

# ---------- 动画 ----------
fig, ax = plt.subplots(figsize=(5.8, 5.6), dpi=110)
vmax = 1.0
im = ax.imshow(snapshots[0], origin="lower", extent=[0, L, 0, L],
               cmap=cmap, vmin=0, vmax=vmax, interpolation="bilinear")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.tick_params(direction="out")
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label("温度 u(x, y, t)")
cbar.ax.tick_params(color="#898781", labelcolor="#898781")
cbar.outline.set_edgecolor("#c3c2b7")

title = ax.set_title("")

def animate(i):
    im.set_array(snapshots[i])
    t = i * frame_step * dt
    title.set_text(f"二维热传导方程  $u_t = a^2(u_{{xx}}+u_{{yy}})$    t = {t:.3f}")
    return [im, title]

anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=100, blit=False)
anim.save(OUT_DIR / "heat_2d.gif", writer="pillow", fps=10)
print(f"完成: {OUT_DIR / 'heat_2d.gif'}  ({len(snapshots)} 帧, t 到 {nsteps*dt:.3f})")
