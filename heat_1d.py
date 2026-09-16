# -*- coding: utf-8 -*-
"""一维热传导方程 u_t = a^2 * u_xx 的可视化。

显式有限差分格式 (FTCS):
    u_i^{n+1} = u_i^n + r * (u_{i+1}^n - 2u_i^n + u_{i-1}^n),  r = a^2*dt/dx^2
稳定性条件: r <= 1/2。

初始条件: 三个高斯"热包"，边界条件: 两端 u = 0（温度恒为零）。
随着时间推移热包逐渐抹平——这就是扩散的直观含义。
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无界面环境，直接渲染到文件
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------- 画布风格（浅色面、弱化坐标轴） ----------
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

# ---------- 网格与参数 ----------
L = 1.0            # 空间区间 [0, L]
nx = 101           # 网格点数
x = np.linspace(0, L, nx)
dx = x[1] - x[0]

a2 = 1.0           # 热传导系数平方 a^2
r = 0.4            # 稳定性条件 r <= 0.5
dt = r * dx * dx / a2
T = 0.12           # 总模拟时间
nsteps = int(T / dt)
frame_step = 40    # 每 40 步存一帧

# ---------- 初始条件：三个高斯热包 ----------
u0 = (0.80 * np.exp(-((x - 0.25) / 0.06) ** 2)
      + 0.55 * np.exp(-((x - 0.60) / 0.04) ** 2)
      + 0.35 * np.exp(-((x - 0.85) / 0.05) ** 2))
u0[0] = u0[-1] = 0.0  # Dirichlet 边界

# ---------- 显式差分迭代 ----------
snapshots = [u0.copy()]
u = u0.copy()
for step in range(1, nsteps + 1):
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    u_new[0] = u_new[-1] = 0.0
    u = u_new
    if step % frame_step == 0:
        snapshots.append(u.copy())

# ---------- 动画 ----------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
ax.set_xlim(0, L)
ax.set_ylim(-0.02, 0.95)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.grid(True, which="both")
ax.tick_params(direction="out")
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

(line,) = ax.plot([], [], color="#2a78d6", linewidth=2)
fill = ax.fill_between(x, 0, snapshots[0], color="#cde2fb", alpha=0.55)
title = ax.set_title("")

def animate(i):
    u = snapshots[i]
    line.set_data(x, u)
    # 更新填充区域（fill_between 对象需要重画）
    global fill
    fill.remove()
    fill = ax.fill_between(x, 0, u, color="#cde2fb", alpha=0.55)
    t = i * frame_step * dt
    title.set_text(f"一维热传导方程  $u_t = a^2u_{{xx}}$    t = {t:.3f}")
    return [line, title]

anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=66, blit=False)
anim.save(OUT_DIR / "heat_1d.gif", writer="pillow", fps=15)
print(f"完成: {OUT_DIR / 'heat_1d.gif'}  ({len(snapshots)} 帧, t 到 {nsteps*dt:.3f})")
