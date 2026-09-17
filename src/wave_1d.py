# -*- coding: utf-8 -*-
"""一维波动方程 u_tt = c^2 * u_xx 的可视化。

中心差分格式:
    u_i^{n+1} = 2u_i^n - u_i^{n-1} + C^2 * (u_{i+1}^n - 2u_i^n + u_{i-1}^n),  C = c*dt/dx
稳定性条件 (CFL): C <= 1。

初始条件: 位于 x=0.3 的高斯波包、初速度为零（波包会分裂成左右两个行波），
边界条件: 两端固定 u = 0（观察波在端点反射时的相位翻转）。
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.titlecolor": "#0b0b0b",
})

# ---------- 网格与参数 ----------
L = 1.0
nx = 101
x = np.linspace(0, L, nx)
dx = x[1] - x[0]

c = 1.0            # 波速
CFL = 0.8          # CFL 数，<= 1 稳定
dt = CFL * dx / c
T = 2.5            # 总模拟时间（波包足够来回跑几趟）
nsteps = int(T / dt)
frame_step = 4

# ---------- 初始条件：高斯波包，初速度为零 ----------
u0 = np.exp(-((x - 0.30) / 0.05) ** 2)
u0[0] = u0[-1] = 0.0

# 初速度为零时，用二阶精度构造 u^{-1}：
# u^{-1} = u^0 - dt*v0 + (dt^2/2)*c^2*u''(0)，其中 v0 = 0
u_prev = u0 + 0.5 * CFL**2 * np.gradient(np.gradient(u0, dx), dx)
u_prev[0] = u_prev[-1] = 0.0

# ---------- 中心差分迭代 ----------
snapshots = [u0.copy()]
u_cur = u0.copy()
for step in range(1, nsteps + 1):
    u_new = 2 * u_cur - u_prev
    u_new[1:-1] += CFL**2 * (u_cur[2:] - 2 * u_cur[1:-1] + u_cur[:-2])
    u_new[0] = u_new[-1] = 0.0
    u_prev, u_cur = u_cur, u_new
    if step % frame_step == 0:
        snapshots.append(u_cur.copy())

# ---------- 动画 ----------
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
ax.set_xlim(0, L)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.grid(True, which="both")
ax.tick_params(direction="out")
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

(line,) = ax.plot([], [], color="#2a78d6", linewidth=2)
fill = ax.fill_between(x, -1.1, 0, color="#cde2fb", alpha=0.4)
title = ax.set_title("")

def animate(i):
    u = snapshots[i]
    line.set_data(x, u)
    global fill
    fill.remove()
    fill = ax.fill_between(x, -1.1, u, color="#cde2fb", alpha=0.4)
    t = i * frame_step * dt
    title.set_text(f"一维波动方程  $u_{{tt}} = c^2u_{{xx}}$    t = {t:.2f}")
    return [line, title]

anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=66, blit=False)
anim.save(OUT_DIR / "wave_1d.gif", writer="pillow", fps=15)
print(f"完成: {OUT_DIR / 'wave_1d.gif'}  ({len(snapshots)} 帧, t 到 {nsteps*dt:.2f})")
