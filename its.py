import time
import numpy as np
import scipy as sp
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec
from matplotlib.animation import FuncAnimation


plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 20
plt.rcParams['axes.linewidth'] = 3


def bin(val, bins, freqs):
    """ Bin a value into an array"""
    idx = np.sum(bins[:-1] <= val)-1
    freqs[idx] += 1
    return idx


FPS = 60
NUMS_PER_DRAW = 20
DIFFHEIGHTSCALE = 4

N_bins = 48

N = 50000

input_bins = np.linspace(0, 1, N_bins+1)
output_bins = np.linspace(0, 1, N_bins+1)
input_freqs = np.zeros(N_bins)
output_freqs = np.zeros(N_bins)

###################################
# DEMO 1 - Normal #################
###################################
distro = "Normal"
dist = sp.stats.norm(loc=0.5, scale=0.1)
##################################

fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(9, 9, wspace=0, hspace=0)

ax_L = fig.add_subplot(gs[:6, :2])
ax_L.xaxis.tick_top()
ax_L.set_ylim([0, 1])
ax_L.set_xlim([0, N/(N_bins-8)])
ax_L.set_xticklabels([])
ax_L.set_xticks([])

ax_B = fig.add_subplot(gs[6:, 2:])
ax_B.yaxis.tick_right()
ax_B.set_xlim([0, 1])
ax_B.set_ylim([0, DIFFHEIGHTSCALE*N/(N_bins-8)])
ax_B.set_yticklabels([])
ax_B.set_yticks([])

ax_M = fig.add_subplot(gs[:6, 2:])
ax_M.set_xticklabels([])
ax_M.set_yticklabels([])
ax_M.set_xlim([0, 1])
ax_M.set_ylim([0, 1])

ax_M.plot(np.linspace(0, 1, 1024), dist.cdf(np.linspace(0, 1, 1024)), lw=3)

container_L = ax_L.barh(input_bins[:-1], input_freqs,
                        height=0.8*np.diff(input_bins),
                        align="edge",
                        alpha=0.5)
container_B = ax_B.bar(output_bins[:-1], output_freqs,
                       width=0.8*np.diff(output_bins),
                       align="edge",
                       alpha=0.5)

txt = ax_M.set_title("")

fig.tight_layout()

np.random.seed(0)
randos = np.random.uniform(0, 1, N)


def update(frame):
    redraw = False
    update_axes = False

    for i in range(frame, frame+NUMS_PER_DRAW):
        if i >= N:
            break
        rand = randos[i]
        idx_L = bin(rand, input_bins, input_freqs)
        idx_B = bin(dist.ppf(rand), output_bins, output_freqs)

        container_L[idx_L].set_width(input_freqs[idx_L])
        container_B[idx_B].set_height(output_freqs[idx_B])

        if np.max(input_freqs) > ax_L.get_xlim()[1]:
            update_axes = True
            redraw = True
        if np.max(output_freqs) > ax_B.get_ylim()[1]:
            update_axes = True
            redraw = True

    if (np.int(frame/NUMS_PER_DRAW)%2 == 0) or (frame  + NUMS_PER_DRAW == N):
        txt.set_text(f"N = {frame + NUMS_PER_DRAW}")
        redraw = True

    if update_axes:
        ax_L.set_xlim([ax_L.get_xlim()[0], np.int(1.5*ax_L.get_xlim()[1])])
        ax_B.set_ylim([ax_B.get_ylim()[0], np.int(1.5*ax_B.get_ylim()[1])])

    if redraw:
        txt.set_text(f"N = {frame + NUMS_PER_DRAW}")
        fig.canvas.draw()

    fig.savefig(f"frames/its_{np.int(frame/NUMS_PER_DRAW):05.0f}.png")

    return (*container_L, *container_B)

ani = FuncAnimation(fig, update, range(0, N, NUMS_PER_DRAW),
                    blit=True, interval=1000./FPS, repeat=False)
