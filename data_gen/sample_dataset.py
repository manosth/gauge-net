# system imports
import os

# python imports
import numpy as np

# plotting imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

# plotting defaults
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)
cmap = plt.get_cmap("twilight")
cmap_t = plt.get_cmap("turbo")
# cmap = plt.get_cmap("hsv")
color_plot = sns.cubehelix_palette(4, reverse=True, rot=0.2)
from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

data = np.load("/Users/manos/data/gauge/data_n=10000.npy", allow_pickle=True)
# data = np.load("data_n=10000.npy", allow_pickle=True)
X_tr, Y_tr = torch.Tensor(data.item()["x"]), torch.Tensor(data.item()["y"])

grid_size = 100
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)

indices = np.random.choice(X_tr.shape[0], 16, replace=False)
for idx in indices:
    print(f"Energy: {Y_tr[idx]}")
    grid = X_tr[idx].view(grid_size, grid_size) % (2 * np.pi)
    s_n = torch.stack([torch.cos(grid), torch.sin(grid)])
    fig_n = plt.figure()
    ax_n = plt.gca()
    quiver_n = ax_n.quiver(X, Y, s_n[0], s_n[1], grid, cmap=cmap, scale=40)
    cbar = plt.colorbar(quiver_n, ax=ax_n, ticks=[0 + 0.05, np.pi, 2 * np.pi - 0.05])
    cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    # plt.title("High energy state")
    plt.title(r"$H(\bm{s})=$" + rf"${Y_tr[idx].item():0.8f}$")
    ax_n.set_xticklabels([])
    ax_n.set_yticklabels([])
    plt.savefig("figs/data_sample=" + str(idx) + ".pdf", bbox_inches="tight")
    plt.show()
    plt.close()
