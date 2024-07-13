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

data = np.load("data_n=10000.npy", allow_pickle=True)
X_tr, Y_tr = torch.Tensor(data.item()["x"]), torch.Tensor(data.item()["y"])

plt.figure()
sns.histplot(
    Y_tr, bins=100, stat="probability", kde=True, color=color_plot[1], ec=color_plot[0]
)
plt.xlabel("Energy")
plt.title("Energy distribution")
plt.savefig("figs/histogram.pdf", bbox_inches="tight")
plt.show()
plt.close()

grid_size = 100
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)

less = 0
more = 0
both = 0
for idx in range(X_tr.shape[0]):
    if (both >= 7) and (less >= 7) and (more >= 7):
        print("All limits reached. Exiting.")
        break

    if Y_tr[idx] > -0.985 and more < 7:
        print(f"High energy: {Y_tr[idx]}\t index: {idx}")
        grid = X_tr[idx].view(grid_size, grid_size) % (2 * np.pi)
        s_n = torch.stack([torch.cos(grid), torch.sin(grid)])
        fig_n = plt.figure()
        ax_n = plt.gca()
        quiver_n = ax_n.quiver(X, Y, s_n[0], s_n[1], grid, cmap=cmap, scale=40)
        cbar = plt.colorbar(
            quiver_n, ax=ax_n, ticks=[0 + 0.05, np.pi, 2 * np.pi - 0.05]
        )
        cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
        plt.title(r"$H(\bm{s})=$" + rf"${Y_tr[idx].item():0.8f}$")
        ax_n.set_xticklabels([])
        ax_n.set_yticklabels([])
        plt.show()
        plt.close()

        more += 1
    elif Y_tr[idx] < -0.9975 and less < 7:
        print(f"Low energy: {Y_tr[idx]}\t index: {idx}")
        grid = X_tr[idx].view(grid_size, grid_size) % (2 * np.pi)
        s_n = torch.stack([torch.cos(grid), torch.sin(grid)])
        fig_n = plt.figure()
        ax_n = plt.gca()
        quiver_n = ax_n.quiver(X, Y, s_n[0], s_n[1], grid, cmap=cmap, scale=40)
        cbar = plt.colorbar(
            quiver_n, ax=ax_n, ticks=[0 + 0.05, np.pi, 2 * np.pi - 0.05]
        )
        cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
        plt.title(r"$H(\bm{s})=$" + rf"${Y_tr[idx].item():0.8f}$")
        ax_n.set_xticklabels([])
        ax_n.set_yticklabels([])
        plt.show()
        plt.close()

        less += 1
    elif Y_tr[idx] < -0.994 and Y_tr[idx] > -0.996 and both < 7:
        print(f"Mid energy: {Y_tr[idx]}\t index: {idx}")
        grid = X_tr[idx].view(grid_size, grid_size) % (2 * np.pi)
        s_n = torch.stack([torch.cos(grid), torch.sin(grid)])
        fig_n = plt.figure()
        ax_n = plt.gca()
        quiver_n = ax_n.quiver(X, Y, s_n[0], s_n[1], grid, cmap=cmap, scale=40)
        cbar = plt.colorbar(
            quiver_n, ax=ax_n, ticks=[0 + 0.05, np.pi, 2 * np.pi - 0.05]
        )
        cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
        plt.title(r"$H(\bm{s})=$" + rf"${Y_tr[idx].item():0.8f}$")
        ax_n.set_xticklabels([])
        ax_n.set_yticklabels([])
        plt.show()
        plt.close()

        both += 1
