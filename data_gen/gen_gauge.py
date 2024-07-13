# system imports
import os

# python imports
import numpy as np

# plotting imports
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

# plotting defaults
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)
cmap = plt.get_cmap("twilight")
color_plot_r = sns.cubehelix_palette(4, reverse=True, rot=0.2)[1]
color_plot_b = sns.cubehelix_palette(4, reverse=True, rot=-0.2)[1]

from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}\usepackage{bm}")

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed = 33616
torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs("figs", exist_ok=True)


class OptimizeField(nn.Module):
    def __init__(self, grid_list, grid_size):
        super().__init__()
        self.grid_list = nn.Parameter(grid_list)
        self.grid_size = grid_size
        L = grid_size
        N = L**2
        self.nbr = self.lattice_nbr(grid_size)

    def normalize(self):
        self.grid_list.data = self.grid_list.data / (2 * np.pi)

    def get_grid(self):
        return self.grid_list.view(self.grid_size, self.grid_size)

    def lattice_nbr(self, grid_size):
        """dxd edge list (periodic)"""
        edg = set()
        for x in range(grid_size):
            for y in range(grid_size):
                v = x + grid_size * y
                for i in [-1, 1]:
                    edg.add((v, ((x + i) % grid_size) + y * grid_size))
                    edg.add((v, x + ((y + i) % grid_size) * grid_size))
        return torch.tensor(np.array(list(edg)), dtype=int)

    def get_nbr(self):
        return self.nbr

    def forward(self):
        return self.grid_list


def energy_loss(grid_list, nbr):
    """
    Computes the energy of the configuration in the XY model.

    Parameters
    ----------
    grid_list: PyTorch tensor of size (grid_size ** 2)
        List containing the spin configuration (angle) of each lattice point.
    nbr: dict
        Dictionary containing the neighbors of each lattice point.

    Returns
    -------
    loss: PyTorch tensor
        Energy of the configuration.
    """

    loss = (
        -1
        / len(nbr)
        * torch.sum(torch.cos(grid_list[nbr[:, 0]] - grid_list[nbr[:, 1]]))
    )
    return loss


# problem parameters
grid_size = 20

# meshgrid is only used for plotting
base_w, base_h = 0.05, 0.05
x = np.arange(grid_size)
y = np.arange(grid_size)
X, Y = np.meshgrid(x, y)

final_lr = 1e-2
opts = ["Adam", "SGD"]
for opt_name in opts:
    if opt_name == "Adam":
        epochs = 1000
        lrs = [1e2, 1e1, 1e0, 1e-1, 1e-2]
    else:
        epochs = 50000
        lrs = [1e4, 1e3, 1e2, 1e1, 1e0]

    for lr in lrs:
        grid_list = 2 * np.pi * torch.rand(grid_size**2)
        grid = grid_list.reshape(grid_size, grid_size)
        s = torch.stack([torch.cos(grid), torch.sin(grid)])

        model = OptimizeField(grid_list, grid_size)
        if opt_name == "Adam":
            opt = optim.Adam(model.parameters(), lr=lr)
        else:
            opt = optim.SGD(model.parameters(), lr=lr)
        updates = 5
        gamma = (final_lr / lr) ** (1 / updates)
        step_size = epochs // updates
        schd = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

        print(f"Optimization: {opt_name} with lr: {lr}")
        print("----")
        losses = []
        grids = []
        for epoch in range(1, epochs + 1):
            model.train()
            s_e = model()
            loss = energy_loss(s_e, model.get_nbr())

            losses.append(loss.item())
            grids.append(model.get_grid().clone().detach() % (2 * np.pi))

            opt.zero_grad()
            loss.backward()
            opt.step()
            schd.step()

            if (epoch - 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch - 1}/{epochs} Loss {loss.item()}")

        # plot using quiver
        s_n = torch.stack([torch.cos(grids[-1]), torch.sin(grids[-1])])
        fig_n = plt.figure()
        ax_n = plt.gca()
        quiver_n = ax_n.quiver(
            X,
            Y,
            s_n[0],
            s_n[1],
            grids[-1],
            cmap=cmap,
            scale=20,
            width=0.005,
        )
        cbar = plt.colorbar(
            quiver_n,
            ax=ax_n,
            ticks=[0 + 0.05, np.pi, 2 * np.pi - 0.05],
        )
        quiver_n.set_clim(0, 2 * np.pi)
        cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
        ax_n.set_xticklabels([])
        ax_n.set_yticklabels([])
        plt.title(r"$H(\bm{s})=$" + rf"${energy_loss_nima(s_e, model.get_nbr()):0.8f}$")
        plt.savefig(
            f"figs/quiver_se_opt={opt_name}_lr={lr}_seed={seed}.pdf",
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
        print("---------------------------------")


# generate gauge field
prod = 0.25
add = 0
plt.figure(figsize=(10, 2))  # WxH
t = np.linspace(0, 1, 1000)
sine = prod * np.cos(2 * np.pi * t) - add
ax = plt.gca()
ax.plot(t, sine, linewidth=4, color=color_plot_r)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
plt.xlim([0, 1])
plt.xlabel("Angle")
plt.ylabel("Amplitude")
plt.savefig(f"figs/x_axis.pdf", bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 2))  # WxH
t = np.linspace(0, 1, 1000)
sine = prod * np.cos(2 * np.pi * 2 * t) - add
ax = plt.gca()
ax.plot(t, sine, linewidth=4, color=color_plot_b)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
plt.xlim([0, 1])
plt.xlabel("Angle")
plt.ylabel("Amplitude")
plt.savefig(f"figs/y_axis.pdf", bbox_inches="tight")
plt.show()
plt.close()

# apply gauge
t = np.linspace(0, 1, grid_size)
sine_x = prod * np.cos(2 * np.pi * t) - add
sine_y = prod * np.cos(2 * np.pi * 2 * t) - add

field_x = sine_x.reshape(1, grid_size)
field_x = np.repeat(field_x, grid_size, axis=0)

field_y = sine_y.reshape(grid_size, 1)
field_y = np.repeat(field_y, grid_size, axis=1)

plt.figure()
sns.heatmap(field_x, cmap=cmap)
plt.axis("off")
plt.savefig(f"figs/field_x.pdf", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
sns.heatmap(field_y, cmap=cmap)
plt.axis("off")
plt.savefig(f"figs/field_y.pdf", bbox_inches="tight")
plt.show()
plt.close()

plt.figure()
ax = sns.heatmap(
    field_x + field_y,
    cmap=cmap,
    vmin=-0.65,
    vmax=0.65,
    cbar_kws={"ticks": [-0.523599, 0, 0.523599]},
)
ax.collections[0].colorbar.set_ticklabels([r"$-\pi/6$", "0", r"$\pi/6$"])
plt.axis("off")
plt.savefig(f"figs/field_xy.pdf", bbox_inches="tight")
plt.show()
plt.close()

n_grids = grids[-1] + field_x + field_y
s_p = torch.stack([torch.cos(n_grids), torch.sin(n_grids)])
fig_n = plt.figure()
ax_n = plt.gca()
quiver_n = ax_n.quiver(
    X,
    Y,
    s_p[0],
    s_p[1],
    n_grids,
    cmap=cmap,
    scale=20,
    width=0.005,
)
cbar = plt.colorbar(
    quiver_n,
    ax=ax_n,
    ticks=[0 + 0.05, np.pi, 2 * np.pi - 0.05],
)
quiver_n.set_clim(0, 2 * np.pi)
cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
plt.title(
    r"$H(\bm{s})=$" + rf"${energy_loss_nima(n_grids.view(-1), model.get_nbr()):0.8f}$"
)
ax_n.set_xticklabels([])
ax_n.set_yticklabels([])
plt.savefig(f"figs/quiver_se_seed={seed}_end.pdf", bbox_inches="tight")
plt.show()
