# system imports
import os

# python imports
import numpy as np
import random

# plotting imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation

# plotting defaults
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=1.4)
cmap = plt.get_cmap("twilight")
# cmap = plt.get_cmap("hsv")

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

    def get_grid_list(self):
        return self.grid_list

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


def energy_loss_nima(grid_list, nbr):
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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

# problem parameters
grid_size = 100
epochs = 10000

# init dataset
N = 500  # number of fields
fields = np.empty((N, grid_size**2))
energies = np.empty(N)
rates = np.empty(N)

final_lr = 1e-2
for data_idx in range(N):
    print(f"Data {data_idx}/{N}")
    grid_list = 2 * np.pi * torch.rand(grid_size**2)

    model = OptimizeField(grid_list, grid_size)
    lrs = [1e2, 1e1, 1e0, 1e-1, 1e-2]
    lr = random.choice(lrs)
    print(f"-- lr = {lr}")

    opt = optim.Adam(model.parameters(), lr=lr)
    updates = 100
    gamma = (final_lr / lr) ** (1 / updates)
    step_size = epochs // updates
    schd = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    # if lr >= 10:
    #     schd = optim.lr_scheduler.MultiStepLR(
    #         opt, [int(0.7 * epochs), int(0.85 * epochs)], gamma=0.1
    #     )
    # elif lr >= 1:
    #     schd = optim.lr_scheduler.MultiStepLR(opt, [int(0.9 * epochs)], gamma=0.1)
    # else:
    #     schd = optim.lr_scheduler.MultiStepLR(opt, [int(0.9 * epochs)], gamma=1)

    for epoch in range(1, epochs + 1):
        model.train()
        s_e = model()
        loss = energy_loss_nima(s_e, model.get_nbr())

        opt.zero_grad()
        loss.backward()
        opt.step()
        schd.step()

        if (epoch - 1) % (epochs // 2) == 0:
            print(f"Epoch {epoch - 1}/{epochs} Loss {loss.item()}")

    fields[data_idx] = model.get_grid_list().clone().detach().cpu().numpy()
    energies[data_idx] = loss.item()
    rates[data_idx] = lr
    print("-------------------------------------")

data = {"x": fields, "y": energies, "lr": rates}
np.save(f"data_n={N}.npy", data)
# sns.histplot(energies)
# plt.show()
