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

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def lattice_nbr(grid_size):
    """dxd edge list (periodic)"""
    edg = set()
    for x in range(grid_size):
        for y in range(grid_size):
            v = x + grid_size * y
            for i in [-1, 1]:
                edg.add((v, ((x + i) % grid_size) + y * grid_size))
                edg.add((v, x + ((y + i) % grid_size) * grid_size))
    return torch.tensor(np.array(list(edg)), dtype=int)


def energy_loss_gauge(grid_list, gauge, nbr):
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

    gauge_f = gauge.flatten()
    loss = (
        -1
        / len(nbr)
        * torch.sum(
            torch.cos(
                (grid_list[nbr[:, 0]] + gauge_f[nbr[:, 0]])
                - (grid_list[nbr[:, 1]] + gauge_f[nbr[:, 1]])
            )
        )
    )
    return loss


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


device = "cuda:0" if torch.cuda.is_available() else "cpu"
seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

data = np.load("data_n=10000.npy", allow_pickle=True)
X, Y = data.item()["x"], data.item()["y"]

# init dataset
N = X.shape[0]  # number of fields
energies = np.empty(N)
grid_size = 100
edge_list = lattice_nbr(grid_size)

# generate gauge field
prod = 0.75
add = 0

# apply gauge
t = np.linspace(0, 1, grid_size)
sine_x = prod * np.cos(2 * np.pi * t) - add
sine_y = prod * np.cos(2 * np.pi * 2 * t) - add

field_x = sine_x.reshape(1, grid_size)
field_x = np.repeat(field_x, grid_size, axis=0).reshape(1, -1, 1)

field_y = sine_y.reshape(grid_size, 1)
field_y = np.repeat(field_y, grid_size, axis=1).reshape(1, -1, 1)
gauge = field_x + field_y

for data_idx in range(N):
    print(f"Data {data_idx}/{N}")
    s_e = X[data_idx, :]
    en = Y[data_idx]

    loss = energy_loss(torch.tensor(s_e), edge_list)
    loss_g = energy_loss_gauge(torch.tensor(s_e), gauge, edge_list)
    print(f"saved energy: {en}")
    print(f"original energy: {loss}\t gauge energy: {loss_g}")

    energies[data_idx] = loss_g
    print("-------------------------------------")

data = {"x": X, "y": energies}
np.save(f"data_n={N}_gauge.npy", data)
