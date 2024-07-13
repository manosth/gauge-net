# system imports
import os

# python imports
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torchinfo import summary

# emlp imports
from emlp.reps import V, Scalar, Vector
from emlp.groups import SO
import emlp

import objax
import jax.numpy as jnp
from tqdm.auto import tqdm

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
color_plot = sns.cubehelix_palette(4, reverse=True, rot=-0.2)
from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

def main():
    def train_model(model):
        opt = objax.optimizer.Adam(model.vars())

        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(x, y):
            yhat = model(x)
            return ((yhat - y) ** 2).mean()

        grad_and_val = objax.GradValues(loss, model.vars())

        @objax.Jit
        @objax.Function.with_vars(model.vars() + opt.vars())
        def train_op(x, y, lr):
            g, v = grad_and_val(x, y)
            opt(lr=lr, grads=g)
            return v

        train_losses, test_losses = [], []
        equiv_errors = []
        for epoch in tqdm(range(NUM_EPOCHS)):
            train_losses.append(
                np.mean(
                    [train_op(jnp.array(x), jnp.array(y), lr) for (x, y) in trainloader]
                )
            )
            if not epoch % 10:
                test_losses.append(
                    np.mean([loss(jnp.array(x), jnp.array(y)) for (x, y) in testloader])
                )
        return train_losses, test_losses

    def evaluate_model(model, loader):
        @objax.Jit
        @objax.Function.with_vars(model.vars())
        def loss(x, y):
            yhat = model(x)
            return ((yhat - y) ** 2).mean()

        return np.mean([loss(jnp.array(x), jnp.array(y)) for (x, y) in loader])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)

    grid_size = 100
    batch_size = 64

    data = np.load("data_n=10000.npy", allow_pickle=True)
    X, Y = data.item()["x"], data.item()["y"]

    tr_idx = np.random.choice(X.shape[0], int(0.8 * X.shape[0]), replace=False)
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[tr_idx] = True
    X_tr, Y_tr = X[mask], Y[mask]
    X_te, Y_te = X[~mask], Y[~mask]

    # reformat to (N, C, W, H)
    X_tr = torch.Tensor(X_tr).view(-1, 1, grid_size, grid_size)
    Y_tr = torch.Tensor(Y_tr).view(-1, 1)
    X_te = torch.Tensor(X_te).view(-1, 1, grid_size, grid_size)
    Y_te = torch.Tensor(Y_te).view(-1, 1)

    bs_tr = X_tr.shape[0]
    bs_te = X_te.shape[0]

    # generate gauge field
    prod = 0.75
    add = 0

    # apply gauge
    t = np.linspace(0, 1, grid_size)
    sine_x = prod * np.cos(2 * np.pi * t) - add
    sine_y = prod * np.cos(2 * np.pi * 2 * t) - add

    field_x = sine_x.reshape(1, grid_size)
    field_x = np.repeat(field_x, grid_size, axis=0).reshape(1, -1)

    field_y = sine_y.reshape(grid_size, 1)
    field_y = np.repeat(field_y, grid_size, axis=1).reshape(1, -1)

    field_x_tr = torch.tensor(np.repeat(field_x, bs_tr, axis=0)).float()
    field_y_tr = torch.tensor(np.repeat(field_y, bs_tr, axis=0)).float()
    field_x_te = torch.tensor(np.repeat(field_x, bs_te, axis=0)).float()
    field_y_te = torch.tensor(np.repeat(field_y, bs_te, axis=0)).float()

    # EMLP expects (N, 2 * data_dim,)
    X_tr = X_tr.view(-1, grid_size * grid_size) + field_x_tr + field_y_tr
    S_tr = torch.cat((torch.cos(X_tr), torch.sin(X_tr)), dim=-1)
    X_te = X_te.view(-1, grid_size * grid_size) + field_x_te + field_y_te
    S_te = torch.cat((torch.cos(X_te), torch.sin(X_te)), dim=-1)

    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(S_tr, Y_tr),
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(S_te, Y_te),
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    NUM_EPOCHS = 200
    lr = 1e-5
    G = SO(2)
    rep_in = 10000 * Vector
    rep_out = 1 * Scalar

    model = emlp.nn.EMLP(rep_in=rep_in, rep_out=rep_out, group=G)

    tr, ts = train_model(model)
    print(tr, ts)


if __name__ == "__main__":
    main()
