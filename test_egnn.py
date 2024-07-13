# system imports
import os

# python imports
import numpy as np
from operator import itemgetter

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from torchinfo import summary

# egnn imports
import egnn_clean as eg

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
# cmap_t = plt.get_cmap("turbo")
# cmap = plt.get_cmap("hsv")
color_plot = sns.cubehelix_palette(4, reverse=True, rot=-0.2)
from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")


def zeromean(X, mean=None, std=None):
    "Expects data in NxCxWxH."
    if mean is None:
        mean = X.mean(axis=(0, 2, 3))
        std = X.std(axis=(0, 2, 3))
        std = torch.ones(std.shape)

    X = torchvision.transforms.Normalize(mean, std)(X)
    return X, mean, std


def standardize(X, mean=None, std=None):
    "Expects data in NxCxWxH."
    if mean is None:
        mean = X.mean(axis=(0, 2, 3))
        std = X.std(axis=(0, 2, 3))

    X = torchvision.transforms.Normalize(mean, std)(X)
    return X, mean, std


def standardize_y(Y, mean=None, std=None):
    "Expects data in Nx1."
    if mean is None:
        mean = Y.min()
        std = Y.max() - Y.min()

    Y = (Y - mean) / std
    return Y, mean, std


def whiten(X, zca=None, mean=None, eps=1e-8):
    "Expects data in NxCxWxH."
    os = X.shape
    X = X.reshape(os[0], -1)

    if zca is None:
        mean = X.mean(dim=0)
        cov = np.cov(X, rowvar=False)
        U, S, V = np.linalg.svd(cov)
        zca = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
    X = torch.Tensor(np.dot(X - mean, zca.T).reshape(os))
    return X, zca, mean


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


def get_edges_batch(edges, n_nodes, batch_size, device):
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1, device=device)
    edges = [
        torch.LongTensor(edges[0]).to(device),
        torch.LongTensor(edges[1]).to(device),
    ]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_norm = "y"
    grid_size = 100
    batch_size = 1

    ### Data loading
    data = np.load("data_n=10000.npy", allow_pickle=True)
    # data = np.load("/Users/manos/data/gauge/data_n=10000.npy", allow_pickle=True)
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

    if data_norm == "standard":
        X_tr, mean, std = standardize(X_tr)
        X_te, _, _ = standardize(X_te, mean, std)
    elif data_norm == "zeromean":
        X_tr, mean, std = zeromean(X_tr)
        X_te, _, _ = zeromean(X_te, mean, std)
    elif data_norm == "whiten":
        X_tr, mean, std = standardize(X_tr)
        X_te, _, _ = standardize(X_te, mean, std)

        X_tr, zca, mean = whiten(X_tr)
        X_te, _, _ = whiten(X_te, zca, mean)
    elif data_norm == "y":
        Y_tr, mean, std = standardize_y(Y_tr)
        Y_te, _, _ = standardize_y(Y_te, mean, std)

    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, Y_tr),
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    test_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_te, Y_te),
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    epochs = 50
    L = lattice_nbr(grid_size)
    sL = sorted(L, key=itemgetter(0))
    rows, cols = [], []
    for item in sL:
        rows.append(item[0])
        cols.append(item[1])
    edges_b = [rows, cols]

    model = eg.EGNN(
        in_node_nf=1,
        hidden_nf=64,
        out_node_nf=1,
        in_edge_nf=1,
        device=device,
        n_layers=2,
    )
    model.load_state_dict(torch.load("best_model_egnn.pth"))
    model.to(device)

    loss_func = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        net_loss = 0.0
        n_total = 0
        # for idx, (x, y) in enumerate(train_dl):
        for idx, (x, y) in enumerate(test_dl):
            x, y = x.to(device), y.to(device)
            batch_size_t = x.shape[0]
            edges, edge_attr = get_edges_batch(
                edges_b, grid_size * grid_size, batch_size_t, device
            )

            # EGNN expects data as (N * grid_size * grid_size, 2)
            x = x.view(batch_size_t * grid_size * grid_size, 1)
            s = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
            h = torch.ones(batch_size_t * grid_size * grid_size, 1, device=device)

            if idx == 0:
                summary(model, input_data=[h, s, edges, edge_attr])
            h_hat, s_hat = model(h, s, edges, edge_attr)

            h_hat = h_hat.view(batch_size_t, grid_size * grid_size)
            h_sum = torch.sum(h_hat, dim=1, keepdim=True)

            loss = loss_func(h_sum, y)

            if idx % 200 == 0:
                print(f"actul energy: {y}\t estimated energy: {h_sum}")
            net_loss += loss.item() * len(x)
            n_total += len(x)
        test_loss = net_loss / n_total
        print(f"loss: {test_loss:.8f}")


if __name__ == "__main__":
    main()
