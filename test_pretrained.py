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
# cmap_t = plt.get_cmap("turbo")
# cmap = plt.get_cmap("hsv")
color_plot = sns.cubehelix_palette(4, reverse=True, rot=-0.2)
from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

from torchsummary import summary


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


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_norm = None  # "y"
    grid_size = 100

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

    batch_size = 1
    test_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_te, Y_te),
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)

    loss_func = torch.nn.MSELoss()
    # model = CNN(grid_size)
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

    # change the output layer
    n_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(n_ftrs, 1)

    # model.load_state_dict(
    #     torch.load("best_model_none.pth", map_location=torch.device("cpu"))
    # )
    model.load_state_dict(torch.load("best_model_none.pth"))
    model.to(device)

    summary(model, input_size=(3, grid_size, grid_size))

    model.eval()
    with torch.no_grad():
        net_loss = 0.0
        n_total = 0
        # for idx, (x, y) in enumerate(train_dl):
        for idx, (x, y) in enumerate(test_dl):
            x, y = x.repeat(1, 3, 1, 1).to(device), y.to(device)
            y_hat = model(x)
            loss = loss_func(y_hat, y)

            # if idx == 100:
            #     grid = x.clone().detach().cpu().view(grid_size, grid_size) % (2 * np.pi)
            #     s_n = torch.stack([torch.cos(grid), torch.sin(grid)])
            #     fig_n = plt.figure()
            #     ax_n = plt.gca()
            #     quiver_n = ax_n.quiver(X, Y, s_n[0], s_n[1], grid, cmap=cmap, scale=40)
            #     cbar = plt.colorbar(
            #         quiver_n, ax=ax_n, ticks=[0 + 0.05, np.pi, 2 * np.pi - 0.05]
            #     )
            #     cbar.ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
            #     plt.title("Sample test data")
            #     ax_n.set_xticklabels([])
            #     ax_n.set_yticklabels([])
            #     plt.show()
            #     plt.close()
            if idx % 100 == 0:
                print(f"actual energy: {y}\t estimated energy: {y_hat}")
            net_loss += loss.item() * len(x)
            n_total += len(x)
        test_loss = net_loss / n_total
        print(f"loss: {test_loss:.8f}")


if __name__ == "__main__":
    main()
