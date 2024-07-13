# system imports
import os
import time

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


def gen_neighbors(grid_size, device):
    """
    Generate edge matrices for a grid of size grid_size x grid_size.
    """

    edg_up = torch.zeros(grid_size * grid_size, grid_size * grid_size, device=device)
    edg_down = torch.zeros(grid_size * grid_size, grid_size * grid_size, device=device)
    edg_left = torch.zeros(grid_size * grid_size, grid_size * grid_size, device=device)
    edg_right = torch.zeros(grid_size * grid_size, grid_size * grid_size, device=device)
    for x in range(grid_size):
        for y in range(grid_size):
            v = x + grid_size * y
            edg_up[v, x + ((y + 1) % grid_size) * grid_size] = 1
            edg_down[v, x + ((y - 1) % grid_size) * grid_size] = 1
            edg_left[v, ((x - 1) % grid_size) + y * grid_size] = 1
            edg_right[v, ((x + 1) % grid_size) + y * grid_size] = 1
    return edg_up, edg_down, edg_left, edg_right


def gen_neighbor_lists(grid_size):
    """
    Generate edge matrices for a grid of size grid_size x grid_size.
    """

    edg_up = set()
    edg_down = set()
    edg_left = set()
    edg_right = set()
    for x in range(grid_size):
        for y in range(grid_size):
            v = x + grid_size * y
            edg_up.add((v, x + ((y + 1) % grid_size) * grid_size))
            edg_down.add((v, x + ((y - 1) % grid_size) * grid_size))
            edg_left.add((v, ((x - 1) % grid_size) + y * grid_size))
            edg_right.add((v, ((x + 1) % grid_size) + y * grid_size))
    return (
        torch.tensor(np.array(list(edg_up)), dtype=int),
        torch.tensor(np.array(list(edg_down)), dtype=int),
        torch.tensor(np.array(list(edg_left)), dtype=int),
        torch.tensor(np.array(list(edg_right)), dtype=int),
    )


class GaugeLayer(nn.Module):
    def __init__(self, hid_dim=64, device="cpu"):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, x):
        out = self.net(x)
        out = x + out
        return out


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
        * torch.sum(torch.cos(grid_list[:, nbr[:, 0]] - grid_list[:, nbr[:, 1]]), dim=1)
    )
    return loss


class GaugeNet(nn.Module):
    def __init__(
        self, in_dim, grid_size, hid_dim=64, out_dim=1, n_layers=2, device="cpu"
    ):
        """
        grid_size needs to be in_dim ** 2
        """
        super().__init__()
        self.grid_size = grid_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.device = device

        up, down, left, right = gen_neighbor_lists(grid_size)
        self.up = torch.sparse_coo_tensor(
            up.t(),
            torch.ones(len(up), device=device),
            (grid_size * grid_size, grid_size * grid_size),
            device=device,
        )
        self.down = torch.sparse_coo_tensor(
            down.t(),
            # up.t(),
            torch.ones(len(down), device=device),
            (grid_size * grid_size, grid_size * grid_size),
            device=device,
        )
        self.left = torch.sparse_coo_tensor(
            # up.t(),
            left.t(),
            torch.ones(len(left), device=device),
            (grid_size * grid_size, grid_size * grid_size),
            device=device,
        )
        self.right = torch.sparse_coo_tensor(
            # up.t(),
            right.t(),
            torch.ones(len(right), device=device),
            (grid_size * grid_size, grid_size * grid_size),
            device=device,
        )

        self.H = torch.eye(2, device=device)

        # lattice_list has 40,000 (4 * grid_size ** 2) elements
        # self.A = -1 / (4 * grid_size**2) * torch.ones(4, 1, device=device)
        # self.s = nn.Parameter(torch.randn(1, device=device))

        ### SAVED
        # self.emb = nn.Linear(4, 1)
        # self.act1 = nn.SiLU()
        # self.post = nn.Linear(grid_size**2, out_dim)
        ### SAVED

        ### SAVED2
        self.emb = nn.Linear(4, hid_dim)
        self.act1 = nn.ReLU()
        self.hidden = nn.Linear(hid_dim, hid_dim)
        self.act2 = nn.ReLU()
        self.post = nn.Linear(hid_dim, out_dim)
        ### SAVED2

        # self.act1 = nn.Identity()
        # self.hidden = nn.Linear(hid_dim, hid_dim)
        # self.act2 = nn.ReLU()
        # self.act2 = nn.Identity()
        # self.final = nn.Linear(grid_size * grid_size, out_dim, bias=False)
        # self.emb = nn.Linear(4, 1, bias=False)
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(hid_dim, hid_dim, bias=False),
        #     nn.SiLU(),
        #     nn.Linear(hid_dim, hid_dim, bias=False),
        # )
        # list = []
        # for i in range(n_layers):
        #     self.add_module("layer_%d" % i, GaugeLayer(hid_dim, device))
        #     # self.add_module("act_%d" % i, nn.SiLU())
        #     # list.append(nn.Linear(hid_dim, hid_dim))
        #     # list.append(nn.SiLU())
        # # self.net = nn.Sequential(*list)
        # self.post_mlp = nn.Sequential(
        #     nn.Dropout(0.4),
        #     nn.Linear(hid_dim, hid_dim),
        #     nn.SiLU(),
        #     nn.Linear(hid_dim, out_dim),
        # )

    def forward(self, x):
        """
        Currentl Torch doesn't support sparse bmm, so we need to do it manually.
        """
        # x is (B, grid_size ** 2, 2)
        batch_size = x.shape[0]

        # do sparse matrix multiplication
        x_p = x.permute(1, 2, 0).reshape(
            self.grid_size * self.grid_size, 2 * batch_size
        )
        Ax_p_up = torch.sparse.mm(self.up, x_p)
        Ax_p_down = torch.sparse.mm(self.down, x_p)
        Ax_p_left = torch.sparse.mm(self.left, x_p)
        Ax_p_right = torch.sparse.mm(self.right, x_p)

        Ax_up = Ax_p_up.view(self.grid_size * self.grid_size, 2, batch_size).permute(
            2, 0, 1
        )
        Ax_down = Ax_p_down.view(
            self.grid_size * self.grid_size, 2, batch_size
        ).permute(2, 0, 1)
        Ax_left = Ax_p_left.view(
            self.grid_size * self.grid_size, 2, batch_size
        ).permute(2, 0, 1)
        Ax_right = Ax_p_right.view(
            self.grid_size * self.grid_size, 2, batch_size
        ).permute(2, 0, 1)

        # s_i is (B, grid_size ** 2)
        s_up = torch.einsum("bim,bin,mn->bi", x, Ax_up, self.H)
        s_down = torch.einsum("bim,bin,mn->bi", x, Ax_down, self.H)
        s_left = torch.einsum("bim,bin,mn->bi", x, Ax_left, self.H)
        s_right = torch.einsum("bim,bin,mn->bi", x, Ax_right, self.H)

        # s_up = torch.einsum("bim,ij,bjn,mn->bi", x, self.up, x, self.H)
        # s_down = torch.einsum("bim,ij,bjn,mn->bi", x, self.down, x, self.H)
        # s_left = torch.einsum("bim,ij,bjn,mn->bi", x, self.left, x, self.H)
        # s_right = torch.einsum("bim,ij,bjn,mn->bi", x, self.right, x, self.H)

        # h is (B, grid_size ** 2, 4)
        h = torch.stack([s_up, s_down, s_left, s_right], dim=2)
        # h = h @ self.A
        # h = h.sum(dim=1)
        # h = self.s * h
        # print(h.shape)

        # h = self.pre_mlp(h) + h
        # h = self.act(h)
        # # print(h.shape)
        # h = self.pre_mlp(h)
        # print(h.shape)
        # for layer in range(self.n_layers):
        #     h = self._modules["layer_%d" % layer](h)
        # #     print(h.shape)
        # # h = self._modules["act_%d" % layer](h) + h
        # # h = self.net(h)
        # h = self.post_mlp(h)
        # h = h.sum(dim=1)
        # h = self.act(h)
        # h = self.final(h.squeeze())
        h = self.act1(self.emb(h))
        h = self.act2(self.hidden(h))
        # h = self.act2(self.hidden(h))  # + h
        h = self.post(h)

        # h = self.act2(h)
        # print(h.shape)
        return h.sum(dim=1)
        # return h

        ### SAVED
        # h = self.act1(self.emb(h)).squeeze()
        # h = self.post(h)
        # return h
        ### SAVED

        ### SAVED2
        # h = self.act1(self.emb(h))
        # h = self.post(h)
        # return h.sum(dim=1)
        ### SAVED2


def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed = 13
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_norm = None  # "y"
    grid_size = 100
    batch_size = 64

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
        Y_tr, mean, std = standardize(Y_tr)
        Y_te, _, _ = standardize(Y_te, mean, std)
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

    epochs = 200
    hidden_nf = 1
    model = GaugeNet(
        grid_size**2, grid_size, hid_dim=hidden_nf, n_layers=1, device=device
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
    # opt = optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.0001, momentum=0.9)
    schd = optim.lr_scheduler.MultiStepLR(
        opt,
        [int(1 / 2 * epochs), int(3 / 4 * epochs)],
        gamma=0.1,
        # opt,
        # [140, 180],
        # gamma=0.1,
    )
    loss_func = torch.nn.MSELoss()

    summary(model, input_size=(batch_size, grid_size * grid_size, 2))
    edge_list = lattice_nbr(grid_size)

    # generate gauge field
    prod = 0.75  # 0.25
    add = 0  # .25

    # apply gauge
    t = np.linspace(0, 1, grid_size)
    sine_x = prod * np.cos(2 * np.pi * t) - add
    sine_y = prod * np.cos(2 * np.pi * 2 * t) - add

    field_x = sine_x.reshape(1, grid_size)
    field_x = np.repeat(field_x, grid_size, axis=0).reshape(1, -1, 1)

    field_y = sine_y.reshape(grid_size, 1)
    field_y = np.repeat(field_y, grid_size, axis=1).reshape(1, -1, 1)

    best_loss = 1e10
    best_epoch = None
    loss_tr = []
    loss_te = []
    start = time.time()
    prev = start
    for epoch in range(1, epochs + 1):
        net_loss = 0.0
        n_total = 0

        model.train()
        for idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)

            x = x.view(-1, grid_size * grid_size, 1)
            bs = x.shape[0]
            # field_x_c = torch.tensor(np.repeat(field_x, bs, axis=0)).float().to(device)
            # field_y_c = torch.tensor(np.repeat(field_y, bs, axis=0)).float().to(device)
            # x_t = x + field_x_c + field_y_c
            # print(energy_loss_nima(x, edge_list)[:4])
            # s = torch.cat((torch.cos(x_t), torch.sin(x_t)), dim=-1)
            s = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)

            h_hat = model(s)

            # h_hat = model(s)
            if epoch == 1 and idx == 0:
                print(h_hat.shape)
            # print(h_hat[:4])
            # print(y[:4])
            loss = loss_func(h_hat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            net_loss += loss.item() * len(x)
            n_total += len(x)
            # break
        train_loss = net_loss / n_total
        loss_tr.append(train_loss)

        current = time.time()
        # break
        net_loss = 0.0
        n_total = 0
        model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dl):
                x, y = x.to(device), y.to(device)
                x = x.view(-1, grid_size * grid_size, 1)
                bs = x.shape[0]
                # field_x_c = (
                #     torch.tensor(np.repeat(field_x, bs, axis=0)).float().to(device)
                # )
                # field_y_c = (
                #     torch.tensor(np.repeat(field_y, bs, axis=0)).float().to(device)
                # )
                # x_t = x + field_x_c + field_y_c
                # s = torch.cat((torch.cos(x_t), torch.sin(x_t)), dim=-1)
                s = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)

                h_hat = model(s)
                loss = loss_func(h_hat, y)

                net_loss += loss.item() * len(x)
                n_total += len(x)
        test_loss = net_loss / n_total
        loss_te.append(test_loss)
        print(
            f"Epoch {epoch} Loss: {train_loss} (train)\t{test_loss} (test)\t({current - prev:3.2f} s/iter)"
        )
        prev = current

        if train_loss <= best_loss:
            best_loss = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"best_model_ours{hidden_nf}.pth")

        with open("log_loss_tr_ours.txt", "a") as file:
            file.write("Epoch " + str(epoch) + ":\t" + str(train_loss) + "\n")

        with open("log_loss_te_ours.txt", "a") as file:
            file.write("Epoch " + str(epoch) + ":\t" + str(test_loss) + "\n")

        schd.step()

    plt.figure()
    plt.yscale("log")
    plt.plot(
        range(len(loss_tr)),
        loss_tr,
        color=color_plot[0],
        label="train",
    )
    plt.plot(
        range(len(loss_te)),
        loss_te,
        color=color_plot[2],
        label="test",
    )
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig("loss_ours.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
