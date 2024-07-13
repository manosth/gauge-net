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
color_plot = sns.cubehelix_palette(4, reverse=True, rot=-0.2)
from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

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

    data_norm = None  # "y"
    grid_size = 100
    batch_size = 64

    ### Data loading
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

    hidden_nf = 64
    model = eg.EGNN(
        in_node_nf=1,
        hidden_nf=hidden_nf,
        out_node_nf=1,
        in_edge_nf=1,
        device=device,
        n_layers=2,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
    schd = optim.lr_scheduler.MultiStepLR(
        opt, [int(1 / 2 * epochs), int(3 / 4 * epochs)], gamma=0.1
    )
    loss_func = torch.nn.MSELoss()

    best_loss = 1e10
    best_epoch = None
    loss_tr = []
    loss_te = []
    start = time.time()
    prev = start

    # generate gauge field
    prod = 0.75
    add = 0

    # apply gauge
    t = np.linspace(0, 1, grid_size)
    sine_x = prod * np.cos(2 * np.pi * t) - add
    sine_y = prod * np.cos(2 * np.pi * 2 * t) - add

    field_x = sine_x.reshape(1, grid_size)
    field_x = np.repeat(field_x, grid_size, axis=0).reshape(1, 1, grid_size, grid_size)

    field_y = sine_y.reshape(grid_size, 1)
    field_y = np.repeat(field_y, grid_size, axis=1).reshape(1, 1, grid_size, grid_size)
    for epoch in range(1, epochs + 1):
        net_loss = 0.0
        n_total = 0

        model.train()
        for idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            batch_size_t = x.shape[0]
            edges, edge_attr = get_edges_batch(
                edges_b, grid_size * grid_size, batch_size_t, device
            )

            field_x_c = (
                torch.tensor(np.repeat(field_x, batch_size_t, axis=0))
                .float()
                .to(device)
            )
            field_y_c = (
                torch.tensor(np.repeat(field_y, batch_size_t, axis=0))
                .float()
                .to(device)
            )
            x_t = x  + field_x_c + field_y_c

            # EGNN expects data as (N * grid_size * grid_size, 2)
            x = x_t.view(batch_size_t * grid_size * grid_size, 1)
            s = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
            h = torch.ones(batch_size_t * grid_size * grid_size, 1, device=device)

            if idx == 0 and epoch == 1:
                summary(model, input_data=[h, s, edges, edge_attr])

            h_hat, s_hat = model(h, s, edges, edge_attr)

            h_hat = h_hat.view(batch_size_t, grid_size * grid_size)
            h_sum = torch.sum(h_hat, dim=1, keepdim=True)

            loss = loss_func(h_sum, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            net_loss += loss.item() * len(x)
            n_total += len(x)
        train_loss = net_loss / n_total
        loss_tr.append(train_loss)

        current = time.time()

        net_loss = 0.0
        n_total = 0
        model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dl):
                x, y = x.to(device), y.to(device)
                batch_size_t = x.shape[0]

                edges, edge_attr = get_edges_batch(
                    edges_b, grid_size * grid_size, batch_size_t, device
                )

                field_x_c = (
                    torch.tensor(np.repeat(field_x, batch_size_t, axis=0))
                    .float()
                    .to(device)
                )
                field_y_c = (
                    torch.tensor(np.repeat(field_y, batch_size_t, axis=0))
                    .float()
                    .to(device)
                )
                x_t = x + field_x_c + field_y_c

                # EGNN expects data as (N * grid_size * grid_size, 2)
                x = x_t.view(batch_size_t * grid_size * grid_size, 1)
                s = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
                h = torch.ones(batch_size_t * grid_size * grid_size, 1, device=device)

                h_hat, s_hat = model(h, s, edges, edge_attr)

                h_hat = h_hat.view(batch_size_t, grid_size * grid_size)
                h_sum = torch.sum(h_hat, dim=1, keepdim=True)

                loss = loss_func(h_sum, y)

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
            torch.save(model.state_dict(), f"best_model_egnn{hidden_nf}.pth")

        with open("log_loss_tr_none.txt", "a") as file:
            file.write("Epoch " + str(epoch) + ":\t" + str(train_loss) + "\n")

        with open("log_loss_te_none.txt", "a") as file:
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
    plt.savefig("loss.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
