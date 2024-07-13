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

        up, down, left, right = gen_neighbors(grid_size, device)
        self.up = up
        self.down = down
        self.left = left
        self.right = right

        self.H = torch.eye(2, device=device)

        self.emb = nn.Linear(4 * in_dim, hid_dim)
        self.pre_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.SiLU(), nn.Linear(hid_dim, hid_dim)
        )
        list = []
        for _ in range(n_layers):
            list.append(nn.Linear(hid_dim, hid_dim))
            list.append(nn.SiLU())
        self.net = nn.Sequential(*list)
        self.post_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        # x is (B, grid_size ** 2, 2)
        batch_size = x.shape[0]

        # s_i is (B, grid_size ** 2)
        s_up = torch.einsum("bim,ij,bjn,mn->bi", x, self.up, x, self.H)
        s_down = torch.einsum("bim,ij,bjn,mn->bi", x, self.down, x, self.H)
        s_left = torch.einsum("bim,ij,bjn,mn->bi", x, self.left, x, self.H)
        s_right = torch.einsum("bim,ij,bjn,mn->bi", x, self.right, x, self.H)

        # h is (B, grid_size ** 2, 4)
        h = torch.stack([s_up, s_down, s_left, s_right], dim=1)

        h = self.emb(h.view(batch_size, -1))
        h = self.pre_mlp(h)
        h = self.net(h)
        h = self.post_mlp(h)
        return h


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

    hidden_nf = 16
    model = GaugeNet(
        grid_size**2, grid_size, hid_dim=hidden_nf, n_layers=2, device=device
    )
    model.load_state_dict(torch.load("best_model_ours16.pth"))
    model.to(device)

    loss_func = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        net_loss = 0.0
        n_total = 0
        for idx, (x, y) in enumerate(test_dl):
            x, y = x.to(device), y.to(device)
            x = x.view(-1, grid_size * grid_size, 1)
            s = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)

            h_hat = model(s)
            loss = loss_func(h_hat, y)

            if idx % 200 == 0:
                print(f"actul energy: {y}\t estimated energy: {h_hat}")
            net_loss += loss.item() * len(x)
            n_total += len(x)
        test_loss = net_loss / n_total
        print(f"loss: {test_loss:.8f}")


if __name__ == "__main__":
    main()
