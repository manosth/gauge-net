# system imports
import os
from typing import Any, Callable, List, Optional, Type, Union
import time

# python imports
import numpy as np

# plotting imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation
from torchvision.models.resnet import Bottleneck

# plotting defaults
sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=2)
cmap = plt.get_cmap("twilight")
color_plot = sns.cubehelix_palette(4, reverse=True, rot=-0.2)
from matplotlib import cm, rc

rc("text", usetex=True)
rc("text.latex", preamble=r"\usepackage{amsmath}")

# torch imports
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torchinfo import summary

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = in_channels
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def main():
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
    train = "pre"

    if train == "train":
        model = ResNet(Bottleneck, [2, 2, 2, 2], in_channels=1)
    else:
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        if train == "pre":
            for param in model.parameters():
                param.requires_grad = False

    # change the output layer
    n_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(n_ftrs, 1)

    model = model.to(device)
    if train == "train":
        summary(model, input_size=(batch_size, 1, grid_size, grid_size))
    else:
        summary(model, input_size=(batch_size, 3, grid_size, grid_size))
        
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
            if train == "train":
                x, y = x.to(device), y.to(device)
                bs = x.shape[0]
                field_x_c = (
                    torch.tensor(np.repeat(field_x, bs, axis=0)).float().to(device)
                )
                field_y_c = (
                    torch.tensor(np.repeat(field_y, bs, axis=0)).float().to(device)
                )
                s = x + field_x_c + field_y_c
            else:
                x, y = x.to(device), y.to(device)
                bs = x.shape[0]
                field_x_c = (
                    torch.tensor(np.repeat(field_x, bs, axis=0)).float().to(device)
                )
                field_y_c = (
                    torch.tensor(np.repeat(field_y, bs, axis=0)).float().to(device)
                )
                s = x + field_x_c + field_y_c
                s = s.repeat(1, 3, 1, 1)

            y_hat = model(s)
            loss = loss_func(y_hat, y)

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
                if train == "train":
                    x, y = x.to(device), y.to(device)
                    bs = x.shape[0]
                    field_x_c = (
                        torch.tensor(np.repeat(field_x, bs, axis=0)).float().to(device)
                    )
                    field_y_c = (
                        torch.tensor(np.repeat(field_y, bs, axis=0)).float().to(device)
                    )
                    s = x + field_x_c + field_y_c
                else:
                    x, y = x.to(device), y.to(device)
                    bs = x.shape[0]
                    field_x_c = (
                        torch.tensor(np.repeat(field_x, bs, axis=0)).float().to(device)
                    )
                    field_y_c = (
                        torch.tensor(np.repeat(field_y, bs, axis=0)).float().to(device)
                    )
                    s = x + field_x_c + field_y_c
                    s = s.repeat(1, 3, 1, 1)

                y_hat = model(s)
                loss = loss_func(y_hat, y)

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
            torch.save(model.state_dict(), f"best_model_resnet-{train}.pth")

        with open(f"log_loss_tr_resnet-{train}.txt", "a") as file:
            file.write("Epoch " + str(epoch) + ":\t" + str(train_loss) + "\n")

        with open(f"log_loss_te_resnet-{train}.txt", "a") as file:
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
    plt.savefig(f"loss_resnet-{train}.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
