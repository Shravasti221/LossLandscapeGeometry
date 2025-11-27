# ===============================================================
# PHASE 1 — Architecture Comparison Experiment  (unchanged)
# ===============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import loss_landscapes as ll
import loss_landscapes.metrics as metrics
from hessian_eigenthings import compute_hessian_eigenthings


# ===============================================================
# MODELS
# ===============================================================

class MLP(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.fc1 = nn.Linear(28*28, width)
        self.fc2 = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width*2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((width*2)*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_mnist(batch=128):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test  = datasets.MNIST("./data", train=False, download=True, transform=tf)

    return (
        DataLoader(train, batch_size=batch, shuffle=True),
        DataLoader(test,  batch_size=batch, shuffle=False)
    )


# ===============================================================
# TRAINING HELPERS
# ===============================================================

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def eval_loss(model, loader, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += F.cross_entropy(model(x), y).item() * x.size(0)
    return total / len(loader.dataset)


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)


# ===============================================================
# LANDSCAPE
# ===============================================================

def plot_landscape(model, loader, num_1d_curves=10, distance=1.5,
                   steps_1d=101, steps_2d=50):
    """
    Plots all figures in ONE ROW:
    [ 1D Loss Curves | 3D Surface | 2D Contour ]

    - 10 random 1D directions (symmetric range)
    - High-res 2D landscape with extended range
    """

    model_cpu = deepcopy(model).cpu()
    batch_x, batch_y = next(iter(loader))
    metric = metrics.Loss(F.cross_entropy, batch_x.cpu(), batch_y.cpu())
    plane = ll.random_plane(
        model=model_cpu,
        metric=metric,
        distance=distance,
        steps=steps_2d,
        normalization='filter'
    )

    Z = np.array(plane)
    A = np.linspace(-distance, distance, steps_2d)
    X, Y = np.meshgrid(A, A)
    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(1, 3, 1)

    for i in range(num_1d_curves):
        line = ll.random_line(
            model_start=model_cpu,
            metric=metric,
            distance=distance,
            steps=steps_1d,
            normalization='filter'
        )
        alphas = np.linspace(-distance, distance, steps_1d)
        ax1.plot(alphas, line, alpha=0.6)

    ax1.set_title("1D Loss Landscapes (10 Random Directions)")
    ax1.set_xlabel("α (direction scale)")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax2.set_title("2D Loss Landscape (Surface)")
    ax2.set_xlabel("α₁")
    ax2.set_ylabel("α₂")
    ax2.set_zlabel("Loss")

    ax3 = fig.add_subplot(1, 3, 3)
    contour = ax3.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar(contour, ax=ax3)
    ax3.set_title("2D Loss Landscape (Contour)")
    ax3.set_xlabel("α₁")
    ax3.set_ylabel("α₂")

    plt.tight_layout()
    plt.show()

def compute_hessian_stats(model, loader, device, k=3, max_batches=5):
    model = model.to(device)
    for p in model.parameters(): p.requires_grad_(True)
    all_eigs = []

    def loss_fn(pred, tgt):
        return F.cross_entropy(pred, tgt)

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        eigenvals, _ = compute_hessian_eigenthings(
            model, [(x, y)], loss_fn, num_eigenthings=k
        )
        all_eigs.append([float(e) for e in eigenvals])
        if i+1 >= max_batches: break

    all_eigs = np.array(all_eigs)
    return {
        "mean": all_eigs.mean(axis=0),
        "std": all_eigs.std(axis=0),
        "min": all_eigs.min(axis=0),
        "max": all_eigs.max(axis=0),
        "all": all_eigs
    }