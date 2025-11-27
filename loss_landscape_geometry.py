# ===============================================================
# Standalone Script: Loss Landscape + Hessian + Sharpness Analysis
# ===============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import loss_landscapes as ll
import loss_landscapes.metrics as metrics
from hessian_eigenthings import compute_hessian_eigenthings


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
        self.conv1 = nn.Conv2d(1, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width*2, 3, padding=1)
        self.fc1 = nn.Linear((width*2)*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_mnist(batch=128):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch, shuffle=True),
        DataLoader(test, batch_size=batch, shuffle=False)
    )


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


def train_model(model, train_loader, test_loader, device, epochs=5):
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    snapshots = []
    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device)
        te = eval_loss(model, test_loader, device)
        print(f"Epoch {ep}: train={tr:.4f}, test={te:.4f}")
        snapshots.append(deepcopy(model.state_dict()))
    return snapshots



def compute_1d_sharpness(model, loader, steps=25, distance=0.5):
    model_cpu = deepcopy(model).cpu()
    x, y = next(iter(loader))
    metric = metrics.Loss(F.cross_entropy, x.cpu(), y.cpu())

    curve = ll.random_line(model_cpu, metric, distance, steps, normalization="filter")
    alphas = np.linspace(-distance, distance, steps)
    slopes = np.abs(np.diff(curve) / np.diff(alphas))
    return float(np.max(slopes))


def get_hessian_sharpness(model, loader, device, k=3, n_batches=20):
    model = model.to(device)
    model.eval()
    eigs = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches: break
        x, y = x.to(device), y.to(device)
        evs, _ = compute_hessian_eigenthings(
            model, [(x, y)],
            loss=lambda o,t: F.cross_entropy(o,t),
            num_eigenthings=k
        )
        eigs.append([float(ev) for ev in evs])
    eigs = np.array(eigs)
    return eigs.mean(0)


def perturb_model(model, scale=1e-3):
    out = deepcopy(model)
    with torch.no_grad():
        for p in out.parameters():
            p.add_(torch.randn_like(p) * scale)
    return out


def compute_flatness(model, loader, device, samples=20):
    base = eval_loss(model, loader, device)
    losses = []
    for _ in range(samples):
        pm = perturb_model(model)
        losses.append(eval_loss(pm.to(device), loader, device))
    losses = np.array(losses)
    return base, float(losses.std()), float(losses.mean())


def flatten_params(state_dict):
    return torch.cat([p.flatten() for p in state_dict.values()])


def interpolate_loss(model_a, model_b, loader, device, steps=30):
    pa = flatten_params(model_a.state_dict())
    pb = flatten_params(model_b.state_dict())
    alphas = np.linspace(0, 1, steps)
    losses = []

    for a in alphas:
        theta = (1-a)*pa + a*pb

        m = deepcopy(model_a)
        idx = 0
        new_state = {}
        for k, v in m.state_dict().items():
            sz = v.numel()
            new_state[k] = theta[idx:idx+sz].view_as(v)
            idx += sz

        m.load_state_dict(new_state)
        losses.append(eval_loss(m.to(device), loader, device))

    return alphas, losses

def stack_snapshots(snaps):
    return torch.stack([flatten_params(s) for s in snaps], dim=0)


def compute_pca_dir(snaps):
    X = stack_snapshots(snaps)
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    return Vt[0] / Vt[0].norm()   # principal direction


def eval_direction(model, loader, device, direction, dist=1.0, steps=41):
    base = flatten_params(model.state_dict())
    alphas = np.linspace(-dist, dist, steps)
    losses = []

    for a in alphas:
        theta = base + a * direction

        m = deepcopy(model)
        idx = 0
        new_state = {}
        for k, v in m.state_dict().items():
            sz = v.numel()
            new_state[k] = theta[idx:idx+sz].view_as(v)
            idx += sz

        m.load_state_dict(new_state)
        losses.append(eval_loss(m.to(device), loader, device))

    return alphas, losses

def run_experiment(model_class, width=128, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_mnist()

    print("\nTraining first model...")
    model1 = model_class(width).to(device)
    snaps1 = train_model(model1, train_loader, test_loader, device, epochs)

    print("\nTraining second model...")
    model2 = model_class(width).to(device)
    snaps2 = train_model(model2, train_loader, test_loader, device, epochs)

    sharp1d = compute_1d_sharpness(model1, test_loader)
    hess_sharp = get_hessian_sharpness(model1, test_loader, device)
    base, flat_std, flat_mean = compute_flatness(model1, test_loader, device)

    print("\n1D sharpness =", sharp1d)
    print("Hessian sharpness =", hess_sharp)
    print("Perturbation flatness std =", flat_std)

    al, lo = interpolate_loss(model1, model2, test_loader, device)
    plt.plot(al, lo)
    plt.title("Mode Connectivity")
    plt.xlabel("α"); plt.ylabel("Loss")
    plt.grid(); plt.show()

    pca_dir = compute_pca_dir(snaps1)
    al, lo = eval_direction(model1, test_loader, device, pca_dir)
    plt.plot(al, lo)
    plt.title("PCA Direction Landscape")
    plt.xlabel("α"); plt.ylabel("Loss")
    plt.grid(); plt.show()


run_experiment(MLP, width=128, epochs=15)

