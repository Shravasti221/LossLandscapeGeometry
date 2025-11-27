

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from helpers import MLP, SimpleCNN, load_mnist, train_one_epoch, eval_loss, accuracy, compute_hessian_stats, plot_landscape

def run_phase2_optimizer_sweep(model_class, width=128, epochs=5):
    """
    Compare SGD, SGD+Momentum, Adam, RMSProp on same model/dataset.
    Uses SAME training, SAME landscape, SAME Hessian code as Phase 1.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = load_mnist()

    optimizers = {
        "SGD":          lambda params: optim.SGD(params, lr=0.01),
        "SGD+Momentum": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        "Adam":         lambda params: optim.Adam(params, lr=0.001),
        # "RMSProp":      lambda params: optim.RMSprop(params, lr=0.001)
    }

    results = {}

    for opt_name, opt_builder in optimizers.items():
        print(f"\n========== Optimizer: {opt_name} ==========")

        # fresh model
        model = model_class(width).to(device)
        opt = opt_builder(model.parameters())

        train_curve = []
        test_curve = []

        # ---- TRAIN ----
        for epoch in range(1, epochs+1):
            tr = train_one_epoch(model, train_loader, opt, device)
            te = eval_loss(model, test_loader, device)
            train_curve.append(tr)
            test_curve.append(te)
            print(f"Epoch {epoch}: train={tr:.4f}, test={te:.4f}")

        # ---- ACCURACY ----
        acc = accuracy(model, test_loader, device)
        print(f"Final Accuracy ({opt_name}): {acc*100:.2f}%")

        # ---- LANDSCAPE ----
        plot_landscape(model, test_loader)

        # ---- HESSIAN ----
        hess = compute_hessian_stats(model, test_loader, device, k=3, max_batches=5)

        # store
        results[opt_name] = {
            "train_curve": train_curve,
            "test_curve":  test_curve,
            "accuracy":    acc,
            "hessian":     hess
        }

    return results

MLP_phase2 = run_phase2_optimizer_sweep(MLP)
CNN_phase2 = run_phase2_optimizer_sweep(SimpleCNN)