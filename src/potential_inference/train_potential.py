"""Train an MLP to predict a scalar potential V(x) such that the negative gradient
of V w.r.t. x matches the provided forces.

Usage (quick):
    python train_potential.py --synthetic --epochs 50

This file is intended to be production-friendly and easy to modify.
Features:
 - Dataset wrapper for (pc_components, forces)
 - MLP returning scalar potentials
 - Force prediction via autograd (force = -grad V)
 - Training + validation loops, checkpointing, early stopping
 - CLI via argparse
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PCForceDataset(Dataset):
    """Simple Dataset for PC-components -> forces mapping.

    pc: shape (M, N_pc)
    forces: shape (M, N_pc)
    """

    def __init__(self, pc: np.ndarray, forces: np.ndarray) -> None:
        assert pc.shape == forces.shape, "pc and forces must have same shape"
        self.pc = torch.from_numpy(pc).float()
        self.forces = torch.from_numpy(forces).float()

    def __len__(self) -> int:
        return len(self.pc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pc[idx], self.forces[idx]


class MLPPotential(nn.Module):
    """MLP mapping input x -> scalar potential V(x).

    Simple, configurable, returns a single scalar per input.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(128, 128),
        activation=nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns shape (batch, 1)
        return self.net(x)


def predict_forces_from_potential(model: nn.Module, x: torch.Tensor, sign: float = -1.0) -> torch.Tensor:
    """Compute predicted forces = sign * grad V(x) w.r.t x.

    The function uses autograd to compute dV/dx for each sample in the batch.
    Return shape: same as x (batch, input_dim)
    """
    # Ensure gradients wrt inputs are tracked
    x = x.clone().detach().requires_grad_(True)
    V = model(x).squeeze(-1)  # (batch,)
    # sum V to compute gradients for each sample efficiently
    grads = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
    return sign * grads


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 10
    save_path: str = "best_potential.pth"
    sign: float = -1.0  # predicted_force = sign * grad V


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: TrainConfig,
) -> None:
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred_forces = predict_forces_from_potential(model, xb, sign=cfg.sign)
            loss = criterion(pred_forces, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss_acc = 0.0
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred_forces = predict_forces_from_potential(model, xb, sign=cfg.sign)
                    l = criterion(pred_forces, yb)
                    val_loss_acc += l.item() * xb.size(0)
                val_loss = val_loss_acc / len(val_loader.dataset)

        # scheduler (use val_loss if available)
        if val_loss is not None:
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        # checkpointing & early stopping
        monitor = val_loss if val_loss is not None else train_loss
        improved = monitor < best_val
        if improved:
            best_val = monitor
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "cfg": cfg}, cfg.save_path)
        else:
            epochs_no_improve += 1

        tqdm.write(f"Epoch {epoch:03d} Train loss: {train_loss:.6f}" + (f" Val loss: {val_loss:.6f}" if val_loss is not None else ""))

        if epochs_no_improve >= cfg.patience:
            tqdm.write(f"Early stopping at epoch {epoch}, best {best_val:.6f}")
            break


def make_synthetic_quadratic(M: int, N: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple quadratic potential dataset for quick verification.

    V(x) = 0.5 * x^T A x + b^T x + c
    force = -grad V = -(A_sym x + b)
    Returns pc (M,N) and forces (M,N).
    """
    rng = np.random.RandomState(seed)
    # Create symmetric positive-definite A
    Q = rng.randn(N, N)
    A = np.dot(Q.T, Q) * 0.1
    b = rng.randn(N) * 0.1
    pc = rng.randn(M, N)
    forces = -(pc @ A.T + b)
    return pc.astype(np.float32), forces.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None, help=".npz file with pc and forces arrays")
    p.add_argument("--synthetic", action="store_true", help="Run a synthetic quick experiment")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dims", type=int, nargs="*", default=[128, 128])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", type=str, default="best_potential.pth")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.synthetic:
        pc, forces = make_synthetic_quadratic(M=2000, N=20, seed=args.seed)
        # split train/val
        split = int(0.9 * len(pc))
        train_pc, val_pc = pc[:split], pc[split:]
        train_f, val_f = forces[:split], forces[split:]
    elif args.data_path is not None:
        d = np.load(args.data_path)
        pc = d["pc"]
        forces = d["forces"]
        split = int(0.9 * len(pc))
        train_pc, val_pc = pc[:split], pc[split:]
        train_f, val_f = forces[:split], forces[split:]
    else:
        raise ValueError("Provide --synthetic or --data-path path.npz containing 'pc' and 'forces'")

    train_ds = PCForceDataset(train_pc, train_f)
    val_ds = PCForceDataset(val_pc, val_f)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = MLPPotential(input_dim=train_pc.shape[1], hidden_dims=tuple(args.hidden_dims))
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, patience=args.patience, save_path=args.save_path)

    train(model, train_loader, val_loader, cfg)


if __name__ == "__main__":
    main()
