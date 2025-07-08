#!/usr/bin/env python
"""Fit a single DendriteConv2d unit so that its **E-I difference** matches a
chosen target receptive field (ON-centre, Gabor, etc.).

Usage
-----
$ python scripts/fit_real_neuron.py --target gabor --ori 45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d
from dendritic_modeling.networks.factories.vision_units import rf_on_center, rf_off_center, rf_gabor
from dendritic_modeling.metrics.visual import orientation_selectivity_index as osi


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["on", "off", "gabor"], default="gabor")
    p.add_argument("--ori", type=float, default=0.0, help="orientation for gabor target")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-1)
    p.add_argument("--save", type=Path, default=Path("fit_result.png"))
    return p.parse_args()


def make_target(args) -> torch.Tensor:
    if args.target == "on":
        return rf_on_center()
    if args.target == "off":
        return rf_off_center()
    return rf_gabor(theta=np.deg2rad(args.ori))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = make_target(args).to(device)

    layer = DendriteConv2d(1, 1, target.shape[0], padding=target.shape[0] // 2, bias=False).to(device)
    opt = optim.Adam(layer.parameters(), lr=args.lr)

    for step in range(args.steps):
        opt.zero_grad()
        # E-I difference image
        E = layer.weight_E[0, 0]  # Remove batch/channel dims
        I = layer.weight_I[0, 0]
        diff = E - I
        loss = F.mse_loss(diff, target)
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4e}")

    # Compute OSI if target is gabor
    if args.target == "gabor":
        # quick test responses
        angles = np.linspace(0, 180, 12, endpoint=False)
        responses = []
        for a in angles:
            g = rf_gabor(theta=np.deg2rad(a)).to(device)
            responses.append((diff * g).sum().item())
        print("OSI:", osi(responses, angles))

    # save figure
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(target.cpu(), cmap="bwr", vmin=-target.abs().max(), vmax=target.abs().max())
    ax[0].set_title("Target RF")
    ax[0].axis("off")
    ax[1].imshow(diff.cpu().detach().numpy(), cmap="bwr", vmin=-target.abs().max(), vmax=target.abs().max())
    ax[1].set_title("Fitted E-I")
    ax[1].axis("off")
    plt.tight_layout()
    fig.savefig(args.save)
    print("Saved", args.save)


if __name__ == "__main__":
    main() 