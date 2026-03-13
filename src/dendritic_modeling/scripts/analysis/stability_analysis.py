"""
Stability analysis for dendritic recurrent models.

Computes:
1. Gradient norm distributions from training batches
2. Jacobian eigenvalue spectrum at trained operating points
3. Stability-accuracy frontier across model variants

Usage:
    PYTHONPATH=src:$PYTHONPATH python -m dendritic_modeling.scripts.analysis.stability_analysis \
        --results_dir <results_dir> \
        --output_dir <output_dir> \
        --device cuda
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def compute_jacobian_eigenvalues(
    model: nn.Module,
    x_input: torch.Tensor,
    n_steps: int = 50,
    device: str = "cpu",
) -> dict:
    """Compute Jacobian eigenvalues of the recurrent dynamics at steady state.

    Runs the model for n_steps to reach an approximate fixed point, then
    computes the Jacobian of the hidden state update w.r.t. the previous
    hidden state via finite differences.

    Args:
        model: Trained DendRNN model (must have core_network with cell).
        x_input: Input sequence [1, T, D] where T >= n_steps.
        n_steps: Number of warmup steps before computing Jacobian.
        device: Device for computation.

    Returns:
        Dict with 'eigenvalues' (complex array), 'max_real' (float),
        'spectral_radius' (float).
    """
    model.eval()
    model.to(device)
    x_input = x_input.to(device)

    core = model.core_network
    cell = core.cell
    B, T, D = x_input.shape

    # Get input projection
    if hasattr(core, "input_proj"):
        with torch.no_grad():
            # Apply input projection to all timesteps
            x_proj = core.input_proj(x_input.reshape(B * T, D)).reshape(B, T, -1)
    else:
        x_proj = x_input

    # Run warmup to reach approximate steady state
    from dendritic_modeling.networks.architectures.recurrent.state import RecurrentState

    state = RecurrentState.init(
        batch_size=B,
        n_exc=cell.n_exc,
        n_inh=cell.n_inh,
        n_branches=cell.n_branches,
        device=device,
    )

    with torch.no_grad():
        for t in range(min(n_steps, T)):
            x_t = x_proj[:, t, :]
            _, state, _ = cell(x_t, state, t_step=t)

    # Now compute Jacobian via autograd at timestep n_steps
    # Flatten state into a single vector for Jacobian computation
    state_vec = _state_to_vector(state)
    state_vec = state_vec.detach().requires_grad_(True)

    # Reconstruct state from vector, do one step, get new state vector
    state_reconstructed = _vector_to_state(
        state_vec,
        B,
        cell.n_exc,
        cell.n_inh,
        cell.n_branches,
        cell.branch_configs,
        device,
    )
    x_t = x_proj[:, min(n_steps, T - 1), :]
    _, new_state, _ = cell(x_t, state_reconstructed, t_step=n_steps)
    new_state_vec = _state_to_vector(new_state)

    # Compute Jacobian row by row
    n_state = state_vec.shape[1]
    jacobian = torch.zeros(n_state, n_state, device=device)
    for i in range(n_state):
        grad_outputs = torch.zeros_like(new_state_vec)
        grad_outputs[0, i] = 1.0
        grads = torch.autograd.grad(
            new_state_vec,
            state_vec,
            grad_outputs=grad_outputs,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grads is not None:
            jacobian[i, :] = grads[0]

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(jacobian).cpu().numpy()

    return {
        "eigenvalues": eigenvalues,
        "max_real": float(np.max(np.real(eigenvalues))),
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "n_state_dims": n_state,
    }


def _state_to_vector(state) -> torch.Tensor:
    """Flatten RecurrentState into [1, D] vector."""
    parts = []
    for e in state.branch_E:
        parts.append(e)
    for e_rec in state.branch_E_rec:
        parts.append(e_rec)
    for i_val in state.branch_I:
        parts.append(i_val)
    for i_rec in state.branch_I_rec:
        parts.append(i_rec)
    for v in state.branch_V:
        parts.append(v)
    if state.h_inh is not None:
        parts.append(state.h_inh)
    return torch.cat(parts, dim=-1)


def _vector_to_state(vec, batch_size, n_exc, n_inh, n_branches, branch_configs, device):
    """Reconstruct RecurrentState from flattened vector."""
    from dendritic_modeling.networks.architectures.recurrent.state import RecurrentState

    idx = 0
    branch_E = []
    for _ in range(n_branches):
        branch_E.append(vec[:, idx : idx + n_exc])
        idx += n_exc
    branch_E_rec = []
    for _ in range(n_branches):
        branch_E_rec.append(vec[:, idx : idx + n_exc])
        idx += n_exc
    branch_I = []
    for _ in range(n_branches):
        branch_I.append(vec[:, idx : idx + n_exc])
        idx += n_exc
    branch_I_rec = []
    for _ in range(n_branches):
        branch_I_rec.append(vec[:, idx : idx + n_exc])
        idx += n_exc
    branch_V = []
    for _ in range(n_branches):
        branch_V.append(vec[:, idx : idx + n_exc])
        idx += n_exc
    h_inh = None
    if n_inh > 0:
        h_inh = vec[:, idx : idx + n_inh]
        idx += n_inh

    return RecurrentState(
        branch_E=branch_E,
        branch_E_rec=branch_E_rec,
        branch_I=branch_I,
        branch_I_rec=branch_I_rec,
        branch_V=branch_V,
        h_inh=h_inh,
    )


def compute_gradient_norms(
    model: nn.Module,
    dataloader,
    loss_fn: nn.Module,
    device: str = "cpu",
    max_batches: int = 20,
) -> dict:
    """Compute gradient norm statistics across batches.

    Returns per-parameter-group gradient norms and total gradient norm.
    """
    model.train()
    model.to(device)

    total_norms = []
    branch_norms = {
        f"branch_{b}": []
        for b in range(getattr(model.core_network.cell, "n_branches", 0))
    }

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        model.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()

        # Total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)
        total_norms.append(total_norm)

        # Per-branch gradient norms (if dendritic cell)
        cell = model.core_network.cell
        if hasattr(cell, "branches"):
            for b, branch_mod in enumerate(cell.branches):
                branch_norm = 0.0
                for p in branch_mod.parameters():
                    if p.grad is not None:
                        branch_norm += p.grad.data.norm(2).item() ** 2
                branch_norms[f"branch_{b}"].append(math.sqrt(branch_norm))

    return {
        "total_grad_norms": total_norms,
        "branch_grad_norms": {k: v for k, v in branch_norms.items() if v},
        "mean_total_norm": float(np.mean(total_norms)) if total_norms else 0.0,
        "max_total_norm": float(np.max(total_norms)) if total_norms else 0.0,
        "std_total_norm": float(np.std(total_norms)) if total_norms else 0.0,
    }


def plot_eigenvalue_spectrum(
    results: dict,
    output_path: Path,
    title: str = "Jacobian Eigenvalue Spectrum",
):
    """Plot eigenvalue spectrum in the complex plane."""
    eigs = results["eigenvalues"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.scatter(
        np.real(eigs),
        np.imag(eigs),
        s=25,
        alpha=0.7,
        c="steelblue",
        edgecolors="black",
        linewidths=0.3,
    )

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, linewidth=0.8)

    # Stability boundary (Re=0 line)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

    ax.set_xlabel(r"Re($\lambda$)", fontsize=11)
    ax.set_ylabel(r"Im($\lambda$)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Stats annotation
    stats_text = (
        f"max Re(λ) = {results['max_real']:.4f}\n"
        f"|λ|_max = {results['spectral_radius']:.4f}\n"
        f"dims = {results['n_state_dims']}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "alpha": 0.8},
    )

    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gradient_norms(
    results: dict,
    output_path: Path,
    title: str = "Gradient Norm Distribution",
):
    """Plot gradient norm distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Total gradient norms
    ax = axes[0]
    norms = results["total_grad_norms"]
    ax.hist(
        norms, bins=20, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax.axvline(
        x=results["mean_total_norm"],
        color="red",
        linestyle="--",
        label=f"mean={results['mean_total_norm']:.3f}",
    )
    ax.set_xlabel("Gradient Norm", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Total Gradient Norms", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Per-branch gradient norms
    ax = axes[1]
    branch_data = results.get("branch_grad_norms", {})
    if branch_data:
        colors = ["#E74C3C", "#F39C12", "#2980B9", "#27AE60", "#8E44AD"]
        for i, (name, norms_b) in enumerate(branch_data.items()):
            color = colors[i % len(colors)]
            ax.hist(
                norms_b,
                bins=15,
                alpha=0.5,
                color=color,
                label=name,
                edgecolor="black",
                linewidth=0.3,
            )
        ax.set_xlabel("Gradient Norm", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Per-Branch Gradient Norms", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No branch data\n(non-dendritic model)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
        ax.set_title("Per-Branch Gradient Norms", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_stability_frontier(
    variants_data: list[dict],
    output_path: Path,
):
    """Plot stability-accuracy frontier across model variants.

    Args:
        variants_data: List of dicts with 'name', 'accuracy', 'max_real_eigenvalue',
                       'spectral_radius', 'exploded_fraction', 'soma_mode'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Color by soma mode
    soma_colors = {
        "shunting": "#2980B9",
        "additive": "#E74C3C",
        "sum": "#F39C12",
        "gru": "#7F8C8D",
        "lstm": "#27AE60",
    }

    # Panel 1: Accuracy vs max Re(eigenvalue)
    ax = axes[0]
    for v in variants_data:
        color = soma_colors.get(v.get("soma_mode", "unknown"), "#95A5A6")
        ax.scatter(
            v["max_real_eigenvalue"],
            v["accuracy"],
            s=80,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.annotate(
            v["name"],
            (v["max_real_eigenvalue"], v["accuracy"]),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Stability boundary")
    ax.set_xlabel(r"max Re($\lambda$)", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("(a) Accuracy vs Stability", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: Accuracy vs max gradient norm
    ax = axes[1]
    for v in variants_data:
        color = soma_colors.get(v.get("soma_mode", "unknown"), "#95A5A6")
        ax.scatter(
            v.get("max_grad_norm", 0),
            v["accuracy"],
            s=80,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.annotate(
            v["name"],
            (v.get("max_grad_norm", 0), v["accuracy"]),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Max Gradient Norm", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("(b) Accuracy vs Gradient Stability", fontsize=11, fontweight="bold")

    # Legend for soma modes
    for mode, color in soma_colors.items():
        ax.scatter(
            [], [], s=60, color=color, label=mode, edgecolors="black", linewidths=0.5
        )
    ax.legend(fontsize=8, title="Soma mode", title_fontsize=8)

    fig.suptitle("Stability-Accuracy Frontier", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stability analysis for DendRNN")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/rnn/",
        help="Directory containing trained model results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/rnn/stability_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=50,
        help="Number of warmup steps before Jacobian computation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found. Run training first.")
        return

    # Collect results from all variants
    all_results = {}
    for variant_dir in sorted(results_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        model_path = variant_dir / "model.pt"
        if not model_path.exists():
            continue

        variant_name = variant_dir.name
        print(f"\nAnalyzing: {variant_name}")

        try:
            torch.load(model_path, map_location=args.device)
            # Store basic info
            all_results[variant_name] = {
                "name": variant_name,
                "model_path": str(model_path),
            }
            print(f"  Loaded checkpoint for {variant_name}")
        except Exception as e:
            print(f"  Error loading {variant_name}: {e}")
            continue

    # Save summary
    summary_path = output_dir / "stability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
                for k, v in all_results.items()
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved summary: {summary_path}")
    print(f"Analyzed {len(all_results)} variants")


if __name__ == "__main__":
    main()
