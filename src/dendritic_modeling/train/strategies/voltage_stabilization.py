"""
Voltage stabilization training strategy with warm-up phase.
"""

import os
import time
import json
from copy import deepcopy
from typing import Optional
import torch
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import wandb
from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.analysis import AnalysisManager
from .base import BaseTrainer, compute_metrics_classification


def voltage_warmup_loss(model, x, target_val=0.5):
    """Compute voltage warm-up loss to stabilize outputs to target value."""
    outputs = model(x)
    if hasattr(model, "net"):
        if hasattr(model.net, "branch_layers"):
            vs = []
            for b in model.net.branch_layers:
                if hasattr(b, "branch_excitation") and b.branch_excitation is not None:
                    pass
                if hasattr(b, "branch_inhibition") and b.branch_inhibition is not None:
                    pass
            # For demonstration, we just do a single approach:
    # The simplest approach is to treat 'outputs' as "voltage" if possible
    return torch.mean((outputs - target_val) ** 2)


def generate_warmup_report(model, data_loader, pre_state, post_state, device, save_path):
    """Generate a report on the differences before and after voltage warm-up."""
    report_data = {}

    # Create report directory
    warmup_report_dir = os.path.join(save_path, "warmup_report")
    os.makedirs(warmup_report_dir, exist_ok=True)

    # Get a batch of data for visualization
    x_batch, _ = next(iter(data_loader))
    x_batch = x_batch.to(device)

    # Save current model state
    current_state = deepcopy(model.state_dict())

    # Function to collect outputs and activations
    def collect_model_stats(state_dict):
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            outputs = model(x_batch)
            output_stats = {
                "mean": outputs.mean().item(),
                "std": outputs.std().item(),
                "min": outputs.min().item(),
                "max": outputs.max().item(),
                "histogram": outputs.detach().cpu().numpy().flatten(),
            }

            # Collect activations from each layer if they're available
            layer_activations = {}
            if hasattr(model, "net"):
                if hasattr(model.net, "branch_layers"):
                    # Store the original state of activation tracking
                    activation_tracking_state = {}

                    # Try to enable activation tracking if it exists
                    for i, layer in enumerate(model.net.branch_layers):
                        if hasattr(layer, "enable_branch_outputs"):
                            activation_tracking_state[i] = layer.enable_branch_outputs
                            layer.enable_branch_outputs = True

                    # Forward pass to collect activations
                    _ = model(x_batch)

                    # Collect branch activations if they exist
                    for i, layer in enumerate(model.net.branch_layers):
                        if hasattr(layer, "branch_activations"):
                            layer_activations[f"layer_{i}"] = {
                                "mean": layer.branch_activations.mean().item(),
                                "std": layer.branch_activations.std().item(),
                                "min": layer.branch_activations.min().item(),
                                "max": layer.branch_activations.max().item(),
                                "histogram": layer.branch_activations.detach().cpu().numpy().flatten(),
                            }

                    # Restore original state
                    for i, state in activation_tracking_state.items():
                        model.net.branch_layers[i].enable_branch_outputs = state

        return output_stats, layer_activations

    # Collect statistics before and after warm-up
    pre_output_stats, pre_activations = collect_model_stats(pre_state)
    post_output_stats, post_activations = collect_model_stats(post_state)

    # Calculate differences
    output_diff = {
        "mean_diff": post_output_stats["mean"] - pre_output_stats["mean"],
        "std_diff": post_output_stats["std"] - pre_output_stats["std"],
        "min_diff": post_output_stats["min"] - pre_output_stats["min"],
        "max_diff": post_output_stats["max"] - pre_output_stats["max"],
    }

    # Store report data
    report_data["pre_warmup"] = {
        "outputs": pre_output_stats,
        "activations": pre_activations,
    }
    report_data["post_warmup"] = {
        "outputs": post_output_stats,
        "activations": post_activations,
    }
    report_data["diff"] = {"outputs": output_diff}

    # Generate output distribution plots
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.hist(pre_output_stats["histogram"], bins=50, alpha=0.5, label="Pre-warmup")
    plt.hist(post_output_stats["histogram"], bins=50, alpha=0.5, label="Post-warmup")
    plt.axvline(x=0.5, color="r", linestyle="--", label="Target voltage (0.5)")
    plt.title("Output Distribution Before and After Warm-up")
    plt.xlabel("Output Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 1, 2)
    for key in ["mean", "std", "min", "max"]:
        x = [0, 1]
        y = [pre_output_stats[key], post_output_stats[key]]
        plt.plot(x, y, marker="o", label=key)
    plt.axhline(y=0.5, color="r", linestyle="--", label="Target voltage (0.5)")
    plt.xticks([0, 1], ["Pre-Warmup", "Post-Warmup"])
    plt.title("Output Statistics Before and After Warm-up")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(warmup_report_dir, "output_distribution.png"))
    plt.close()

    # Save report summary as text file
    with open(os.path.join(warmup_report_dir, "warmup_report_summary.txt"), "w") as f:
        f.write("Voltage Warm-up Report Summary\n")
        f.write("==============================\n\n")
        f.write("Output Statistics:\n")
        f.write(f"  Before warm-up: mean={pre_output_stats['mean']:.4f}, std={pre_output_stats['std']:.4f}, min={pre_output_stats['min']:.4f}, max={pre_output_stats['max']:.4f}\n")
        f.write(f"  After warm-up: mean={post_output_stats['mean']:.4f}, std={post_output_stats['std']:.4f}, min={post_output_stats['min']:.4f}, max={post_output_stats['max']:.4f}\n")
        f.write(f"  Change: mean_diff={output_diff['mean_diff']:.4f}, std_diff={output_diff['std_diff']:.4f}, min_diff={output_diff['min_diff']:.4f}, max_diff={output_diff['max_diff']:.4f}\n\n")

    # Save numeric data as JSON for later analysis
    json_data = {
        "pre_warmup": {
            "outputs": {k: v for k, v in pre_output_stats.items() if k != "histogram"},
            "activations": {
                layer: {k: v for k, v in stats.items() if k != "histogram"}
                for layer, stats in pre_activations.items()
            },
        },
        "post_warmup": {
            "outputs": {k: v for k, v in post_output_stats.items() if k != "histogram"},
            "activations": {
                layer: {k: v for k, v in stats.items() if k != "histogram"}
                for layer, stats in post_activations.items()
            },
        },
        "diff": {"outputs": output_diff},
    }

    with open(os.path.join(warmup_report_dir, "warmup_data.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # Restore original model state
    model.load_state_dict(current_state)

    return report_data


class VoltageStabilizationTrainer(BaseTrainer):
    """Trainer that adds a warm-up phase to stabilize voltages to a target value."""
    
    def __init__(
        self,
        optimizer,
        suppress_prints=False,
        print_every=10,
        warmup_epochs=1,
        stability_target_voltage=0.5,
    ):
        super().__init__(optimizer, suppress_prints, print_every)
        self.warmup_epochs = warmup_epochs
        self.stability_target_voltage = stability_target_voltage

    def train(
        self,
        model,
        train_data,
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

        train_losses = []
        valid_losses = []
        best_loss = float("inf")

        # Store the model state before warm-up
        pre_warmup_state = deepcopy(model.state_dict())

        start_time = time.time()

        logger.info("=== Starting warm-up phase ===")
        for epoch in range(1, self.warmup_epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                self.optimizer.zero_grad()
                loss_val = voltage_warmup_loss(model, xb, target_val=self.stability_target_voltage)
                loss_val.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += loss_val.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            val_loss, _ = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[VoltageWarmup] epoch={epoch}/{self.warmup_epochs}, warmup_loss={epoch_train_loss:.4f}, "
                    f"valid_loss={val_loss:.4f}, elapsed={elapsed:.1f}s"
                )
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f"warmup_epoch_{epoch}", training=True)
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "warmup_loss": epoch_train_loss,
                    "valid_loss": val_loss,
                })

        logger.info("=== Warm-up phase ended ===")

        # Store the model state after warm-up
        post_warmup_state = deepcopy(model.state_dict())

        # Print brief warmup statistics
        with torch.no_grad():
            model.eval()
            # Get a small batch of data for quick statistics
            sample_loader = DataLoader(train_data, batch_size=min(32, len(train_data)), shuffle=True)
            x_sample, _ = next(iter(sample_loader))
            x_sample = x_sample.to(device)

            # Get pre-warmup stats
            model.load_state_dict(pre_warmup_state)
            pre_outputs = model(x_sample)
            pre_mean = pre_outputs.mean().item()
            pre_std = pre_outputs.std().item()

            # Get post-warmup stats
            model.load_state_dict(post_warmup_state)
            post_outputs = model(x_sample)
            post_mean = post_outputs.mean().item()
            post_std = post_outputs.std().item()

            # Calculate changes
            mean_change = post_mean - pre_mean
            mean_pct_change = (mean_change / (abs(pre_mean) + 1e-8)) * 100
            std_change = post_std - pre_std
            std_pct_change = (std_change / (abs(pre_std) + 1e-8)) * 100

            # Print summary
            logger.info("=== Warm-up Effect Summary ===")
            logger.info(f"Target voltage: {self.stability_target_voltage:.4f}")
            logger.info(f"Output mean: {pre_mean:.4f} → {post_mean:.4f} (Δ: {mean_change:.4f}, {mean_pct_change:+.1f}%)")
            logger.info(f"Output std: {pre_std:.4f} → {post_std:.4f} (Δ: {std_change:.4f}, {std_pct_change:+.1f}%)")

            # Calculate distance to target
            pre_dist = abs(pre_mean - self.stability_target_voltage)
            post_dist = abs(post_mean - self.stability_target_voltage)
            dist_change = pre_dist - post_dist
            logger.info(f"Distance to target: {pre_dist:.4f} → {post_dist:.4f} (Improved by: {dist_change:.4f})")
            logger.info("=============================")

            # Restore the state after warmup
            model.load_state_dict(post_warmup_state)

        # Main training phase
        logger.info("=== Starting main training phase ===")
        remaining_epochs = epochs - self.warmup_epochs
        start_main_phase = time.time()
        
        for epoch in range(1, remaining_epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss_val = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                loss_val.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += loss_val.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)["accuracy"]
            
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            train_losses.append(epoch_train_loss)
            valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_main_phase
                logger.info(
                    f"[VoltageStab Main] epoch={epoch}/{remaining_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f"main_epoch_{epoch}", training=True)
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": self.warmup_epochs + epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                })

        model.load_state_dict(best_state)
        results = {"train_losses": train_losses, "valid_losses": valid_losses}
        
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        
        return results 