import json
import logging
import os
from math import log
from typing import Callable, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import MultiplicativeGainAnalysisParams
from dendritic_modeling.datasets.standard_datasets import PoissonGeneratorDataset
from dendritic_modeling.models import BaseModel, Classifier, Regressor
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


METRIC_MAPPING = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "categorical_loglikelihood": "Log Likelihood",
    "mse": "Mean Squared Error",
    "cosine_similarity": "Cosine Similarity",
}


class MultiplicativeGainAnalyzer(AbstractAnalyzer):
    def __init__(self, params: MultiplicativeGainAnalysisParams):
        super().__init__("MultiplicativeGainAnalyzer")

        self.uniform_gain = getattr(params, "uniform_gain", True)
        self.gain_min = getattr(params, "gain_min", 0.0)
        self.gain_max = getattr(params, "gain_max", 10.0)
        self.n_steps = getattr(params, "n_steps", 10)
        self.logspace = getattr(params, "logspace", False)

        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_loglikelihood = getattr(
            params, "categorical_loglikelihood", False
        )
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)

    def _apply_gain_to_inputs(
        self, dataset: torch.utils.data.Dataset, gain_factor: float
    ) -> torch.Tensor:
        if isinstance(dataset, PoissonGeneratorDataset):
            if self.uniform_gain:
                dataset._set_gain_factor(gain_factor)
            else:
                dataset._set_max_gain_factor(gain_factor)

            inputs = dataset[:][0]
            return inputs

        else:
            inputs = dataset[:][0]

            if not self.uniform_gain:
                gain_factor = (torch.rand_like(inputs) * (gain_factor - 1)) + 1
                # gain_factor = torch.exp(
                #     ((torch.rand_like(inputs) * 2) - 1
                #     ) * log(gain_factor)
                # )

            inputs = inputs * gain_factor
            return inputs

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
    ):
        if not isinstance(model, Classifier):
            self.accuracy = False
            self.auc = False
            self.categorical_loglikelihood = False

        if not isinstance(model, Regressor):
            self.mse = False
            self.cosine_similarity = False

        if self.logspace:
            log_gain_max = log(self.gain_max)

            if self.gain_min > 0:
                log_gain_min = log(self.gain_min)
            else:
                log_gain_min = -log_gain_max

            gain_factors = torch.linspace(
                log_gain_min, log_gain_max, self.n_steps
            ).exp()

        else:
            gain_factors = torch.linspace(self.gain_min, self.gain_max, self.n_steps)

        data_dict = {"gain_factors": gain_factors.tolist()}

        if self.accuracy:
            data_dict["accuracy"] = []
        if self.auc:
            data_dict["auc"] = []
        if self.categorical_loglikelihood:
            data_dict["categorical_loglikelihood"] = []
        if self.mse:
            data_dict["mse"] = []
        if self.cosine_similarity:
            data_dict["cosine_similarity"] = []

        model = model.to(device)
        model.eval()

        test_labels: torch.Tensor = test_dataset[:][1].to(device)

        def _evaluate_metric(
            metric_func: Callable, metric_key: str, dataset: torch.utils.data.Dataset
        ):
            metric_score = metric_func(model, test_ds=dataset, move_device=False)[-1]
            data_dict[metric_key].append(metric_score)

        for gain_factor in data_dict["gain_factors"]:
            scaled_test_inputs = self._apply_gain_to_inputs(test_dataset, gain_factor)
            scaled_test_inputs = scaled_test_inputs.to(device)
            scaled_test_dataset = TensorDataset(scaled_test_inputs, test_labels)

            if self.accuracy:
                _evaluate_metric(
                    metric_func=evaluate_accuracy,
                    metric_key="accuracy",
                    dataset=scaled_test_dataset,
                )
            if self.auc:
                _evaluate_metric(
                    metric_func=evaluate_auc,
                    metric_key="auc",
                    dataset=scaled_test_dataset,
                )
            if self.categorical_loglikelihood:
                _evaluate_metric(
                    metric_func=evaluate_categorical_loglikelihood,
                    metric_key="categorical_loglikelihood",
                    dataset=scaled_test_dataset,
                )
            if self.mse:
                _evaluate_metric(
                    metric_func=evaluate_mse,
                    metric_key="mse",
                    dataset=scaled_test_dataset,
                )
            if self.cosine_similarity:
                _evaluate_metric(
                    metric_func=evaluate_cosine_similarity,
                    metric_key="cosine_similarity",
                    dataset=scaled_test_dataset,
                )

        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")
            else:
                metrics = list(data_dict.keys())
                metrics.remove("gain_factors")

                for metric in metrics:
                    plot_performance_vs_gain_final(
                        data_dict=data_dict,
                        metric=metric,
                        logspace=self.logspace,
                        save_path=save_path,
                    )

                epochs_data, epoch_numbers = self.load_epochs_data(save_path)
                if epochs_data:
                    for metric in metrics:
                        plot_performance_vs_gain_training(
                            epochs_data=epochs_data,
                            epoch_numbers=epoch_numbers,
                            metric=metric,
                            logspace=self.logspace,
                            save_path=save_path,
                        )

            save_dict(data_dict, save_path, f"{filename}.json")

        return data_dict

    def _initialize_agg_structure(self, data: dict) -> dict:
        """
        Recursively initialize an aggregate dictionary structure with empty lists at leaf nodes.

        Args:
            data: The dictionary structure to replicate

        Returns:
            Dictionary with same structure but empty lists at leaf nodes
        """
        agg_structure = {}
        for key, value in data.items():
            if isinstance(value, dict):
                agg_structure[key] = self._initialize_agg_structure(value)
            else:
                # Leaf node - initialize as empty list
                if key == "gain_factors":
                    agg_structure[key] = data[key]
                else:
                    agg_structure[key] = []
        return agg_structure

    def _populate_agg_dict(self, agg_dict: dict, epoch_data: dict):
        """
        Recursively populate aggregate dictionary by appending leaf values from epoch_data.

        Args:
            agg_dict: The aggregate dictionary to populate (modified in place)
            epoch_data: The epoch data dictionary to extract values from
        """
        for key, value in epoch_data.items():
            if isinstance(value, dict):
                if key in agg_dict:
                    self._populate_agg_dict(agg_dict[key], value)
            else:
                # Leaf node - append value to list
                if key in agg_dict and key != "gain_factors":
                    agg_dict[key].append(value)

    def load_epochs_data(self, save_path: str):
        training_path = os.path.join(save_path, "epochs")

        if not os.path.exists(training_path):
            return None, None

        agg_dict_initialized = False
        agg_dict = {}

        # Collect all epoch files and sort by epoch number
        epoch_files = []
        for filename in os.listdir(training_path):
            # Check if file matches the "epoch#" naming convention
            if filename.startswith("epoch") and filename.endswith(".json"):
                # Extract the number part (everything between "epoch" and ".json")
                epoch_number_str = filename[
                    5:-5
                ]  # Remove "epoch" prefix and ".json" suffix
                try:
                    epoch_number = int(epoch_number_str)
                    epoch_files.append((epoch_number, filename))
                except ValueError:
                    # Skip files that don't have numeric epoch numbers
                    continue

        # Sort by epoch number to process in order
        epoch_files.sort(key=lambda x: x[0])

        # Iterate through epoch files in sorted order
        epoch_numbers = []
        for epoch_number, filename in epoch_files:
            epoch_numbers.append(epoch_number)
            # Load the JSON file
            file_path = os.path.join(training_path, filename)
            with open(file_path) as f:
                epoch_data = json.load(f)

            if not agg_dict_initialized:
                # Initialize aggregate dictionary with same structure, empty lists at leaves
                agg_dict = self._initialize_agg_structure(epoch_data)
                agg_dict_initialized = True

            # Populate aggregate dictionary by appending values from this epoch
            self._populate_agg_dict(agg_dict, epoch_data)

        return agg_dict, epoch_numbers


def plot_performance_vs_gain_final(
    data_dict: dict, metric: str, logspace: bool, save_path: str
):
    """
    Plot performance metric vs gain factor.

    Args:
        data_dict: Dictionary containing 'gain_factors' and metric values
        metric: Name of the metric to plot (e.g., 'accuracy', 'auc', 'mse', etc.)
        logspace: Whether to plot the gain factors on a log scale
        save_path: Optional path to save the figure
    """
    gain_factors = data_dict["gain_factors"]
    metric_values = data_dict[metric]

    plt.figure(figsize=(10, 6))
    plt.plot(gain_factors, metric_values, marker="o", linewidth=2, markersize=4)
    plt.xlabel("Gain Factor", fontsize=12)
    plt.ylabel(METRIC_MAPPING[metric], fontsize=12)
    plt.title(f"{METRIC_MAPPING[metric]} vs Gain Factor", fontsize=14)

    # Set logarithmic scale on x-axis if logspace is True
    if logspace:
        plt.xscale("log")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{metric}_vs_gain_final"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, f"{filename}.png"), dpi=300, bbox_inches="tight"
    )
    logger.info(f"Plot saved to {os.path.join(save_path, f'{filename}.png')}")

    plt.close()


def plot_performance_vs_gain_training(
    epochs_data: dict, epoch_numbers: list, metric: str, logspace: bool, save_path: str
):
    """
    Create an animated video showing performance metric vs gain factor across training epochs.

    Args:
        epochs_data: Dictionary containing 'gain_factors' and metric values for each epoch
        epoch_numbers: List of epoch numbers corresponding to the data
        metric: Name of the metric to plot (e.g., 'accuracy', 'auc', 'mse', etc.)
        logspace: Whether to plot the gain factors on a log scale
        save_path: Path to save the video
    """
    gain_factors = epochs_data["gain_factors"]
    metric_values_list = epochs_data[metric]

    # Calculate global y-axis limits across all epochs
    all_metric_values = [
        val for epoch_values in metric_values_list for val in epoch_values
    ]
    y_min = min(all_metric_values)
    y_max = max(all_metric_values)
    # Add small padding to y-axis limits
    y_range = y_max - y_min
    y_padding = y_range * 0.05 if y_range > 0 else 0.1
    y_min -= y_padding
    y_max += y_padding

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    def animate(frame):
        """Update the plot for each frame (epoch)."""
        ax.clear()

        epoch_num = epoch_numbers[frame]
        metric_values = metric_values_list[frame]

        # Plot with same style as plot_performance_vs_gain_final
        ax.plot(gain_factors, metric_values, marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Gain Factor", fontsize=12)
        ax.set_ylabel(METRIC_MAPPING[metric], fontsize=12)
        ax.set_title(
            f"{METRIC_MAPPING[metric]} vs Gain Factor (Epoch {epoch_num})", fontsize=14
        )

        # Set logarithmic scale on x-axis if logspace is True
        if logspace:
            ax.set_xscale("log")

        ax.set_ylim(y_min, y_max)  # Set consistent y-axis limits
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return []

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(epoch_numbers), interval=100, repeat=True, blit=True
    )

    # Save animation as video
    video_filename = f"{metric}_vs_gain_training"
    os.makedirs(save_path, exist_ok=True)

    ffmpeg_available = animation.writers.is_available("ffmpeg")
    if ffmpeg_available:
        try:
            writer = animation.FFMpegWriter(fps=10, bitrate=1800)
            video_path = os.path.join(save_path, f"{video_filename}.mp4")
            anim.save(video_path, writer=writer)
            logger.info(f"Video saved to {video_path}")
        except Exception as e:
            logger.warning(f"Error saving MP4 video: {e}")
            logger.info("Falling back to GIF format...")
            try:
                writer = animation.PillowWriter(fps=10)
                video_path = os.path.join(save_path, f"{video_filename}.gif")
                anim.save(video_path, writer=writer)
                logger.info(f"GIF saved to {video_path}")
            except Exception as e2:
                logger.error(f"Error saving GIF video: {e2}")
    else:
        logger.info("ffmpeg writer not available; saving GIF format...")
        try:
            writer = animation.PillowWriter(fps=10)
            video_path = os.path.join(save_path, f"{video_filename}.gif")
            anim.save(video_path, writer=writer)
            logger.info(f"GIF saved to {video_path}")
        except Exception as e2:
            logger.error(f"Error saving GIF video: {e2}")

    plt.close(fig)
