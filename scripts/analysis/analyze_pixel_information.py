import argparse
import multiprocessing
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.cluster import SpectralClustering
from sklearn.metrics import mutual_info_score

# torchvision.transforms might be used later for CIFAR grayscale if needed
# from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    print(
        "tqdm not found. Progress bar will not be available for parallel PID calculation."
    )
    print("Consider installing it: pip install tqdm")
    tqdm = None  # Fallback if tqdm is not installed

# Add external BROJA_2PID to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXTERNAL_BROJA_PATH = os.path.join(PROJECT_ROOT, "external")
if EXTERNAL_BROJA_PATH not in sys.path:
    sys.path.append(EXTERNAL_BROJA_PATH)

try:
    # Import the BROJA_2PID.py file (module) from the BROJA_2PID package
    from BROJA_2PID import BROJA_2PID as broja_module
except ImportError as e:
    print(
        f"Could not import BROJA_2PID.BROJA_2PID module from {EXTERNAL_BROJA_PATH}. Error: {e}"
    )
    print(
        "Ensure BROJA_2PID package is in external/BROJA_2PID, contains BROJA_2PID.py, and external/ is in sys.path."
    )
    print(
        "BROJA_2PID also requires the 'ecos' package. Install with 'pip install ecos'."
    )
    sys.exit(1)

# Ensure ecos is installed, as BROJA_2PID.py imports it directly
try:
    import ecos
except ImportError:
    print(
        "The 'ecos' package is required by BROJA_2PID. Please install it: pip install ecos"
    )
    sys.exit(1)

# Data loading utility from the dendritic-modeling codebase
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
try:
    from dendritic_modeling.datasets import get_unified_datasets

    # load_config is not strictly necessary if we load yaml directly as planned
    # from dendritic_modeling.utils import load_config
except ImportError as e:
    print(
        f"Could not import get_unified_datasets from dendritic_modeling (tried path: {SRC_PATH}). Error: {e}"
    )
    sys.exit(1)


# Top-level helper function for multiprocessing to avoid lambda pickling issues
def _starmap_worker_helper(task_args):
    """Helper to unpack arguments for the actual worker for imap_unordered"""
    return _calculate_pid_for_single_pair_worker(*task_args)


def load_data(dataset_name, config_file_path):
    """
    Loads dataset using the project's utilities.
    Ensures data is not flattened and not normalized for pixel-level analysis.
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    task_cfg = config.get("task", {})
    train_cfg_dict = config.get("train", {})

    task_cfg["dataset"] = dataset_name

    if "parameters" not in task_cfg:
        task_cfg["parameters"] = {}
    task_cfg["parameters"]["flatten"] = False
    task_cfg["parameters"]["normalize"] = False  # MNIST ToTensor output is [0,1]

    class ConfigNamespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            # Ensure seed is present, defaulting if not in config_file
            if "seed" not in self.__dict__:
                self.seed = 0
            if "train_valid_split" not in self.__dict__:
                self.train_valid_split = 1.0

    train_cfg = ConfigNamespace(**train_cfg_dict)

    print(f"Loading {dataset_name} with flatten=False, normalize=False...")
    train_ds, _, _ = get_unified_datasets(
        task_cfg, train_cfg
    )  # Use _ for valid_ds, test_ds

    if hasattr(train_ds, "tensors"):
        images, labels = train_ds.tensors[0], train_ds.tensors[1]
    else:
        images_list = []
        labels_list = []
        # Fallback for datasets not immediately providing .tensors (e.g. raw torchvision)
        # Limit to a reasonable number for quick analysis if it's a huge dataset without .tensors
        num_samples_to_load = len(train_ds)
        print(
            f"Dataset does not have .tensors attribute, iterating through {num_samples_to_load} samples..."
        )
        for i in range(num_samples_to_load):
            img, lbl = train_ds[i]
            images_list.append(img)
            labels_list.append(lbl)
            if i % 1000 == 0 and i > 0:
                print(f"  Loaded {i} samples...")
        images = torch.stack(images_list)
        labels = torch.tensor(labels_list, dtype=torch.long)

    print(f"Loaded {images.shape[0]} images of shape {images.shape[1:]}")
    return images.numpy(), labels.numpy()


def discretize_pixels(pixel_data, num_bins):
    """
    Discretizes pixel values into num_bins.
    Pixel data can be a single array of pixel values or an entire image array.
    """
    if num_bins <= 1:  # Must have at least 2 bins for meaningful discretization
        print("Warning: num_bins <= 1, defaulting to binarization (2 bins).")
        num_bins = 2

    min_val = np.min(pixel_data)
    max_val = np.max(pixel_data)

    if num_bins == 2:
        # Binarize around the midpoint of the actual range of this specific pixel_data
        threshold = min_val + (max_val - min_val) / 2.0
        # Handle case where min_val == max_val (e.g. all pixels are 0 or all are 1)
        if min_val == max_val:
            return (
                np.zeros_like(pixel_data, dtype=int)
                if threshold == 0
                else (pixel_data >= threshold).astype(int)
            )
        return (pixel_data > threshold).astype(int)
    else:
        # np.linspace creates edges like [min, ..., max] for num_bins+1 edges
        bins = np.linspace(min_val, max_val, num_bins + 1)

        # Ensure the max value is included in the last bin if it's exactly at the edge
        # Add a small epsilon to the max_val for the last bin edge to include the max_val itself if it sits on the edge.
        if bins[-1] == max_val:
            bins[-1] = max_val + 1e-6
        else:  # If linspace already made the last bin edge > max_val, ensure it covers max_val
            bins[-1] = np.nextafter(max_val, np.inf)

        discretized = np.digitize(pixel_data, bins[:-1], right=False) - 1
        # Clip to ensure values are within [0, num_bins-1]
        return np.clip(discretized, 0, num_bins - 1)


def calculate_single_pixel_mi(images, labels, num_bins_pixel, output_dir, dataset_name):
    N, C, H, W = images.shape
    if C > 1 and dataset_name == "mnist":  # MNIST should be grayscale
        print(f"Warning: MNIST data has {C} channels. Taking the first channel.")
        images_for_mi = images[:, 0:1, :, :]
    elif C > 1:  # For other datasets like CIFAR
        print(
            f"Warning: Dataset {dataset_name} has {C} channels. Converting to grayscale using average for MI."
        )
        images_for_mi = np.mean(images, axis=1, keepdims=True)
    else:
        images_for_mi = images

    mi_map = np.zeros((H, W))

    print("Calculating single pixel mutual information...")
    for r_idx in range(H):
        for c_idx in range(W):
            pixel_values_at_rc = images_for_mi[:, 0, r_idx, c_idx]
            discretized_pixel_values = discretize_pixels(
                pixel_values_at_rc, num_bins_pixel
            )
            mi = mutual_info_score(labels, discretized_pixel_values)
            mi_map[r_idx, c_idx] = mi
        if (r_idx + 1) % (
            H // 10 if H >= 10 else 1
        ) == 0 or r_idx == H - 1:  # Print progress periodically
            print(f"  Processed MI for row {r_idx+1}/{H}")

    plt.figure(figsize=(10, 8))
    plt.imshow(mi_map, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Mutual Information (bits)")
    title_str = f"Mutual Information: I(Pixel Value; Class Label) for {dataset_name.upper()}\nPixel Bins: {num_bins_pixel}"
    plt.title(title_str)
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir, f"{dataset_name}_single_pixel_mi_bins{num_bins_pixel}.png"
    )
    plt.savefig(plot_path)
    print(f"Saved single pixel MI map to {plot_path}")
    plt.close()
    return mi_map


def get_pixel_pair_iterator(H, W, mode, num_pairs_to_analyze):
    total_pixels = H * W
    if mode == "all_pairs":
        print(
            f"Preparing to iterate through all {total_pixels * (total_pixels -1) // 2} unique pixel pairs."
        )
        # Iterate through unique pairs (p1_flat_idx < p2_flat_idx)
        for p1_flat_idx in range(total_pixels):
            for p2_flat_idx in range(p1_flat_idx + 1, total_pixels):
                yield p1_flat_idx, p2_flat_idx  # Yield only flat indices
    elif mode == "random_sample":
        processed_coord_pairs_flat = set()
        for _ in range(num_pairs_to_analyze):
            attempts = 0
            while attempts < num_pairs_to_analyze * 3:  # Try to find a new unique pair
                p1_flat_idx = np.random.randint(total_pixels)
                p2_flat_idx = np.random.randint(total_pixels)
                if p1_flat_idx == p2_flat_idx:
                    attempts += 1
                    continue
                current_pair_flat = tuple(sorted((p1_flat_idx, p2_flat_idx)))
                if current_pair_flat not in processed_coord_pairs_flat:
                    processed_coord_pairs_flat.add(current_pair_flat)
                    yield p1_flat_idx, p2_flat_idx  # Yield only flat indices
                    break
                attempts += 1
            else:
                print(
                    f"  Warning: Could not find enough unique random pairs after many attempts. Found {len(processed_coord_pairs_flat)}."
                )
                return  # Stop yielding if we can't find more unique pairs


def _calculate_pid_for_single_pair_worker(
    p1_flat_idx,
    p2_flat_idx,
    images_for_pid_shape,
    images_for_pid_flat,
    labels,
    num_bins_pixel,
    num_classes,
    H,
    W,
):
    current_images_for_pid = images_for_pid_flat.reshape(images_for_pid_shape)
    N_samples = current_images_for_pid.shape[0]
    r1, c1 = np.unravel_index(p1_flat_idx, (H, W))
    r2, c2 = np.unravel_index(p2_flat_idx, (H, W))
    pixel1_values = current_images_for_pid[:, 0, r1, c1]
    pixel2_values = current_images_for_pid[:, 0, r2, c2]
    discrete_p1 = discretize_pixels(pixel1_values, num_bins_pixel)
    discrete_p2 = discretize_pixels(pixel2_values, num_bins_pixel)
    joint_pdf_counts = np.zeros((num_classes, num_bins_pixel, num_bins_pixel))
    for sample_idx in range(N_samples):
        l = int(labels[sample_idx])
        v1 = int(discrete_p1[sample_idx])
        v2 = int(discrete_p2[sample_idx])
        joint_pdf_counts[l, v1, v2] += 1
    joint_pdf_probs = joint_pdf_counts / N_samples
    pdf_dict = {}
    for l_idx in range(num_classes):
        for v1_idx in range(num_bins_pixel):
            for v2_idx in range(num_bins_pixel):
                prob = joint_pdf_probs[l_idx, v1_idx, v2_idx]
                if prob > 1e-12:
                    pdf_dict[(l_idx, v1_idx, v2_idx)] = float(prob)

    pdf_sum = sum(pdf_dict.values())
    if not pdf_dict or abs(pdf_sum - 1.0) > 1e-6:
        # This print might be too verbose if it happens for every pair in parallel mode.
        # Consider adding a flag or sampling if this becomes an issue.
        # For now, let's print for the first few failures in a run or if num_pairs is small.
        # However, for debugging this specific issue, printing it is useful.
        print(
            f"  [Worker p{p1_flat_idx},p{p2_flat_idx}] Invalid PDF: Empty or sum is {pdf_sum:.6f}. pdf_dict sample (first 5): {list(pdf_dict.items())[:5]}"
        )
        return p1_flat_idx, p2_flat_idx, np.nan, np.nan, np.nan, np.nan

    try:
        pid_result_dict = broja_module.pid(
            pdf_dict, output=0
        )  # output=0 to suppress BROJA's own prints
        # Check if BROJA indicates an error (e.g. from its Num_err field if it has one, or if results are suspect)
        # For now, we assume if it doesn't raise an exception, the values are usable.
        # More sophisticated error checking based on BROJA's output could be added.
        return (
            p1_flat_idx,
            p2_flat_idx,
            pid_result_dict["SI"],
            pid_result_dict["UIY"],
            pid_result_dict["UIZ"],
            pid_result_dict["CI"],
        )
    except Exception as e:
        print(
            f"  [Worker p{p1_flat_idx},p{p2_flat_idx}] Error during broja_module.pid call: {e}"
        )
        print(
            f"    PDF sum: {pdf_sum:.6f}, PDF dict (first 5 items): {list(pdf_dict.items())[:5]}"
        )
        # It might be useful to see the full traceback from the worker in some cases
        # import traceback
        # traceback.print_exc() # This might be too much for parallel execution stdout
        return p1_flat_idx, p2_flat_idx, np.nan, np.nan, np.nan, np.nan


def calculate_pairwise_pixel_pid(
    images,
    labels,
    num_bins_pixel,
    num_pairs_to_analyze,
    pid_analysis_mode,
    num_workers,
    num_redundancy_clusters,
    output_dir,
    dataset_name,
):
    N_samples, C, H, W = images.shape
    if C > 1 and dataset_name == "mnist":
        print(
            f"Warning: MNIST data has {C} channels for PID. Taking the first channel."
        )
        images_for_pid = images[:, 0:1, :, :]
    elif C > 1:
        print(
            f"Warning: Dataset {dataset_name} has {C} channels for PID. Converting to grayscale using average."
        )
        images_for_pid = np.mean(images, axis=1, keepdims=True)
    else:
        images_for_pid = images

    num_classes = len(np.unique(labels))
    pid_results_list = []  # For random_sample text output
    total_pixels = H * W

    if total_pixels < 2:
        print("Image too small for pairwise analysis (need at least 2 pixels).")
        return

    # Initialize matrices for 'all_pairs' mode or for sparse visualization in 'random_sample'
    synergy_matrix = np.full((total_pixels, total_pixels), np.nan)
    redundancy_matrix = np.full((total_pixels, total_pixels), np.nan)
    unique_info_matrix_p1_given_p2 = np.full(
        (total_pixels, total_pixels), np.nan
    )  # UI(P1 | P2)
    unique_info_matrix_p2_given_p1 = np.full(
        (total_pixels, total_pixels), np.nan
    )  # UI(P2 | P1)

    print(
        f"\nCalculating pairwise pixel PID (mode: {pid_analysis_mode}, workers: {num_workers}) for {dataset_name.upper()}..."
    )
    if pid_analysis_mode == "all_pairs":
        actual_num_pairs = total_pixels * (total_pixels - 1) // 2
        print(
            f"WARNING: 'all_pairs' mode selected. This will compute PID for {actual_num_pairs} pairs."
        )
        print("This can be VERY time-consuming and memory-intensive.")
    else:  # random_sample
        actual_num_pairs = num_pairs_to_analyze

    pair_coords_iterator_gen = get_pixel_pair_iterator(
        H, W, pid_analysis_mode, num_pairs_to_analyze
    )
    pair_coords_list = list(pair_coords_iterator_gen)
    actual_num_pairs_to_process = len(pair_coords_list)

    if actual_num_pairs_to_process == 0 and pid_analysis_mode == "random_sample":
        print(
            "No pairs selected for random_sample mode (num_pixel_pairs might be 0 or too few unique pairs found)."
        )
        # Proceed to save empty/NaN matrices and attempt clustering if requested, though it won't be meaningful.
    elif actual_num_pairs_to_process == 0:
        print("No pixel pairs to process.")
        return

    if pid_analysis_mode == "all_pairs" and actual_num_pairs_to_process > 0:
        print(
            f"Processing {actual_num_pairs_to_process} pairs using {num_workers} workers."
        )
        images_for_pid_shape = images_for_pid.shape
        images_for_pid_flat = images_for_pid.ravel()
        tasks = [
            (
                p1_loop_idx,
                p2_loop_idx,
                images_for_pid_shape,
                images_for_pid_flat,
                labels,
                num_bins_pixel,
                num_classes,
                H,
                W,
            )
            for p1_loop_idx, p2_loop_idx in pair_coords_list
        ]
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator_gen = pool.imap_unordered(_starmap_worker_helper, tasks)

            results_iterable_for_loop = results_iterator_gen
            if tqdm:
                results_iterable_for_loop = tqdm(
                    results_iterator_gen,
                    total=actual_num_pairs_to_process,
                    desc="Processing PID for pairs",
                )

            successful_calculations = 0  # Counter for successful PID computations
            for p1_idx, p2_idx, si, ui_p1, ui_p2, ci in results_iterable_for_loop:
                if not np.isnan(si):  # Check if calculation was successful
                    successful_calculations += 1
                    synergy_matrix[p1_idx, p2_idx] = synergy_matrix[p2_idx, p1_idx] = si
                    redundancy_matrix[p1_idx, p2_idx] = redundancy_matrix[
                        p2_idx, p1_idx
                    ] = ci
                    unique_info_matrix_p1_given_p2[p1_idx, p2_idx] = ui_p1
                    unique_info_matrix_p2_given_p1[p1_idx, p2_idx] = ui_p2
                    # Also fill the symmetric part for the other unique info perspective
                    unique_info_matrix_p1_given_p2[p2_idx, p1_idx] = ui_p2  # UI(P2|P1)
                    unique_info_matrix_p2_given_p1[p2_idx, p1_idx] = ui_p1  # UI(P1|P2)

                    if pid_analysis_mode == "random_sample":
                        r1, c1 = np.unravel_index(p1_idx, (H, W))
                        r2, c2 = np.unravel_index(p2_idx, (H, W))
                        pid_results_list.append(
                            {
                                "pair_coords_p1": (r1, c1),
                                "pair_coords_p2": (r2, c2),
                                "SI": si,
                                "UI_p1": ui_p1,
                                "UI_p2": ui_p2,
                                "CI": ci,
                            }
                        )

    else:  # Sequential for random_sample or if all_pairs had 0 pairs (edge case)
        print(f"Processing {actual_num_pairs} pairs sequentially...")
        iterator_to_use = pair_coords_list
        if tqdm and actual_num_pairs > 0:
            iterator_to_use = tqdm(
                pair_coords_list,
                total=actual_num_pairs,
                desc="Processing PID for pairs (sequential)",
            )

        successful_calculations = 0  # Counter for successful PID computations
        for p1_idx, p2_idx in iterator_to_use:
            _, _, si, ui_p1, ui_p2, ci = _calculate_pid_for_single_pair_worker(
                p1_idx,
                p2_idx,
                images_for_pid.shape,
                images_for_pid.ravel(),
                labels,
                num_bins_pixel,
                num_classes,
                H,
                W,
            )
            if not np.isnan(si):
                successful_calculations += 1
                synergy_matrix[p1_idx, p2_idx] = synergy_matrix[p2_idx, p1_idx] = si
                redundancy_matrix[p1_idx, p2_idx] = redundancy_matrix[
                    p2_idx, p1_idx
                ] = ci
                unique_info_matrix_p1_given_p2[p1_idx, p2_idx] = ui_p1
                unique_info_matrix_p2_given_p1[p1_idx, p2_idx] = ui_p2
                unique_info_matrix_p1_given_p2[p2_idx, p1_idx] = ui_p2
                unique_info_matrix_p2_given_p1[p2_idx, p1_idx] = ui_p1

                if pid_analysis_mode == "random_sample":
                    r1, c1 = np.unravel_index(p1_idx, (H, W))
                    r2, c2 = np.unravel_index(p2_idx, (H, W))
                    pid_results_list.append(
                        {
                            "pair_coords_p1": (r1, c1),
                            "pair_coords_p2": (r2, c2),
                            "SI": si,
                            "UI_p1": ui_p1,
                            "UI_p2": ui_p2,
                            "CI": ci,
                        }
                    )

    print(
        f"Finished PID calculations. Attempted: {actual_num_pairs_to_process}, Successfully calculated: {successful_calculations}"
    )

    # Save text results for random_sample mode
    if pid_analysis_mode == "random_sample" and pid_results_list:
        results_path = os.path.join(
            output_dir,
            f"{dataset_name}_pairwise_pid_bins{num_bins_pixel}_pairs{len(pid_results_list)}.txt",
        )
        with open(results_path, "w") as f:
            f.write(
                "P1_Coord; P2_Coord; Synergy(SI); Unique_P1(UIY); Unique_P2(UIZ); Redundancy(CI)\n"
            )
            for res in pid_results_list:
                f.write(
                    f"({res['pair_coords_p1'][0]},{res['pair_coords_p1'][1]}); ({res['pair_coords_p2'][0]},{res['pair_coords_p2'][1]}); "
                    f"{res['SI']:.6f}; {res['UI_p1']:.6f}; {res['UI_p2']:.6f}; {res['CI']:.6f}\n"
                )
        print(
            f"Saved {len(pid_results_list)} pairwise PID text results to {results_path}"
        )
    elif pid_analysis_mode == "random_sample" and not pid_results_list:
        print("No PID results were generated for random_sample mode.")

    # Save and plot matrices
    pid_matrices = {
        "synergy": synergy_matrix,
        "redundancy": redundancy_matrix,
        "unique_info_p1_given_p2": unique_info_matrix_p1_given_p2,  # UI(P1 | P2)
        "unique_info_p2_given_p1": unique_info_matrix_p2_given_p1,  # UI(P2 | P1)
    }

    for name, matrix_data in pid_matrices.items():
        if pid_analysis_mode == "all_pairs" and not np.all(np.isnan(matrix_data)):
            npy_path = os.path.join(
                output_dir, f"{dataset_name}_pid_matrix_{name}_bins{num_bins_pixel}.npy"
            )
            np.save(npy_path, matrix_data)
            print(f"Saved PID matrix ({name}) to {npy_path}")

        plt.figure(figsize=(12, 10))
        valid_values = matrix_data[~np.isnan(matrix_data)]
        num_valid_values = len(valid_values)

        print(
            f"  Plotting heatmap for '{name}': {num_valid_values} valid data points."
        )  # Debug print
        if num_valid_values > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            print(f"    Min valid: {min_val}, Max valid: {max_val}")  # Debug print
            if min_val == max_val:
                # If all valid values are the same, create a small range around it for plotting
                # unless it's zero, then plot from 0 to a small positive.
                if min_val == 0:
                    vmin, vmax = 0, 0.1
                else:
                    epsilon = abs(min_val) * 0.1 if min_val != 0 else 0.1
                    vmin = min_val - epsilon
                    vmax = max_val + epsilon
            else:
                vmin, vmax = min_val, max_val
        else:
            print(
                f"    Note: PID matrix for '{name}' is all NaNs or empty. Plot will use default range [0,1]."
            )
            vmin, vmax = 0, 1  # Default for empty plot

        print(f"    Using vmin: {vmin}, vmax: {vmax} for heatmap.")  # Debug print

        plt.imshow(
            matrix_data, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        plt.colorbar(label=f'{name.replace("_", " ").title()} (bits)')
        title_str = f'Pairwise PID: {name.replace("_", " ").title()} between Pixels for {dataset_name.upper()}\nPixel Bins: {num_bins_pixel}, Mode: {pid_analysis_mode}'
        if pid_analysis_mode == "random_sample":
            title_str += f" (Sampled {num_pairs_to_analyze} pairs)"
        plt.title(title_str)
        plt.xlabel("Pixel Index (Flat)")
        plt.ylabel("Pixel Index (Flat)")
        plot_path = os.path.join(
            output_dir,
            f"{dataset_name}_pid_heatmap_{name}_bins{num_bins_pixel}_mode_{pid_analysis_mode}.png",
        )
        plt.savefig(plot_path)
        print(f"Saved PID heatmap ({name}) to {plot_path}")
        plt.close()

    # Perform redundancy clustering if requested and matrix is available
    if num_redundancy_clusters > 0:
        if np.all(np.isnan(redundancy_matrix)) and pid_analysis_mode == "random_sample":
            print(
                "\nSkipping redundancy clustering: redundancy matrix from 'random_sample' mode is empty/all NaNs."
            )
            print("  Clustering is most meaningful with 'all_pairs' mode.")
        elif np.all(np.isnan(redundancy_matrix)):
            print(
                "\nSkipping redundancy clustering: redundancy matrix is all NaNs (PID calculations might have failed)."
            )
        else:
            cluster_and_plot_pixel_redundancy(
                redundancy_matrix,
                H,
                W,
                num_redundancy_clusters,
                output_dir,
                dataset_name,
                num_bins_pixel,
                pid_analysis_mode,
            )

    return (
        pid_matrices if pid_analysis_mode == "all_pairs" else None
    )  # Return matrices for 'all_pairs'


def cluster_and_plot_pixel_redundancy(
    redundancy_matrix,
    H,
    W,
    num_clusters,
    output_dir,
    dataset_name,
    num_bins_pixel,
    pid_mode,
):
    print(
        f"\nClustering pixels based on pairwise redundancy into {num_clusters} clusters..."
    )
    if np.all(np.isnan(redundancy_matrix)):
        print("  Redundancy matrix is all NaNs. Skipping clustering.")
        return

    # Preprocessing for SpectralClustering
    # Higher redundancy = higher affinity. Replace NaNs with 0 (no affinity).
    affinity_matrix = np.nan_to_num(redundancy_matrix, nan=0.0)
    # Ensure diagonal is strong for self-affinity
    diag_val = np.nanmax(
        affinity_matrix[~np.isinf(affinity_matrix)]
    )  # Max non-inf, non-nan value
    if (
        np.isnan(diag_val) or np.isinf(diag_val) or diag_val == 0
    ):  # Handle case where matrix was all NaN or all zero
        diag_val = 1.0
    np.fill_diagonal(affinity_matrix, diag_val)

    # Ensure matrix is symmetric (it should be, but good practice after manipulations)
    affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2

    try:
        print("  Running Spectral Clustering...")
        sc_model = SpectralClustering(
            n_clusters=num_clusters,
            assign_labels="kmeans",
            affinity="precomputed",
            random_state=42,
        )  # For reproducibility
        cluster_labels_flat = sc_model.fit_predict(affinity_matrix)
    except Exception as e:
        print(f"  Error during Spectral Clustering: {e}")
        print("  Skipping redundancy clustering plot.")
        return

    cluster_map_2d = cluster_labels_flat.reshape((H, W))

    plt.figure(figsize=(10, 8))
    # Choose a colormap suitable for discrete categories
    if num_clusters <= 10:
        cmap = plt.cm.get_cmap("tab10", num_clusters)
    elif num_clusters <= 20:
        cmap = plt.cm.get_cmap("tab20", num_clusters)
    else:
        cmap = plt.cm.get_cmap("viridis", num_clusters)  # Fallback for many clusters

    plt.imshow(cluster_map_2d, cmap=cmap, interpolation="nearest")
    # Create a colorbar with integer ticks for cluster labels
    formatter = plt.FuncFormatter(lambda val, loc: int(val))
    plt.colorbar(ticks=np.arange(num_clusters), format=formatter, label="Cluster ID")

    title_str = f"Pixel Clusters by Redundancy ({num_clusters} clusters) for {dataset_name.upper()}\nPixel Bins: {num_bins_pixel}, Mode: {pid_mode}"
    plt.title(title_str)
    plt.xlabel("Pixel Column")
    plt.ylabel("Pixel Row")

    plot_filename = f"{dataset_name}_redundancy_clusters_k{num_clusters}_bins{num_bins_pixel}_mode_{pid_mode}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    print(f"Saved redundancy cluster map to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pixel-wise and pairwise pixel information for image datasets."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Name of the dataset to analyze (default: mnist).",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to a YAML experiment configuration file (e.g., .vscode/config_exp.yaml).",
    )
    parser.add_argument(
        "--num_bins_pixel",
        type=int,
        default=2,
        help="Number of bins for pixel value discretization (default: 2 for binarization).",
    )
    parser.add_argument(
        "--pid_analysis_mode",
        type=str,
        default="random_sample",
        choices=["random_sample", "all_pairs"],
        help="Mode for pairwise PID analysis: 'random_sample' (default) or 'all_pairs' ( computationally intensive).",
    )
    parser.add_argument(
        "--num_pixel_pairs",
        type=int,
        default=100,
        help="Number of random pixel pairs for PID (if mode is 'random_sample', default: 100).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, os.cpu_count() // 2),
        help="Number of worker processes for parallel PID calculation in 'all_pairs' mode (default: half of CPU cores).",
    )
    parser.add_argument(
        "--num_redundancy_clusters",
        type=int,
        default=0,
        help="Number of clusters for pixel redundancy clustering (0 to skip, default: 0). Best with 'all_pairs' mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pixel_analysis_results",
        help="Directory to save plots and results (default: ./pixel_analysis_results).",
    )

    args = parser.parse_args()

    print(f"Starting pixel information analysis with arguments: {args}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.pid_analysis_mode == "all_pairs":
        print("\nWARNING: 'all_pairs' mode for PID analysis is selected.")
        print(
            "This can be EXTREMELY time-consuming. For a 28x28 image, this is approx. 300,000 pairs."
        )
        if args.num_redundancy_clusters > 0:
            print(
                "Pixel redundancy clustering will also be performed after PID calculations."
            )
        user_confirmation = input("Do you want to continue? (yes/no): ")
        if user_confirmation.lower() != "yes":
            print("Aborting analysis.")
            return

    try:
        images, labels = load_data(args.dataset_name, args.config_file)
    except Exception as e:
        print(f"Critical error loading data: {e}")
        return

    try:
        calculate_single_pixel_mi(
            images, labels, args.num_bins_pixel, args.output_dir, args.dataset_name
        )
    except Exception as e:
        print(f"Error in single pixel MI calculation: {e}")
        traceback.print_exc()

    if args.pid_analysis_mode == "random_sample" and args.num_pixel_pairs <= 0:
        print(
            "Skipping pairwise pixel PID analysis as num_pixel_pairs is 0 for random_sample mode."
        )
        # Still attempt clustering if requested, though it will be on an empty/NaN matrix
        if args.num_redundancy_clusters > 0:
            print(
                "Attempting redundancy clustering on potentially empty/sparse matrix due to num_pixel_pairs=0."
            )
            H, W = (
                (images.shape[2], images.shape[3]) if len(images.shape) == 4 else (0, 0)
            )  # Get H,W for empty matrix shape
            if H > 0 and W > 0:
                dummy_redundancy_matrix = np.full((H * W, H * W), np.nan)
                cluster_and_plot_pixel_redundancy(
                    dummy_redundancy_matrix,
                    H,
                    W,
                    args.num_redundancy_clusters,
                    args.output_dir,
                    args.dataset_name,
                    args.num_bins_pixel,
                    args.pid_analysis_mode,
                )
            else:
                print(
                    "Could not determine image dimensions for dummy matrix for clustering."
                )

    else:
        try:
            calculate_pairwise_pixel_pid(
                images,
                labels,
                args.num_bins_pixel,
                args.num_pixel_pairs,
                args.pid_analysis_mode,
                args.num_workers,
                args.num_redundancy_clusters,
                args.output_dir,
                args.dataset_name,
            )
        except Exception as e:
            print(f"Error in pairwise pixel PID calculation: {e}")
            traceback.print_exc()

    print("\nPixel information analysis finished.")


if __name__ == "__main__":
    # Guard for multiprocessing on Windows/other platforms that use spawn
    multiprocessing.freeze_support()
    main()
