#!/usr/bin/env python3
"""
This script generates placeholder images for the documentation.
Run this script before building the documentation to ensure all referenced images exist.
"""

import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir_exists(directory):
    """Ensure directory exists, create if it doesn't."""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory exists or created: {directory}")
        return True
    except Exception as e:
        print(f"❌ ERROR creating directory {directory}: {str(e)}")
        traceback.print_exc()
        return False


def create_placeholder_image(
    filename, width=800, height=600, title="Placeholder Image"
):
    """Create a placeholder image with random data."""
    try:
        print(f"Creating image: {filename}")
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Generate random data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)

        plt.plot(x, y)
        plt.title(title)
        plt.grid(True)

        # Add text explaining this is a placeholder
        plt.figtext(
            0.5,
            0.01,
            "This is a placeholder image for documentation",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
        )

        # Save the figure
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Verify file was created
        if os.path.exists(filename):
            print(f"✓ Created image: {filename} ({os.path.getsize(filename)} bytes)")
            return True
        else:
            print(
                f"❌ Failed to create image: {filename} (file doesn't exist after save)"
            )
            return False
    except Exception as e:
        print(f"❌ ERROR creating image {filename}: {str(e)}")
        traceback.print_exc()
        return False


def create_network_diagram(
    filename, width=800, height=600, title="Network Architecture"
):
    """Create a simple network diagram placeholder."""
    try:
        print(f"Creating network diagram: {filename}")
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Create a diagram with nodes and edges
        layer_sizes = [4, 8, 6, 3]
        layer_positions = [1, 2, 3, 4]

        # Draw nodes
        for i, (size, pos) in enumerate(zip(layer_sizes, layer_positions)):
            for j in range(size):
                y_pos = (j - size / 2 + 0.5) * 0.8
                plt.scatter(
                    pos, y_pos, s=500, color=f"C{i}", edgecolors="black", zorder=2
                )

        # Draw connections between layers
        for i in range(len(layer_sizes) - 1):
            for j in range(layer_sizes[i]):
                y1 = (j - layer_sizes[i] / 2 + 0.5) * 0.8
                for k in range(layer_sizes[i + 1]):
                    y2 = (k - layer_sizes[i + 1] / 2 + 0.5) * 0.8
                    plt.plot(
                        [layer_positions[i], layer_positions[i + 1]],
                        [y1, y2],
                        "k-",
                        alpha=0.1,
                        zorder=1,
                    )

        plt.title(title)
        plt.axis("off")

        # Add text explaining this is a placeholder
        plt.figtext(
            0.5,
            0.01,
            "This is a placeholder network diagram for documentation",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
        )

        # Save the figure
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Verify file was created
        if os.path.exists(filename):
            print(
                f"✓ Created network diagram: {filename} ({os.path.getsize(filename)} bytes)"
            )
            return True
        else:
            print(
                f"❌ Failed to create network diagram: {filename} (file doesn't exist after save)"
            )
            return False
    except Exception as e:
        print(f"❌ ERROR creating network diagram {filename}: {str(e)}")
        traceback.print_exc()
        return False


def create_dendritic_diagram(filename, width=800, height=600):
    """Create a simplified dendritic neuron diagram."""
    try:
        print(f"Creating dendritic diagram: {filename}")
        plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # Draw soma
        soma_x, soma_y = 5, 5
        soma_radius = 1
        soma = plt.Circle((soma_x, soma_y), soma_radius, color="gray", alpha=0.8)
        plt.gca().add_patch(soma)

        # Draw dendrites (as branches)
        branch_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for angle in branch_angles:
            # Primary branch
            end_x = soma_x + 3 * np.cos(angle)
            end_y = soma_y + 3 * np.sin(angle)
            plt.plot([soma_x, end_x], [soma_y, end_y], "k-", linewidth=2)

            # Secondary branches
            for sub_angle in [angle - 0.3, angle + 0.3]:
                sub_end_x = end_x + 2 * np.cos(sub_angle)
                sub_end_y = end_y + 2 * np.sin(sub_angle)
                plt.plot([end_x, sub_end_x], [end_y, sub_end_y], "k-", linewidth=1.5)

                # Synapses
                plt.scatter(sub_end_x, sub_end_y, color="red", s=50, zorder=3)

        # Draw axon
        axon_end_x = soma_x - 5
        axon_end_y = soma_y
        plt.plot(
            [soma_x - soma_radius, axon_end_x],
            [soma_y, axon_end_y],
            "b-",
            linewidth=2.5,
        )

        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.title("Simplified Dendritic Neuron")
        plt.axis("off")

        # Add text explaining this is a placeholder
        plt.figtext(
            0.5,
            0.01,
            "This is a simplified dendritic neuron diagram for documentation",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "lightblue", "alpha": 0.2, "pad": 5},
        )

        # Save the figure
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Verify file was created
        if os.path.exists(filename):
            print(
                f"✓ Created dendritic diagram: {filename} ({os.path.getsize(filename)} bytes)"
            )
            return True
        else:
            print(
                f"❌ Failed to create dendritic diagram: {filename} (file doesn't exist after save)"
            )
            return False
    except Exception as e:
        print(f"❌ ERROR creating dendritic diagram {filename}: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Generate all placeholder images for the documentation."""
    print("=" * 80)
    print(f"STARTING IMAGE GENERATION")
    print(f"Current working directory: {os.getcwd()}")
    print("=" * 80)

    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # Change to the script directory to ensure relative paths work correctly
    os.chdir(script_dir)
    print(f"Changed working directory to: {os.getcwd()}")

    # Define image directories relative to the script directory
    dirs = [
        "_static",
        "source/_static",
        "source/images",
        "source/_static/images",
    ]

    # Ensure directories exist
    for directory in dirs:
        ensure_dir_exists(directory)

    success_count = 0
    failure_count = 0

    # Create general placeholder
    if create_placeholder_image("_static/placeholder.png", title="Placeholder Image"):
        success_count += 1
    else:
        failure_count += 1

    if create_placeholder_image(
        "source/_static/placeholder.png", title="Placeholder Image"
    ):
        success_count += 1
    else:
        failure_count += 1

    # Create dendritic diagram
    if create_dendritic_diagram("_static/dendritic_diagram.png"):
        success_count += 1
    else:
        failure_count += 1

    if create_dendritic_diagram("source/_static/dendritic_diagram.png"):
        success_count += 1
    else:
        failure_count += 1

    # Create network architecture diagrams
    if create_network_diagram(
        "_static/einet_architecture.png", title="EINet Architecture"
    ):
        success_count += 1
    else:
        failure_count += 1

    if create_network_diagram(
        "source/_static/einet_architecture.png", title="EINet Architecture"
    ):
        success_count += 1
    else:
        failure_count += 1

    # List of specific images referenced in the documentation
    referenced_images = [
        "dendritic_diagram.png",
        "einet_architecture.png",
        "input_network_architecture.png",
        "dendritic_processing.png",
        "training_strategies.png",
        "sequential_vs_joint.png",
        "head_training_strategies.png",
        "input_modes_comparison.png",
        "learning_strategies.png",
        "dendritic_computation.png",
        "ablation_visualization.png",
        "computational_properties.png",
        "input_modes_complete.png",
        "installation_workflow.png",
    ]

    # Create each referenced image
    for img in referenced_images:
        # Skip images we've already created
        if img in ["dendritic_diagram.png", "einet_architecture.png"]:
            continue

        if create_placeholder_image(
            f"_static/{img}", title=img.replace(".png", "").replace("_", " ").title()
        ):
            success_count += 1
        else:
            failure_count += 1

        if create_placeholder_image(
            f"source/_static/{img}",
            title=img.replace(".png", "").replace("_", " ").title(),
        ):
            success_count += 1
        else:
            failure_count += 1

    print("=" * 80)
    print(f"IMAGE GENERATION SUMMARY:")
    print(f"Successfully created: {success_count} images")
    print(f"Failed to create: {failure_count} images")
    print("=" * 80)

    # List files in directories to verify
    for directory in dirs:
        print(f"\nContents of {directory}:")
        try:
            files = os.listdir(directory)
            for file in files:
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    print(f"  - {file} ({os.path.getsize(file_path)} bytes)")
                else:
                    print(f"  - {file}/ (directory)")
            if not files:
                print("  (empty directory)")
        except Exception as e:
            print(f"  ERROR listing directory: {str(e)}")

    return success_count, failure_count


if __name__ == "__main__":
    try:
        success, failure = main()
        if failure > 0:
            sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
