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
import contextlib
import importlib.util
import shutil

# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------


def set_default_style():
    """Apply a clean, publication-ready matplotlib style."""
    # Use seaborn if available, otherwise rely on built-in styles
    try:
        import seaborn as sns  # noqa: F401

        plt.style.use("seaborn-v0_8-whitegrid")
    except ModuleNotFoundError:
        # Fallback to a built-in style that is still pleasant
        plt.style.use("ggplot")

    # Global rcParams tweaks
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.autolayout": True,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )


@contextlib.contextmanager
def _figure_context(width, height):
    """Context manager creating and closing a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    yield fig, ax
    plt.close(fig)


def _save_figure_multiformat(fig, filename: str):
    """Save *fig* as PNG and SVG for crisp raster and vector outputs."""
    base, _ = os.path.splitext(filename)
    png_path = f"{base}.png"
    svg_path = f"{base}.svg"
    fig.savefig(png_path, transparent=False)
    fig.savefig(svg_path, transparent=True)
    return png_path, svg_path

# Apply style once at import time
set_default_style()


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
    filename: str,
    width: int = 800,
    height: int = 600,
    title: str = "Sample Output Curve",
):
    """Create a polished placeholder line chart."""

    try:
        print(f"Creating image: {filename}")

        with _figure_context(width, height) as (fig, ax):
            # Generate smooth data and confidence interval
            x = np.linspace(0, 10, 200)
            y = np.sin(x)
            noise = 0.15 * np.random.randn(len(x))
            y_obs = y + noise

            ax.plot(x, y, label="True signal", color="#2a9d8f", linewidth=2)
            ax.scatter(x, y_obs, label="Observations", s=10, color="#264653", alpha=0.6)

            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()

            _save_figure_multiformat(fig, filename)

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

        with _figure_context(width, height) as (fig, ax):
            layer_sizes = [4, 8, 6, 3]
            layer_x = np.arange(len(layer_sizes)) * 2  # horizontal spacing

            # Draw connections
            for i in range(len(layer_sizes) - 1):
                for j in range(layer_sizes[i]):
                    for k in range(layer_sizes[i + 1]):
                        y1 = (j + 0.5) / layer_sizes[i]
                        y2 = (k + 0.5) / layer_sizes[i + 1]
                        ax.plot(
                            [layer_x[i], layer_x[i + 1]],
                            [y1, y2],
                            color="#999999",
                            alpha=0.3,
                            linewidth=0.8,
                        )

            # Draw nodes on top of edges
            cmap = plt.get_cmap("viridis")
            for idx, (n_nodes, x) in enumerate(zip(layer_sizes, layer_x)):
                color = cmap(idx / len(layer_sizes))
                y_positions = (np.arange(n_nodes) + 0.5) / n_nodes
                ax.scatter(
                    np.full(n_nodes, x),
                    y_positions,
                    s=400,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=3,
                )

                # Annotate layer label
                ax.text(
                    x,
                    1.05,
                    f"Layer {idx + 1}\n({n_nodes})",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            ax.set_xlim(layer_x[0] - 1, layer_x[-1] + 1)
            ax.set_ylim(-0.05, 1.15)
            ax.axis("off")
            ax.set_title(title)

            _save_figure_multiformat(fig, filename)

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

        with _figure_context(width, height) as (fig, ax):
            soma_x, soma_y = 0, 0
            soma_radius = 0.4

            # Draw soma
            soma_circle = plt.Circle((soma_x, soma_y), soma_radius, color="#6d6875")
            ax.add_patch(soma_circle)

            # Generate branch angles and lengths
            rng = np.random.default_rng(seed=42)
            n_primary = 10
            primary_angles = np.linspace(0, 2 * np.pi, n_primary, endpoint=False)

            for angle in primary_angles:
                length1 = rng.uniform(2.0, 3.0)
                end1 = soma_x + length1 * np.cos(angle), soma_y + length1 * np.sin(angle)
                ax.plot([soma_x, end1[0]], [soma_y, end1[1]], color="#1d3557", linewidth=2)

                # Secondary branches
                for _ in range(rng.integers(2, 4)):
                    delta = rng.uniform(-0.5, 0.5)
                    sub_angle = angle + delta
                    length2 = rng.uniform(1.0, 1.8)
                    end2 = end1[0] + length2 * np.cos(sub_angle), end1[1] + length2 * np.sin(sub_angle)
                    ax.plot([end1[0], end2[0]], [end1[1], end2[1]], color="#457b9d", linewidth=1.2)

                    # Synapse dots
                    ax.scatter(*end2, s=25, color="#e63946", zorder=3)

            # Axon
            ax.plot([soma_x, soma_x - 4], [soma_y, soma_y - 0.5], color="#2a9d8f", linewidth=2.5)

            ax.set_aspect("equal")
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.axis("off")
            ax.set_title("Stylized Dendritic Neuron")

            _save_figure_multiformat(fig, filename)

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


# -----------------------------------------------------------------------------
# Optional: integrate advanced figure generators from scripts/
# -----------------------------------------------------------------------------


def _run_external_figure_generators():
    """Import and run the advanced figure generators if the script exists.

    The external script lives at ../../scripts/generate_documentation_figures.py
    (relative to this file). We dynamically load it, override its OUTPUT_DIR so
    it writes directly into the local ``_static`` directory, then invoke every
    callable whose name starts with ``create_``.
    """

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "generate_documentation_figures.py"

    if not script_path.exists():
        print("External figure generation script not found – skipping advanced diagrams.")
        return 0

    print(f"Found external figure generator: {script_path}")

    spec = importlib.util.spec_from_file_location("doc_ext_figures", script_path)
    if spec is None:
        print("Unable to create import spec – skipping.")
        return 0

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to import external figure generator: {exc}")
        traceback.print_exc()
        return 0

    # Redirect its OUTPUT_DIR attribute (if present) to our _static directory
    target_dir = Path(__file__).parent / "_static"
    if hasattr(module, "OUTPUT_DIR"):
        setattr(module, "OUTPUT_DIR", str(target_dir))
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Redirected external OUTPUT_DIR to: {target_dir}")

    created = 0
    for name in dir(module):
        if name.startswith("create_") and callable(getattr(module, name)):
            func = getattr(module, name)
            try:
                print(f"Running external generator: {name}()")
                func()
                created += 1
            except Exception as exc:  # pylint: disable=broad-except
                print(f"⚠️  External generator {name} failed: {exc}")
                traceback.print_exc()

    # Copy generated PNGs/SVGs (if any) into source/_static for Sphinx builds
    source_dir = target_dir
    sphinx_static_dir = Path(__file__).parent / "source" / "_static"
    sphinx_static_dir.mkdir(parents=True, exist_ok=True)

    for file_path in source_dir.glob("*.*"):
        dest = sphinx_static_dir / file_path.name
        try:
            shutil.copy2(file_path, dest)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Could not copy {file_path} to Sphinx static dir: {exc}")

    return created


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
        # new placeholders replacing legacy PDFs
        "einet_classifier_dataflow.png",
        "dendrinet_1.png",
        "dendrinet_2.png",
        "experiment_tracking.png",
        "configuration_system.png",
        "sweep_job_submission.png",
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

    # Run external advanced generators at the end so they can override simple placeholders
    print("\n" + "-" * 80)
    print("Running external advanced figure generators (if available)…")
    extra = _run_external_figure_generators()
    print(f"Generated {extra} advanced figures")

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
