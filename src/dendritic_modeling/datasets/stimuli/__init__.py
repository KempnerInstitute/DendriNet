"""Visual stimulus generators for testing dendritic neurons."""

# Import specific functions to avoid star imports
from .bars import create_length_sweep_bars, create_moving_bar
from .gratings import create_drifting_grating, create_phase_offset_gratings, create_static_grating
from .noise import create_colored_noise, create_sparse_dots, create_white_noise
from .textures import create_contour_fragments, create_portilla_simoncelli_texture

__all__ = [
    # noise
    "create_colored_noise",
    "create_sparse_dots", 
    "create_white_noise",
    # bars
    "create_length_sweep_bars",
    "create_moving_bar",
    # gratings
    "create_drifting_grating",
    "create_phase_offset_gratings",
    "create_static_grating",
    # textures
    "create_contour_fragments",
    "create_portilla_simoncelli_texture",
]
