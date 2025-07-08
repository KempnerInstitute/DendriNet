"""Visual stimulus generators for testing dendritic neurons."""

from .gratings import *
from .noise import *
from .bars import *
from .textures import *

__all__ = [
    # gratings
    "create_drifting_grating",
    "create_static_grating",
    "create_phase_offset_gratings",
    # noise
    "create_white_noise",
    "create_colored_noise",
    "create_sparse_dots",
    # bars
    "create_moving_bar",
    "create_length_sweep_bars",
    # textures
    "create_portilla_simoncelli_texture",
    "create_contour_fragments",
] 