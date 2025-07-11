"""Grating stimulus generators for visual neuron testing."""

from typing import Optional

import numpy as np

__all__ = [
    "create_drifting_grating",
    "create_phase_offset_gratings",
    "create_static_grating",
]


def create_static_grating(
    size: int,
    orientation: float,
    spatial_frequency: float,
    phase: float = 0.0,
    contrast: float = 1.0,
    aperture: Optional[str] = "circular",
) -> np.ndarray:
    """Create a static sinusoidal grating.

    Parameters
    ----------
    size : int
        Image size (square)
    orientation : float
        Orientation in degrees (0 = vertical)
    spatial_frequency : float
        Cycles per image
    phase : float
        Phase offset in radians
    contrast : float
        Contrast (0-1)
    aperture : str or None
        'circular' or None

    Returns
    -------
    grating : np.ndarray
        2D grating image
    """
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)

    # Rotate coordinates
    theta = np.deg2rad(orientation)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)

    # Create grating
    wavelength = size / spatial_frequency
    grating = np.sin(2 * np.pi * X_rot / wavelength + phase)

    # Apply contrast
    grating = grating * contrast

    # Apply aperture
    if aperture == "circular":
        radius = size // 2 - 5
        mask = (X**2 + Y**2) <= radius**2
        grating = grating * mask

    return grating


def create_drifting_grating(
    size: int,
    orientation: float,
    spatial_frequency: float,
    temporal_frequency: float,
    duration: float,
    fps: float = 30.0,
    contrast: float = 1.0,
    aperture: Optional[str] = "circular",
) -> np.ndarray:
    """Create a drifting sinusoidal grating movie.

    Parameters
    ----------
    size : int
        Image size (square)
    orientation : float
        Orientation in degrees
    spatial_frequency : float
        Cycles per image
    temporal_frequency : float
        Cycles per second
    duration : float
        Duration in seconds
    fps : float
        Frames per second
    contrast : float
        Contrast (0-1)
    aperture : str or None
        'circular' or None

    Returns
    -------
    movie : np.ndarray
        3D array (frames, height, width)
    """
    n_frames = int(duration * fps)
    movie = np.zeros((n_frames, size, size))

    # Phase increment per frame
    phase_increment = 2 * np.pi * temporal_frequency / fps

    for t in range(n_frames):
        phase = t * phase_increment
        movie[t] = create_static_grating(
            size, orientation, spatial_frequency, phase, contrast, aperture
        )

    return movie


def create_phase_offset_gratings(
    size: int,
    orientation: float,
    spatial_frequency: float,
    phases: Optional[np.ndarray] = None,
    contrast: float = 1.0,
    aperture: Optional[str] = "circular",
) -> np.ndarray:
    """Create gratings at multiple phase offsets.

    Parameters
    ----------
    size : int
        Image size
    orientation : float
        Orientation in degrees
    spatial_frequency : float
        Cycles per image
    phases : np.ndarray or None
        Phase values in radians. If None, uses 8 evenly spaced phases
    contrast : float
        Contrast (0-1)
    aperture : str or None
        'circular' or None

    Returns
    -------
    gratings : np.ndarray
        3D array (n_phases, height, width)
    """
    if phases is None:
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    gratings = np.zeros((len(phases), size, size))

    for i, phase in enumerate(phases):
        gratings[i] = create_static_grating(
            size, orientation, spatial_frequency, phase, contrast, aperture
        )

    return gratings
