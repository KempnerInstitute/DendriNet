"""Bar stimulus generators for motion and length tuning."""

from typing import Optional, Union

import numpy as np

__all__ = [
    "create_length_sweep_bars",
    "create_moving_bar",
]

def create_moving_bar(
    size: Union[int, tuple[int, int]],
    bar_width: int,
    bar_length: Optional[int] = None,
    direction: float = 0.0,
    speed: float = 1.0,
    n_frames: Optional[int] = None,
    contrast: float = 1.0,
) -> np.ndarray:
    """Create a moving bar stimulus.

    Parameters
    ----------
    size : int or tuple
        Image size
    bar_width : int
        Bar width in pixels
    bar_length : int or None
        Bar length (None for full extent)
    direction : float
        Motion direction in degrees (0 = rightward)
    speed : float
        Speed in pixels/frame
    n_frames : int or None
        Number of frames (None to cross full field)
    contrast : float
        Bar contrast

    Returns
    -------
    movie : np.ndarray
        Shape (n_frames, height, width)
    """
    if isinstance(size, int):
        size = (size, size)

    height, width = size

    if bar_length is None:
        bar_length = max(height, width)

    # Calculate trajectory
    theta = np.deg2rad(direction)
    dx = np.cos(theta) * speed
    dy = np.sin(theta) * speed

    # Starting position (outside frame)
    margin = max(bar_length, bar_width) + 10
    start_x = -margin if dx > 0 else width + margin
    start_y = -margin if dy > 0 else height + margin

    # Number of frames to cross field
    if n_frames is None:
        distance = np.sqrt((width + 2*margin)**2 + (height + 2*margin)**2)
        n_frames = int(distance / speed) + 1

    movie = np.zeros((n_frames, height, width))

    for t in range(n_frames):
        # Current bar center
        cx = start_x + dx * t
        cy = start_y + dy * t

        # Create bar mask
        img = np.zeros((height, width))

        # Bar orientation perpendicular to motion
        bar_theta = theta + np.pi/2

        # Create coordinates relative to bar center
        y, x = np.ogrid[:height, :width]
        x_rot = (x - cx) * np.cos(bar_theta) + (y - cy) * np.sin(bar_theta)
        y_rot = -(x - cx) * np.sin(bar_theta) + (y - cy) * np.cos(bar_theta)

        # Bar mask
        mask = (np.abs(x_rot) <= bar_width/2) & (np.abs(y_rot) <= bar_length/2)
        img[mask] = contrast

        movie[t] = img

    return movie

def create_length_sweep_bars(
    size: Union[int, tuple[int, int]],
    bar_width: int,
    lengths: Optional[list[int]] = None,
    orientation: float = 90.0,
    contrast: float = 1.0,
) -> np.ndarray:
    """Create bars of different lengths for length tuning.

    Parameters
    ----------
    size : int or tuple
        Image size
    bar_width : int
        Bar width in pixels
    lengths : list or None
        Bar lengths to test (None for automatic range)
    orientation : float
        Bar orientation in degrees (90 = vertical)
    contrast : float
        Bar contrast

    Returns
    -------
    bars : np.ndarray
        Shape (n_lengths, height, width)
    """
    if isinstance(size, int):
        size = (size, size)

    height, width = size

    if lengths is None:
        # Automatic range from small to full extent
        min_length = bar_width * 2
        max_length = min(height, width) - 10
        lengths = np.linspace(min_length, max_length, 15).astype(int)

    bars = np.zeros((len(lengths), height, width))

    # Bar center
    cx, cy = width // 2, height // 2

    # Bar orientation
    theta = np.deg2rad(orientation)

    for i, length in enumerate(lengths):
        img = np.zeros((height, width))

        # Create coordinates relative to bar center
        y, x = np.ogrid[:height, :width]
        x_rot = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
        y_rot = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)

        # Bar mask
        mask = (np.abs(x_rot) <= bar_width/2) & (np.abs(y_rot) <= length/2)
        img[mask] = contrast

        bars[i] = img

    return bars
