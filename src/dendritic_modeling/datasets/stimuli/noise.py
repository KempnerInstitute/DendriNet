"""Noise stimulus generators for receptive field mapping."""

from typing import Union

import numpy as np

__all__ = [
    "create_colored_noise"
    "create_sparse_dots"
    "create_white_noise"
]

def create_white_noise(
    size: Union[int, tuple[int, int]],
    n_frames: int = 1,
    mean: float = 0.5,
    std: float = 0.2,
) -> np.ndarray:
    """Create white noise stimulus.

    Parameters
    ----------
    size : int or tuple
        Image size (height, width) or single int for square
    n_frames : int
        Number of frames
    mean : float
        Mean pixel value
    std : float
        Standard deviation

    Returns
    -------
    noise : np.ndarray
        Shape (n_frames, height, width) or (height, width) if n_frames=1
    """
    if isinstance(size, int):
        size = (size, size)

    shape = (n_frames, *size) if n_frames > 1 else size
    noise = np.random.randn(*shape) * std + mean

    # Clip to valid range
    noise = np.clip(noise, 0, 1)

    return noise

def create_colored_noise(
    size: Union[int, tuple[int, int]],
    n_frames: int = 1,
    spectral_slope: float = -1.0,
    mean: float = 0.5,
    std: float = 0.2,
) -> np.ndarray:
    """Create colored (1/f) noise.

    Parameters
    ----------
    size : int or tuple
        Image size
    n_frames : int
        Number of frames
    spectral_slope : float
        Power spectrum slope (e.g., -1 for pink noise)
    mean : float
        Mean pixel value
    std : float
        Standard deviation

    Returns
    -------
    noise : np.ndarray
        Colored noise images
    """
    if isinstance(size, int):
        size = (size, size)

    def generate_frame():
        # Generate white noise
        white = np.random.randn(*size)

        # FFT
        fft = np.fft.fft2(white)

        # Create frequency grid
        fx = np.fft.fftfreq(size[1])[:, np.newaxis]
        fy = np.fft.fftfreq(size[0])[np.newaxis, :]
        f = np.sqrt(fx**2 + fy**2)

        # Apply spectral shaping
        f[0, 0] = 1  # Avoid division by zero
        fft = fft * (f ** (spectral_slope / 2))

        # Inverse FFT
        colored = np.real(np.fft.ifft2(fft))

        # Normalize
        colored = (colored - colored.mean()) / colored.std()
        colored = colored * std + mean

        return np.clip(colored, 0, 1)

    if n_frames > 1:
        noise = np.array([generate_frame() for _ in range(n_frames)])
    else:
        noise = generate_frame()

    return noise

def create_sparse_dots(
    size: Union[int, tuple[int, int]],
    n_dots: int,
    dot_size: int = 3,
    n_frames: int = 1,
    polarity: str = "both",
) -> np.ndarray:
    """Create sparse dot stimulus.

    Parameters
    ----------
    size : int or tuple
        Image size
    n_dots : int
        Number of dots per frame
    dot_size : int
        Dot diameter in pixels
    n_frames : int
        Number of frames
    polarity : str
        'both', 'bright', or 'dark'

    Returns
    -------
    dots : np.ndarray
        Sparse dot images
    """
    if isinstance(size, int):
        size = (size, size)

    def generate_frame():
        img = np.zeros(size)

        # Random positions
        y_pos = np.random.randint(dot_size//2, size[0] - dot_size//2, n_dots)
        x_pos = np.random.randint(dot_size//2, size[1] - dot_size//2, n_dots)

        # Random polarities
        if polarity == "both":
            values = np.random.choice([-1, 1], n_dots)
        elif polarity == "bright":
            values = np.ones(n_dots)
        else:  # dark
            values = -np.ones(n_dots)

        # Draw dots
        for y, x, val in zip(y_pos, x_pos, values):
            yy, xx = np.ogrid[:size[0], :size[1]]
            mask = (yy - y)**2 + (xx - x)**2 <= (dot_size/2)**2
            img[mask] = val

        # Normalize to [0, 1]
        img = (img + 1) / 2

        return img

    if n_frames > 1:
        dots = np.array([generate_frame() for _ in range(n_frames)])
    else:
        dots = generate_frame()

    return dots
