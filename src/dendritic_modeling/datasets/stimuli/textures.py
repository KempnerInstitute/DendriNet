"""Texture stimulus generators for higher-order visual processing."""

import numpy as np
from scipy import signal, ndimage
from typing import Optional, Tuple, Union

__all__ = [
    "create_portilla_simoncelli_texture",
    "create_contour_fragments",
]


def create_portilla_simoncelli_texture(
    size: Union[int, Tuple[int, int]],
    n_scales: int = 4,
    n_orientations: int = 4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Create synthetic texture using simplified Portilla-Simoncelli statistics.
    
    This is a simplified version that captures key texture properties
    without full PS synthesis.
    
    Parameters
    ----------
    size : int or tuple
        Image size
    n_scales : int
        Number of spatial scales
    n_orientations : int
        Number of orientations
    seed : int or None
        Random seed
        
    Returns
    -------
    texture : np.ndarray
        Synthetic texture image
    """
    if isinstance(size, int):
        size = (size, size)
        
    if seed is not None:
        np.random.seed(seed)
        
    # Start with filtered noise
    noise = np.random.randn(*size)
    
    # Build steerable pyramid filters (simplified)
    texture = np.zeros(size)
    
    for scale in range(n_scales):
        scale_factor = 2 ** scale
        
        for ori_idx in range(n_orientations):
            # Create oriented filter
            theta = ori_idx * np.pi / n_orientations
            
            # Gaussian derivative filter (simplified steerable filter)
            sigma = 1.5 * scale_factor
            
            # Create coordinate grid
            ks = int(6 * sigma) | 1  # Ensure odd
            x = np.arange(ks) - ks // 2
            y = np.arange(ks) - ks // 2
            X, Y = np.meshgrid(x, y)
            
            # Rotate coordinates
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
            
            # Oriented Gaussian derivative
            gaussian = np.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
            derivative = X_rot * gaussian / sigma**2
            
            # Normalize
            derivative = derivative - derivative.mean()
            derivative = derivative / (np.abs(derivative).sum() + 1e-10)
            
            # Filter noise
            filtered = signal.convolve2d(noise, derivative, mode='same', boundary='symm')
            
            # Add to texture with scale-dependent weight
            texture += filtered / (scale_factor + 1)
            
    # Apply nonlinearity to increase texture-like appearance
    texture = np.tanh(texture / texture.std())
    
    # Normalize to [0, 1]
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-10)
    
    return texture


def create_contour_fragments(
    size: Union[int, Tuple[int, int]],
    n_fragments: int = 5,
    fragment_length: Tuple[int, int] = (10, 30),
    curvature_range: Tuple[float, float] = (0.0, 0.1),
    width: int = 3,
    contrast: float = 1.0,
) -> np.ndarray:
    """Create image with curved contour fragments.
    
    Parameters
    ----------
    size : int or tuple
        Image size
    n_fragments : int
        Number of contour fragments
    fragment_length : tuple
        Min and max fragment length in pixels
    curvature_range : tuple
        Min and max curvature (1/radius)
    width : int
        Contour width in pixels
    contrast : float
        Contour contrast
        
    Returns
    -------
    image : np.ndarray
        Image with contour fragments
    """
    if isinstance(size, int):
        size = (size, size)
        
    height, width_img = size
    image = np.zeros(size)
    
    for _ in range(n_fragments):
        # Random fragment parameters
        length = np.random.randint(fragment_length[0], fragment_length[1])
        curvature = np.random.uniform(curvature_range[0], curvature_range[1])
        
        # Starting position and orientation
        start_x = np.random.randint(width, width_img - width)
        start_y = np.random.randint(width, height - width)
        start_angle = np.random.uniform(0, 2 * np.pi)
        
        # Generate curved path
        t = np.linspace(0, length, length)
        
        if curvature > 0:
            # Circular arc
            radius = 1 / curvature
            angles = start_angle + t / radius
            x_path = start_x + radius * (np.cos(angles) - np.cos(start_angle))
            y_path = start_y + radius * (np.sin(angles) - np.sin(start_angle))
        else:
            # Straight line
            x_path = start_x + t * np.cos(start_angle)
            y_path = start_y + t * np.sin(start_angle)
            
        # Draw contour
        for i in range(len(t) - 1):
            x0, y0 = int(x_path[i]), int(y_path[i])
            x1, y1 = int(x_path[i+1]), int(y_path[i+1])
            
            # Check bounds
            if (0 <= x0 < width_img and 0 <= y0 < height and
                0 <= x1 < width_img and 0 <= y1 < height):
                
                # Draw thick line segment using distance transform
                yy, xx = np.ogrid[:height, :width_img]
                
                # Distance to line segment
                dx = x1 - x0
                dy = y1 - y0
                
                if dx == 0 and dy == 0:
                    continue
                    
                # Point-to-line distance
                t_param = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx**2 + dy**2), 0, 1)
                nearest_x = x0 + t_param * dx
                nearest_y = y0 + t_param * dy
                
                dist = np.sqrt((xx - nearest_x)**2 + (yy - nearest_y)**2)
                
                # Update image
                mask = dist <= width / 2
                image[mask] = np.maximum(image[mask], contrast * (1 - dist[mask] / (width / 2)))
                
    return image 