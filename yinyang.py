# -*- coding: utf-8 -*-
#
# Description:  Generate parametric Yin Yang symbols with varying morphology
# Author:       Jarosław Bulat (kwant@agh.edu.pl, kwanty@gmail.com)
# Agents:       Gemini 3.0 Pro, Claude Haiku 4.5
# Created:      15.12.2025
# Changelog:
#   - Added type hints and improved documentation
#   - Refactored code structure and comments
#
# License:      GPLv3
# File:         yinyang.py

from typing import Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt


def generate_yinyang(
    p: float,
    size: int = 28,
    supersample: int = 4,
) -> np.ndarray:
    """
    Generate a grayscale image of a parametric Yin Yang symbol.

    The symbol morphs continuously based on the parameter p, which controls
    the relative sizes of the black and white lobes and their accompanying dots.

    Args:
        p: Morphology parameter in range [0, 1] controlling lobe proportions.
        size: Output image size in pixels (default: 28x28).
        supersample: Supersampling factor for anti-aliasing (default: 4x).

    Returns:
        Grayscale numpy array of shape (size, size) with values in [0, 1].
    """
    # Setup grid with supersampling for smooth rendering
    grid_size = size * supersample
    y, x = np.ogrid[:grid_size, :grid_size]

    # Normalize coordinates to [-0.9, 0.9] range centered at origin
    center = grid_size / 2.0
    scale = (grid_size / 2.0) * 0.9

    xx_raw = (x - center) / scale
    yy_raw = -(y - center) / scale

    # Apply global rotation (swirl) based on parameter p
    angle = np.pi * p
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    xx = xx_raw * cos_angle - yy_raw * sin_angle
    yy = xx_raw * sin_angle + yy_raw * cos_angle

    # Calculate radial distance from center
    r_dist = np.sqrt(xx**2 + yy**2)
    R = 1.0  # Outer circle radius

    # Initialize canvas with zeros (black background)
    canvas = np.zeros_like(r_dist)
    mask_main = r_dist <= R

    # Step 1: Create base vertical split
    x_thresh = R * (1.0 - 2.0 * p)
    mask_split = (xx > x_thresh) & mask_main
    canvas[mask_split] = 1.0

    # Step 2: Draw black lobe (lower half)
    y_b = p * R
    r_b = R * (1.0 - p)
    r_b = max(0.0, r_b)  # Clamp to prevent negative radius

    if r_b > 0:
        dist_b = np.sqrt(xx**2 + (yy - y_b)**2)
        canvas[(dist_b <= r_b) & mask_main] = 0.0

    # Step 3: Draw white lobe (upper half)
    y_w = (p - 1.0) * R
    r_w = R * p
    r_w = max(0.0, r_w)  # Clamp to prevent negative radius

    if r_w > 0:
        dist_w = np.sqrt(xx**2 + (yy - y_w)**2)
        canvas[(dist_w <= r_w) & mask_main] = 1.0

    # Step 4: Add decorative dots with size threshold
    # This prevents visual artifacts when lobes become too small
    dot_target = 0.15 * R


    def get_dot_radius(lobe_radius: float) -> float:
        """
        Calculate the radius of the decorative dot based on lobe size.

        The dot grows linearly from no dot (when lobe < 0.3R) to full size
        (when lobe >= 0.5R), providing smooth morphing transitions.

        Args:
            lobe_radius: Radius of the containing lobe.

        Returns:
            Radius of the dot to draw, or 0 if no dot should be drawn.
        """
        min_lobe = 0.3 * R
        full_lobe = 0.5 * R

        if lobe_radius < min_lobe:
            return 0.0
        elif lobe_radius >= full_lobe:
            return dot_target
        else:
            # Linear interpolation between min and full thresholds
            factor = (lobe_radius - min_lobe) / (full_lobe - min_lobe)
            return dot_target * factor

    # Draw white dot in black lobe
    r_wd = get_dot_radius(r_b)
    if r_wd > 0:
        dist_wd = np.sqrt(xx**2 + (yy - y_b)**2)
        canvas[dist_wd <= r_wd] = 1.0

    # Draw black dot in white lobe
    r_bd = get_dot_radius(r_w)
    if r_bd > 0:
        dist_bd = np.sqrt(xx**2 + (yy - y_w)**2)
        canvas[dist_bd <= r_bd] = 0.0

    # Mask background (everything outside the main circle)
    canvas[r_dist > R] = 0.0

    # Downsample from supersampled grid using averaging (anti-aliasing)
    img = canvas.reshape(size, supersample, size, supersample).mean(axis=(1, 3))

    return img


def visualize_16x16() -> None:
    """
    Generate and visualize a 16x16 grid of Yin Yang symbols with varying parameters.

    Creates a 16x16 grid of images showing the continuous morphology of the
    symbol from p=0 to p=1, and saves the composite image.
    """
    # Generate parameter values for grid
    rows, cols = 16, 16
    total = rows * cols
    ps = np.linspace(0, 1, total)

    # Generate all images
    imgs = [generate_yinyang(p, size=28) for p in ps]

    # Create composite image by tiling
    composite = np.zeros((rows * 28, cols * 28))

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < total:
                composite[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = imgs[idx]

    # Plot and save
    plt.figure(figsize=(10, 10))
    plt.imshow(composite, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Yin Yang Parametric Grid (16x16)")
    plt.tight_layout()
    plt.savefig("images/yinyang_progression_16x16.png")
    print("Saved images/yinyang_progression_16x16.png")


def visualize_4x4() -> None:
    """
    Generate and visualize a 4x4 grid of Yin Yang symbols with parameter labels.

    Creates a 4x4 grid of images with text labels showing the parameter p value
    above each symbol, and displays the figure interactively.
    """
    # Generate parameter values for grid
    rows, cols = 4, 4
    total = rows * cols
    ps = np.linspace(0, 1, total)

    # Generate all images
    imgs = [generate_yinyang(p, size=28) for p in ps]

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing

    for idx in range(total):
        ax = axes[idx]
        ax.imshow(imgs[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"p = {ps[idx]:.2f}", fontsize=10)
        ax.axis("off")

    plt.suptitle("Yin Yang Parametric Preview (4x4)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("images/yinyang_preview_4x4.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved images/yinyang_preview_4x4.png")


def generate_dataset(
    num_samples: int = 10000,
    filename: str = "data/yinyang_10k.npz",
) -> None:
    """
    Generate a dataset of Yin Yang images and save to NPZ format.

    Creates a batch of grayscale images with random parameter values uniformly
    distributed in [0, 1], along with corresponding parameter labels. The dataset
    is suitable for regression tasks (predicting p from image) or generative modeling.

    Args:
        num_samples: Number of images to generate (default: 10000).
        filename: Path to save the compressed NPZ file (default: 'yinyang.npz').

    Output file structure:
        x_train: Array of shape (num_samples, 28, 28) with dtype float32
        y_train: Array of shape (num_samples,) with dtype float32
    """
    print(f"Generating {num_samples} samples...")

    # Sample parameter values uniformly across [0, 1]
    p_values = np.random.uniform(0.0, 1.0, num_samples)

    images = []

    # Generate all images
    for i, p in enumerate(p_values):
        if i % 1000 == 0:
            print(f"Generated {i}/{num_samples}")
        img = generate_yinyang(p, size=28)
        images.append(img)

    # Convert to numpy arrays with appropriate dtypes
    images = np.array(images, dtype=np.float32)
    labels = p_values.astype(np.float32)

    # Save to compressed NPZ format
    np.savez_compressed(filename, x_train=images, y_train=labels)
    print(f"Saved dataset to {filename}")
    print(f"Shapes: x_train {images.shape}, y_train {labels.shape}")


if __name__ == "__main__":
    # Uncomment to generate visualization grids
    visualize_16x16()
    visualize_4x4()

    # Generate dataset for training
    generate_dataset()
