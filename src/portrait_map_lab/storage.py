"""Storage utilities for saving and loading images and arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save an image to disk using OpenCV.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    image
        Image array to save (BGR format expected for color images).
    path
        Path where the image should be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_array(array: np.ndarray, path: str | Path) -> None:
    """Save a numpy array to disk as .npy file.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    array
        Array to save.
    path
        Path where the array should be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Parameters
    ----------
    path
        Path to the image file.

    Returns
    -------
    np.ndarray
        Loaded image in BGR format (OpenCV default).

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")

    return image


def ensure_output_dir(base: str | Path, image_name: str) -> Path:
    """Create output directory structure for an image.

    Creates a directory at <base>/<image_name>/ and returns the path.

    Parameters
    ----------
    base
        Base output directory.
    image_name
        Name of the image (used as subdirectory name).

    Returns
    -------
    Path
        The created output directory path.
    """
    output_dir = Path(base) / image_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
