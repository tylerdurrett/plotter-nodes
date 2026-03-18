"""Shared test fixtures for portrait_map_lab tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def test_image_path() -> Path:
    """Path to a portrait test image."""
    return Path(__file__).resolve().parent.parent / "test_images" / "20230427-171404.JPG"


@pytest.fixture
def test_image(test_image_path: Path) -> np.ndarray:
    """Load the test portrait image as a BGR numpy array."""
    image = cv2.imread(str(test_image_path))
    if image is None:
        pytest.fail(f"Failed to load test image: {test_image_path}")
    return image
