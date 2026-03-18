"""Tests for portrait_map_lab.landmarks module."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.models import LandmarkResult


class TestDetectLandmarks:
    def test_returns_landmark_result(self, test_image: np.ndarray):
        result = detect_landmarks(test_image)
        assert isinstance(result, LandmarkResult)
        assert result.landmarks.shape == (478, 2)
        assert result.landmarks.dtype == np.float64

    def test_coordinates_within_bounds(self, test_image: np.ndarray):
        result = detect_landmarks(test_image)
        h, w = result.image_shape
        assert np.all(result.landmarks[:, 0] >= 0)
        assert np.all(result.landmarks[:, 0] < w)
        assert np.all(result.landmarks[:, 1] >= 0)
        assert np.all(result.landmarks[:, 1] < h)

    def test_image_shape_matches_input(self, test_image: np.ndarray):
        result = detect_landmarks(test_image)
        assert result.image_shape == (test_image.shape[0], test_image.shape[1])

    def test_confidence_is_valid(self, test_image: np.ndarray):
        result = detect_landmarks(test_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_raises_valueerror_on_blank_image(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="No face detected"):
            detect_landmarks(blank)
