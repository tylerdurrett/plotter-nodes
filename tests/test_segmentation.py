"""Tests for image segmentation module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from portrait_map_lab.face_contour import rasterize_contour_mask, rasterize_filled_mask
from portrait_map_lab.segmentation import (
    SEGMENTATION_ACCESSORIES,
    SEGMENTATION_FACE_SKIN,
    SEGMENTATION_HAIR,
    extract_segmentation_polygon,
    segment_image,
)


class TestSegmentImage:
    """Tests for segment_image function."""

    def test_returns_2d_uint8(self, test_image):
        """Should return a 2D uint8 array."""
        mask = segment_image(test_image)
        assert mask.ndim == 2
        assert mask.dtype == np.uint8

    def test_shape_matches_input(self, test_image):
        """Category mask should match input image dimensions."""
        h, w = test_image.shape[:2]
        mask = segment_image(test_image)
        assert mask.shape == (h, w)

    def test_valid_class_values(self, test_image):
        """All values should be in [0, 5]."""
        mask = segment_image(test_image)
        assert mask.min() >= 0
        assert mask.max() <= 5

    def test_has_face_skin(self, test_image):
        """Portrait image should have face skin pixels."""
        mask = segment_image(test_image)
        assert np.any(mask == SEGMENTATION_FACE_SKIN)

    def test_blank_image_has_no_face(self):
        """Blank image should have no face skin pixels."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        mask = segment_image(blank)
        assert not np.any(mask == SEGMENTATION_FACE_SKIN)


class TestExtractSegmentationPolygon:
    """Tests for extract_segmentation_polygon function."""

    def test_returns_nx2_float64(self, test_image):
        """Should return an Nx2 float64 array."""
        mask = segment_image(test_image)
        polygon = extract_segmentation_polygon(mask, [SEGMENTATION_FACE_SKIN])
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        assert polygon.dtype == np.float64
        assert polygon.shape[0] >= 3

    def test_face_skin_polygon_within_bounds(self, test_image):
        """Face skin polygon should be within image bounds."""
        h, w = test_image.shape[:2]
        mask = segment_image(test_image)
        polygon = extract_segmentation_polygon(mask, [SEGMENTATION_FACE_SKIN])
        assert np.all(polygon[:, 0] >= 0) and np.all(polygon[:, 0] <= w)
        assert np.all(polygon[:, 1] >= 0) and np.all(polygon[:, 1] <= h)

    def test_head_polygon_larger_than_face(self, test_image):
        """Head polygon (hair+face+accessories) should enclose more area than face alone."""
        mask = segment_image(test_image)
        face_poly = extract_segmentation_polygon(mask, [SEGMENTATION_FACE_SKIN])
        head_poly = extract_segmentation_polygon(
            mask, [SEGMENTATION_HAIR, SEGMENTATION_FACE_SKIN, SEGMENTATION_ACCESSORIES]
        )
        face_area = cv2.contourArea(np.round(face_poly).astype(np.int32))
        head_area = cv2.contourArea(np.round(head_poly).astype(np.int32))
        assert head_area >= face_area

    def test_simplification_reduces_vertices(self, test_image):
        """Non-zero epsilon should reduce vertex count vs no simplification."""
        mask = segment_image(test_image)
        full = extract_segmentation_polygon(mask, [SEGMENTATION_FACE_SKIN], epsilon_factor=0)
        simplified = extract_segmentation_polygon(
            mask, [SEGMENTATION_FACE_SKIN], epsilon_factor=0.01
        )
        assert simplified.shape[0] <= full.shape[0]

    def test_no_contour_raises(self):
        """Should raise ValueError when no contour found."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="No contour found"):
            extract_segmentation_polygon(empty_mask, [SEGMENTATION_FACE_SKIN])

    def test_compatible_with_rasterize(self, test_image):
        """Polygon should work with existing rasterize functions."""
        h, w = test_image.shape[:2]
        mask = segment_image(test_image)
        polygon = extract_segmentation_polygon(mask, [SEGMENTATION_FACE_SKIN])

        contour_mask = rasterize_contour_mask(polygon, (h, w))
        assert contour_mask.shape == (h, w)
        assert np.sum(contour_mask > 0) > 0

        filled_mask = rasterize_filled_mask(polygon, (h, w))
        assert np.sum(filled_mask > 0) > 0
