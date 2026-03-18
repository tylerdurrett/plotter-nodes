"""Tests for face contour distance field computation."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.face_contour import (
    average_signed_distances,
    compute_signed_distance,
    derive_contour_from_sdf,
    get_face_oval_polygon,
    prepare_directional_distance,
    rasterize_contour_mask,
    rasterize_filled_mask,
)
from portrait_map_lab.models import LandmarkResult


class TestGetFaceOvalPolygon:
    """Tests for get_face_oval_polygon function."""

    def test_returns_nx2_array(self):
        """Should return an Nx2 array."""
        landmarks = np.random.rand(478, 2) * 100
        landmark_result = LandmarkResult(
            landmarks=landmarks, image_shape=(480, 640), confidence=0.95
        )

        polygon = get_face_oval_polygon(landmark_result)

        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        # Convex hull has at least 3 vertices
        assert polygon.shape[0] >= 3

    def test_correct_dtype(self):
        """Should return float64 array."""
        landmarks = np.random.rand(478, 2).astype(np.float32) * 100
        landmark_result = LandmarkResult(
            landmarks=landmarks, image_shape=(480, 640), confidence=0.95
        )

        polygon = get_face_oval_polygon(landmark_result)

        assert polygon.dtype == np.float64

    def test_coordinates_within_image_bounds(self):
        """Returned coordinates should be within image dimensions."""
        h, w = 480, 640
        landmarks = np.random.rand(478, 2) * [w, h]
        landmark_result = LandmarkResult(landmarks=landmarks, image_shape=(h, w), confidence=0.95)

        polygon = get_face_oval_polygon(landmark_result)

        assert np.all(polygon[:, 0] >= 0)
        assert np.all(polygon[:, 0] <= w)
        assert np.all(polygon[:, 1] >= 0)
        assert np.all(polygon[:, 1] <= h)

    def test_hull_encloses_all_landmarks(self):
        """Convex hull should enclose all landmark points."""
        from scipy.spatial import Delaunay

        np.random.seed(42)
        landmarks = np.random.rand(478, 2) * 100
        landmark_result = LandmarkResult(
            landmarks=landmarks, image_shape=(480, 640), confidence=0.95
        )

        polygon = get_face_oval_polygon(landmark_result)

        # All original landmarks should be inside or on the hull
        hull = Delaunay(polygon)
        inside = hull.find_simplex(landmarks) >= 0
        assert np.all(inside)


class TestRasterizeContourMask:
    """Tests for rasterize_contour_mask function."""

    def test_dtype_and_shape(self):
        """Should return uint8 array with correct shape."""
        # Create simple square polygon
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
        image_shape = (50, 50)

        mask = rasterize_contour_mask(polygon, image_shape)

        assert mask.dtype == np.uint8
        assert mask.shape == image_shape

    def test_binary_values(self):
        """Should only contain 0 or 255 values."""
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
        mask = rasterize_contour_mask(polygon, (50, 50))

        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values)

    def test_nonzero_pixels(self):
        """Should have nonzero pixels along the contour."""
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
        mask = rasterize_contour_mask(polygon, (50, 50))

        assert np.sum(mask > 0) > 0

    def test_thickness_thin_vs_thick(self):
        """Thicker lines should produce more nonzero pixels."""
        polygon = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.float64)

        thin_mask = rasterize_contour_mask(polygon, (50, 50), thickness=1)
        thick_mask = rasterize_contour_mask(polygon, (50, 50), thickness=3)

        thin_pixels = np.sum(thin_mask > 0)
        thick_pixels = np.sum(thick_mask > 0)

        assert thick_pixels > thin_pixels

    def test_closed_polygon(self):
        """Should produce a closed contour."""
        # Triangle polygon
        polygon = np.array([[25, 10], [40, 40], [10, 40]], dtype=np.float64)
        mask = rasterize_contour_mask(polygon, (50, 50))

        # Check that we have pixels forming a closed shape
        nonzero_count = np.sum(mask > 0)
        assert nonzero_count > 20  # Should have pixels along all three edges


class TestRasterizeFilledMask:
    """Tests for rasterize_filled_mask function."""

    def test_dtype_and_shape(self):
        """Should return uint8 array with correct shape."""
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
        image_shape = (50, 50)

        mask = rasterize_filled_mask(polygon, image_shape)

        assert mask.dtype == np.uint8
        assert mask.shape == image_shape

    def test_binary_values(self):
        """Should only contain 0 or 255 values."""
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
        mask = rasterize_filled_mask(polygon, (50, 50))

        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values)

    def test_filled_area_larger_than_contour(self):
        """Filled mask should have more pixels than contour mask."""
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)

        contour_mask = rasterize_contour_mask(polygon, (50, 50))
        filled_mask = rasterize_filled_mask(polygon, (50, 50))

        contour_pixels = np.sum(contour_mask > 0)
        filled_pixels = np.sum(filled_mask > 0)

        assert filled_pixels > contour_pixels

    def test_interior_is_filled(self):
        """Interior of polygon should be completely filled."""
        # Create a simple square
        polygon = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64)
        mask = rasterize_filled_mask(polygon, (50, 50))

        # Check that the interior region (e.g., center of square) is filled
        interior_region = mask[22:28, 22:28]
        assert np.all(interior_region == 255)


class TestComputeSignedDistance:
    """Tests for compute_signed_distance function."""

    def test_shape_and_dtype(self):
        """Should return float64 array with same shape as input."""
        contour_mask = np.zeros((50, 50), dtype=np.uint8)
        filled_mask = np.zeros((50, 50), dtype=np.uint8)

        signed_dist = compute_signed_distance(contour_mask, filled_mask)

        assert signed_dist.shape == (50, 50)
        assert signed_dist.dtype == np.float64

    def test_zero_on_contour(self):
        """Pixels on the contour should have distance ~0.0."""
        # Create a simple contour
        polygon = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64)
        contour_mask = rasterize_contour_mask(polygon, (50, 50), thickness=1)
        filled_mask = rasterize_filled_mask(polygon, (50, 50))

        signed_dist = compute_signed_distance(contour_mask, filled_mask)

        # Check contour pixels
        contour_pixels = contour_mask > 0
        contour_distances = signed_dist[contour_pixels]

        assert np.allclose(contour_distances, 0.0, atol=1e-10)

    def test_negative_inside(self):
        """Interior pixels should have negative distances."""
        polygon = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64)
        contour_mask = rasterize_contour_mask(polygon, (50, 50), thickness=1)
        filled_mask = rasterize_filled_mask(polygon, (50, 50))

        signed_dist = compute_signed_distance(contour_mask, filled_mask)

        # Check interior point (center of square)
        center_dist = signed_dist[25, 25]
        assert center_dist < 0

    def test_positive_outside(self):
        """Exterior pixels should have positive distances."""
        polygon = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float64)
        contour_mask = rasterize_contour_mask(polygon, (50, 50), thickness=1)
        filled_mask = rasterize_filled_mask(polygon, (50, 50))

        signed_dist = compute_signed_distance(contour_mask, filled_mask)

        # Check exterior points
        exterior_corners = [
            signed_dist[5, 5],  # Top-left corner
            signed_dist[5, 45],  # Top-right corner
            signed_dist[45, 5],  # Bottom-left corner
            signed_dist[45, 45],  # Bottom-right corner
        ]

        assert all(d > 0 for d in exterior_corners)

    def test_magnitude_increases_from_contour(self):
        """Absolute distance should increase moving away from contour."""
        polygon = np.array([[25, 25], [35, 25], [35, 35], [25, 35]], dtype=np.float64)
        contour_mask = rasterize_contour_mask(polygon, (60, 60), thickness=1)
        filled_mask = rasterize_filled_mask(polygon, (60, 60))

        signed_dist = compute_signed_distance(contour_mask, filled_mask)

        # Check distances at increasing distances from contour
        # Center is farther from contour than mid_inside
        center = signed_dist[30, 30]  # Inside, far from contour
        mid_inside = signed_dist[28, 30]  # Inside, closer to contour
        far_outside = signed_dist[10, 10]  # Outside, far from contour

        # Center (farther from contour) should have larger absolute distance
        assert abs(center) > abs(mid_inside)
        # Far outside point should have large absolute distance
        assert abs(far_outside) > abs(mid_inside)


class TestPrepareDirectionalDistance:
    """Tests for prepare_directional_distance function."""

    def setup_method(self):
        """Create test signed distance field."""
        # Create a simple signed distance field
        # Interior (negative), contour (zero), exterior (positive)
        self.signed_dist = np.array(
            [
                [3.0, 2.0, 1.0, 0.0, -1.0],
                [2.0, 1.0, 0.0, -1.0, -2.0],
                [1.0, 0.0, -1.0, -2.0, -3.0],
                [0.0, -1.0, -2.0, -3.0, -4.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0],
            ]
        )

    def test_inward_clamps_exterior(self):
        """Inward mode should clamp exterior pixels."""
        result = prepare_directional_distance(self.signed_dist, mode="inward", clamp_value=999.0)

        # Exterior pixels (positive values) should be clamped
        assert result[0, 0] == 999.0  # Was 3.0
        assert result[0, 1] == 999.0  # Was 2.0

        # Interior pixels (negative values) should be absolute value
        assert result[4, 4] == 5.0  # Was -5.0
        assert result[3, 3] == 3.0  # Was -3.0

        # Zero values should be clamped (they're on the boundary, not inside)
        assert result[0, 3] == 999.0  # Was 0.0

    def test_outward_clamps_interior(self):
        """Outward mode should clamp interior pixels."""
        result = prepare_directional_distance(self.signed_dist, mode="outward", clamp_value=999.0)

        # Interior pixels (negative values) should be clamped
        assert result[4, 4] == 999.0  # Was -5.0
        assert result[3, 3] == 999.0  # Was -3.0

        # Exterior pixels (positive values) should be kept
        assert result[0, 0] == 3.0
        assert result[0, 1] == 2.0

        # Zero values should be clamped (they're on the boundary)
        assert result[0, 3] == 999.0  # Was 0.0

    def test_both_is_all_positive(self):
        """Both mode should use absolute values everywhere."""
        result = prepare_directional_distance(self.signed_dist, mode="both")

        # All values should be absolute
        assert np.all(result >= 0)
        assert result[0, 0] == 3.0
        assert result[4, 4] == 5.0
        assert result[0, 3] == 0.0

    def test_band_clamps_beyond_width(self):
        """Band mode should clamp distances beyond band_width."""
        result = prepare_directional_distance(
            self.signed_dist, mode="band", band_width=2.5, clamp_value=999.0
        )

        # Within band (abs <= 2.5)
        assert result[0, 1] == 2.0  # abs(2.0) = 2.0 <= 2.5
        assert result[3, 2] == 2.0  # abs(-2.0) = 2.0 <= 2.5

        # Beyond band (abs > 2.5)
        assert result[0, 0] == 999.0  # abs(3.0) = 3.0 > 2.5
        assert result[4, 4] == 999.0  # abs(-5.0) = 5.0 > 2.5

    def test_band_requires_band_width(self):
        """Band mode should raise error without band_width."""
        with pytest.raises(ValueError, match="band_width is required"):
            prepare_directional_distance(self.signed_dist, mode="band")

    def test_raises_on_unknown_mode(self):
        """Should raise ValueError for unknown mode."""
        with pytest.raises(ValueError, match="Unknown direction mode"):
            prepare_directional_distance(self.signed_dist, mode="invalid")

    def test_preserves_shape_and_dtype(self):
        """Output should have same shape and be float64."""
        for mode in ["inward", "outward", "both"]:
            result = prepare_directional_distance(self.signed_dist, mode=mode)
            assert result.shape == self.signed_dist.shape
            assert result.dtype == np.float64

        # Also test band mode
        result = prepare_directional_distance(self.signed_dist, mode="band", band_width=3.0)
        assert result.shape == self.signed_dist.shape
        assert result.dtype == np.float64


class TestAverageSignedDistances:
    """Tests for average_signed_distances function."""

    def test_identical_sdfs_return_same(self):
        """Averaging identical SDFs should return the same SDF."""
        sdf = np.array([[1.0, -2.0], [3.0, -4.0]])
        result = average_signed_distances([sdf, sdf, sdf])
        np.testing.assert_array_almost_equal(result, sdf)

    def test_two_sdfs_mean(self):
        """Should compute elementwise mean of two SDFs."""
        sdf1 = np.array([[2.0, -4.0], [6.0, -8.0]])
        sdf2 = np.array([[4.0, -2.0], [0.0, -4.0]])
        result = average_signed_distances([sdf1, sdf2])
        expected = np.array([[3.0, -3.0], [3.0, -6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_sdf(self):
        """Single SDF should return itself."""
        sdf = np.array([[1.0, -1.0], [0.0, 2.0]])
        result = average_signed_distances([sdf])
        np.testing.assert_array_almost_equal(result, sdf)

    def test_empty_list_raises(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="At least one"):
            average_signed_distances([])

    def test_shape_mismatch_raises(self):
        """Should raise ValueError for mismatched shapes."""
        sdf1 = np.zeros((10, 10))
        sdf2 = np.zeros((10, 20))
        with pytest.raises(ValueError, match="shape mismatch"):
            average_signed_distances([sdf1, sdf2])

    def test_preserves_dtype(self):
        """Result should be float64."""
        sdf = np.array([[1.0, -1.0]], dtype=np.float64)
        result = average_signed_distances([sdf, sdf])
        assert result.dtype == np.float64


class TestDeriveContourFromSdf:
    """Tests for derive_contour_from_sdf function."""

    def _make_circle_sdf(self, shape=(100, 100), center=(50, 50), radius=20):
        """Create a signed distance field for a circle."""
        h, w = shape
        y, x = np.mgrid[:h, :w]
        dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) - radius
        return dist.astype(np.float64)

    def test_returns_correct_types(self):
        """Should return (polygon, contour_mask, filled_mask) with correct types."""
        sdf = self._make_circle_sdf()
        polygon, contour_mask, filled_mask = derive_contour_from_sdf(sdf)

        assert polygon.ndim == 2
        assert polygon.shape[1] == 2
        assert polygon.dtype == np.float64
        assert contour_mask.dtype == np.uint8
        assert filled_mask.dtype == np.uint8

    def test_filled_mask_matches_sdf_interior(self):
        """Filled mask should match negative region of SDF."""
        sdf = self._make_circle_sdf()
        _, _, filled_mask = derive_contour_from_sdf(sdf)

        expected_interior = sdf < 0
        actual_interior = filled_mask > 0
        # Allow small border differences from rasterization
        agreement = np.mean(expected_interior == actual_interior)
        assert agreement > 0.99

    def test_contour_mask_has_pixels(self):
        """Contour mask should have nonzero pixels."""
        sdf = self._make_circle_sdf()
        _, contour_mask, _ = derive_contour_from_sdf(sdf)
        assert np.sum(contour_mask > 0) > 0

    def test_no_contour_raises(self):
        """Should raise ValueError if SDF has no interior."""
        sdf = np.ones((50, 50), dtype=np.float64)  # All positive = no interior
        with pytest.raises(ValueError, match="No contour found"):
            derive_contour_from_sdf(sdf)
