"""Visualization utilities for the portrait map lab pipeline."""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np

from portrait_map_lab.models import LandmarkResult


def draw_landmarks(image: np.ndarray, landmarks: LandmarkResult) -> np.ndarray:
    """Draw landmark points on a copy of the image.

    Parameters
    ----------
    image
        BGR image array to annotate.
    landmarks
        Face landmark detection result containing pixel coordinates.

    Returns
    -------
    np.ndarray
        New image with landmarks drawn as small green circles.
    """
    # Create copy to avoid mutation
    annotated = image.copy()

    # Draw each landmark as a small green circle
    for point in landmarks.landmarks:
        x, y = int(point[0]), int(point[1])
        # Ensure coordinates are within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

    return annotated


def colorize_map(array: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Apply a matplotlib colormap to a grayscale array.

    Parameters
    ----------
    array
        Float array with values in [0, 1].
    colormap
        Name of matplotlib colormap to use.

    Returns
    -------
    np.ndarray
        BGR uint8 image suitable for OpenCV.
    """
    # Ensure array is in [0, 1] range
    array_norm = np.clip(array, 0, 1)

    # Get colormap and apply
    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(array_norm)  # Returns RGBA in [0, 1]

    # Convert RGBA to RGB, then to BGR for OpenCV
    rgb = rgba[..., :3]  # Drop alpha channel
    bgr = rgb[..., ::-1]  # Reverse channel order

    # Convert to uint8
    bgr_uint8 = (bgr * 255).astype(np.uint8)

    return bgr_uint8


def make_contact_sheet(images: dict[str, np.ndarray], columns: int = 4) -> np.ndarray:
    """Create a grid of labeled images.

    Parameters
    ----------
    images
        Dictionary mapping labels to image arrays.
    columns
        Number of columns in the grid.

    Returns
    -------
    np.ndarray
        Single composite image with all inputs arranged in a grid.
    """
    if not images:
        # Return small white image if no images provided
        return np.full((512, 512, 3), 255, dtype=np.uint8)

    # Calculate grid dimensions
    n_images = len(images)
    rows = (n_images + columns - 1) // columns  # Ceiling division

    # Cell dimensions with space for labels
    cell_width = 512
    cell_height = 512
    label_height = 60
    total_cell_height = cell_height + label_height

    # Create white canvas
    canvas_height = rows * total_cell_height
    canvas_width = columns * cell_width
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    # Place each image
    for idx, (label, img) in enumerate(images.items()):
        row = idx // columns
        col = idx % columns

        # Calculate position
        x = col * cell_width
        y = row * total_cell_height + label_height

        # Resize image to fit cell while preserving aspect ratio
        img_h, img_w = img.shape[:2]
        scale = min(cell_width / img_w, cell_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Center the resized image within the cell (white padding)
        pad_x = (cell_width - new_w) // 2
        pad_y = (cell_height - new_h) // 2
        canvas[y + pad_y : y + pad_y + new_h, x + pad_x : x + pad_x + new_w] = resized

        # Add label above image
        text_y = row * total_cell_height + 40  # Position text in label area
        cv2.putText(
            canvas,
            label,
            (x + 20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),  # Black text
            2,
            cv2.LINE_AA,
        )

    return canvas


def draw_contour(
    image: np.ndarray,
    polygon: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a face contour polygon on a copy of the image.

    Parameters
    ----------
    image
        BGR image array to annotate.
    polygon
        Nx2 array of polygon vertices in pixel coordinates.
    color
        BGR color tuple for the contour. Default is cyan (0, 255, 255).
    thickness
        Line thickness in pixels. Default is 2.

    Returns
    -------
    np.ndarray
        New image with contour drawn as a closed polyline.
    """
    # Create copy to avoid mutation
    annotated = image.copy()

    # Convert polygon to integer coordinates for cv2
    pts = np.round(polygon).astype(np.int32)

    # Draw closed polyline
    cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=thickness)

    return annotated


def visualize_flow_field(
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    image: np.ndarray | None = None,
    step: int = 16,
    scale: float = 10.0,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw flow field vectors as arrows on a grid.

    Parameters
    ----------
    flow_x
        X-component of the flow field (unit vectors).
    flow_y
        Y-component of the flow field (unit vectors).
    image
        Optional BGR image to draw on. If None, creates a white canvas.
    step
        Grid spacing in pixels between arrows.
    scale
        Length scaling factor for arrows.
    color
        BGR color tuple for arrows. Default is green (0, 255, 0).

    Returns
    -------
    np.ndarray
        BGR uint8 image with flow field arrows drawn.
    """
    h, w = flow_x.shape

    # Create canvas if no image provided
    if image is None:
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    else:
        canvas = image.copy()

    # Draw arrows at grid points
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            # Get flow vector at this position
            vx = flow_x[y, x]
            vy = flow_y[y, x]

            # Calculate arrow endpoints
            start_point = (x, y)
            end_x = int(x + scale * vx)
            end_y = int(y + scale * vy)
            end_point = (end_x, end_y)

            # Draw arrow
            cv2.arrowedLine(
                canvas,
                start_point,
                end_point,
                color,
                thickness=1,
                tipLength=0.3,
            )

    return canvas


def overlay_lic(
    lic_image: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay LIC texture on an image with alpha blending.

    Parameters
    ----------
    lic_image
        Grayscale LIC texture with values in [0, 1].
    image
        BGR image to overlay onto.
    alpha
        Blending weight for LIC texture. 0 = only image, 1 = only LIC.

    Returns
    -------
    np.ndarray
        BGR uint8 image with LIC overlay.
    """
    # Convert LIC to BGR
    lic_uint8 = (lic_image * 255).astype(np.uint8)
    lic_bgr = cv2.cvtColor(lic_uint8, cv2.COLOR_GRAY2BGR)

    # Alpha blend
    result = alpha * lic_bgr + (1 - alpha) * image
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
