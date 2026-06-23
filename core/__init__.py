"""Core processing layer: pure NumPy / OpenCV / SciPy math.

This package is intentionally agnostic of PyQt6 and of every hardware SDK so that
the optical-sectioning mathematics can be unit-tested and reused in isolation.
"""

from .image_processing import (
    process_dsi,
    compute_dsi_images,
    normalize_to_8bit,
    save_dsi_results,
    save_raw_stack_tiff,
    save_volume_tiff,
    save_axial_sectioning_plot,
    save_parameter_log,
    scale_16bit_image,
    accumulate_event_frame,
    filter_crazy_pixels,
    apply_smoothing,
    save_mat_tif,
)

__all__ = [
    "process_dsi",
    "compute_dsi_images",
    "normalize_to_8bit",
    "save_dsi_results",
    "save_raw_stack_tiff",
    "save_volume_tiff",
    "save_axial_sectioning_plot",
    "save_parameter_log",
    "scale_16bit_image",
    "accumulate_event_frame",
    "filter_crazy_pixels",
    "apply_smoothing",
    "save_mat_tif",
]
