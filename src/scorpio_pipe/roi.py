"""Region-of-interest (ROI) primitives.

The pipeline needs a small, GUI-independent representation of the object and
sky regions.

The GUI includes an interactive ROI picker, but science stages and unit tests
must not depend on Qt. Keep this module dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ROI:
    """Y-ranges (inclusive) spanning the full X extent.

    Attributes are named using the pipeline-internal convention:
      - ``obj_y0``, ``obj_y1``: object band.
      - ``sky_top_y0``, ``sky_top_y1``: sky band above the object.
      - ``sky_bot_y0``, ``sky_bot_y1``: sky band below the object.

    The code typically *clips* indices to [0, ny-1] and *sorts* endpoints, so it
    is safe if users provide y0>y1.
    """

    obj_y0: int = 0
    obj_y1: int = 0
    sky_top_y0: int = 0
    sky_top_y1: int = 0
    sky_bot_y0: int = 0
    sky_bot_y1: int = 0
