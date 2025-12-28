"""Line identification stage.

This stage is currently implemented as an interactive GUI (PySide6) that writes
`hand_pairs.txt`.
"""

from __future__ import annotations


from .lineid_gui import prepare_lineid

__all__ = ["prepare_lineid"]
