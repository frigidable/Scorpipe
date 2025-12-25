"""UI helpers for scorpio-pipe.

This package is intentionally small and focused on user-facing widgets.
"""

from __future__ import annotations

__all__ = ["PdfViewer", "LauncherWindow"]

# Allow importing the package in headless environments (CI) where PySide6 may
# be absent. GUI entry points should import the needed modules directly.
try:
    from .launcher_window import LauncherWindow  # noqa: F401
except Exception:  # pragma: no cover
    LauncherWindow = None  # type: ignore

try:
    from .pdf_viewer import PdfViewer  # noqa: F401
except Exception:  # pragma: no cover
    PdfViewer = None  # type: ignore
