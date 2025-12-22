from __future__ import annotations

"""Matplotlib style used across Scorpio Pipe plots.

This style is intentionally *light* (white background + black axes) to remain
readable in papers, PDFs and quick QA screenshots, even when the Qt UI runs in
dark mode.
"""

from contextlib import contextmanager

STYLE: dict[str, object] = {
    "font.family":         "DejaVu Serif",
    "font.size":           13,
    "axes.titlesize":      15,
    "axes.labelsize":      14,
    "xtick.labelsize":     12,
    "ytick.labelsize":     12,
    "legend.fontsize":     11,
    "figure.titlesize":    16,
    "axes.linewidth":      1.0,
    "xtick.major.width":   1.0,
    "ytick.major.width":   1.0,
    "xtick.minor.width":   0.8,
    "ytick.minor.width":   0.8,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.top":           True,
    "ytick.right":         True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.grid":           False,
    "grid.alpha":          0.25,
    "grid.linestyle":      "-",
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.04,
    "figure.dpi":          120,
    "figure.facecolor":    "white",
}

@contextmanager
def mpl_style():
    """Context manager that temporarily applies :data:`STYLE`."""
    import matplotlib as mpl
    with mpl.rc_context(STYLE):
        yield
