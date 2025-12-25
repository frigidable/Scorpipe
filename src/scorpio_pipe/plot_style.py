from __future__ import annotations

"""Shared Matplotlib style for Scorpio Pipe.

All pipeline plots should look consistent and remain readable in papers, PDFs
and quick QA screenshots.

This module defines a single style dictionary and a context manager that
temporarily applies it.
"""

from contextlib import contextmanager

# User-approved pipeline style (v5+)
STYLE_PIPELINE: dict[str, object] = {
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,

    "figure.facecolor": "white",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,

    "axes.linewidth": 1.0,
    "axes.grid": False,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": False,

    # --- Ticks on all sides
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.labeltop": False,
    "ytick.labelright": False,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,

    "legend.frameon": False,
    "legend.fontsize": 10,

    "path.simplify": True,
    "agg.path.chunksize": 20000,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Backward compatibility: older code imports STYLE
STYLE = STYLE_PIPELINE


@contextmanager
def mpl_style():
    """Context manager that temporarily applies :data:`STYLE_PIPELINE`."""
    import matplotlib as mpl
    with mpl.rc_context(STYLE_PIPELINE):
        yield
