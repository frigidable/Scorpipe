"""Project version.

Bumped as part of the release packaging process.
"""

__version__ = "5.40.34"

# Backward-compatible alias used by the UI/CI
PIPELINE_VERSION = __version__


def as_header_cards(prefix: str = "SCORP") -> dict[str, str]:
    """Minimal FITS provenance cards.

    We keep the key namespace short to avoid header bloat.
    """

    p = str(prefix).strip().upper()
    return {
        f"{p}VER": __version__,
    }
