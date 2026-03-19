from __future__ import annotations

"""SCORPIO-2 instrument helpers (P0-ARCH-05.1).

Thin wrapper around :mod:`scorpio_pipe.metadata.frame_meta` instrument-specific
parser.
"""

from typing import Any, Mapping

from scorpio_pipe.metadata.frame_meta import FrameMeta, parse_scorpio2_header


def parse_header(
    header: Mapping[str, Any],
    *,
    strict: bool = True,
    policy: Any | None = None,
    fallback_sources: Any | None = None,
) -> FrameMeta:
    pol = policy
    if pol is None:
        from scorpio_pipe.metadata.policy import default_metadata_policy

        pol = default_metadata_policy()
    return parse_scorpio2_header(header, policy=pol, strict=bool(strict), fallback=fallback_sources)
