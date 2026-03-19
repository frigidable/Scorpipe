from __future__ import annotations

"""SCORPIO (SCORPIO-1) instrument helpers (P0-ARCH-05.1).

This module is intentionally thin. The authoritative header parsing logic lives
in :mod:`scorpio_pipe.metadata.frame_meta`. We keep this wrapper so that
instrument adapters can depend on a stable import path under
:mod:`scorpio_pipe.instruments`.
"""

from typing import Any, Mapping

from scorpio_pipe.metadata.frame_meta import FrameMeta, parse_scorpio1_header


def parse_header(
    header: Mapping[str, Any],
    *,
    strict: bool = True,
    policy: Any | None = None,
    fallback_sources: Any | None = None,
) -> FrameMeta:
    # policy/fallback are passed through (they are types from scorpio_pipe.metadata)
    pol = policy
    if pol is None:
        # import lazily to keep imports light
        from scorpio_pipe.metadata.policy import default_metadata_policy

        pol = default_metadata_policy()
    return parse_scorpio1_header(header, policy=pol, strict=bool(strict), fallback=fallback_sources)
