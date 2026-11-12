"""Key-based mapping between legacy Qt pages and scientific GUI stages.

Navigation keys are stable identities; a page's QStackedWidget index and its
position in a sidebar are implementation details. This helper is deliberately
Qt-free to permit early validation and lightweight regression tests.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_page_index(
    page_specs: Mapping[int, tuple[str, str, Any]],
    stage_keys: Sequence[str],
) -> dict[str, int]:
    """Return stage key -> internal stacked-page index, checking full coverage.

    Neither the ordering of `stage_keys` nor scientific execution order is
    inferred from `page_specs`. Duplicate/unknown/missing keys fail closed.
    """
    stages = tuple(str(key) for key in stage_keys)
    if len(stages) != len(set(stages)) or not all(stages):
        raise ValueError("GUI stage keys must be unique and nonempty")
    result: dict[str, int] = {}
    for index, (stage_key, _attr, _builder) in page_specs.items():
        key = str(stage_key)
        if key in result:
            raise ValueError(f"GUI page key is duplicated: {key!r}")
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Invalid GUI page index: {index!r}")
        result[key] = index
    if set(result) != set(stages):
        missing = sorted(set(stages) - set(result))
        extra = sorted(set(result) - set(stages))
        raise ValueError(f"GUI page mapping mismatch (missing={missing}, extra={extra})")
    if set(result.values()) != set(range(len(result))):
        raise ValueError("GUI page indices must be contiguous from zero")
    return result
