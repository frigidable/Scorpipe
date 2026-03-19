from __future__ import annotations

"""Instrument catalog / registry (P0-ARCH-05.1).

The registry is the single place where instrument adapters are declared.

Notes
-----
- Keep this module import-light: it must not eagerly import heavy instrument code.
- We use lazy import via entrypoint strings (module:object).
"""

from dataclasses import dataclass
from typing import Any, Mapping

from scorpio_pipe.ports.instrument import InstrumentAdapter


@dataclass(frozen=True)
class InstrumentInfo:
    key: str
    display_name: str
    entrypoint: str
    # Substrings (normalized) to match against INSTRUME/INSTRUMENT.
    header_match_any: tuple[str, ...]


def _norm_upper_nospace(s: str) -> str:
    return "".join(ch for ch in (s or "").strip().upper() if not ch.isspace())


# Registry: adding a new instrument = new adapter + new record here.
CATALOG: tuple[InstrumentInfo, ...] = (
    InstrumentInfo(
        key="scorpio2",
        display_name="SCORPIO-2",
        entrypoint="scorpio_pipe.adapters.instrument_scorpio2:Scorpio2Adapter",
        header_match_any=("SCORPIO-2", "SCORPIO2"),
    ),
    InstrumentInfo(
        key="scorpio1",
        display_name="SCORPIO",
        entrypoint="scorpio_pipe.adapters.instrument_scorpio1:Scorpio1Adapter",
        header_match_any=("SCORPIO",),
    ),
)

_BY_KEY: dict[str, InstrumentInfo] = {x.key: x for x in CATALOG}

# Common aliases found in configs/headers.
_ALIASES: dict[str, str] = {
    "SCORPIO": "scorpio1",
    "SCORPIO1": "scorpio1",
    "SCORPIO-1": "scorpio1",
    "SCORPIO2": "scorpio2",
    "SCORPIO-2": "scorpio2",
}


_ADAPTER_CACHE: dict[str, InstrumentAdapter] = {}


def list_instruments() -> list[InstrumentInfo]:
    return list(CATALOG)


def normalize_instrument_key(name: str | None) -> str | None:
    if not name:
        return None
    n = _norm_upper_nospace(str(name))
    if not n:
        return None
    if n in _ALIASES:
        return _ALIASES[n]
    # If user already passes registry key.
    low = str(name).strip().lower()
    if low in _BY_KEY:
        return low
    return None


def detect_instrument_key_from_header(header: Mapping[str, Any]) -> str | None:
    """Detect registry key from FITS header."""

    raw = header.get("INSTRUME") or header.get("INSTRUMENT") or ""
    instr = _norm_upper_nospace(str(raw))
    if not instr:
        return None

    # Match catalog entries. Prefer more specific entries first (SCORPIO-2 before SCORPIO).
    for info in CATALOG:
        for token in info.header_match_any:
            if _norm_upper_nospace(token) in instr:
                # Guard: SCORPIO-2 should not be swallowed by the SCORPIO catch-all.
                if info.key == "scorpio1" and ("SCORPIO-2" in instr or "SCORPIO2" in instr):
                    continue
                return info.key
    return None


def _parse_entrypoint(ep: str) -> tuple[str, str]:
    if ":" not in ep:
        raise ValueError(f"Invalid entrypoint: {ep!r} (expected module:attr)")
    mod, attr = ep.split(":", 1)
    mod = mod.strip()
    attr = attr.strip()
    if not mod or not attr:
        raise ValueError(f"Invalid entrypoint: {ep!r}")
    return mod, attr


def get_adapter_by_key(key: str) -> InstrumentAdapter:
    k = str(key).strip().lower()
    if not k:
        raise ValueError("Empty instrument key")
    if k in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[k]
    info = _BY_KEY.get(k)
    if info is None:
        raise KeyError(f"Unknown instrument key: {key!r}")
    mod, attr = _parse_entrypoint(info.entrypoint)
    from importlib import import_module

    m = import_module(mod)
    obj = getattr(m, attr)
    adapter: InstrumentAdapter
    if isinstance(obj, type):
        adapter = obj()  # type: ignore[call-arg]
    else:
        adapter = obj  # type: ignore[assignment]
    _ADAPTER_CACHE[k] = adapter
    return adapter


def resolve_adapter(
    *,
    header: Mapping[str, Any] | None = None,
    instrument_name: str | None = None,
) -> InstrumentAdapter:
    """Resolve an adapter either from header or from an instrument name."""

    key: str | None = None
    if instrument_name:
        key = normalize_instrument_key(instrument_name)
    if key is None and header is not None:
        key = detect_instrument_key_from_header(header)
    if key is None:
        raise KeyError(
            "Cannot resolve instrument adapter (missing/unknown INSTRUME or instrument.name)"
        )
    return get_adapter_by_key(key)


def parse_header_to_meta(
    header: Mapping[str, Any],
    *,
    strict: bool = True,
    policy: Any | None = None,
    fallback_sources: Any | None = None,
) -> Any:
    """Parse header to FrameMeta via the resolved instrument adapter."""

    try:
        adapter = resolve_adapter(header=header)
    except KeyError as e:
        from scorpio_pipe.metadata import HeaderContractError
        raw = header.get("INSTRUME") or header.get("INSTRUMENT") or None
        raise HeaderContractError(
            "Unknown instrument (cannot resolve adapter)",
            instrument=str(raw) if raw else None,
            context={"INSTRUME": raw},
        ) from e
    return adapter.parse_header(header, strict=strict, policy=policy, fallback_sources=fallback_sources)
