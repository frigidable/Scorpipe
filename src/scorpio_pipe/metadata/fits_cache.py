from __future__ import annotations

"""Small LRU cache for FITS headers and parsed FrameMeta.

Goal
----
Avoid repeated FITS open/getheader calls across UI and preflight-style scans.
The cache key is derived from ``(path, mtime_ns, size)`` so edits on disk
naturally invalidate the entry.

Notes
-----
- The cached header is stored as a plain mapping (``dict``), not as an Astropy
  ``Header`` instance. All current consumers only need mapping semantics.
- We cache a best-effort lenient ``FrameMeta`` eagerly and a strict
  ``FrameMeta`` lazily on demand. This keeps the first access useful for UI
  browsing while still supporting strict validation paths without re-opening the
  FITS file.
- Policy / fallback overrides are not cached (but they still reuse the cached
  header mapping and therefore avoid disk I/O).
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import os
import threading
from typing import Any, Mapping

from .frame_meta import FrameMeta, parse_frame_meta

_DEFAULT_MAX_ITEMS = 4096


@dataclass(frozen=True)
class FitsHeaderMetaSnapshot:
    path: str
    mtime_ns: int
    size: int
    header_fields: Mapping[str, Any]
    frame_meta: FrameMeta | None


@dataclass(frozen=True)
class FitsHeaderMetaCacheStats:
    entries: int
    max_items: int
    hits: int
    misses: int
    disk_loads: int
    evictions: int


class _CacheEntry:
    __slots__ = (
        "snapshot",
        "strict_meta",
        "strict_error",
        "header_error",
    )

    def __init__(
        self,
        *,
        snapshot: FitsHeaderMetaSnapshot | None,
        strict_meta: FrameMeta | None = None,
        strict_error: BaseException | None = None,
        header_error: BaseException | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.strict_meta = strict_meta
        self.strict_error = strict_error
        self.header_error = header_error


_LOCK = threading.RLock()
_CACHE: "OrderedDict[tuple[str, int, int], _CacheEntry]" = OrderedDict()
_STATS: dict[str, int] = {"hits": 0, "misses": 0, "disk_loads": 0, "evictions": 0}


def _max_items() -> int:
    raw = os.getenv("SCORPIPE_FITS_META_CACHE_ITEMS", str(_DEFAULT_MAX_ITEMS))
    try:
        return max(16, int(raw))
    except Exception:
        return _DEFAULT_MAX_ITEMS


def _norm_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _cache_key(path: str | Path) -> tuple[str, int, int]:
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return (str(p), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))), int(st.st_size))


def _raise(err: BaseException) -> None:
    raise err


def _header_to_mapping(hdr: Any) -> dict[str, Any]:
    try:
        return dict(hdr)
    except Exception:
        out: dict[str, Any] = {}
        try:
            for k in hdr.keys():
                out[str(k)] = hdr.get(k)
        except Exception:
            pass
        return out


def _load_header_from_disk(path: str | Path) -> dict[str, Any]:
    from astropy.io import fits

    from scorpio_pipe.fits_utils import open_fits_smart

    p = Path(path).expanduser().resolve()
    try:
        hdr = fits.getheader(p, 0, memmap=False)
        return _header_to_mapping(hdr)
    except Exception:
        # Fall back to a more permissive open path for slightly malformed files.
        with open_fits_smart(
            p,
            memmap="auto",
            ignore_missing_end=True,
            ignore_missing_simple=True,
        ) as hdul:
            return _header_to_mapping(hdul[0].header)


def _build_entry(path: str | Path) -> tuple[tuple[str, int, int], _CacheEntry]:
    key = _cache_key(path)
    with _LOCK:
        _STATS["disk_loads"] += 1
    p_s = key[0]
    try:
        header_fields = _load_header_from_disk(p_s)
    except BaseException as e:
        return key, _CacheEntry(snapshot=None, header_error=e)

    meta_relaxed: FrameMeta | None = None
    try:
        meta_relaxed = parse_frame_meta(header_fields, strict=False)
    except BaseException:
        meta_relaxed = None

    snap = FitsHeaderMetaSnapshot(
        path=p_s,
        mtime_ns=key[1],
        size=key[2],
        header_fields=header_fields,
        frame_meta=meta_relaxed,
    )
    return key, _CacheEntry(snapshot=snap)


def _remember(key: tuple[str, int, int], entry: _CacheEntry) -> _CacheEntry:
    _CACHE[key] = entry
    _CACHE.move_to_end(key)
    while len(_CACHE) > _max_items():
        _CACHE.popitem(last=False)
        _STATS["evictions"] += 1
    return entry


def _resolve_meta_from_snapshot(
    snap: FitsHeaderMetaSnapshot,
    *,
    strict: bool,
    policy: Any | None = None,
    fallback_sources: Any | None = None,
) -> FrameMeta:
    if policy is not None or fallback_sources is not None:
        return parse_frame_meta(
            snap.header_fields,
            strict=strict,
            policy=policy,
            fallback_sources=fallback_sources,
        )

    if strict:
        key = (snap.path, snap.mtime_ns, snap.size)
        with _LOCK:
            entry = _CACHE.get(key)
            if entry is not None:
                if entry.strict_error is not None:
                    _raise(entry.strict_error)
                if entry.strict_meta is not None:
                    _CACHE.move_to_end(key)
                    return entry.strict_meta

        try:
            meta = parse_frame_meta(snap.header_fields, strict=True)
        except BaseException as e:
            with _LOCK:
                entry = _CACHE.get(key)
                if entry is not None:
                    entry.strict_error = e
                    _CACHE.move_to_end(key)
            _raise(e)

        with _LOCK:
            entry = _CACHE.get(key)
            if entry is not None:
                entry.strict_meta = meta
                entry.strict_error = None
                _CACHE.move_to_end(key)
        return meta

    if snap.frame_meta is not None:
        return snap.frame_meta
    return parse_frame_meta(snap.header_fields, strict=False)


def get_fits_header_meta_cache_stats() -> FitsHeaderMetaCacheStats:
    with _LOCK:
        return FitsHeaderMetaCacheStats(
            entries=len(_CACHE),
            max_items=_max_items(),
            hits=int(_STATS.get("hits", 0)),
            misses=int(_STATS.get("misses", 0)),
            disk_loads=int(_STATS.get("disk_loads", 0)),
            evictions=int(_STATS.get("evictions", 0)),
        )


def reset_fits_header_meta_cache_stats() -> None:
    with _LOCK:
        for k in list(_STATS.keys()):
            _STATS[k] = 0


def get_cached_snapshot(path: str | Path) -> FitsHeaderMetaSnapshot:
    """Return cached FITS header/meta snapshot for *path*.

    Raises the original FITS-open exception when header loading failed.
    """

    key = _cache_key(path)
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            _STATS["hits"] += 1
            _CACHE.move_to_end(key)
            if entry.header_error is not None:
                _raise(entry.header_error)
            if entry.snapshot is None:
                raise RuntimeError("FITS cache invariant broken: missing snapshot")
            return entry.snapshot
        _STATS["misses"] += 1

    key2, entry2 = _build_entry(path)
    with _LOCK:
        entry = _remember(key2, entry2)
        if entry.header_error is not None:
            _raise(entry.header_error)
        if entry.snapshot is None:
            raise RuntimeError("FITS cache invariant broken: missing snapshot")
        return entry.snapshot


def get_header_fields(path: str | Path) -> Mapping[str, Any]:
    return get_cached_snapshot(path).header_fields


def get_header_and_frame_meta(
    path: str | Path,
    *,
    strict: bool = True,
    policy: Any | None = None,
    fallback_sources: Any | None = None,
) -> tuple[Mapping[str, Any], FrameMeta]:
    """Return cached FITS header mapping and parsed ``FrameMeta`` together.

    This is the most efficient accessor for scan-style callers that need both
    header fields and parsed metadata for the same file, because it reuses a
    single snapshot lookup and avoids an extra cache-key/stat round-trip.
    """

    snap = get_cached_snapshot(path)
    return snap.header_fields, _resolve_meta_from_snapshot(
        snap,
        strict=strict,
        policy=policy,
        fallback_sources=fallback_sources,
    )


def get_frame_meta(
    path: str | Path,
    *,
    strict: bool = True,
    policy: Any | None = None,
    fallback_sources: Any | None = None,
) -> FrameMeta:
    """Return parsed ``FrameMeta`` without re-opening the FITS file."""

    _hdr, meta = get_header_and_frame_meta(
        path,
        strict=strict,
        policy=policy,
        fallback_sources=fallback_sources,
    )
    return meta


def clear_fits_header_meta_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def clear_fits_header_meta_cache_all() -> None:
    clear_fits_header_meta_cache()
    reset_fits_header_meta_cache_stats()


__all__ = [
    "FitsHeaderMetaSnapshot",
    "FitsHeaderMetaCacheStats",
    "get_cached_snapshot",
    "get_header_fields",
    "get_header_and_frame_meta",
    "get_frame_meta",
    "get_fits_header_meta_cache_stats",
    "reset_fits_header_meta_cache_stats",
    "clear_fits_header_meta_cache",
    "clear_fits_header_meta_cache_all",
]
