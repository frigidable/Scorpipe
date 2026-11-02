"""Run-scoped Preflight selection and input fingerprinting.

The night-wide inventory is retained for audit, but only explicitly assigned
frames determine the processing gate. Missing, excluded or unscanned assigned
frames fail closed. Legacy configurations without a frame-role inventory
continue to gate on the whole-night result.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

FRAME_ROLES = (
    "obj", "bias", "neon", "flat", "sky", "sunsky",
    "standard", "telluric_standard",
)


def assigned_frames(cfg: Mapping[str, Any], data_dir: Path) -> tuple[bool, list[tuple[str, Path]]]:
    frames = cfg.get("frames")
    if not isinstance(frames, dict):
        return False, []
    explicit = any(role in frames for role in FRAME_ROLES)
    assignments: list[tuple[str, Path]] = []
    for role in FRAME_ROLES:
        raw = frames.get(role, [])
        if raw is None:
            continue
        if not isinstance(raw, list):
            # Invalid role definition must not silently switch to night scope.
            raise ValueError(f"frames.{role} must be a list")
        for value in raw:
            s = str(value).strip()
            if not s:
                raise ValueError(f"frames.{role} contains an empty path")
            p = Path(s.replace("\\", "/")).expanduser()
            if not p.is_absolute():
                p = data_dir / p
            assignments.append((role, p.resolve()))
    return explicit, assignments


def snapshot_fingerprint(
    cfg: Mapping[str, Any],
    data_dir: Path,
    scanned: Sequence[Path],
    excluded_abs: set[str],
    *,
    recursive: bool,
) -> str:
    """Detect changes to the selected roles, excludes, and night-wide FITS files."""
    _explicit, assigned = assigned_frames(cfg, data_dir)
    paths = set(p.resolve() for p in scanned)
    paths.update(p for _, p in assigned)
    states = []
    for p in sorted(paths, key=lambda x: x.as_posix()):
        try:
            stat = p.stat()
            states.append((p.as_posix(), int(stat.st_size), int(stat.st_mtime_ns)))
        except OSError:
            states.append((p.as_posix(), None, None))
    payload = {
        "data_dir": data_dir.resolve().as_posix(),
        "recursive": bool(recursive),
        "assigned": [(r, p.as_posix()) for r, p in assigned],
        "excluded": sorted(excluded_abs),
        "fits": states,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assess_selection(
    cfg: Mapping[str, Any],
    data_dir: Path,
    scanned: Sequence[Path],
    quarantine_entries: Sequence[Mapping[str, Any]],
    excluded_abs: set[str],
) -> dict[str, Any] | None:
    """Return effective selected-frame status, or None for legacy night-only mode."""
    explicit, assigned = assigned_frames(cfg, data_dir)
    if not explicit:
        return None

    scanned_set = {p.resolve() for p in scanned}
    entry_by_path = {
        (data_dir / str(e.get("path") or "")).resolve(): e
        for e in quarantine_entries
    }
    multiplicity = Counter(p for _, p in assigned)
    statuses = Counter()
    reasons = Counter()
    samples: dict[tuple[str, str], list[str]] = {}
    roles = Counter(role for role, _ in assigned)

    def add(severity: str, code: str, path: str) -> None:
        key = (severity, code)
        reasons[key] += 1
        examples = samples.setdefault(key, [])
        if len(examples) < 5 and path not in examples:
            examples.append(path)

    if not assigned:
        statuses["ERROR"] += 1
        add("ERROR", "NO_ASSIGNED_FRAMES", "<config>")

    if "obj" in cfg.get("frames", {}) and not cfg["frames"]["obj"]:
        statuses["ERROR"] += 1
        add("ERROR", "NO_SCIENCE_FRAMES", "<config>")

    for role, path in assigned:
        path_text = path.as_posix()
        severity = "OK"
        if multiplicity[path] > 1:
            severity = "ERROR"
            add("ERROR", "DUPLICATE_ROLE_ASSIGNMENT", path_text)
        if path not in scanned_set or not path.is_file():
            severity = "ERROR"
            add("ERROR", "ASSIGNED_FRAME_NOT_SCANNED", path_text)
        elif str(path) in excluded_abs:
            severity = "ERROR"
            add("ERROR", "ASSIGNED_FRAME_EXCLUDED", path_text)
        else:
            entry = entry_by_path.get(path)
            if entry:
                ticket_status = str(entry.get("severity") or "UNKNOWN").upper()
                if ticket_status not in {"WARN", "ERROR", "INFO"}:
                    severity = "ERROR"
                    add("ERROR", "UNKNOWN_FRAME_STATUS", path_text)
                else:
                    if ticket_status == "ERROR":
                        severity = "ERROR"
                    elif ticket_status == "WARN" and severity != "ERROR":
                        severity = "WARN"
                    for code in entry.get("codes", []):
                        add(ticket_status, str(code), path_text)
        statuses[severity] += 1

    status = "ERROR" if statuses["ERROR"] else ("WARN" if statuses["WARN"] else "OK")
    reason_items = [
        {
            "severity": severity,
            "code": code,
            "n": count,
            "sample_paths": samples.get((severity, code), []),
        }
        for (severity, code), count in sorted(
            reasons.items(),
            key=lambda item: (
                {"ERROR": 0, "WARN": 1, "INFO": 2}.get(item[0][0], 3),
                -item[1],
                item[0][1],
            ),
        )
    ]
    return {
        "status": status,
        "n_assigned": len(assigned),
        "n_ok": statuses["OK"],
        "n_warn": statuses["WARN"],
        "n_error": statuses["ERROR"],
        "roles": dict(sorted(roles.items())),
        "reasons": reason_items,
    }
