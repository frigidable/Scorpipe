#!/usr/bin/env python3
"""Safely repair one *saved* SCORPIO-1/2 wave_reference=UNKNOWN to air.

Use only on explicit user action. No FITS or other config fields are edited.
Checks actual YAML, exact block, full resulting config schema, preserves line
endings/comments, backs up exact original bytes outside the workspace.

Usage: python scripts/qa/sc1_repair_air_wave_reference.py config.yaml qa_dir
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


WAVEREF_LINE = re.compile(
    r"(?P<prefix>[ \t]+wave_reference[ \t]*:[ \t]*)"
    r"(?P<quote>['\"]?)(?i:UNKNOWN)(?P=quote)"
    r"(?P<tail>[ \t]*(?:#.*)?)\Z"
)
PROJECT_LINE = re.compile(r"project:[ \t]*(?:#.*)?\Z")


def prepare_patch(original: bytes) -> tuple[bytes, str]:
    text = original.decode("utf-8-sig")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or not isinstance(doc.get("project"), dict):
        raise ValueError("Expected a project: YAML mapping")
    current = str(doc["project"].get("wave_reference") or "").strip().lower()
    if current == "air":
        return original, "ALREADY_AIR"
    if current != "unknown":
        raise ValueError(f"Refuse to edit project.wave_reference={current!r}; expected UNKNOWN")

    lines = text.splitlines(keepends=True)
    project_lines = [
        i for i, line in enumerate(lines)
        if PROJECT_LINE.fullmatch(line.rstrip("\r\n"))
    ]
    if len(project_lines) != 1:
        raise ValueError("Expected exactly one top-level 'project:' block")
    start = project_lines[0]
    end = len(lines)
    for i in range(start + 1, len(lines)):
        x = lines[i].rstrip("\r\n")
        if x and not x[0].isspace() and not x.startswith("#"):
            end = i
            break

    matches = []
    for i in range(start + 1, end):
        line = lines[i]
        body = line.rstrip("\r\n")
        match = WAVEREF_LINE.fullmatch(body)
        if match:
            matches.append((i, match, line[len(body):]))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one project.wave_reference: UNKNOWN line; found {len(matches)}"
        )
    i, m, newline = matches[0]
    line = m.group("prefix") + m.group("quote") + "air" + m.group("quote")
    lines[i] = line + m.group("tail") + newline
    patched = "".join(lines).encode("utf-8")
    if original.startswith(b"\xef\xbb\xbf"):
        patched = b"\xef\xbb\xbf" + patched

    after = yaml.safe_load(patched.decode("utf-8-sig"))
    if after["project"]["wave_reference"] != "air":
        raise RuntimeError("Patch did not set expected air reference")
    expected = yaml.safe_load(text)
    expected["project"]["wave_reference"] = "air"
    if after != expected:
        raise RuntimeError("Patch would alter other YAML values; refusing")
    return patched, "PATCH_TO_AIR"


def repair(config: Path, backup_dir: Path) -> dict:
    config = config.expanduser().resolve(strict=True)
    backup_dir = backup_dir.expanduser().resolve()
    if config.name.lower() != "config.yaml":
        raise ValueError("Only a saved config.yaml can be changed")
    original = config.read_bytes()
    patched, mode = prepare_patch(original)
    report = {
        "status": mode,
        "config": str(config),
        "old_sha256": hashlib.sha256(original).hexdigest(),
        "backup": None,
        "new_sha256": None,
        "field_changed": "project.wave_reference" if mode == "PATCH_TO_AIR" else None,
        "other_yaml_fields_changed": False,
    }
    from scorpio_pipe.config import load_config_any
    from scorpio_pipe.typed_config import build_typed_config

    # Validate *in memory* before writing the file.
    cfg = load_config_any(config)
    cfg.setdefault("project", {})["wave_reference"] = "air"
    build_typed_config(cfg, config_path=config)
    if mode == "ALREADY_AIR":
        report["new_sha256"] = report["old_sha256"]
        return report

    # Only operate when no stage has started in the saved working directory.
    wd = Path(str(cfg.get("work_dir") or config.parent)).resolve()
    if wd != config.parent:
        raise RuntimeError("Refusing config edit outside its own run root")
    if list(wd.glob("??_*")) or list((wd / "products").glob("??_*")):
        raise RuntimeError("Stage directories exist; refusing pre-run config repair")
    if config.read_bytes() != original:
        raise RuntimeError("Config changed during validation; refusing repair")

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_dir / (
        f"sc1_config_before_air_{stamp}_{report['old_sha256'][:10]}.yaml.bak"
    )
    with backup.open("xb") as f:
        f.write(original)
    report["backup"] = str(backup)

    # Atomic replace on same filesystem; on failure original remains untouched.
    tmp = config.with_name(config.name + ".airfix.tmp")
    try:
        with tmp.open("xb") as f:
            f.write(patched)
        if config.read_bytes() != original:
            raise RuntimeError("Config changed since backup; refusing atomic replace")
        os.replace(tmp, config)
        effective = load_config_any(config)
        build_typed_config(effective, config_path=config)
        if config.read_bytes() != patched:
            raise RuntimeError("Config differs from expected one-line patch")
    except BaseException:
        # If the patch was written, restore exact original backup bytes.
        if config.exists() and config.read_bytes() == patched:
            restore = config.with_name(config.name + ".airfix.rollback.tmp")
            try:
                with restore.open("xb") as f:
                    f.write(original)
                os.replace(restore, config)
            finally:
                restore.unlink(missing_ok=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)

    report["status"] = "CONFIG_AIR_PATCHED_AND_VALIDATED"
    report["new_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    return report


def selftest():
    x = (
        b"data_dir: D:/raw\r\nproject:\r\n  wave_reference: 'UNKNOWN' # old\r\n"
        b"  barycentric_correction: 'OFF'\r\ncalib:\r\n  bias_policy: strict\r\n"
    )
    fixed, state = prepare_patch(x)
    assert state == "PATCH_TO_AIR"
    assert b"  wave_reference: 'air' # old\r\n" in fixed
    assert fixed.replace(b"'air'", b"'UNKNOWN'") == x
    assert prepare_patch(fixed) == (fixed, "ALREADY_AIR")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 3:
        result = repair(Path(sys.argv[1]), Path(sys.argv[2]))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        raise SystemExit("Usage: script.py --selftest | config.yaml QA_backup_directory")
