from __future__ import annotations

"""Dataset manifest schema validation (P0-B4).

This module is the *single* place that defines the on-disk contract for
``dataset_manifest.json``.

Goals
-----
- Fail fast on inconsistent/unknown schema versions.
- Keep validation lightweight (Pydantic only; no Astropy).
- Provide clear error messages for operators.

Schema
------
Current schema is **v3**: ``scorpio-pipe.dataset-manifest.v3``.

Backward compatibility:
- v1/v2 manifests can still be loaded by :class:`~scorpio_pipe.dataset.manifest.DatasetManifest`
  (it auto-populates list-valued fields).
- However, *validators* can be configured to require v3 for strict workflows.
"""

import json
from pathlib import Path
from typing import Any

from scorpio_pipe.dataset.manifest import DatasetManifest


SCHEMA_ID_V3 = "scorpio-pipe.dataset-manifest.v3"
SCHEMA_VERSION_V3 = 3


class DatasetManifestSchemaError(ValueError):
    """Raised when ``dataset_manifest.json`` violates the schema contract."""


def validate_dataset_manifest(payload: Any, *, require_v3: bool = True) -> DatasetManifest:
    """Validate payload and return a parsed :class:`DatasetManifest`.

    Parameters
    ----------
    payload:
        Parsed JSON object.
    require_v3:
        If True, require ``schema``/``schema_version`` to match v3.
        If False, allow v1/v2 as long as it parses.
    """

    man = DatasetManifest.model_validate(payload)

    schema_id = str(getattr(man, "schema_id", "") or "")
    schema_ver = int(getattr(man, "schema_version", 0) or 0)

    if require_v3:
        if schema_id != SCHEMA_ID_V3 or schema_ver != SCHEMA_VERSION_V3:
            raise DatasetManifestSchemaError(
                f"Unsupported dataset_manifest schema: schema={schema_id!r}, schema_version={schema_ver}. "
                f"Expected {SCHEMA_ID_V3!r} (v{SCHEMA_VERSION_V3})."
            )

    # Minimal internal consistency checks.
    ss_ids = {str(s.science_set_id) for s in (man.science_sets or [])}
    match_ids = {str(m.science_set_id) for m in (man.matches or [])}
    missing = sorted(ss_ids - match_ids)
    extra = sorted(match_ids - ss_ids)
    if missing or extra:
        raise DatasetManifestSchemaError(
            "Inconsistent dataset_manifest: science_sets and matches disagree. "
            f"missing_matches_for={missing[:10]!r}, extra_matches_for={extra[:10]!r}"
        )

    return man


def validate_dataset_manifest_file(path: str | Path, *, require_v3: bool = True) -> DatasetManifest:
    """Read and validate a manifest from disk."""

    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise DatasetManifestSchemaError(f"Failed to read dataset_manifest JSON: {p}: {e}") from e

    try:
        return validate_dataset_manifest(payload, require_v3=require_v3)
    except DatasetManifestSchemaError:
        raise
    except Exception as e:
        raise DatasetManifestSchemaError(f"Failed to validate dataset_manifest: {p}: {e}") from e


# Backward-compatible names
SCHEMA_ID = SCHEMA_ID_V3
SCHEMA_VERSION = SCHEMA_VERSION_V3

__all__ = [
    "DatasetManifestSchemaError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SCHEMA_ID_V3",
    "SCHEMA_VERSION_V3",
    "validate_dataset_manifest",
    "validate_dataset_manifest_file",
]
