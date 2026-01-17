from __future__ import annotations

from pathlib import Path

from scorpio_pipe.dataset.builder import resolve_global_exclude


def test_p0_k_resolve_global_exclude_expands_files_and_globs(tmp_path: Path):
    # Create dummy files (no FITS parsing required for this unit test).
    (tmp_path / "a.fits").write_text("x", encoding="utf-8")
    (tmp_path / "b.fits").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.fits").write_text("x", encoding="utf-8")

    (tmp_path / "project_manifest.yaml").write_text(
        """schema: scorpio-pipe.project-manifest.v1.1
exclude:
  files: [b.fits]
  globs: [sub/*.fits, missing/*.fits]
roles: {}
""",
        encoding="utf-8",
    )

    merged, summary = resolve_global_exclude(tmp_path)

    # Exclude must include b.fits and sub/c.fits.
    merged_set = {str(Path(p)) for p in merged}
    assert str((tmp_path / "b.fits").resolve()) in merged_set
    assert str((tmp_path / "sub" / "c.fits").resolve()) in merged_set

    assert isinstance(summary, dict)
    assert summary.get("excluded_n") == 2

    rels = set(summary.get("excluded_paths") or [])
    assert "b.fits" in rels
    assert "sub/c.fits" in rels

    assert "missing/*.fits" in set(summary.get("unmatched_globs") or [])


def test_p0_k_resolve_global_exclude_merges_cli_exclude_paths(tmp_path: Path):
    (tmp_path / "a.fits").write_text("x", encoding="utf-8")

    merged, summary = resolve_global_exclude(tmp_path, exclude_paths=["a.fits"])
    assert str((tmp_path / "a.fits").resolve()) in set(merged)
    # No manifest means empty summary.
    assert summary == {}
