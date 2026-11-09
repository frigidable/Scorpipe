import numpy as np

from scorpio_pipe.long_setup_priors import filter_lab_lines_for_setup


def test_filter_lab_lines_for_setup_keeps_coverage_and_boosts_strong_lines():
    lam = np.array([3500.0, 3888.65, 4200.0, 5852.49, 7300.0, 9000.0], dtype=float)
    w = np.ones_like(lam)
    lam2, w2, meta = filter_lab_lines_for_setup(lam, w, setup_or_cfg='VPHG1200@540')

    assert meta['applied'] is True
    assert 3500.0 not in lam2
    assert 9000.0 not in lam2
    assert 3888.65 in lam2
    assert 5852.49 in lam2
    i = int(np.where(np.isclose(lam2, 5852.49))[0][0])
    assert w2[i] > 1.0


def test_packaged_neon_linelist_comments_do_not_corrupt_wavelengths():
    """Regression: numpy names=True previously used '# waveref: air' as a header."""
    from pathlib import Path
    from scorpio_pipe.stages.lineid_auto_backup import _load_neon_lines_csv

    resource = (
        Path(__file__).resolve().parents[1]
        / "src/scorpio_pipe/resources/neon_lines.csv"
    )
    wavelengths, weights = _load_neon_lines_csv(resource)
    assert wavelengths.size > 50
    assert np.isfinite(wavelengths).all()
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert np.all(np.diff(wavelengths) > 0)
    assert wavelengths[0] == 3888.65
    assert np.max(wavelengths) > 10000.0


def test_lineid_curated_csv_preserves_metadata_priorities_and_safe_flags(tmp_path):
    from scorpio_pipe.stages.lineid_auto_backup import _load_neon_lines_csv

    csv_file = tmp_path / "curated.csv"
    csv_file.write_text(
        "# waveref: air\n# unit: angstrom\n"
        "wavelength,intensity,priority,safe_for_auto,blend_class\n"
        "4300,2.0,2,1,strong\n"
        "4200,1.0,0,0,weak_blend\n"
        "4100,3.0,0,1,weak_blend\n",
        encoding="utf-8",
    )
    lam, w = _load_neon_lines_csv(csv_file)
    assert lam.tolist() == [4100.0, 4300.0]
    assert np.allclose(w, [1.5, 3.4])


def test_lineid_single_row_and_explicit_nanometre_units(tmp_path):
    from scorpio_pipe.stages.lineid_auto_backup import _load_neon_lines_csv

    p = tmp_path / "single.csv"
    p.write_text(
        "# unit: nm\nwavelength,intensity\n450,7\n", encoding="utf-8",
    )
    lam, w = _load_neon_lines_csv(p)
    assert lam.tolist() == [4500.0]
    assert w.tolist() == [7.0]

    p.write_text("4300,3\n4200,2\n", encoding="utf-8")
    lam, w = _load_neon_lines_csv(p)
    assert lam.tolist() == [4200.0, 4300.0]
    assert w.tolist() == [2.0, 3.0]


def test_lineid_parser_fails_closed_on_empty_nan_or_unsafe_catalogue(tmp_path):
    import pytest
    from scorpio_pipe.stages.lineid_auto_backup import _load_neon_lines_csv

    file = tmp_path / "invalid.csv"
    for source in (
        "# waveref: air\n",
        "# unit: angstrom\nwavelength\nNaN\n",
        "# unit: angstrom\nwavelength,safe_for_auto\n4300,0\n",
        "broken_name\n4200\n",
        "450\n480\n",  # nm/angstrom ambiguous without explicit units
    ):
        file.write_text(source, encoding="utf-8")
        with pytest.raises(ValueError):
            _load_neon_lines_csv(file)
