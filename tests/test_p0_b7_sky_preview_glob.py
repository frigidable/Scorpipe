from __future__ import annotations

import fnmatch

from scorpio_pipe.pipeline.engine import STAGE_SPECS


def test_sky_stage_validates_preview_fits_name() -> None:
    """P0-B7 regression: sky stage writes preview.fits and engine must validate it."""

    spec = STAGE_SPECS["sky"]
    globs = tuple(spec.validate_globs or ())

    assert globs, "sky StageSpec.validate_globs must not be empty"

    assert any(fnmatch.fnmatch("preview.fits", pat) for pat in globs), (
        "sky StageSpec.validate_globs must match the produced preview product name 'preview.fits'. "
        f"validate_globs={globs!r}"
    )
