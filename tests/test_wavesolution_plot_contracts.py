"""Check internal Wavesolution call signatures and residual diagnostic products.

The Golden mini-night exposed successive mismatched keyword names between
helper functions and calls late in the stage, after successful scientific fits.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from scorpio_pipe.stages import wavesolution as ws


def test_wavesolution_internal_calls_use_declared_keyword_names() -> None:
    """Static contract for direct, same-module function calls (no dynamic calls)."""
    tree = ast.parse(Path(ws.__file__).read_text(encoding="utf-8"))
    funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        fn = funcs.get(node.func.id)
        if fn is None or fn.args.kwarg is not None:
            continue
        allowed = {
            arg.arg
            for arg in (
                fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
            )
        }
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in allowed:
                bad.append(
                    f"line {node.lineno}: {node.func.id}({kw.arg}=...) "
                    f"does not accept that keyword"
                )
    assert not bad, "\n".join(bad)


def test_wavesolution_residual_diagnostics_generate_pngs(tmp_path: Path) -> None:
    """Exercise diagnostic helpers with independently generated residual samples."""
    wavelengths = np.repeat(np.array([5000.0, 5020.0, 5040.0]), 20)
    y_pix = np.tile(np.arange(20.0), 3)
    dlam = 0.025 * np.sin(y_pix / 4.0)
    used = np.ones(wavelengths.shape, dtype=bool)
    used[::7] = False

    versus_lambda = tmp_path / "residuals_vs_lambda.png"
    versus_y = tmp_path / "residuals_vs_y.png"

    ws._plot_residuals_vs_lambda(
        lam_key=np.round(wavelengths, 3),
        lam_all=wavelengths,
        dlam_all=dlam,
        used_mask=used,
        rejected_lines_A=[],
        outpng=versus_lambda,
        title="Synthetic residuals vs wavelength",
    )
    ws._plot_residuals_vs_y(
        y_all=y_pix,
        dlam_all=dlam,
        used_mask=used,
        outpng=versus_y,
        title="Synthetic residuals vs slit coordinate",
    )
    assert versus_lambda.stat().st_size > 0
    assert versus_y.stat().st_size > 0
