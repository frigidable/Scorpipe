"""Regression: NumPy must not be shadowed inside the Wavesolution stage.

Python treats an import anywhere in a function as an assignment to that name
throughout the entire function, even if the import occurs in a late try block.
That previously made the first np.sum() raise UnboundLocalError.
"""

from scorpio_pipe.stages.wavesolution import build_wavesolution


def test_wavesolution_uses_module_numpy_not_a_local_import() -> None:
    code = build_wavesolution.__code__
    assert "np" not in code.co_varnames, (
        "build_wavesolution shadows module-level NumPy alias, "
        "which makes earlier np calls raise UnboundLocalError"
    )
    assert "np" in code.co_names, "Expected NumPy use from module scope"
