import sys

import pytest


pytest.importorskip("PySide6")


def test_launcher_window_import_is_lightweight() -> None:
    """Importing the GUI module must not pull in heavy scientific stack.

    This protects frozen-build startup time: the splash should disappear quickly
    and the main window should become visible even on slow machines.
    """

    # Ensure a clean import.
    sys.modules.pop("scorpio_pipe.ui.launcher_window", None)
    sys.modules.pop("scorpio_pipe.inspect", None)
    sys.modules.pop("scorpio_pipe.ui.pipeline_runner", None)

    __import__("scorpio_pipe.ui.launcher_window")

    assert "scorpio_pipe.inspect" not in sys.modules
    assert "scorpio_pipe.ui.pipeline_runner" not in sys.modules
