"""PyInstaller entry-point for the Scorpipe GUI.

This wrapper exists to keep the frozen (windowed) executable startup path as
robust as possible.
"""

from __future__ import annotations


def main() -> None:
    # Windows frozen builds (PyInstaller) + multiprocessing: prevent spawn recursion/hangs.
    try:
        import multiprocessing as _mp

        _mp.freeze_support()
    except Exception:
        pass

    from scorpio_pipe.ui.bootstrap import run_gui

    run_gui()


if __name__ == "__main__":
    main()
