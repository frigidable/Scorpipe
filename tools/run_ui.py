"""Thin entry script for PyInstaller.

PyInstaller works best when you point it to a .py file.
"""

from scorpio_pipe.ui.bootstrap import run_gui


if __name__ == "__main__":
    run_gui()
