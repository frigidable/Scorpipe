"""Robust GUI bootstrap for Windows (and PyInstaller).

Problem: when the UI is launched as a "windowed" executable (no console),
any early exception can be effectively invisible to the user, leading to the
classic symptom "process appears in Task Manager but no window opens".

This module ensures:
  - a startup log file is written to a predictable per-user location
  - uncaught exceptions are captured with a traceback (and shown via MessageBox on Windows)
  - Qt plugin paths are set correctly in frozen (PyInstaller) builds
"""

from __future__ import annotations

import os
import sys
import traceback
import logging
import faulthandler
from pathlib import Path
from datetime import datetime


def _user_log_dir() -> Path:
    # Prefer LOCALAPPDATA on Windows, else fall back to ~/.local/share.
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        p = Path(base) / "Scorpipe" / "logs"
    else:
        p = Path.home() / ".local" / "share" / "Scorpipe" / "logs"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        # Last resort: temp directory
        import tempfile
        p = Path(tempfile.gettempdir()) / "Scorpipe" / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p


def _setup_file_logging(log_path: Path) -> None:
    # A minimal file logger that won't depend on Rich / Qt.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called twice.
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path:
            return

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    # Also log Python warnings.
    logging.captureWarnings(True)


def _show_messagebox(title: str, message: str) -> None:
    # Best-effort: show a native Windows MessageBox, otherwise print.
    if os.name != "nt":
        try:
            print(f"{title}: {message}", file=sys.stderr)
        except Exception:
            pass
        return

    try:
        import ctypes  # noqa: PLC0415

        MB_ICONERROR = 0x10
        MB_OK = 0x0
        ctypes.windll.user32.MessageBoxW(None, str(message), str(title), MB_OK | MB_ICONERROR)
    except Exception:
        try:
            print(f"{title}: {message}", file=sys.stderr)
        except Exception:
            pass


def _configure_qt_plugin_paths() -> None:
    # In PyInstaller onefile builds, Qt plugins live inside sys._MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    base = Path(meipass)
    # Try common PySide6 layouts.
    candidates = [
        base / "PySide6" / "plugins",
        base / "PySide6" / "Qt" / "plugins",
        base / "Qt" / "plugins",
    ]
    plugins_dir = next((p for p in candidates if p.exists()), None)
    if plugins_dir:
        # Qt respects these env vars.
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins_dir))
        # Platform plugins (qwindows.dll) are usually under plugins/platforms
        plat = plugins_dir / "platforms"
        if plat.exists():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plat))

    # Ensure Windows can locate bundled DLLs.
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(base))
    except Exception:
        pass


def _install_excepthook(log_path: Path) -> None:
    def _hook(exc_type, exc, tb):
        try:
            msg = "".join(traceback.format_exception(exc_type, exc, tb))
            logging.getLogger("scorpipe.gui").error("Uncaught exception:\n%s", msg)
            _show_messagebox(
                "Scorpipe: ошибка запуска",
                f"Приложение не смогло запуститься.\n\nЛог: {log_path}\n\n{msg[-1200:]}",
            )
        finally:
            # Let default handler run too (in case we're in console mode).
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


def run_gui() -> None:
    log_dir = _user_log_dir()
    log_path = log_dir / "scorpipe_gui.log"

    # Add a clear session header for each start.
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"Scorpipe GUI start: {datetime.now().isoformat()}\n")
            f.write(f"argv: {sys.argv!r}\n")
            f.write(f"frozen: {getattr(sys, 'frozen', False)}  _MEIPASS: {getattr(sys, '_MEIPASS', None)!r}\n")
    except Exception:
        pass

    _setup_file_logging(log_path)
    _configure_qt_plugin_paths()
    _install_excepthook(log_path)

    # Capture hard crashes / segfaults into the log when possible.
    try:
        faulthandler.enable(open(log_path, "a", encoding="utf-8"))
    except Exception:
        pass

    try:
        from scorpio_pipe.ui.launcher_app import main  # noqa: PLC0415

        main()
    except SystemExit:
        raise
    except Exception:
        # Force it through excepthook for MessageBox + logging.
        exc_type, exc, tb = sys.exc_info()
        assert exc_type is not None and exc is not None and tb is not None
        sys.excepthook(exc_type, exc, tb)
        raise


# Convenience: allow `python -m scorpio_pipe.ui.bootstrap`
if __name__ == "__main__":
    run_gui()
