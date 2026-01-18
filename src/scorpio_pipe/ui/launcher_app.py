from __future__ import annotations

"""GUI launcher.

This module is intentionally **lightweight at import time**.

In the Windows installer build, Scorpipe is packaged as a *windowed* executable
(no console). If startup is slow (Qt init and/or heavy scientific imports), the
user symptom is the classic:

  "process exists in Task Manager, but no window opens"

To make startup observable we:
- import Qt lazily inside :func:`main`
- show a simple splash screen while heavy imports / window construction happen
- write timestamped progress to the main GUI log (set up by ui.bootstrap)
"""

from pathlib import Path
import logging
import sys
import time
import traceback


def _log_progress(msg: str) -> None:
    # ui.bootstrap configures file logging early; this will go to scorpipe_gui.log.
    try:
        logging.getLogger("scorpipe.gui.start").info(msg)
    except Exception:
        pass


def _guess_log_path() -> Path | None:
    """Best-effort location of the GUI log file.

    This duplicates the path logic from :mod:`scorpio_pipe.ui.bootstrap` to avoid
    import cycles.
    """

    try:
        import os

        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "Scorpipe" / "logs" / "scorpipe_gui.log"
        return Path.home() / ".local" / "share" / "Scorpipe" / "logs" / "scorpipe_gui.log"
    except Exception:
        return None


def main() -> None:
    t0 = time.perf_counter()
    _log_progress("launcher_app: entering main()")

    # Lazy import Qt to keep import-time costs down.
    from PySide6 import QtCore, QtGui, QtWidgets  # noqa: PLC0415

    app = QtWidgets.QApplication.instance()
    if app is None:
        _log_progress("launcher_app: creating QApplication")
        app = QtWidgets.QApplication(sys.argv)

    # Force dot as decimal separator for all numeric widgets (independent of OS locale).
    try:
        QtCore.QLocale.setDefault(QtCore.QLocale.c())
        app.setLocale(QtCore.QLocale.c())
    except Exception:
        pass

    # --- Splash (gives immediate feedback while importing/building the main UI) ---
    splash: QtWidgets.QSplashScreen | None = None
    try:
        pm = QtGui.QPixmap(520, 320)
        pm.fill(QtGui.QColor(30, 30, 30))
        splash = QtWidgets.QSplashScreen(pm)
        splash.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        splash.showMessage(
            "Scorpipe запускается...\n(первый запуск может занять до 1-2 минут)",
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
            QtGui.QColor(220, 220, 220),
        )
        splash.show()
        app.processEvents()
    except Exception:
        splash = None

    _log_progress(f"launcher_app: Qt ready, splash={bool(splash)}")

    # Theme / settings are small, but keep them after splash for responsiveness.
    _log_progress("launcher_app: loading theme/settings")
    from scorpio_pipe.ui.theme import apply_theme, load_ui_settings  # noqa: PLC0415

    st = load_ui_settings()
    theme_mode = st.value("ui/theme", "dark")
    try:
        apply_theme(app, mode=str(theme_mode))
    except Exception:
        pass

    try:
        if splash is not None:
            splash.showMessage(
                "Загрузка интерфейса...",
                QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
                QtGui.QColor(220, 220, 220),
            )
            app.processEvents()
    except Exception:
        pass

    # Heavy import: main window (pulls in most of the pipeline + UI widgets).
    try:
        _log_progress("launcher_app: importing LauncherWindow")
        from scorpio_pipe.ui.launcher_window import LauncherWindow  # noqa: PLC0415

        _log_progress("launcher_app: constructing LauncherWindow")
        w = LauncherWindow()
    except Exception:
        tb = traceback.format_exc()
        _log_progress("launcher_app: FAILED to initialize UI\n" + tb)
        lp = _guess_log_path()
        msg = "Не удалось запустить интерфейс Scorpipe.\n\n" + tb
        if lp:
            msg += f"\n\nЛог: {lp}"
        try:
            QtWidgets.QMessageBox.critical(None, "Scorpipe", msg)
        except Exception:
            pass
        try:
            if splash is not None:
                splash.close()
        except Exception:
            pass
        raise SystemExit(1)

    # User preference: open maximized by default.
    try:
        w.showMaximized()
    except Exception:
        w.show()

    # Bring the window to the foreground (helps when launched from Explorer).
    try:
        w.raise_()
        w.activateWindow()
    except Exception:
        pass

    # If bootstrap enabled faulthandler.dump_traceback_later(), cancel it once UI is up.
    try:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass

    # Hide splash only after the window is shown.
    try:
        if splash is not None:
            splash.finish(w)
    except Exception:
        pass

    dt = time.perf_counter() - t0
    _log_progress(f"launcher_app: main window shown in {dt:.2f}s")

    # If, for some reason, the window didn't become visible, show a hint.
    try:
        if not w.isVisible():
            lp = _guess_log_path()
            hint = "Окно не появилось.\n\n"
            if lp:
                hint += f"Проверьте лог: {lp}"
            QtWidgets.QMessageBox.critical(None, "Scorpipe", hint)
    except Exception:
        pass

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
