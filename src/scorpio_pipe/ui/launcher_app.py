from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets

from scorpio_pipe.ui.launcher_window import LauncherWindow
from scorpio_pipe.ui.theme import apply_theme, load_ui_settings


def main() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Force dot as decimal separator for all numeric widgets (independent of OS locale).
    try:
        QtCore.QLocale.setDefault(QtCore.QLocale.c())
        app.setLocale(QtCore.QLocale.c())
    except Exception:
        pass

    st = load_ui_settings()
    theme_mode = st.value("ui/theme", "dark")
    apply_theme(app, mode=str(theme_mode))

    w = LauncherWindow()
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
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
