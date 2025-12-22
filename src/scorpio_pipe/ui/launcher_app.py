from __future__ import annotations

import sys

from PySide6 import QtWidgets

from scorpio_pipe.ui.launcher_window import LauncherWindow
from scorpio_pipe.ui.theme import apply_theme, load_ui_settings


def main() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    st = load_ui_settings()
    theme_mode = st.value("ui/theme", "dark")
    apply_theme(app, mode=str(theme_mode))

    w = LauncherWindow()
    w.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
