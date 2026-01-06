from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.ui.launcher_window import LauncherWindow
from scorpio_pipe.ui.theme import apply_theme, load_ui_settings


def _get_log_file() -> Path:
    """Получить путь к лог-файлу ошибок."""
    log_dir = Path.home() / ".scorpipe_logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return log_dir / "gui_crash.log"


def _write_crash_log(message: str) -> None:
    """Записать сообщение об ошибке в лог."""
    try:
        log_file = _get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(message)
            f.write("\n" + "=" * 80 + "\n")
    except Exception:
        pass


def _show_error_dialog(title: str, message: str, details: str = "") -> None:
    """Показать диалог ошибки пользователю."""
    try:
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if details:
            msg_box.setDetailedText(details)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.setMinimumWidth(500)
        msg_box.exec()
    except Exception:
        # Если даже диалог не работает — просто логируем
        pass


def main() -> None:
    """Главная точка входа GUI с полной обработкой ошибок."""

    try:
        # 1. Создать QApplication
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # 2. Попытка установить локаль (некритично)
        try:
            QtCore.QLocale.setDefault(QtCore.QLocale.c())
            app.setLocale(QtCore.QLocale.c())
        except Exception as e:
            # Логируем, но не прерываем
            pass

        # 3. Загрузить параметры UI (некритично)
        try:
            st = load_ui_settings()
            theme_mode = st.value("ui/theme", "dark")
        except Exception as e:
            theme_mode = "dark"

        # 4. Применить тему (некритично)
        try:
            apply_theme(app, mode=str(theme_mode))
        except Exception as e:
            # Тема не критична
            pass

        # 5. Создать главное окно (КРИТИЧНО!)
        try:
            w = LauncherWindow()
        except Exception as e:
            error_msg = traceback.format_exc()
            _write_crash_log(f"FAILED TO CREATE LAUNCHERWINDOW:\n{error_msg}")

            # Показать диалог ошибки
            log_file = _get_log_file()
            user_message = (
                "Scorpio Pipe GUI failed to initialize.\n\n"
                f"Error log saved to:\n{log_file}\n\n"
                "Please check the log file for details."
            )
            _show_error_dialog(
                "Scorpio Pipe — Initialization Error",
                user_message,
                error_msg
            )
            raise SystemExit(1)

        # 6. Показать окно
        try:
            w.showMaximized()
        except Exception as e:
            try:
                w.show()
            except Exception:
                pass

        # 7. Запустить event loop
        exit_code = app.exec()
        raise SystemExit(exit_code)

    except SystemExit as e:
        # Re-raise SystemExit (не логируем как ошибку)
        raise
    except Exception as e:
        # Ловим ВСЕ необработанные исключения
        error_msg = traceback.format_exc()
        _write_crash_log(f"UNCAUGHT EXCEPTION:\n{error_msg}")

        log_file = _get_log_file()
        user_message = (
            "Scorpio Pipe encountered an unexpected error.\n\n"
            f"Error log saved to:\n{log_file}\n\n"
            "Please check the log file and restart the application."
        )

        try:
            _show_error_dialog(
                "Scorpio Pipe — Critical Error",
                user_message,
                error_msg
            )
        except Exception:
            # Даже если диалог не сработал, выходим gracefully
            pass

        raise SystemExit(1)


if __name__ == "__main__":
    main()