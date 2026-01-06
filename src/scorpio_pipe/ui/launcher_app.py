from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from scorpio_pipe.ui.launcher_window import LauncherWindow
from scorpio_pipe.ui.theme import apply_theme, load_ui_settings


def _write_crash_log(exc_info: str) -> Path:
    """Запишем ошибку в лог-файл для отладки."""
    log_dir = Path.home() / ".scorpipe_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "gui_crash.log"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"CRASH at {Path.cwd()}\n")
            f.write(exc_info)
            f.write("\n" + "=" * 80 + "\n")
    except Exception:
        pass
    return log_file


def _show_error_dialog(title: str, message: str, details: str = "") -> None:
    """Показать диалог ошибки пользователю."""
    try:
        # Убедимся, что QApplication существует
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # Создаём диалог ошибки
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if details:
            msg_box.setDetailedText(details)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec()
    except Exception:
        # Если даже этот дополнительный диалог не работает — выводим в stderr
        print(f"ERROR: {title}", file=sys.stderr)
        print(f"{message}", file=sys.stderr)
        if details:
            print(f"Details:\n{details}", file=sys.stderr)


def main() -> None:
    """Главная точка входа GUI приложения с обработкой ошибок."""

    try:
        # 1. Создать QApplication (если ещё не создан)
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        # 2. Установить локаль
        try:
            QtCore.QLocale.setDefault(QtCore.QLocale.c())
            app.setLocale(QtCore.QLocale.c())
        except Exception as e:
            print(f"[Warning] Failed to set locale: {e}", file=sys.stderr)
            # Продолжаем несмотря на ошибку локали

        # 3. Загрузить параметры темы
        try:
            st = load_ui_settings()
            theme_mode = st.value("ui/theme", "dark")
        except Exception as e:
            print(f"[Warning] Failed to load settings: {e}", file=sys.stderr)
            theme_mode = "dark"

        # 4. Применить тему
        try:
            apply_theme(app, mode=str(theme_mode))
        except Exception as e:
            print(f"[Warning] Failed to apply theme: {e}", file=sys.stderr)
            # Тема не критична, продолжаем

        # 5. Создать главное окно
        try:
            w = LauncherWindow()
        except Exception as e:
            error_details = traceback.format_exc()
            log_file = _write_crash_log(error_details)
            _show_error_dialog(
                "Scorpio Pipe — Initialization Error",
                f"Failed to initialize the GUI.\n\nLog saved to:\n{log_file}",
                error_details
            )
            raise SystemExit(1)

        # 6. Показать окно
        try:
            w.showMaximized()
        except Exception:
            try:
                w.show()
            except Exception as e:
                print(f"[Error] Failed to show window: {e}", file=sys.stderr)
                raise SystemExit(1)

        # 7. Запустить event loop
        exit_code = app.exec()
        raise SystemExit(exit_code)

    except SystemExit:
        raise  # Re-raise SystemExit
    except Exception as e:
        # Ловим ВСЕ необработанные исключения
        error_details = traceback.format_exc()
        log_file = _write_crash_log(error_details)

        try:
            _show_error_dialog(
                "Scorpio Pipe — Critical Error",
                f"An unexpected error occurred.\n\nLog saved to:\n{log_file}",
                error_details
            )
        except Exception:
            print(f"CRITICAL ERROR:\n{error_details}", file=sys.stderr)
            print(f"Log file: {log_file}", file=sys.stderr)

        raise SystemExit(1)


if __name__ == "__main__":
    main()