# Scorpipe — установка (Windows)

## Вариант №1 (рекомендуется): setup.exe из Releases

1) Скачайте **Windows release zip** из GitHub Releases (обычно называется вроде `Scorpipe-Windows-x64.zip`).
2) Распакуйте архив.
3) Запустите **setup.exe**.
4) После установки запускайте **scorpipe** из **Start Menu** (или по ярлыку на рабочем столе, если включили его в установщике).

Важно: кнопка **Code → Download ZIP** на GitHub скачивает **исходники** (source code). В таком архиве **не будет** `setup.exe`.

Это «идеальный» сценарий: пользователю не нужен Python, venv и команды.

## Вариант №2 (для разработчика): сборка из исходников

Если вы работаете с репозиторием и хотите собрать EXE локально, используйте скрипты в:

- `scripts/windows/setup.bat` (быстрый dev-путь)
- `tools/setup.ps1` и `tools/setup_ui.ps1` (PowerShell)

Сборка релизного инсталлятора (setup.exe) оформлена в `packaging/windows`.

## Где лежат сборочные файлы

- `packaging/windows/scorpipe.spec` — PyInstaller spec для `scorpipe.exe`
- `packaging/windows/scorpipe.iss` — Inno Setup script для `setup.exe`
- `.github/workflows/windows_release.yml` — GitHub Actions, который собирает артефакты на Windows
