# Scorpipe — установка (Windows)

## Вариант №1 (рекомендуется): инсталлятор из Releases

1) Откройте **GitHub Releases** и скачайте установщик вида `ScorpioPipe-Setup-x64-<версия>.exe`.
2) Запустите установщик.
3) После установки запускайте **scorpipe** из **Start Menu** (или по ярлыку).

Если нужна portable‑сборка без установки, скачайте `Scorpipe-Windows-x64-<версия>.zip` и распакуйте его.


## Вариант №2 (для разработчика): сборка из исходников

Если вы работаете с репозиторием и хотите собрать EXE локально, используйте:

- `setup.bat` (в корне проекта)
  - `setup.bat`            — install/update + build `scorpipe.exe` + run
  - `setup.bat --build`    — только build `scorpipe.exe`
  - `setup.bat --installer`— build `scorpipe.exe` **и затем** build `packaging\\windows\\Output\\ScorpioPipe-Setup-x64-<версия>.exe`

Сборка релизного инсталлятора оформлена в `packaging/windows` (PyInstaller + Inno Setup).

## Где лежат сборочные файлы

- `packaging/windows/scorpipe.spec` — PyInstaller spec для `scorpipe.exe`
- `packaging/windows/scorpipe.iss` — Inno Setup script для установщика `ScorpioPipe-Setup-x64-<версия>.exe`
- `.github/workflows/windows_release.yml` — GitHub Actions, который собирает артефакты на Windows


## Если EXE запускается, но окна нет

В версии 5.40.40 добавлен диагностический лог запуска GUI.

- Лог: `%LOCALAPPDATA%\Scorpipe\logs\scorpipe_gui.log`
- Если папка недоступна, используется `%TEMP%\Scorpipe\logs`

Если окно не открылось, откройте лог и посмотрите traceback — это обычно указывает на отсутствующий DLL/Qt-плагин или ошибку импорта.

## Документация по семантике запуска

Если вам важно, *когда* пайплайн пропускает стадию, что означает `resume/force`, и где смотреть причины (`done.json`, `manifest/stage_state.json`), см.:

- [`docs/EXECUTION_CONTRACT.md`](docs/EXECUTION_CONTRACT.md)
