# Где взять setup.exe

Если вы нажимаете **Code → Download ZIP**, GitHub скачивает **исходники** (source code). В таком архиве *не бывает* `setup.exe`.

`setup.exe` появляется только после сборки на Windows (PyInstaller → Inno Setup).

## Для обычного пользователя (без Python)

1) Откройте вкладку **Releases** на GitHub.
2) Скачайте **Release asset** `Scorpipe-Windows-x64.zip`.
3) Внутри будет `setup.exe` → запускайте его.
4) После установки запускайте **Scorpipe** из Start Menu.

Если в Releases ещё нет готового архива, откройте вкладку **Actions** → workflow `windows-release` → **Artifacts** (можно скачать `Scorpipe-Windows-x64.zip` напрямую).

## Для разработчика (локальная сборка на Windows)

1) Установите Python 3.12.
2) Установите Inno Setup 6 (чтобы был `iscc.exe`).
3) В корне проекта выполните:

**Самый простой путь:**

```bat
setup.bat --installer
```

Или вручную (PowerShell):

```powershell
python -m pip install -U pip
pip install -e ".[gui]"
pwsh packaging/windows/build.ps1
```

Результат: `packaging\windows\Output\setup.exe`.
