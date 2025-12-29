# Где взять установщик Windows

Если вы нажимаете **Code → Download ZIP**, GitHub скачивает **исходники** (source code). В таком архиве *не бывает* установщика.

Установщик появляется только после сборки на Windows (PyInstaller → Inno Setup) и публикуется в **Releases**.

---

## Для обычного пользователя (без Python)

1) Откройте вкладку **Releases** на GitHub.
2) Скачайте установщик вида `ScorpioPipe-Setup-x64-<версия>.exe` (версия соответствует тегу `vX.Y.Z`).
3) Запустите установщик.
4) После установки запускайте **Scorpipe** из Start Menu.

Portable‑сборка (без установки): `Scorpipe-Windows-x64-<версия>.zip`.

---

## Для разработчика (сборка локально)

1) Подготовьте окружение (один раз):
```powershell
python -m pip install -U pip
pip install -e ".[gui]"
```

2) Соберите:
```powershell
pwsh packaging/windows/build.ps1
```

Результат:
- `packaging\windows\Output\ScorpioPipe-Setup-x64-<версия>.exe`
