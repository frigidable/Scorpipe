# Contributing to Scorpio Pipe

Ниже — минимальные правила, чтобы PR проходили быстро и без сюрпризов.

## Быстрый старт (dev)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install -e ".[dev,science]"
```

## Проверки перед PR

1) Линтер
```bash
ruff check .
```

2) Тесты
```bash
pytest -q
```

3) Сборка и установка из wheel (важно для релизов)
```bash
python -m pip install -U build
python -m build
python -m pip install dist/*.whl
scorpio-pipe --help
```

## Стиль и качество

- Предпочитай маленькие PR (1 логическая тема).
- Для изменений в стадиях: добавляй/обновляй тест (или объясняй, почему тест невозможен).
- Для изменения файлов форматов/путей: обновляй `docs/RUNBOOK.md` и/или `docs/MANUAL.md`.

## Что прикладывать к issue/PR

- Версия (`scorpio-pipe --version` или `pyproject.toml`).
- ОС, Python.
- Логи запуска (консольный вывод).
- Минимальный пример данных/файлов (или структуру папок + FITS headers, если данные нельзя прикладывать).

Спасибо — каждый хороший PR экономит нам ночи на телескопе.
