# RUNBOOK — Scorpio Pipe v5.13

Этот файл — короткая шпаргалка «как прогнать пайплайн» и где искать продукты.

## 1) Установка (dev-режим)

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e .[science,gui]
```

## 2) Запуск GUI

```bash
python -m scorpio_pipe.ui.launcher_app
```

В GUI:
1. **Inspect** — выбери `data_dir` и объект.
2. Нажми **Save config** (получишь `work/.../config.yaml`).
3. Дальше запускай стадии кнопками **Run ...**.

## 3) CLI / headless запуск (doit)

В корне проекта:

```bash
python -m pip install -e .[science,gui,dev]
python -m doit -f workflow/dodo.py inspect
python -m doit -f workflow/dodo.py wavesolution
python -m doit -f workflow/dodo.py linearize
python -m doit -f workflow/dodo.py sky_sub
python -m doit -f workflow/dodo.py stack2d
python -m doit -f workflow/dodo.py extract1d
```

## 4) Структура выходных данных

Канонические продукты (v5.13+) пишутся в `work_dir/products/...`. Часть стадий дополнительно создаёт legacy mirror в старых папках, чтобы не ломать привычные пути.

Типично:

```
work/<run>/
  products/
    lin/
      lin_preview.fits
      per_exp/              # per-exposure rectified SCI/VAR/MASK
    sky/
      per_exp/              # per-exposure sky-subtracted frames
    stack/
      stacked2d.fits         # final stacked 2D (SCI/VAR/MASK/COV)
      coverage.png
    spec/
      spec1d.fits            # final 1D (FLUX/VAR/MASK)
      spec1d.png
      trace.json
```

## 5) Smoke-check

После установки зависимостей можно выполнить:

```bash
python scripts/smoke_check.py --help
```
