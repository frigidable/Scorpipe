# RUNBOOK — Scorpio Pipe v5.12

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
python -m doit -f workflow/dodo.py sky
python -m doit -f workflow/dodo.py extract1d
```

## 4) Структура выходных данных

Канонические продукты (v5.12+) пишутся в `work_dir/products/...`, но для обратной совместимости также копируются в старые папки.

Типично:

```
work/<run>/
  products/
    sky/
      obj_sky_sub.fits
      sky_sub_done.json
      per_exp/
        <exp>_sky_sub.fits
        <exp>_sky_spectrum.csv
    stack/
      stacked2d.fits
      stacked2d_cov.fits
    spec/
      spectrum_1d.fits
      extract1d_done.json

  sky/   (legacy mirror)
  spec/  (legacy mirror)
```

## 5) Smoke-check

После установки зависимостей можно выполнить:

```bash
python scripts/smoke_check.py --help
```
