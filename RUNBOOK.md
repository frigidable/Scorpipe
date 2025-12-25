# RUNBOOK — Scorpio Pipe v5.16

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

### Полезные P1-опции (по умолчанию выключены)

В `config.yaml` (или через GUI → Advanced):

- Flexure/Δλ correction по sky-линиям (**субпиксельный** сдвиг на λ-сетке):
  - legacy-ключи (остались поддержаны):
    - `sky.flexure_enabled: true`
    - `sky.flexure_max_shift_pix: 5`  (типичный диапазон: 0–10)
  - расширенный блок (рекомендуется в v5.16+):
    ```yaml
    sky:
      flexure:
        enabled: true
        max_shift_pix: 5
        # режим для максимального S/N по линиям:
        mode: windows          # full | windows
        windows_unit: auto    # auto | A | pix (auto: Å если есть WCS, иначе px)
        windows_A:
          - [6260, 6360]       # пример: [O I] 6300 + соседство
          - [6860, 6920]       # пример: O2 band / яркие sky особенности
        # новинка v5.15: Δλ(y) модель (полином по y)
        y_dependent: true
        y_poly_deg: 1
        y_smooth_bins: 5
        y_bin: 24
        y_step: 24
        save_curve: true
    ```

- Y-alignment перед stacking (**субпиксельный** сдвиг по y):
  - legacy-ключи:
    - `stack2d.y_align_enabled: true`
    - `stack2d.y_align_max_shift_pix: 10` (типичный диапазон: 0–20)
  - расширенный блок:
    ```yaml
    stack2d:
      y_align:
        enabled: true
        max_shift_pix: 10
        mode: windows          # full | windows
        windows_unit: auto    # auto | A | pix (auto: Å если есть WCS, иначе px)
        windows_A:
          - [6500, 6620]       # пример: Halpha+[NII] (если объект с эмиссией)
        use_positive_flux: true
    ```

- Маска насыщения (v5.15+):
  - `linearize.mask_saturation: true`
  - `linearize.saturation_adu: 65535`  # опционально (если нет в header)
  - `linearize.saturation_margin_adu: 0`


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
