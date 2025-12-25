# Scorpio Pipe — Workflow Audit (v5.17)

Цель этого файла — **честно описать** текущий workflow: какие стадии реально делают вычисления и создают продукты, а какие остаются «каркасом/заглушками» (или нуждаются в научном усилении).

## Стандартный workflow (doit / workflow/dodo.py)

Ниже — перечень ключевых задач (в терминах `task_*`) и ожидаемых продуктов.

### Calibs / Preprocessing
- **superbias** → `work/calibs/superbias.fits` (+ legacy `work/calib/superbias.fits`)
- **superflat** → `work/calibs/superflat.fits` (+ legacy `work/calib/superflat.fits`)
- **clean_cosmics** → очищенные кадры (SCI/VAR/MASK) в `work/cosmics/` *(реальная стадия)*

### Wavesolution (каркас + базовые продукты)
- **lineid_prepare / lineid_solve / wavesolution** → `work/wavesol/` и продукты RMS/линии/λ‑карты
  - **Комментарий:** точность и интерактивность — зона активной разработки; структура данных стабилизирована.

### Science core (v5.17+ минимально рабочий скелет)
- **linearize** → `work/products/lin/lin_preview.fits` (+ PNG), `per_exp/*.fits`
- **sky_sub** → `work/products/sky/preview.fits` (+ PNG), `per_exp/*.fits`, `sky_model/*.fits`
- **stack2d** → `work/products/stack/stacked2d.fits` (+ `coverage.png`)
- **extract1d** → `work/products/extract/spec1d.fits` (+ PNG)

### QC / Reproducibility
- **manifest** → `work/qc/manifest.json` (+ legacy `work/report/manifest.json`)
- **qc_report** → `work/qc/index.html`, `work/qc/qc_report.json` (+ legacy `work/report/*`)
- **timings** → `work/qc/timings.json` (+ legacy `work/report/timings.json`)

## Статус «заглушек»
В v5.17 сознательно делается упор на **инфраструктуру** (единый формат I/O + структура work + QC каркас). Некоторые части остаются намеренно упрощёнными (научная оптимальность будет улучшаться в следующих релизах).

## Что добавлено в v5.17
- Единый MEF I/O: `SCI/VAR/MASK` (+ метаданные wave‑grid в заголовках)
- `work/` структура: `raw/ calibs/ science/ products/ qc/` (+ legacy алиасы)
- `smoke_test.py` на синтетике для быстрого end‑to‑end прогона
