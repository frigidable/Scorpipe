# Scorpio Pipe / Scorpipe

Пайплайн первичной редукции **длиннощелевых спектров** (long‑slit) для инструментов **SCORPIO‑1 / SCORPIO‑2** на 6‑м телескопе БТА (САО РАН).

Проект объединяет:
- **инспекцию ночи** (классификация кадров по FITS‑заголовкам),
- **автосборку конфигурации** под выбранный объект/настройку прибора,
- базовые калибровки (**superbias / superflat / flatfield**),
- очистку **cosmics** (стековый MAD/median‑подход),
- построение **superneon** + детекцию линий,
- интерактивную **LineID GUI** (ручные пары x↔λ + библиотека пар),
- **wavesolution** (1D/2D продукты: RMS, λ‑карта),
- научное ядро v5.x: **sky_sub → linearize → stack2d → extract1d**,
- **QC и воспроизводимость** (manifest, отчёт, timings).

> В терминах “дойти до научного результата”: пайплайн в активной разработке, но уже даёт воспроизводимый, управляемый и расширяемый workflow.  
> Это «скелет, который не стыдно наращивать» — как металлокаркас будущей обсерватории: строгий, но с запасом прочности.


## Воспроизводимость

В каждом **GitHub Release** публикуется не только сам артефакт, но и «пакет доверия» — чтобы результат можно было
проверить и воспроизвести:

- **Assets**: `SHA256SUMS.txt` и `*.sbom.spdx.json` (SBOM для **каждого** артефакта: `*.exe`, `*.zip`, и при наличии — `*.whl`, `*.tar.gz`).
- **Attestations (provenance/SBOM)**: GitHub Artifact Attestations, создаются workflow релиза; смотреть в интерфейсе релиза/Actions или проверять через `gh attestation verify`.

Подробности: [`docs/RELEASING.md`](docs/RELEASING.md)


---

## Кому это нужно

- **Наблюдателям SCORPIO**: быстро разобраться с ночью, собрать конфиг и получить первые продукты (2D/1D) с контролем параметров.
- **Разработчикам**: стабильная структура work‑директории, единый реестр продуктов, CLI/GUI вокруг одной логики стадий, удобная база для научных улучшений.

---

## Содержание

- [Быстрый старт (GUI)](#быстрый-старт-gui)
- [Быстрый старт (CLI)](#быстрый-старт-cli)
- [Установка](#установка)
- [Структура work‑директории](#структура-workдиректории)
- [Стадии пайплайна](#стадии-пайплайна)
- [Конфигурация](#конфигурация)
- [Продукты (outputs)](#продукты-outputs)
- [Воспроизводимость](#воспроизводимость)
- [Как цитировать](#как-цитировать)
- [Диагностика и типовые проблемы](#диагностика-и-типовые-проблемы)
- [Разработка и сборка Windows‑инсталлятора](#разработка-и-сборка-windowsинсталлятора)
- [Материалы и благодарности](#материалы-и-благодарности)

---

## Быстрый старт (GUI)

### 1) Установить
На Windows для обычного пользователя рекомендуемый путь — **готовый установщик из Releases** (см. раздел [Установка](#установка)).

### 2) Запустить Scorpipe
После установки запускайте **Scorpipe** из Start Menu (или ярлыка).

### 3) Выбрать ночь и объект
1. На странице проекта укажите папку с данными ночи (где лежат FITS).
2. Нажмите **Inspect** (инспекция ночи).
3. Выберите объект (или набор, если у вас несколько целей в ночь).
4. Нажмите **Create New Config** (создаст `config.yaml` в work‑директории).

### 4) Запустить стадии
Обычно порядок такой:
1. `manifest`
2. `superbias`
3. `cosmics`
4. `superflat` → `flatfield` (опционально, но рекомендуется перед sky)
5. `superneon`
6. `lineid_prepare` → **LineID GUI** (привязка) → сохранить пары
7. `wavesolution`
8. `sky`
9. `linearize`
10. `stack2d`
11. `extract1d`
12. `qc_report`

GUI использует режим **resume**: если продукт уже существует и хэш стадии совпадает, стадия будет пропущена.

---

## Быстрый старт (CLI)

CLI удобен для быстрых проверок и автоматизации.

### Команды

```bash
# версия
scorpio-pipe version

# инспекция ночи (создаст nightlog)
scorpio-pipe inspect --data-dir <PATH_TO_NIGHT>

# валидация конфига
scorpio-pipe validate --config <WORK_DIR>/config.yaml

# диагностика окружения/ресурсов (опционально с автофиксом)
scorpio-pipe doctor --config <WORK_DIR>/config.yaml --fix

# список ожидаемых продуктов для данного config
scorpio-pipe products --config <WORK_DIR>/config.yaml

# автоконфиг + (опционально) запуск неинтерактивных стадий
scorpio-pipe run --data-dir <PATH_TO_NIGHT> --object "<OBJECT_NAME>" --execute
```

> Под капотом пайплайн использует `doit` (см. `workflow/dodo.py`). GUI — это «штурвал», CLI — «автопилот».

---

## Установка

### Вариант A (рекомендуется): Windows инсталлятор из Releases
1. Откройте вкладку **Releases** вашего репозитория.
2. Скачайте установщик вида `ScorpioPipe-Setup-x64-<версия>.exe` (версия соответствует тегу `vX.Y.Z`).
3. Запустите установщик.
4. Запускайте **Scorpipe** из Start Menu.

Если нужна portable‑сборка без установки — скачайте `Scorpipe-Windows-x64-<версия>.zip`.

Важно: **Code → Download ZIP** на GitHub скачивает *исходники* и **не** содержит установщик.

Подробности: [`INSTALL.md`](INSTALL.md), [`GET_SETUP_EXE.md`](docs/GET_SETUP_EXE.md).

### Вариант B (разработчик): установка из исходников (Python)
Требования: **Python ≥ 3.10**.

```bash
python -m venv .venv
python -m pip install -U pip

# минимально (CLI)
python -m pip install -e .

# GUI + научные зависимости
python -m pip install -e ".[gui,science]"

# dev‑инструменты (тесты/ruff)
python -m pip install -e ".[gui,science,dev]"
```

---

## Структура work‑директории

Пайплайн пишет результаты в выбранную рабочую папку (work dir). Внутри создаётся стандартная структура (v5.17+):

```
work/
  raw/        # (опционально) ссылки/копии исходников
  calibs/     # superbias/superflat и др.
  science/    # промежуточные научные шаги (если применимо)
  products/   # стабильные научные продукты (lin/sky/stack/spec)
  qc/         # отчёты, manifest, timings
  # legacy‑зеркала (для обратной совместимости):
  calib/
  report/
```

---

## Стадии пайплайна

Ниже — «канонический» набор стадий (имена соответствуют runner/CLI):

### Calibs / preprocessing
- **manifest** — манифест воспроизводимости (входы/параметры/версии).
- **superbias** — суммарный bias.
- **superflat** — суммарный flat (если используется).
- **flatfield** — применение flatfield (если включено).
- **cosmics** — маскирование/замена космиков по стеку экспозиций.

### Wavesolution
- **superneon** — суммарный неон + детекция кандидатов линий.
- **lineid_prepare** — подготовка к LineID (таблицы, preview, путь wavesol).
- **LineID GUI (интерактивно)** — ручные пары x↔λ, библиотека пар.
- **wavesolution** — построение волнового решения (1D/2D продукты).

### Science core (v5.x)
- **linearize** — линеаризация (preview + per‑exposure дерево).
- **sky** — вычитание неба (Kelson‑like baseline: S(λ) + a(y), b(y)).
- **stack2d** — суммирование 2D (опционально с y‑alignment).
- **extract1d** — извлечение 1D спектра.

### QC
- **qc_report** — отчёт (HTML + JSON) и сводные метрики.

---

## Конфигурация

Конфиг хранится в `config.yaml` (создаётся в GUI или через `scorpio-pipe run`).

Ключевые идеи:
- `frames` описывает выбранные файлы (science/calibs) и setup (`frames.__setup__`).
- для каждой стадии есть свой раздел параметров (`cosmics`, `superneon`, `wavesol`, `linearize`, `sky`, `stack2d`, `extract1d`…).
- параметры в GUI применяются через **Apply**, и только затем используются **Run**.

Полезные команды:
- `scorpio-pipe validate --config ...` — валидация + предупреждения о типичных ошибках.
- `scorpio-pipe products --config ...` — список ожидаемых файлов (и где они будут лежать).

---

## Продукты (outputs)

Пайплайн поддерживает явный «реестр продуктов», который используют UI/QC/CLI.

Главные ожидаемые артефакты:

### QC / служебные
- `work/qc/manifest.json` — манифест воспроизводимости
- `work/qc/products_manifest.json` — индекс продуктов (в т.ч. per‑exposure деревья)
- `work/qc/index.html`, `work/qc/qc_report.json` — отчёт
- `work/qc/timings.json` — тайминги стадий

### Calibs
- `work/calibs/superbias.fits`
- `work/calibs/superflat.fits` (если используется)

### Wavesolution
- `work/superneon/superneon.fits` (+ PNG preview)
- `work/lineid/hand_pairs.txt` — сохранённые ручные пары
- `work/08_wavesol/` (или `work/08_wavesol/<disperser>/`) — продукты решения (JSON/PNG/λ‑карта)

### Science
- `work/10_linearize/lin_preview.fits` (+ PNG)
- `work/09_sky/preview.fits` (+ PNG)
- `work/11_stack/stacked2d.fits` (+ coverage.png)
- `work/12_extract/spec1d.fits` (+ PNG)

> Точный список под ваш конфиг: `scorpio-pipe products --config <...>`.

---

## Диагностика и типовые проблемы

- **“setup.bat мигнул и исчез”** → используйте `setup_debug.bat` (пишет `setup_debug.log`).  
  См. [`SETUP_TROUBLESHOOTING.md`](docs/SETUP_TROUBLESHOOTING.md).

- **Проблемы с ресурсами/путями/параметрами** →  
  `scorpio-pipe doctor --config <...> --fix`

- **FITS preview показывает “No image HDU”** → в v5.25 исправлена типичная ситуация со SCORPIO‑заголовками и scaling‑ключами; если всё ещё воспроизводится — приложите пример FITS + лог.

---

## Разработка и сборка Windows‑инсталлятора

Ключевые файлы:
- `setup.bat` / `setup_debug.bat` — сборка/запуск на Windows
- `packaging/windows/scorpipe.spec` — PyInstaller
- `packaging/windows/scorpipe.iss` — Inno Setup (установщик)
- `.github/workflows/windows_release.yml` — сборка релиз‑артефактов (CI)

Быстрые команды (из корня проекта):
```bat
setup.bat
setup.bat --build
setup.bat --installer
```

---


## Как цитировать

На странице репозитория GitHub будет доступна кнопка **“Cite this repository”** — она формируется автоматически из файла `CITATION.cff`.

- Быстрый способ: нажмите **Cite this repository** → выберите формат (BibTeX / APA / …) и укажите версию релиза (tag).
- Исходник метаданных: [`CITATION.cff`](CITATION.cff).

Если вы используете Scorpipe в научной работе, пожалуйста, указывайте **версию** (релизный тег) — это делает результаты воспроизводимыми.

## Материалы и благодарности

- **SCORPIO‑2: краткое руководство наблюдателя (рабочая версия 2022‑12)** — хороший контекст по режимам, решёткам, калибровкам и особенностям наблюдений.
- **SCORPIO / SCORPIO‑1: руководство 2013** — полезно для исторической совместимости и понимания эволюции instrument‑режимов.

Внутренние документы проекта:
- [`MANUAL.md`](docs/MANUAL.md) — руководство пользователя
- [`RUNBOOK.md`](docs/RUNBOOK.md) — шпаргалка “как прогнать пайплайн”
- [`CHANGELOG.md`](CHANGELOG.md) — заметки релизов и hotfix‑ов
- [`AUDIT.md`](docs/AUDIT.md) — честный аудит стадий и статуса “заглушек/готового”

---

### Примечание о лицензии
В репозитории должен быть явный файл `LICENSE`. Если его нет — добавьте, чтобы статус использования был однозначным.
