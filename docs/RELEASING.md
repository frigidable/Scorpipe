# RELEASING — Scorpio Pipe

Короткий, практичный чек‑лист релиза (без лишней магии). Идея простая:
**сначала** убеждаемся, что сборка/установка/тесты проходят, **потом** создаём тег/релиз.

---

## 0) Перед релизом (локально)

1) Обнови окружение
```bash
python -m pip install -U pip
```

2) Прогони линтер и тесты
```bash
ruff check .
pytest -q
```

3) Проверь, что пакет собирается и устанавливается из wheel
```bash
python -m pip install -U build
python -m build
python -m pip install dist/*.whl
scorpio-pipe --help
python -c "import importlib.metadata as m; print(m.version('scorpio-pipe'))"
```

---

## 1) Версия

- Версия задаётся в `pyproject.toml`:
  - `[project].version = "X.Y.Z"`

Правило: версия в `pyproject.toml` и заголовок в `CHANGELOG.md` должны быть синхронизированы.

---

## 2) CHANGELOG

1) Добавь секцию **в начало** файла `CHANGELOG.md`:
- `## vX.Y.Z`
- короткие пункты по изменениям (пользователь‑ориентированные + внутренние)

2) Если изменение касается UI/научных стадий — добавь 1–2 строки “почему это важно”.

---

## 3) CI / PR

- Убедись, что на GitHub зелёные:
  - `lint`
  - `tests`
  - `build_check` (сборка sdist/wheel + установка из wheel + smoke)

---

## 4) Тег и релиз на GitHub

Рекомендуемый поток:

1) Создай тег и запушь его
```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

2) Создай GitHub Release по тегу `vX.Y.Z` и вставь краткие notes из `CHANGELOG.md`.

3) Убедись, что workflow релиза публикует полный набор артефактов:
- Windows: `*.exe`, `*.zip`
- Python (если присутствуют): `*.whl`, `*.tar.gz`
- Доверие: `*.sbom.spdx.json`, `SHA256SUMS.txt` (и соответствующие attestations)

---

## 5) После релиза

- Проверь, что артефакты релиза загрузились (setup.exe / zip и т.п.).
- Проверь “пакет доверия” для артефактов:
  - в Assets есть `release/*.sbom.spdx.json` и `SHA256SUMS.txt`;
  - в `SHA256SUMS.txt` присутствуют хэши **и артефактов**, и их **SBOM**;
  - в workflow релиза успешно прошли шаги **Attest SBOM** и **Attest build provenance** (attestations).
- Если в релизе присутствуют Python‑дистрибутивы (`*.whl`, `*.tar.gz`), они должны публиковаться вместе с соответствующими SBOM/attestations (по умолчанию это уже поддержано release‑workflow’ом; важно лишь, чтобы файлы лежали в `release/`).
- Если релиз включает миграции форматов/папок: добавь заметку в `docs/RUNBOOK.md`.
