# RELEASING — Scorpipe

Короткий, практичный ритуал релиза без ручного “подкручивания версии”.
Релизы делает **Release Please** (Release PR → merge → published GitHub Release → Windows артефакты).

---

## 0) Один раз настроить (после внедрения Release Please)

1) Добавьте секрет репозитория `RELEASE_PLEASE_TOKEN` (fine‑grained PAT).
2) Включите: **Settings → Actions → General → “Allow GitHub Actions to create and approve pull requests”**.
3) Убедитесь, что workflow `release-please` зелёный.

---

## 1) Как выпускать релиз (обычный день)

### Шаг A — разработка
- Делайте изменения через PR.
- PR title должен быть **Conventional** (`fix: ...`, `feat: ...`, `feat!: ...`).
- Merge делайте через **Squash & Merge**.

### Шаг B — Release PR
Release Please автоматически откроет PR вида “release …”.
В нём будут:
- обновление версии,
- обновление `CHANGELOG.md`.

### Шаг C — публикация релиза
1) Проверьте CI на Release PR.
2) **Squash & Merge** Release PR в `main`.
3) Release Please опубликует GitHub Release с тегом `vX.Y.Z` (publish сразу).

После публикации автоматически запустится `Windows Release` и прикрепит артефакты к релизу.

---

## 2) Что должно появиться в GitHub Release

Assets (имена без `v`, версия = `X.Y.Z`):
- `ScorpioPipe-Setup-x64-X.Y.Z.exe`
- `Scorpipe-Windows-x64-X.Y.Z.zip`
- `SHA256SUMS.txt`
- `*.sbom.spdx.json` для каждого артефакта

Attestations (на странице релиза):
- SBOM attestation
- Build provenance attestation

---

## 3) Если что-то пошло не так

- Если Release PR не появляется: проверьте, что PR title/коммиты соответствуют Conventional Commits и что Release Please workflow зелёный.
- Если Windows Release не стартует: проверьте, что релиз **published** (не draft) и что тег имеет вид `vX.Y.Z`.
- Если “installer not found”: значит mismatch имён/путей (смотрите шаги сборки Inno Setup в `Windows Release` логах).

---

## 4) Мини‑чек перед нажатием Merge на Release PR

- [ ] `CI` зелёный.
- [ ] В `CHANGELOG.md` корректно описаны изменения.
- [ ] Версия в `src/scorpio_pipe/version.py` соответствует Release PR.
