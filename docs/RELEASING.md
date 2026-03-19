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


## Observatory-ready wording

Do not describe a build as **LONG-level** or **observatory-ready** unless the release also satisfies the gates in `docs/spec/OBSERVATORY_RELEASE_GATES.md`.
Passing the public synthetic LONG fixture alone is only a check of the harness and report plumbing.


## Release-qualification package (P5)

Before calling a build **LONG-level** or **observatory-ready**, generate a certification bundle:

```bash
scorpio-pipe certify-long-parity   --registry benchmarks/long_parity/cases/registry.json   --golden-compare golden_artifacts/perf/golden_time_budget_compare.json   --known-limitations benchmarks/long_parity/certification/known_limitations.template.md   --observer-signoff "<observer name>"   --output release_qualification
```

If the summary says `observatory_ready: false`, do not use LONG-level wording in notes or releases.
The certification summary now also prints explicit `blocker:` lines so the missing evidence is visible immediately.

To fill one pending private registry slot without hand-editing JSON files, use:

```bash
scorpio-pipe prepare-long-case   --config work/<night>/<run>/config.yaml   --long-bundle /private/long/<case>/long_bundle.json   --registry benchmarks/long_parity/cases/registry.json   --case-id <registry_slot>   --title "<human title>"   --category <reference|wavesolution|sky|extraction|geometry|calibration|repeatability>
```


## LONG-parity bootstrap helpers

Before the final certification pass you can reduce manual work with two helper commands:

```bash
scorpio-pipe inspect-long-params --long-root /path/to/LONG --output bench_work/long_setup_summary
scorpio-pipe bootstrap-night-cases --nightlog /path/to/s251216.txt --long-root /path/to/LONG --output bench_work/night_bootstrap --write-case-drafts
```

The first command extracts setup coverage / dispersion / line-table information already present in LONG.
The second command mines a night log and prepares `bootstrap-only` registry drafts. These drafts do **not** count as real certification evidence until Scorpipe and LONG observer bundles are attached.
