# VERSIONING — Scorpipe

В проекте используется **SemVer** (MAJOR.MINOR.PATCH) и **Conventional Commits** в заголовках PR (потому что merge делается через **Squash** и PR title становится финальным commit message).

Release Please читает историю коммитов и автоматически:
- поднимает версию,
- обновляет `CHANGELOG.md`,
- создаёт **Release PR**,
- после merge публикует GitHub Release с тегом `vX.Y.Z`.

---

## Правила повышения версии

### PATCH (x.y.Z) — исправления
Используйте `fix:` для багфиксов без новых возможностей и без ломания интерфейсов.

Примеры PR title:
- `fix(qc): handle missing unit keyword`
- `fix(windows): correct installer path`

### MINOR (x.Y.z) — новые возможности без breaking
Используйте `feat:` для добавления возможностей, новых стадий/опций, улучшений UI/CLI, которые не ломают существующий сценарий.

Примеры:
- `feat(ui): add Run switcher recent list`
- `feat(sky): add rectified sky subtraction stage`

### MAJOR (X.y.z) — breaking changes
Используйте `feat!:` **или** добавляйте в описание PR строку `BREAKING CHANGE: ...`.

Это нужно, если вы ломаете совместимость (примеры ниже).

Примеры:
- `feat!(layout): change stage output paths`
- `feat(cli)!: rename command and remove legacy flags`

---

## Что считается breaking для Scorpipe

Считайте это MAJOR, если меняется хотя бы одно из:

1) **File-layout contract**: имена/пути ключевых outputs по стадиям (например `*_done.json`, `lambda_map.fits`, структура `run_root/NN_stage/...`).
2) **Форматы маркеров и отчётов**: структура JSON в `*_done.json`, QC JSON/HTML, поля, на которые опираются тесты/GUI.
3) **CLI/конфиг**: удаление флагов, смена смысла параметров, переименование ключей конфигурации.
4) **Порядок стадий/семантика**: когда “то же имя стадии” начинает означать другой алгоритм/результат.

---

## Небольшие соглашения по заголовкам PR

Рекомендуемые типы: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Scope (в скобках) опционален: `fix(windows): ...` или `fix: ...`.
