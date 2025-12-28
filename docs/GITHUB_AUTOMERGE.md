# GitHub Auto‑merge (Dependabot)

Этот проект настроен так, чтобы **Dependabot‑PR** для безопасных обновлений (обычно *patch/minor*) проходили путь
«создался → прошёл required checks → автоматически смёрджился» с минимальным ручным участием.

## 1) Что нужно включить в Settings (1 минута)

1. **Settings → General (или Pull Requests) → Allow auto‑merge = ON**  
   Без этого GitHub не позволит включать auto‑merge на PR программно.

2. **Settings → Actions → General → Workflow permissions → Read and write permissions**  
   Нужно, чтобы workflow мог:
   - ставить approve,
   - включать auto‑merge,
   - (опционально) добавлять labels.

> Если включён “Require approval”, то approve делает workflow из `.github/workflows/dependabot-automerge.yml`.

## 2) Какие required checks должны быть зелёными

В ruleset для `main` обычно требуется:
- **CI / gate** (сводный гейт: lint + tests + build_check)
- **CI / dependency_review** (проверка уязвимостей/опасных изменений зависимостей)
- **CodeQL / analyze**

Пока они не зелёные — auto‑merge не произойдёт.

## 3) Как Dependabot становится “одним PR в неделю”

Файл: `.github/dependabot.yml`

- Для `pip` и `github-actions` включены **groups**.
- Для Python‑зависимостей по умолчанию группируются только:
  - `patch`
  - `minor`
- `major` обновления по умолчанию **игнорируются**, чтобы не ловить внезапные breaking changes.

### Если нужны major‑апдейты отдельно (по желанию)
В `.github/dependabot.yml`:
- убери блок `ignore` для `version-update:semver-major`,
- добавь отдельную группу `python-deps-major` (пример уже есть в комментарии рядом).

## 4) Где живёт автоматика auto‑merge

Workflow: `.github/workflows/dependabot-automerge.yml`

Типичная логика:
- срабатывает на PR‑события,
- проверяет, что PR автор — `dependabot[bot]`,
- делает approve,
- включает auto‑merge (обычно squash),
- если “Allow auto‑merge” выключен — пишет warning и не валит workflow.

## 5) Быстрая диагностика (30 секунд)

Если Dependabot‑PR не смёрджился сам:

1) Проверь, что **Allow auto‑merge = ON**.  
2) Проверь **Actions → Workflow permissions: Read and write**.  
3) Открой PR и убедись, что required checks действительно зелёные.  
4) Посмотри логи workflow **Dependabot auto‑approve & auto‑merge**.
