from __future__ import annotations

from pathlib import Path

import yaml

try:
    import grimp
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "grimp is required for architecture import enforcement. Install with: pip install .[dev]"
    ) from e


def _is_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _load_rules() -> list[dict]:
    repo_root = Path(__file__).resolve().parents[1]
    rules_path = repo_root / "tools" / "arch_imports.yml"
    data = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list):
        raise AssertionError("tools/arch_imports.yml: expected top-level 'rules: [ ... ]'")
    return rules


def test_no_forbidden_imports() -> None:
    # Build a graph of our package imports.
    graph = grimp.build_graph(
        "scorpio_pipe",
        include_external_packages=True,
        exclude_type_checking_imports=True,
    )

    rules = _load_rules()
    violations: list[str] = []

    for rule in rules:
        rid = str(rule.get("id") or "RULE")
        src_prefix = str(rule.get("from") or "").strip()
        forbid = rule.get("forbid")
        allow = set(rule.get("allow") or [])

        if not src_prefix:
            raise AssertionError(f"{rid}: missing 'from' prefix")
        if not isinstance(forbid, list) or not forbid:
            raise AssertionError(f"{rid}: missing/non-list 'forbid'")

        src_modules = [m for m in graph.modules if _is_under(m, src_prefix)]
        if not src_modules:
            # If the prefix is valid but absent, it's most likely a configuration mistake.
            raise AssertionError(f"{rid}: no modules found under prefix '{src_prefix}'")

        for m in sorted(src_modules):
            if m in allow:
                continue
            direct = graph.find_modules_directly_imported_by(m)
            for d in sorted(direct):
                for bad in forbid:
                    bad = str(bad)
                    if _is_under(d, bad):
                        violations.append(f"{rid}: {m} imports {d} (forbidden by {bad})")

    if violations:
        msg = "\n".join(["Forbidden imports detected:", *violations])
        raise AssertionError(msg)
