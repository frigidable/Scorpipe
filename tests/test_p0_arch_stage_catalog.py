from __future__ import annotations


def test_stage_catalog_covers_all_engine_tasks() -> None:
    from scorpio_pipe.catalog.stages import CATALOG
    from scorpio_pipe.pipeline import engine

    assert set(CATALOG.task_keys()) == set(engine.CANONICAL_TASKS)
    assert set(engine.STAGE_SPECS) == set(engine.CANONICAL_TASKS)
    # TASKS may contain compatibility aliases, but must cover all canonicals.
    assert set(engine.TASKS).issuperset(set(engine.CANONICAL_TASKS))


def test_stage_catalog_gui_stage_table_is_unique_and_complete() -> None:
    from scorpio_pipe.catalog.stages import CATALOG

    stages = CATALOG.gui_stages()
    assert len(stages) >= 12
    assert len({s.id for s in stages}) == len(stages)
    assert len({s.key for s in stages}) == len(stages)
    assert all(s.slug for s in stages)


def test_stage_registry_is_thin_shim_over_catalog() -> None:
    from scorpio_pipe.catalog.stages import CATALOG
    from scorpio_pipe import stage_registry

    cat_keys = [s.key for s in CATALOG.gui_stages()]
    reg_keys = [s.key for s in stage_registry.STAGES]
    assert reg_keys == cat_keys

    # A few critical aliases must exist for backward compatibility.
    assert stage_registry.ALIASES["biascorr"] == "biascorr"
    assert stage_registry.ALIASES["superbias"] == "biascorr"
    assert stage_registry.ALIASES["wavesolution"] == "wavesol"
    assert stage_registry.ALIASES["lineid_prepare"] == "arclineid"
    # Regression: legacy alias previously present in engine tables must be catalog-driven.
    assert CATALOG.canonical_task_name("wavesolution2d") == "wavesolution"
    assert stage_registry.ALIASES["wavesolution2d"] == "wavesol"


def test_stage_contracts_are_derived_and_validate_against_product_registry() -> None:
    from scorpio_pipe import stage_contracts

    # Must include canonical keys + legacy superbias aliases.
    assert "superbias" in stage_contracts.CONTRACTS
    assert "bias_combine" in stage_contracts.CONTRACTS
    assert "masterbias" in stage_contracts.CONTRACTS

    # Validate against product registry using an empty cfg (product list is static).
    stage_contracts.validate_contracts({})


def test_stage_catalog_registers_calibration_domain_tasks() -> None:
    from scorpio_pipe.catalog.stages import CATALOG

    expected = {
        "standard_prepare",
        "sensfunc",
        "telluric_model",
        "fluxcal_apply",
        "line_measure",
        "science_report",
    }

    assert expected.issubset(set(CATALOG.task_keys()))
    assert CATALOG.get_task("fluxcal").key == "fluxcal_apply"
    assert CATALOG.get_task("telluric").key == "telluric_model"
