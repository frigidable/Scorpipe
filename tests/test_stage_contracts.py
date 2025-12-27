from __future__ import annotations

from pathlib import Path

from scorpio_pipe.stage_contracts import list_stage_contracts, validate_contracts
from scorpio_pipe.work_layout import ensure_work_layout


def _minimal_cfg(tmp_path: Path) -> dict:
    work_dir = tmp_path / "work"
    ensure_work_layout(work_dir)
    return {"config_dir": str(tmp_path), "work_dir": str(work_dir)}


def test_stage_contracts_validate(tmp_path: Path):
    cfg = _minimal_cfg(tmp_path)
    contracts = list_stage_contracts()
    assert contracts, "contracts registry must not be empty"
    validate_contracts(cfg)


def test_stage_contract_keys_unique():
    contracts = list_stage_contracts()
    assert len(set(contracts.keys())) == len(contracts)
