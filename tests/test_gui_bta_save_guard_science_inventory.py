"""Guard against saving the all-empty SCI frame list seen in BTA field QA."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / "src" / "scorpio_pipe" / "ui" / "launcher_window.py"


def _method(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "LauncherWindow")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                  and x.name == name)
    env = {"Path": Path}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[method], type_ignores=[])), str(SOURCE), "exec"), env)
    return env[name]


def _night():
    setup = {
        "instrument": "SCORPIO", "mode": "Spectra",
        "disperser": "VPHG1200B", "slit": "1.2",
        "binning": "1x2", "window": "", "shape": "1032x2080",
    }
    rows = []
    for i in (4, 7, 8, 9, 10, 11, 12):
        rows.append(dict(setup, kind="obj", object="N2146sat",
                         object_norm="N2146SAT", exptime=1200,
                         path=f"s228006{i:02d}.fts"))
    rows.append(dict(setup, kind="neon", object="N2146sat",
                     object_norm="N2146SAT", exptime=60,
                     path="s22800606.fts"))
    return pd.DataFrame(rows)


class _FakeWindow:
    _validate_science_frames_before_save = _method("_validate_science_frames_before_save")

    def __init__(self, frames):
        self._inspect = SimpleNamespace(table=_night())
        self._cfg = {
            "object": "N2146sat", "data_dir": ".",
            "work_dir": ".",
            "frames": {"obj": frames, "__setup__": {
                "disperser": "VPHG1200B", "slit": "1.2", "binning": "1x2",
            }},
        }
        self.errors = []

    def _log_error(self, s):
        self.errors.append(s)


def test_guards_against_saving_zero_when_inspector_has_seven_science():
    win = _FakeWindow([])
    assert not win._validate_science_frames_before_save()
    assert len(win.errors) == 1
    assert "0 SCI frames" in win.errors[0]
    assert "found 7" in win.errors[0]


def test_valid_seven_frames_can_be_saved_without_warning():
    win = _FakeWindow([f"s228006{i:02d}.fts" for i in (4, 7, 8, 9, 10, 11, 12)])
    assert win._validate_science_frames_before_save()
    assert win.errors == []


def test_earlier_config_without_inspection_can_be_saved():
    win = _FakeWindow([])
    win._inspect = None
    assert win._validate_science_frames_before_save()
    assert not win.errors


def test_mismatched_target_no_false_positive():
    win = _FakeWindow([])
    win._cfg["object"] = "NOT_PRESENT"
    assert win._validate_science_frames_before_save()
    assert not win.errors


def test_guard_runs_before_any_filesystem_write():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "LauncherWindow")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
              and x.name == "_do_save_cfg")
    body = ast.unparse(fn)
    assert body.index("self._validate_science_frames_before_save()") < body.index("self._cfg_path.write_text")


def test_regenerate_config_backups_old_yaml_before_overwrite():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "LauncherWindow")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
              and x.name == "_do_make_cfg")
    src = ast.unparse(fn)
    assert src.index('if not ac.frames.get(\'obj\')') < src.index('cfg_path.write_text')
    assert src.index('shutil.copy2(cfg_path, backup)') < src.index('cfg_path.write_text')
    assert 'config.yaml.bak.' in src
