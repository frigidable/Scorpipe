"""LineID22 source manifest and immutable ARC read-only QA contracts."""
import ast
from pathlib import Path
import re

from scorpio_pipe.lineid_atlas_review import _REFS, SOURCES


QA_SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"qa"/(
    "lineid22_original_anchor_review.py"
)


def test_lineid22_original_arc_hashes_are_full_and_distinct():
    source=ast.parse(QA_SCRIPT.read_text(encoding="utf-8"))
    matches=[
        ast.literal_eval(item.value)
        for item in source.body if isinstance(item,ast.Assign)
        and any(isinstance(name,ast.Name) and name.id=="_ARC_HASHES"
                for name in item.targets)
    ]
    assert len(matches)==1
    hashes=matches[0]
    assert set(hashes)=={"06","13"}
    assert all(re.fullmatch(r"[0-9a-f]{64}",v) for v in hashes.values())
    assert len(set(hashes.values()))==2


def test_lineid22_documented_19_laboratory_ids_are_distinct_and_ordered():
    wavelengths=[row[0] for row in _REFS]
    assert len(wavelengths)==19
    assert len(set(wavelengths))==19
    assert wavelengths==sorted(wavelengths)
    assert wavelengths[0]==3888.65 and wavelengths[-1]==5341.60
    assert any(item[1]=="He I" for item in _REFS)
    assert any(item[1]=="Ar I" for item in _REFS)
    assert any(item[1]=="Ne I" for item in _REFS)
    assert SOURCES["SCORPIO1"].endswith("/BSAO_eng.pdf")


def test_lineid22_diagnostic_script_never_changes_original_fits():
    source=QA_SCRIPT.read_text(encoding="utf-8")
    assert "output.mkdir(exist_ok=False)" in source
    assert 'if output.exists() or not output.parent.is_dir():' in source
    assert "source SHA mismatch" in source
    assert "full ASD export not yet obtained" in source
    for forbidden in ("fits.writeto", ".writeto(", "os.replace(", "shutil.copyfile("):
        assert forbidden not in source
