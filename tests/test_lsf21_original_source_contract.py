"""Original N2146sat v21 Ne sources and no-silent-wave-correction gates."""
import ast
from pathlib import Path
import re


SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"qa"/(
    "lsf21_original_neon_multiplet_probe.py"
)


def test_nist_v21_historic_sources_sha_have_complete_canonical_hashes():
    code=ast.parse(SCRIPT.read_text(encoding="utf-8"))
    manifests=[
        ast.literal_eval(node.value)
        for node in code.body
        if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=="EXPECTED_SHA256"
                for target in node.targets)
    ]
    assert len(manifests)==1
    hashes=manifests[0]
    assert set(hashes)=={"arc06","arc13","map","hand_pairs"}
    assert all(re.fullmatch("[0-9a-f]{64}",v) for v in hashes.values())
    assert hashes["map"]==(
        "37808e73c00279ced67ea520e2245767427bd71dc7edec9af22177fbc8df07ab"
    )
    assert hashes["hand_pairs"]==(
        "2d88b6948d9225326bcbf16125dfa844544da6a44ec1e5e8af6438975433d623"
    )


def test_nist_v21_original_arc_diagnostic_cannot_write_or_modify_fits():
    script=SCRIPT.read_text(encoding="utf-8")
    from scorpio_pipe.stages import lsf_model, wavesolution
    for module in (lsf_model,wavesolution):
        source=Path(module.__file__).read_text(encoding="utf-8")
        assert "lsf_neon_multiplet_diagnostic" not in source
    assert "output.mkdir(exist_ok=False)" in script
    assert "auto_replace_production_lsf" in script
    assert "Original source SHA256 mismatch" in script
    assert ".writeto(" not in script
