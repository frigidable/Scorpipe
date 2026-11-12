from pathlib import Path


def test_superflat_policy_is_strict():
    source = (Path(__file__).resolve().parents[1] / 'src/scorpio_pipe/stages/calib.py').read_text(encoding='utf-8')
    section = source[source.index('def build_superflat('):]
    assert 'policy_override="strict"' in section
    assert 'ref_sig.diff(sb_sig)' in section
    assert 'allow_readout_diff=False' in section
