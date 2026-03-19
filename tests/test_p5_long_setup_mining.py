from pathlib import Path

from scorpio_pipe.benchmark.long_params import discover_long_setups


def test_discover_long_setups(tmp_path: Path):
    long_root = tmp_path / 'LONG'
    long_root.mkdir()
    (long_root / 'VPHG1200@540.txt').write_text(
        'VPGH1200@540\npx A\n516 3887\n743 4044\n1140 4334\n4300 7032\n',
        encoding='utf-8',
    )
    (long_root / 'lines_tab_VPHG1200@540.tab').write_text(
        '3888.65 3 0.4\n4471.48 3 2.5\n5852.49 1 12.0\n7032.41 3 7.0\n',
        encoding='utf-8',
    )
    payload = discover_long_setups(long_root)
    assert payload['n_setups'] == 1
    row = payload['setups'][0]
    assert row['setup'] == 'VPHG1200@540'
    assert row['n_anchor_points'] == 4
    assert row['lambda_min_A'] == 3887.0
    assert row['line_table_count'] == 4
    assert 5852.49 in row['strongest_lines_A']
