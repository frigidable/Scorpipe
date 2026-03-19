from pathlib import Path

from scorpio_pipe.benchmark.night_bootstrap import build_night_bootstrap_pack


NIGHTLOG = '''
|s23841404|ngc2784|obj |02:42:56|  300| Spectra|VPHG1200@540| 1.00|             |         |       |09:13:31.03| -24:17:02.3|1.5   | 69| -15|348.2|313.8|
|s23841405|ngc2784|obj |02:49:02|  300| Spectra|VPHG1200@540| 1.00|             |         |       |09:13:30.99| -24:17:03.9|1.5   | 69| -12|349.3|314.9|
|s23841406|ngc2784|neon|02:54:45|   40| Spectra|VPHG1200@540| 1.00|             |         |       |09:13:30.95| -24:17:04.6|1.5   | 68| -11|350.4|316.0|
|s23841310|mage0740+16|flat|02:33:05|   20| Spectra|VPHG1200@540| 1.00|             |         |       |09:13:23.83| -24:16:34.6|1.5   | 69| -16|346.3|334.0|
|s23841306|mage0740+16|obj |01:44:37|  900| Spectra|VPHG1200@540| 1.00|             |         |       |07:42:12.74| +16:47:22.3|1.5   | 27| -14|348.7|314.9|
|s23841307|mage0740+16|obj |02:00:08|  900| Spectra|VPHG1200@540| 1.00|             |         |       |07:42:12.61| +16:47:22.1|1.5   | 26|  -7|354.7|320.9|
|s23841308|mage0740+16|obj |02:15:46|  900| Spectra|VPHG1200@540| 1.00|             |         |       |07:42:12.55| +16:47:20.8|1.5   | 26|   1|  1.0|327.2|
|s23841309|mage0740+16|neon|02:31:40|   40| Spectra|VPHG1200@540| 1.00|             |         |       |07:42:12.50| +16:47:20.5|1.5   | 27|   9|  7.3|333.5|
|s23841312|mage0740+16|flat|02:34:27|   20| Spectra|VPHG1200@540| 1.00|             |         |       |09:13:23.83| -24:16:34.6|1.5   | 69| -17|346.6|333.4|
'''

HEADER = '''
OBJECT  = 'ngc2784 '
FILE    = 's23841404.fts'
INSTRUME= 'SCORPIO-2'
DISPERSE= 'VPHG1200@540'
SLITWID = 1.00019
BINNING = '1x2'
SEEING  = '1.5'
SPERANGE= '3600-7070 AA'
'''


def test_build_night_bootstrap_pack(tmp_path: Path):
    nightlog = tmp_path / 's251216.txt'
    nightlog.write_text(NIGHTLOG, encoding='utf-8')
    hdr = tmp_path / 'ngc2784_science.txt'
    hdr.write_text(HEADER, encoding='utf-8')
    payload = build_night_bootstrap_pack(nightlog_path=nightlog, header_text_paths=[hdr])
    slots = {row['slot']: row for row in payload['candidate_suggestions']}
    assert 'reference_candidate' in slots
    assert 'repeatability_multi_exposure' in slots
    assert 'extended_object_sky_stress' in slots
    ref = slots['reference_candidate']
    assert ref['category'] == 'reference'
    assert ref['evidence']['object'] in {'ngc2784', 'mage0740+16'}
