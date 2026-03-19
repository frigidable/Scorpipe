# LONG setup priors

Scorpipe now ships a built-in summary of empirical setup parameters mined from the
legacy LONG pipeline anchor tables and curated line tables.

## Source

- `src/scorpio_pipe/resources/setup_priors/long_setup_priors.json`
- generated from LONG `*.txt` anchor tables and `lines_tab_*.tab` line lists

## What is used automatically

For a known disperser (for example `VPHG1200@540`, `VPHG1800@590`, `VPHG1026@735`):

- expected wavelength coverage
- trimmed linearization limits
- median dispersion in Å/pix
- recommended `wavesol.pair_tol_A`
- recommended `wavesol.pair_top_lines`
- recommended peak target counts for auto LineID
- strongest setup-specific arc lines for auto LineID weighting
- empirical LSF fallback for setups missing manual FWHM priors

## Runtime integration points

- `autocfg.py` / `operational_presets.py`
- `stages/lineid_auto_backup.py`
- `stages/lsf_model.py`

## Safety

These priors are **additive defaults only**:

- explicit user config still wins;
- certification does not treat these priors as proof of LONG parity;
- missing real-night cases still block formal observatory certification.
