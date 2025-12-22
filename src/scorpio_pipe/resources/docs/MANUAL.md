# Scorpio Pipe — Quick manual

This GUI is a guided front-end for the SCORPIO/SCORPIO‑2 long‑slit reduction pipeline.

## Workflow (recommended)

1. **Project & Data**
   - Select the folder with your raw FITS.
   - Press **Inspect**. The pipeline parses FITS headers + the nightlog, groups frames by setup.

2. **Config & Setup**
   - Choose **Object**, **Disperser**, (optionally) **Slit** and **Binning**.
   - Pick or **Suggest** a Work directory.
   - Press **Create new config**.
   - Use the **Parameters** tab for common options, or the **YAML** tab for full editing.

3. **Calibrations**
   - Run **Calibrations** (bias/flat/etc according to the config).
   - Open **QC** to verify master frames.

4. **SuperNeon**
   - Run **SuperNeon** to produce the high‑S/N neon product.
   - QC: check line sharpness and coverage.

5. **Line identification**
   - Run **Prepare LineID** to build a 1D profile and detect peaks.
   - Build or select line pairs (pixel ↔ wavelength).
   - Optionally copy built‑in pairs and edit locally.

6. **Wavelength solution**
   - Optionally **Clean pairs** (reject bad lines interactively).
   - Run **Wavelength solution** (1D λ(x) + 2D λ(x,y)).
   - QC: inspect residuals and λ‑maps.

## QC panel

Use **QC Viewer** at any time. If **Auto QC** is enabled (toolbar), the viewer pops up after each step.

## Instrument database

Tools → **Instrument database** opens a read‑only reference. It is used for hints/sanity checks and can be refined by updating:

`resources/instruments/scorpio_instruments.yaml`
