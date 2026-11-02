# Real BTA test: selected target determines run directory

The first BTA Scorpipe GUI run covers 306 FITS from 6–7 December 2024.
When selecting N2146sat, VPHG1200B, slit 1.2 and binning 1x2,
the application could still show a run folder ending with unknown_01.

The previous GUI sampled the first science frame from the entire
mixed-object night. The run-name builder also ignored SCORPIO-1's
DISPERSE header and did not try the valid ISO DATE after failing
to parse DATE-OBS=2024/06/12.

The corrected GUI filters the inspected table by the selected object's
SCI classification and full setup, validates the original FITS OBJECT,
and regenerates auto-suggested run directories on selection changes.
Manual run-folder overrides are preserved. The run-name engine
recognizes DISPERSE and tries DATE when DATE-OBS is not parseable.
An unknown observing date now fails closed instead of selecting today.

Expected run name for the BTA 2024-12-06 science frames:

    06_12_2024/N2146sat_VPHG1200B_01

The actual workspace root is chosen by the user and must not be inside
the raw FITS folder or any Git checkout. Original data are never edited.

This PR is stacked on #67 and #66 (GUI startup, inspection completion
and lazy Setup lifecycle fixes). Run-path semantics need Windows GUI
qualification on the real BTA night before scientific reduction starts.
