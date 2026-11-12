# BTA field test — compact, readable Setup filesystem paths

Real Windows test with Scorpipe 8.0.5 and N2146sat showed two independent
geometry defects. The initial 430px Setup pane clipped the filesystem
editor; the first PR #69 version improved width but introduced a second
tiny, low-contrast line duplicating the same path. Users confirmed that
the fields remain too low in vertical height, even after widening.

The current design retains the full-width editor rows and flexible
splitter (initial ~700px, minimum 570px), but removes both repeated
gray path readbacks. Each Qt QLineEdit is **36px tall**, with 36px
browse buttons aligned to the field. The full exact path remains
available via keyboard selection (Ctrl+A/C), horizontal scrolling,
and a tooltip synchronized on every text change. QLineEdit itself
retains its native filename editing and undo behavior; there is no
extra second text source that can become stale.

The buttons Initialize run, Switch and Open have nontrivial minimum
widths, preserving readability at Windows display scaling.

Real BTA workspace:

    C:\Users\frigi\Documents\PhD\ScorpipeWork

Generated run:

    C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_01

Qt offscreen tests assert path editor widths, exact 36px heights,
absence of duplicate labels, tooltip synchronization and action
button widths. Windows screenshot verification still required.
No raw FITS, saved config, stage math or pipeline science is modified.
