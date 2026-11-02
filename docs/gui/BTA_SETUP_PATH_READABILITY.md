# BTA field test — Setup path readability

On Windows, Scorpipe 8.0.5's Setup page kept its left panel
initially 430px wide, then used a QFormLayout with its filesystem
editors placed only in the narrow second column. In the real
N2146sat run, this made the current workspace and Run folder
visually impossible to verify.

This patch gives the two path editors full form-width rows. Each
has its label above the editor, a browse button, and a word-wrapped,
mouse-selectable full path readback with a matching tooltip.
Programmatic changes (Suggest, Open, Initialize) update the readback
without marking the field as manually edited. The Setup side of the
QSplitter grows as the window grows (initial ~700px, min 570px)
without eliminating the editable YAML panel. The action controls for
Initialize run, Switch and Open also have nontrivial minimum widths.

Expected real BTA workspace:

    C:\Users\frigi\Documents\PhD\ScorpipeWork

Expected generated run:

    C:\Users\frigi\Documents\PhD\ScorpipeWork\06_12_2024\N2146sat_VPHG1200B_01

PySide6 offscreen tests validate true Qt widget widths and readbacks.
These tests need the GUI extra (Windows), because default Linux
continuous integration does not install PySide6. No input FITS is
modified by this work.
