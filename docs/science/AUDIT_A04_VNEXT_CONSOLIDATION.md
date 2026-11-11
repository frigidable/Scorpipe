# A04 — boxcar output usability and aperture DQ provenance

**Historical source:** [PR #56](https://github.com/frigidable/Scorpipe/pull/56). **Current integration:** [PR #180](https://github.com/frigidable/Scorpipe/pull/180), SCIENCE HOLD.

The essential A04 production behavior is **already present** in current vNext: while processing a boxcar extraction, `NO_COVERAGE` from one excluded spatial input row is not blindly OR-ed into the final 1D usability mask. Instead, `NO_COVERAGE` on the 1D output is derived from the number and required fraction of accepted input pixels. Other input DQ bits may be preserved as provenance.

This scoped consolidation transports test coverage rather than reapplying a stale main-based code diff. Synthetic test cases cover one missing input row with otherwise usable output, insufficient coverage, all input rows missing, retention of informational/provenance DQ flags and consistent absence of `NO_COVERAGE` between boxcar and optimal extraction.

Limitations: an aperture provenance bit such as `BADPIX` on the **1D output** is not necessarily itself an unusable output; downstream readers must use the versioned DQ1D usability contract rather than interpreting every provenance bit as a fatal defect. These tests do not validate flux restoration for arbitrary partially masked profiles, spatial covariance, end-to-end velocity and line-flux uncertainty, or real SCORPIO observations. They do not change the scientific pipeline behavior. **SCIENCE HOLD** remains.
