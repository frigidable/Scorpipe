# Observer dashboard and recovery workflows

P4 introduces an observer-facing cockpit above the normal QC report.

Outputs:

- `observer_dashboard/observer_dashboard.json`
- `observer_dashboard/observer_dashboard.md`
- `observer_dashboard/observer_dashboard.html`
- `observer_dashboard/recovery_plan.json`

The dashboard combines:

- setup-aware preset selection;
- slit sanity / geometry / wavelength / sky refine / extraction summaries;
- an overall observer trust score;
- recovery workflows derived from the operational failure taxonomy.

The dashboard is not a new source of scientific truth. It is a compact
observer-oriented layer built from canonical products and QC JSON outputs.
