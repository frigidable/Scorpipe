# Scorpio Pipe resources

This folder contains optional static assets used by the GUI/pipeline.

## Built-in hand pairs

If you want to ship "ready" LineID pair lists for specific SCORPIO gratings,
put them into:

`resources/pairs/<DISPERSER_SLUG>.txt`

Where `DISPERSER_SLUG` matches the output of `slugify_disperser()`.

Example:

`resources/pairs/VPHG1200_540.txt`

The UI will offer these files as "Built-in pairs".
