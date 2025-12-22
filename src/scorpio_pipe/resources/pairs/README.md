# Built-in manual pair sets

This folder can optionally contain pre-defined manual pair lists (x-pixel ↔ wavelength).

Layout:

```
resources/pairs/<disperser-slug>/*.txt
```

Where `<disperser-slug>` is the same slug as `wavesol/<disperser-slug>/`.

The UI will display any `*.txt` files found here as "built-in pairs".

Note: built-ins are optional; you can always create and store your own pair sets in the user library
(`~/.scorpio_pipe/pairs`).
