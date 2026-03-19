"""Catalog subpackage.

Important
---------
This package must remain *import-light*.

Several core modules (workspace_paths, stage_registry, products) form a tight
dependency graph. Importing both StageCatalog and ProductCatalog eagerly from
``catalog.__init__`` can create circular imports.

Therefore we expose convenience symbols via *lazy* ``__getattr__``.
Prefer importing directly from submodules:

- :mod:`scorpio_pipe.catalog.stages`
- :mod:`scorpio_pipe.catalog.products`
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover
    if name in {"STAGE_CATALOG", "GuiStageDef", "StageCatalog", "StageInfo", "iter_gui_stages"}:
        from .stages import CATALOG as STAGE_CATALOG, GuiStageDef, StageCatalog, StageInfo, iter_gui_stages

        return {
            "STAGE_CATALOG": STAGE_CATALOG,
            "GuiStageDef": GuiStageDef,
            "StageCatalog": StageCatalog,
            "StageInfo": StageInfo,
            "iter_gui_stages": iter_gui_stages,
        }[name]

    if name in {"PRODUCT_CATALOG", "ProductSpec", "ProductCatalog", "Product"}:
        from .products import CATALOG as PRODUCT_CATALOG, Product, ProductCatalog, ProductSpec

        return {
            "PRODUCT_CATALOG": PRODUCT_CATALOG,
            "ProductSpec": ProductSpec,
            "ProductCatalog": ProductCatalog,
            "Product": Product,
        }[name]


    if name in {"INSTRUMENT_CATALOG", "InstrumentInfo", "list_instruments", "resolve_adapter", "parse_header_to_meta"}:
        from .instruments import CATALOG as INSTRUMENT_CATALOG, InstrumentInfo, list_instruments, resolve_adapter, parse_header_to_meta

        return {
            "INSTRUMENT_CATALOG": INSTRUMENT_CATALOG,
            "InstrumentInfo": InstrumentInfo,
            "list_instruments": list_instruments,
            "resolve_adapter": resolve_adapter,
            "parse_header_to_meta": parse_header_to_meta,
        }[name]

    raise AttributeError(name)


__all__ = [
    # stages
    "STAGE_CATALOG",
    "GuiStageDef",
    "StageCatalog",
    "StageInfo",
    "iter_gui_stages",
    # products
    "PRODUCT_CATALOG",
    "ProductSpec",
    "ProductCatalog",
    "Product",
    # instruments
    "INSTRUMENT_CATALOG",
    "InstrumentInfo",
    "list_instruments",
    "resolve_adapter",
    "parse_header_to_meta",
]
