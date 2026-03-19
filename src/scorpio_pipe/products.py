"""Registry of pipeline products (deprecated shim).

P0-ARCH-01.2
------------
Single source of truth moved to :mod:`scorpio_pipe.catalog.products`.

This module remains as a thin compatibility layer because a lot of code/tests
import:

- ``scorpio_pipe.products.Product``
- ``scorpio_pipe.products.list_products``
- ``scorpio_pipe.products.task_is_complete`` / ``products_for_task``

The *specs* and the *path resolution* now live in ProductCatalog.
"""

from __future__ import annotations

from typing import Any, Iterable

from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG, Product


def list_products(cfg: dict[str, Any]) -> list[Product]:
    """Return stable products for QC/UI listings.

    Delegates to :class:`~scorpio_pipe.catalog.products.ProductCatalog`.
    """

    return PRODUCT_CATALOG.list_products(cfg)


def products_by_stage(products: Iterable[Product]) -> dict[str, list[Product]]:
    out: dict[str, list[Product]] = {}
    for p in products:
        out.setdefault(p.stage, []).append(p)
    return out


# Compatibility alias used by qc_report.py (older name).
def group_by_stage(products: Iterable[Product]) -> dict[str, list[Product]]:
    return products_by_stage(products)


# -----------------------------------------------------------------------------
# Task completion heuristics (UI skip logic)
# -----------------------------------------------------------------------------

_TASK_COMPLETION: dict[str, list[list[str]]] = {
    # Each inner list is an OR-group (any existing product from the group satisfies that requirement).
    "manifest": [["manifest"]],
    # P0-C1: prefer readout-aware master bias index, keep legacy superbias.fits fallback
    "superbias": [["master_bias_index", "superbias_fits"]],
    "bias_combine": [["master_bias_index", "superbias_fits"]],
    "masterbias": [["master_bias_index", "superbias_fits"]],
    "superflat": [["superflat_fits"]],
    "flatfield": [["flatfield_done"]],
    "cosmics": [["cosmics_summary"]],
    "superneon": [["superneon_fits", "superneon_png"]],
    "slit_sanity": [["slit_sanity_json"]],
    "geometry_prior": [["geometry_model_json"]],
    "lineid_prepare": [["lineid_template_csv"], ["lineid_auto_csv"]],
    "lineid": [["hand_pairs_txt"]],
    "wavesolution": [["lambda_map"], ["wavesolution_2d_json"]],
    "lsf_model": [["lsf_map_fits"], ["lsf_model_json"]],
    "linearize": [["linearize_done", "lin_preview_fits"]],
    "sky": [["sky_done"]],
    "sky_refine": [["sky_refine_done"]],
    "stack2d": [["stacked2d_fits"]],
    "extract1d": [["spec1d_fits"]],
    "publication_ready": [["publication_spectrum_fts"], ["publication_wl_fts"], ["publication_dismap_fts"]],
    "kinematics_prep": [["spec_kin_fits"], ["kinematics_masks_json"], ["kinematics_readme_md"]],
    "qc_report": [["qc_json"], ["qc_html"]],
    "observer_dashboard": [["observer_dashboard_json"], ["observer_dashboard_html"]],
}


def products_for_task(cfg: dict[str, Any], task: str) -> list[Product]:
    """Return products that represent completion of a task."""

    t = (task or "").strip().lower()
    rules = _TASK_COMPLETION.get(t)
    if not rules:
        return []
    all_products = {p.key: p for p in list_products(cfg)}
    keys: set[str] = set(k for group in rules for k in group)
    return [all_products[k] for k in keys if k in all_products]


def task_is_complete(cfg: dict[str, Any], task: str) -> bool:
    """Return True if task outputs exist on disk."""

    t = (task or "").strip().lower()
    rules = _TASK_COMPLETION.get(t)
    if not rules:
        return False
    prods = {p.key: p for p in list_products(cfg)}
    for group in rules:
        ok = False
        for k in group:
            p = prods.get(k)
            if p is not None and p.exists():
                ok = True
                break
        if not ok:
            return False
    return True


__all__ = [
    "Product",
    "list_products",
    "products_by_stage",
    "group_by_stage",
    "products_for_task",
    "task_is_complete",
]
