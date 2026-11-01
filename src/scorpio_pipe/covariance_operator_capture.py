"""Opt-in, versioned capture of frozen Stack2D / Extract1D coefficients.

Never interpret these diagnostics as complete end-to-end noise propagation.
FITS COV always denotes spatial/spectral coverage, not covariance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np


SCHEMA = "SCORPIPE_FROZEN_OPERATOR_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    target = Path(path)
    temp = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


class StackOperatorCapture:
    """Disk-backed O(chunk*nlambda) stack-weight capture, opt-in only.

    Every supplied block must be the *actual last-pass* weight divided by
    last-pass total weight. No reconstruction from output VAR is allowed.
    """

    def __init__(self, out_dir: Path, paths: list[Path], *, ny: int, nlam: int,
                 max_bytes: int = 2_147_483_648):
        if not paths or ny <= 0 or nlam <= 0:
            raise ValueError("Stack operator capture needs nonempty inputs and shape")
        self.paths = [Path(p) for p in paths]
        self.shape = (int(ny), int(nlam))
        size = len(paths) * ny * nlam * np.dtype("float64").itemsize
        if size > int(max_bytes):
            raise ValueError(f"Frozen stack weights require {size} bytes; limit={max_bytes}")
        self.id = uuid.uuid4().hex
        self.out_dir = Path(out_dir)
        self.pending = self.out_dir / (".covariance_operator_pending_" + self.id)
        self.final = self.out_dir / ("covariance_operator_stack_" + self.id)
        self.pending.mkdir(parents=True, exist_ok=False)
        self.arrays = [
            np.lib.format.open_memmap(self.pending / f"stack_weight_{i:04d}.npy",
                                      mode="w+", dtype="float64", shape=self.shape)
            for i in range(len(paths))
        ]
        self.next_row = [0] * len(paths)
        self.finished = False

    def write(self, exposure_index: int, y0: int, y1: int, weights: np.ndarray) -> None:
        if self.finished:
            raise ValueError("Capture already finalized")
        i = int(exposure_index)
        if not 0 <= i < len(self.arrays) or int(y0) != self.next_row[i] or not y0 < y1 <= self.shape[0]:
            raise ValueError("Out-of-order stack-weight block")
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (y1 - y0, self.shape[1]) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("Nonfinite, negative, or wrong-shaped frozen stack weights")
        self.arrays[i][y0:y1] = w
        self.next_row[i] = int(y1)

    def finalize(self, *, stack_fits: Path, meta: dict) -> Path:
        if self.finished or any(k != self.shape[0] for k in self.next_row):
            raise ValueError("Incomplete frozen stack operator capture")
        # Verify normalization with row streaming, never constructing an exposure cube.
        for y0 in range(0, self.shape[0], 16):
            y1 = min(self.shape[0], y0 + 16)
            total = np.zeros((y1 - y0, self.shape[1]), dtype=np.float64)
            for arr in self.arrays:
                total += arr[y0:y1]
            supported = total > 0
            if np.any(~np.isclose(total[supported], 1., rtol=2e-6, atol=2e-8)):
                raise ValueError("Frozen stack weights do not normalize to one")
        for arr in self.arrays:
            arr.flush()
        self.arrays.clear()
        manifest = {
            "schema": SCHEMA,
            "stage": "stack2d",
            "status": "CONDITIONAL_FROZEN_OPERATOR_ONLY",
            "shape_y_lambda": list(self.shape),
            "weights_dtype": "float64",
            "weights_normalized": True,
            "output": {"path": str(stack_fits), "sha256": sha256_file(stack_fits)},
            "inputs": [
                {"index": i, "path": str(p), "sha256": sha256_file(p),
                 "weights_file": f"stack_weight_{i:04d}.npy",
                 "weights_sha256": sha256_file(self.pending / f"stack_weight_{i:04d}.npy")}
                for i, p in enumerate(self.paths)
            ],
            "frozen_parameters": meta,
            "limitations": ["frozen robust selection", "data-derived shifts",
                            "upstream sky/flat/bias noise excluded",
                            "empirical eta not a covariance operator"],
        }
        _atomic_json(self.pending / "manifest.json", manifest)
        os.replace(self.pending, self.final)
        self.finished = True
        return self.final / "manifest.json"

    def abort(self) -> None:
        self.arrays.clear()
        if not self.finished:
            shutil.rmtree(self.pending, ignore_errors=True)


def verify_spatial_operator(sci: np.ndarray, weights: np.ndarray,
                            flux: np.ndarray, *, rtol: float = 2e-9) -> None:
    """Verify the frozen *linear* part independently reproduces extracted flux."""
    sci = np.asarray(sci, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if sci.shape != weights.shape or sci.ndim != 2 or flux.shape != (sci.shape[1],):
        raise ValueError("Extraction operator shape mismatch")
    if not np.isfinite(weights).all():
        raise ValueError("Invalid extraction coefficients")
    pred = np.sum(np.where(weights != 0, weights * np.where(np.isfinite(sci), sci, 0.), 0.), axis=0)
    valid = np.isfinite(flux)
    if np.any(valid & ~np.isclose(pred, flux, rtol=rtol, atol=1e-8)):
        raise ValueError("Frozen extraction operator fails to reproduce output flux")
    if np.any(~valid & np.any(weights != 0, axis=0)):
        raise ValueError("Undefined extraction flux has nonzero operator coefficients")


def save_extraction_operators(out_dir: Path, *, input_fits: Path,
                              spec_fits: Path, E_trace: np.ndarray,
                              E_fixed: np.ndarray, metadata: dict,
                              max_bytes: int = 536_870_912) -> Path:
    """Write exact effective extraction maps as separate versioned diagnostics."""
    E_trace = np.asarray(E_trace, dtype=np.float64)
    E_fixed = np.asarray(E_fixed, dtype=np.float64)
    if E_trace.ndim != 2 or E_trace.shape != E_fixed.shape:
        raise ValueError("TRACE/FIXED extraction operator shapes differ")
    if E_trace.nbytes + E_fixed.nbytes > int(max_bytes):
        raise ValueError("Frozen extraction operators exceed explicit byte limit")
    if not np.isfinite(E_trace).all() or not np.isfinite(E_fixed).all():
        raise ValueError("Nonfinite extraction coefficients")
    uid = uuid.uuid4().hex
    root = Path(out_dir) / ("covariance_operator_extract_" + uid)
    root.mkdir(parents=True, exist_ok=False)
    try:
        archive = root / "extraction_weights.npz"
        np.savez(archive, E_trace=E_trace, E_fixed=E_fixed)
        manifest = {
            "schema": SCHEMA,
            "stage": "extract1d",
            "status": "CONDITIONAL_FROZEN_OPERATOR_ONLY",
            "shape_y_lambda": list(E_trace.shape),
            "weights_dtype": "float64",
            "input": {"path": str(input_fits), "sha256": sha256_file(input_fits)},
            "output": {"path": str(spec_fits), "sha256": sha256_file(spec_fits)},
            "weights_file": archive.name,
            "weights_sha256": sha256_file(archive),
            "frozen_parameters": metadata,
            "limitations": ["data-derived trace and spatial profile",
                            "sky-fit and calibration uncertainty excluded",
                            "local sky correction unsupported"],
        }
        _atomic_json(root / "manifest.json", manifest)
        return root / "manifest.json"
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
