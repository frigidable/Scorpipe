from __future__ import annotations

"""Filesystem-backed Workspace adapter.

P0-ARCH-03.1
------------
Implements the :class:`~scorpio_pipe.ports.workspace.Workspace` interface for a
local filesystem run_root.

Key feature: a simple "single writer" lock (run.lock) to prevent two processes
from writing into the same run_root concurrently.
"""

import json
import os
import socket
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scorpio_pipe.io.atomic import atomic_write_json, atomic_write_text
from scorpio_pipe.ports.workspace import WorkspaceLockedError
from scorpio_pipe.workspace_paths import extract_stem_short, stage_dir as _stage_dir


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_best_effort(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _pid_is_running(pid: int) -> bool:
    """Best-effort check whether a PID exists on the *current* host."""

    if pid <= 0:
        return False

    # POSIX: os.kill(pid, 0) checks existence.
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # Process exists but we can't signal it.
        return True
    except Exception:
        pass

    # Windows fallback: use OpenProcess.
    try:
        import ctypes  # type: ignore

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def _format_lock_diag(info: dict[str, Any] | None, *, lock_path: Path) -> str:
    if not info:
        return f"Workspace is locked (cannot parse lock file): {lock_path}"

    pid = info.get("pid")
    host = info.get("host")
    started = info.get("started_utc") or info.get("time")
    user = info.get("user")
    argv = info.get("argv")

    parts = ["Workspace is locked by another process:"]
    parts.append(f"  lock: {lock_path}")
    if host:
        parts.append(f"  host: {host}")
    if user:
        parts.append(f"  user: {user}")
    if pid is not None:
        parts.append(f"  pid: {pid}")
    if started:
        parts.append(f"  started_utc: {started}")
    if argv:
        try:
            if isinstance(argv, list):
                parts.append("  argv: " + " ".join(str(x) for x in argv[:20]))
            else:
                parts.append(f"  argv: {argv}")
        except Exception:
            pass
    parts.append("\nTo override (UNSAFE), use an explicit override option.")
    return "\n".join(parts)


# Process-local re-entrancy guard: lock_path -> count
_LOCK_COUNTS: dict[str, int] = {}


class FSWorkspace:
    """Filesystem workspace adapter."""

    def __init__(self, run_root: str | Path) -> None:
        self._run_root = Path(run_root).expanduser().resolve()

    @property
    def run_root(self) -> Path:
        return self._run_root

    # ---- layout ----
    def stage_dir(self, key: str) -> Path:
        return _stage_dir(self._run_root, key)

    def per_exp_dir(self, stem: str) -> Path:
        # Stage-agnostic per-exposure area (adapter-level convention).
        # Stages may still use stage_dir(...)/<stem>/ or stage_dir(...)/per_exp/.
        s = extract_stem_short(stem)
        return self._run_root / "per_exp" / s

    # ---- atomic IO ----
    def atomic_write_text(self, path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        return atomic_write_text(path, text, encoding=encoding)

    def atomic_write_json(
        self,
        path: str | Path,
        obj: Any,
        *,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> Path:
        return atomic_write_json(path, obj, indent=indent, ensure_ascii=ensure_ascii)

    def atomic_write_bytes(self, path: str | Path, data: bytes) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(path))
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return path

    # ---- products ----
    def _load_cfg_from_run_root(self) -> dict[str, Any] | None:
        cfg_path = self._run_root / "config.yaml"
        if not cfg_path.exists():
            return None
        try:
            from scorpio_pipe.config import load_config

            return load_config(cfg_path)
        except Exception:
            return None

    def open_product(self, key: str) -> Path:
        cfg = self._load_cfg_from_run_root() or {"work_dir": str(self._run_root)}
        try:
            from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

            return PRODUCT_CATALOG.best_path(cfg, key, run_root=self._run_root)
        except Exception as e:
            raise RuntimeError(f"Failed to resolve product {key!r} in {self._run_root}: {e}") from e

    def list_products(self) -> list[Any]:
        cfg = self._load_cfg_from_run_root()
        if cfg is None:
            return []
        try:
            from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

            return PRODUCT_CATALOG.list_products(cfg, run_root=self._run_root)
        except Exception:
            return []

    # ---- retention/cleanup ----
    def cleanup(self, policy: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Apply a retention policy to the workspace and return a report.

        Policies
        --------
        all
            Keep everything (no-op).
        canonical
            Keep only canonical products (ProductCatalog non-collections) + qc + provenance.
        none
            Aggressively remove heavy intermediates (whitelist-based).

        Notes
        -----
        - This is best-effort and defensive: it should never crash a pipeline run.
        - Deletions are restricted to paths under run_root.
        """

        pol = str(policy or "all").strip().lower()
        if pol not in {"all", "canonical", "none"}:
            raise ValueError(f"Unknown retention policy: {policy!r} (expected all|canonical|none)")

        rr = self._run_root
        report: dict[str, Any] = {
            "policy": pol,
            "run_root": str(rr),
            "dry_run": bool(dry_run),
            "started_utc": _now_utc_iso(),
            "deleted_files": [],
            "deleted_bytes": 0,
            "errors": [],
        }

        if pol == "all":
            return report

        # Always keep these roots (publish/provenance).
        keep_dirs: set[Path] = {
            rr / "manifest",
            rr / "qc",
            rr / "ui",
        }
        keep_files: set[Path] = {
            rr / "config.yaml",
            rr / "run.json",
            rr / "index.html",  # QC HTML (legacy location)
        }

        # Product-driven keep set (lazy import).
        cfg = self._load_cfg_from_run_root()
        if cfg is not None:
            try:
                from scorpio_pipe.catalog.products import CATALOG as PRODUCT_CATALOG

                products = PRODUCT_CATALOG.list_products(cfg, include_collections=False, run_root=rr)

                essential_keys = {
                    # publish bundle
                    "manifest",
                    "products_manifest",
                    "timings",
                    "qc_json",
                    "qc_html",
                    "linearize_qc",
                    "navigator_dir",
                    # key science deliverables
                    "lambda_map",
                    "eta_lambda_fits",
                    "wavesolution_report_txt",
                    "wavesolution_1d_png",
                    "wavesolution_1d_json",
                    "wavesolution_2d_json",
                    "residuals_vs_lambda_png",
                    "stacked2d_fits",
                    "stack2d_fits",
                    "spec1d_fits",
                    "spec1d_txt",
                    "spec1d_csv",
                    "spec1d_png",
                }

                for pr in products:
                    try:
                        k = str(getattr(pr, "key", "") or "")
                        if pol == "none" and k and k not in essential_keys:
                            continue
                        p = Path(getattr(pr, "path"))
                        kind = str(getattr(pr, "kind", "") or "")
                        if kind == "dir":
                            keep_dirs.add(p)
                        else:
                            keep_files.add(p)
                    except Exception:
                        continue
            except Exception:
                # Never crash cleanup because catalog resolution failed.
                report["errors"].append({"path": "<catalog>", "error": "failed to resolve products"})
        else:
            # If no config, keep only standard publish roots.
            pass

        # Treat any files under keep_dirs as preserved.
        def _under_keep_dir(p: Path) -> bool:
            for d in keep_dirs:
                try:
                    if p == d or p.is_relative_to(d):
                        return True
                except Exception:
                    # Fallback for older pathlib: string prefix check
                    try:
                        if str(p).startswith(str(d)):
                            return True
                    except Exception:
                        pass
            return False

        deleted: list[dict[str, Any]] = []
        bytes_deleted = 0
        errors: list[dict[str, Any]] = []

        # Enumerate files in a stable order.
        try:
            all_paths = sorted(rr.rglob("*"), key=lambda x: (x.is_dir(), str(x)))
        except Exception:
            all_paths = []

        for p in all_paths:
            try:
                if p.is_dir():
                    continue

                # Never touch lock files (or their backups) during retention.
                if p.name.startswith("run.lock"):
                    continue

                # Only operate inside run_root (defense-in-depth).
                try:
                    _ = p.resolve().relative_to(rr)
                except Exception:
                    continue

                if p in keep_files or _under_keep_dir(p):
                    continue

                sz = 0
                try:
                    sz = int(p.stat().st_size)
                except Exception:
                    sz = 0

                if not dry_run:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        # On Windows a file may be locked; record and continue.
                        raise

                try:
                    rel = str(p.relative_to(rr))
                except Exception:
                    rel = str(p)

                deleted.append({"path": rel, "bytes": sz})
                bytes_deleted += int(sz)
            except Exception as e:
                errors.append({"path": str(p), "error": str(e)})

        # Remove empty directories bottom-up (best-effort).
        if not dry_run:
            try:
                dirs = [d for d in rr.rglob("*") if d.is_dir()]
                dirs.sort(key=lambda x: len(x.parts), reverse=True)
                for d in dirs:
                    if d == rr:
                        continue
                    if _under_keep_dir(d):
                        continue
                    try:
                        if not any(d.iterdir()):
                            d.rmdir()
                    except Exception:
                        pass
            except Exception:
                pass

        report["deleted_files"] = deleted
        report["deleted_bytes"] = int(bytes_deleted)
        report["errors"] = errors

        # Persist report for auditability (under manifest/).
        try:
            man = rr / "manifest"
            man.mkdir(parents=True, exist_ok=True)
            rep_path = man / "cleanup_report.json"
            if not dry_run:
                atomic_write_json(rep_path, report, indent=2, ensure_ascii=False)
            report["report_path"] = str(rep_path)
        except Exception:
            pass

        return report
    # ---- locking ----
    def _lock_path(self) -> Path:
        return self._run_root / "run.lock"

    def _lock_payload(self) -> dict[str, Any]:
        try:
            import getpass

            user = getpass.getuser()
        except Exception:
            user = None

        try:
            import sys

            argv = list(sys.argv)
        except Exception:
            argv = None

        return {
            "pid": int(os.getpid()),
            "host": str(socket.gethostname()),
            "started_utc": _now_utc_iso(),
            "user": user,
            "argv": argv,
        }

    def _try_acquire_lock(self, *, override: bool) -> None:
        self._run_root.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path()

        # Re-entrant within a single process.
        k = str(lock_path)
        if _LOCK_COUNTS.get(k, 0) > 0:
            _LOCK_COUNTS[k] += 1
            return

        payload = self._lock_payload()

        # If lock exists, decide what to do.
        if lock_path.exists():
            info = _read_json_best_effort(lock_path)
            host = str(info.get("host") or "") if info else ""
            pid = int(info.get("pid") or 0) if info and str(info.get("pid") or "").isdigit() else 0

            # If the lock is from this host and the pid is not running, treat as stale.
            if host and host == payload.get("host") and pid and (not _pid_is_running(pid)):
                try:
                    stale = lock_path.with_name(lock_path.name + f".stale_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
                    lock_path.replace(stale)
                except Exception:
                    try:
                        lock_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            elif not override:
                raise WorkspaceLockedError(_format_lock_diag(info, lock_path=lock_path), lock_path=lock_path)
            else:
                # Explicit override: keep a copy for diagnostics.
                try:
                    bak = lock_path.with_name(lock_path.name + f".overridden_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
                    lock_path.replace(bak)
                except Exception:
                    try:
                        lock_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        # Now create the lock file atomically.
        try:
            with lock_path.open("x", encoding="utf-8") as f:
                f.write(json.dumps(payload, indent=2, ensure_ascii=False))
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError:
            info = _read_json_best_effort(lock_path)
            if not override:
                raise WorkspaceLockedError(_format_lock_diag(info, lock_path=lock_path), lock_path=lock_path)
            # Override race: try once more by removing and creating.
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            with lock_path.open("x", encoding="utf-8") as f:
                f.write(json.dumps(payload, indent=2, ensure_ascii=False))
                f.flush()
                os.fsync(f.fileno())

        _LOCK_COUNTS[k] = 1

    def _release_lock(self) -> None:
        lock_path = self._lock_path()
        k = str(lock_path)
        n = _LOCK_COUNTS.get(k, 0)
        if n <= 1:
            _LOCK_COUNTS.pop(k, None)
        else:
            _LOCK_COUNTS[k] = n - 1
            return

        # Only remove if we still own it.
        try:
            info = _read_json_best_effort(lock_path) or {}
            if str(info.get("host") or "") != str(socket.gethostname()):
                return
            if int(info.get("pid") or 0) != int(os.getpid()):
                return
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    @contextmanager
    def lock(self, *, override: bool = False) -> Iterator[None]:
        # Environment variable is an additional explicit override mechanism.
        env_override = str(os.environ.get("SCORPIO_OVERRIDE_LOCK", "") or "").strip().lower() in {"1", "true", "yes", "y"}
        do_override = bool(override) or bool(env_override)

        self._try_acquire_lock(override=do_override)
        try:
            yield None
        finally:
            self._release_lock()


__all__ = ["FSWorkspace"]
