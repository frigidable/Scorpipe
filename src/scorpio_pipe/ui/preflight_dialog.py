from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtWidgets


@dataclass
class PreflightGateResult:
    allowed: bool
    preflight_override: bool = False
    override_reason: str | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _fmt_status(s: str) -> str:
    s = str(s or "").strip().upper()
    if s not in {"OK", "WARN", "ERROR"}:
        return "UNKNOWN"
    return s


class PreflightGateDialog(QtWidgets.QDialog):
    """Modal preflight gate (GUI Wizard step #1).

    Shows preflight status, reasons, quarantine, and a compact "what will be
    counted" summary.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        run_root: Path,
        report: dict[str, Any] | None,
        night_profile: dict[str, Any] | None,
        quarantine_index: dict[str, Any] | None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preflight")
        self.setModal(True)

        self._run_root = Path(run_root)
        self._report = report or {}
        self._night = night_profile or {}
        self._qidx = quarantine_index or {}

        self.preflight_override: bool = False
        self.override_reason: str | None = None

        status = _fmt_status(str(self._report.get("status") or ""))

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(12)

        # Header
        hdr = QtWidgets.QLabel(self._build_header_text(status))
        hdr.setWordWrap(True)
        root.addWidget(hdr)

        # Tabs
        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs, 1)

        tabs.addTab(self._build_summary_tab(), "Summary")
        tabs.addTab(self._build_reasons_tab(), "Reasons")
        tabs.addTab(self._build_quarantine_tab(), "Quarantine")

        # Override row (visible only on ERROR)
        self._chk_override = QtWidgets.QCheckBox(
            "Override ERROR and proceed (recorded in provenance)"
        )
        self._chk_override.setVisible(status == "ERROR")
        self._chk_override.stateChanged.connect(self._update_buttons)

        self._edit_reason = QtWidgets.QLineEdit()
        self._edit_reason.setPlaceholderText("Override reason (optional)")
        self._edit_reason.setVisible(status == "ERROR")

        ov_row = QtWidgets.QWidget()
        ov_lay = QtWidgets.QHBoxLayout(ov_row)
        ov_lay.setContentsMargins(0, 0, 0, 0)
        ov_lay.addWidget(self._chk_override)
        ov_lay.addWidget(self._edit_reason, 1)
        root.addWidget(ov_row)

        # Buttons
        btns = QtWidgets.QDialogButtonBox()
        self._btn_cancel = btns.addButton("Cancel", QtWidgets.QDialogButtonBox.RejectRole)
        self._btn_proceed = btns.addButton("Proceed to Run", QtWidgets.QDialogButtonBox.AcceptRole)
        self._btn_cancel.clicked.connect(self.reject)
        self._btn_proceed.clicked.connect(self._accept)
        root.addWidget(btns)

        # Gate: only allow Proceed if status!=ERROR or override checked.
        self._status = status
        self._update_buttons()

        self.resize(900, 640)

    def _build_header_text(self, status: str) -> str:
        p = self._run_root
        rep_p = p / "manifest" / "preflight_report.json"
        q_p = p / "manifest" / "quarantine_index.json"
        night_p = p / "manifest" / "night_profile.json"
        return (
            f"<b>Status:</b> <span style='font-size:16px'>{status}</span><br>"
            f"<b>Run:</b> {p.as_posix()}<br>"
            f"<b>Artifacts:</b> {rep_p.name}, {q_p.name}, {night_p.name}"
        )

    def _compute_counts(self) -> dict[str, int | None]:
        counts = self._night.get("counts") if isinstance(self._night.get("counts"), dict) else {}
        total = counts.get("n_total_fits") if isinstance(counts.get("n_total_fits"), int) else None
        q_entries = self._qidx.get("entries") if isinstance(self._qidx.get("entries"), list) else []
        n_error = sum(1 for e in q_entries if str((e or {}).get("severity") or "").upper() == "ERROR")
        n_excl_policy = sum(
            1
            for e in q_entries
            if "RC_EXCLUDED_BY_POLICY" in ((e or {}).get("codes") or [])
        )
        n_excluded = n_error + n_excl_policy
        n_included = (total - n_excluded) if isinstance(total, int) else None
        return {
            "total": total,
            "included": n_included,
            "excluded": n_excluded,
            "excluded_by_policy": n_excl_policy,
            "error": n_error,
            "quarantine": len(q_entries),
        }

    def _build_summary_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        counts = self._compute_counts()

        lines: list[str] = []
        lines.append("What will be processed")
        lines.append("-")
        if counts.get("total") is not None:
            lines.append(f"Total FITS: {counts['total']}")
        if counts.get("included") is not None:
            lines.append(f"Included (OK + WARN): {counts['included']}")
        lines.append(f"Excluded: {counts['excluded']}")
        lines.append(f"  • by policy: {counts['excluded_by_policy']}")
        lines.append(f"  • ERROR: {counts['error']}")
        lines.append(f"Quarantine tickets: {counts['quarantine']}")
        lines.append("")

        inst = self._night.get("instrument")
        mode = self._night.get("mode")
        grism = self._night.get("grism")
        binning = self._night.get("binning")
        window = self._night.get("window")
        shape = self._night.get("shape")
        readout = self._night.get("readout")

        lines.append("Night profile")
        lines.append("-")
        if inst is not None:
            lines.append(f"Instrument: {inst}")
        if mode is not None:
            lines.append(f"Mode: {mode}")
        if grism is not None:
            lines.append(f"Grism: {grism}")
        if binning is not None:
            lines.append(f"Binning: {binning}")
        if window is not None:
            lines.append(f"Window: {window}")
        if shape is not None:
            lines.append(f"Shape: {shape}")
        if readout is not None:
            lines.append(f"Readout: {readout}")

        lamp_hints = self._night.get("lamp_hints")
        if isinstance(lamp_hints, dict) and lamp_hints:
            lines.append("")
            lines.append("Lamp hints")
            lines.append("-")
            for k in sorted(lamp_hints.keys()):
                lines.append(f"{k}: {lamp_hints.get(k)}")

        t = QtWidgets.QPlainTextEdit("\n".join(lines))
        t.setReadOnly(True)
        t.setMinimumHeight(220)
        lay.addWidget(t)

        hint = QtWidgets.QLabel(
            "Full details are stored in run_root/manifest/*.json and run_root/quarantine/*.json"
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addStretch(1)
        return w

    def _build_reasons_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        reasons = self._report.get("reasons")
        items = reasons if isinstance(reasons, list) else []

        table = QtWidgets.QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Severity", "Code", "Count", "Details"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setRowCount(len(items))
        for r, it in enumerate(items):
            it = it if isinstance(it, dict) else {}
            sev = str(it.get("severity") or "")
            code = str(it.get("code") or "")
            cnt = str(it.get("count") or "")
            msg = str(it.get("message") or "")
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(sev))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(code))
            table.setItem(r, 2, QtWidgets.QTableWidgetItem(cnt))
            table.setItem(r, 3, QtWidgets.QTableWidgetItem(msg))

        lay.addWidget(table, 1)
        return w

    def _build_quarantine_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        entries = self._qidx.get("entries")
        items = entries if isinstance(entries, list) else []

        # Sort: ERROR first, then WARN, then INFO, stable by path.
        def _rank(e: dict[str, Any]) -> tuple[int, str]:
            sev = str(e.get("severity") or "").upper()
            r = {"ERROR": 0, "WARN": 1, "INFO": 2}.get(sev, 3)
            return (r, str(e.get("rel_path") or ""))

        items2 = [it for it in items if isinstance(it, dict)]
        items2.sort(key=_rank)

        table = QtWidgets.QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Severity", "Path", "Codes", "Ticket"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setRowCount(min(len(items2), 500))

        for r, it in enumerate(items2[:500]):
            sev = str(it.get("severity") or "")
            rel = str(it.get("rel_path") or "")
            codes = ",".join([str(c) for c in (it.get("codes") or [])])
            ticket = str(it.get("ticket") or "")
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(sev))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(rel))
            table.setItem(r, 2, QtWidgets.QTableWidgetItem(codes))
            table.setItem(r, 3, QtWidgets.QTableWidgetItem(ticket))

        lay.addWidget(table, 1)

        if len(items2) > 500:
            lay.addWidget(
                QtWidgets.QLabel(
                    f"Showing first 500 entries out of {len(items2)}. See manifest/quarantine_index.json for full list."
                )
            )
        return w

    def _update_buttons(self) -> None:
        if self._status == "ERROR":
            self._btn_proceed.setEnabled(bool(self._chk_override.isChecked()))
        else:
            self._btn_proceed.setEnabled(True)

    def _accept(self) -> None:
        if self._status == "ERROR":
            self.preflight_override = bool(self._chk_override.isChecked())
            self.override_reason = str(self._edit_reason.text() or "").strip() or None
        self.accept()


def run_preflight_gate(
    parent: QtWidgets.QWidget | None,
    *,
    cfg: dict[str, Any],
    run_root: Path,
    show_on_ok: bool = False,
) -> PreflightGateResult:
    """Run preflight and (optionally) show the gate dialog.

    - If status is OK and ``show_on_ok=False``: returns allowed=True silently.
    - If status is WARN/ERROR: shows dialog.
    """

    run_root = Path(run_root)
    try:
        from scorpio_pipe.preflight import ensure_preflight

        # Ensure artifacts exist.
        _ = ensure_preflight(cfg, run_root)
    except Exception:
        # If preflight itself crashed, treat as ERROR and surface the report if any.
        pass

    report = _read_json(run_root / "manifest" / "preflight_report.json")
    night = _read_json(run_root / "manifest" / "night_profile.json")
    qidx = _read_json(run_root / "manifest" / "quarantine_index.json")

    status = _fmt_status(str((report or {}).get("status") or ""))
    if status == "OK" and not show_on_ok:
        return PreflightGateResult(allowed=True)

    dlg = PreflightGateDialog(
        parent,
        run_root=run_root,
        report=report,
        night_profile=night,
        quarantine_index=qidx,
    )
    res = dlg.exec()
    if res != QtWidgets.QDialog.Accepted:
        return PreflightGateResult(allowed=False)
    return PreflightGateResult(
        allowed=True,
        preflight_override=bool(dlg.preflight_override),
        override_reason=dlg.override_reason,
    )
