from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from scorpio_pipe.instrument_db import find_grism, load_instrument_db


class InstrumentBrowserDialog(QtWidgets.QDialog):
    """Read-only viewer of instrument/grism reference tables."""

    def __init__(self, *, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Instrument database")
        self.resize(980, 670)
        # User preference: open windows maximized by default.
        try:
            self.setWindowState(self.windowState() | QtCore.Qt.WindowMaximized)
        except Exception:
            pass

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        top = QtWidgets.QLabel(
            "Reference: instrument/detector + grism tables used for UI hints and sanity checks.\n"
            "Ground truth remains FITS headers and your night-specific calibrations."
        )
        top.setWordWrap(True)
        outer.addWidget(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 1)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(320)
        split.addWidget(self.tree)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        split.addWidget(self.details)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        db = load_instrument_db()
        for inst_name in sorted(db.keys()):
            inst = db[inst_name]
            title = inst.display_name or inst.name
            if title != inst.name:
                title = f"{title} ({inst.name})"

            it_inst = QtWidgets.QTreeWidgetItem([title])
            it_inst.setData(0, QtCore.Qt.UserRole, ("instrument", inst.name, None))
            self.tree.addTopLevelItem(it_inst)

            if inst.grisms:
                for gid in sorted(inst.grisms.keys()):
                    it_g = QtWidgets.QTreeWidgetItem([gid])
                    it_g.setData(0, QtCore.Qt.UserRole, ("grism", inst.name, gid))
                    it_inst.addChild(it_g)

            it_inst.setExpanded(True)

        self.tree.currentItemChanged.connect(self._on_pick)
        if self.tree.topLevelItemCount() > 0:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.accept)
        outer.addWidget(btn, alignment=QtCore.Qt.AlignRight)

    def _on_pick(
        self,
        cur: QtWidgets.QTreeWidgetItem | None,
        _prev: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        if cur is None:
            return
        kind, inst_name, grism_id = cur.data(0, QtCore.Qt.UserRole)

        db = load_instrument_db()
        inst = db.get(inst_name)
        if inst is None:
            self.details.setPlainText("—")
            return

        if kind == "instrument":
            lines: list[str] = [f"Instrument: {inst.name}"]
            if inst.display_name and inst.display_name != inst.name:
                lines.append(f"Display: {inst.display_name}")

            if inst.plate_scale_arcsec_per_pix is not None:
                lines.append(
                    f"Plate scale: {inst.plate_scale_arcsec_per_pix} arcsec/pix"
                )
            if inst.fov_arcmin is not None:
                lines.append(
                    f"FOV (IM): {inst.fov_arcmin[0]} × {inst.fov_arcmin[1]} arcmin"
                )
            if inst.slit_length_arcmin is not None:
                lines.append(f"Slit length (LS): {inst.slit_length_arcmin} arcmin")

            if inst.detector:
                lines.append("")
                lines.append("Detector")
                for k, v in inst.detector.items():
                    lines.append(f"  {k}: {v}")

            if inst.grisms:
                lines.append("")
                lines.append(f"Grisms: {len(inst.grisms)}")

            self.details.setPlainText("\n".join(lines))
            return

        g = find_grism(inst.name, grism_id) or (
            inst.grisms.get(grism_id) if inst.grisms else None
        )
        if g is None:
            self.details.setPlainText("—")
            return

        lines: list[str] = [f"Instrument: {inst.name}", f"Grism: {g.id}", ""]
        if g.grooves_lmm is not None:
            lines.append(f"Grooves: {g.grooves_lmm} l/mm")
        if g.range_A is not None:
            lines.append(f"Range: {g.range_A[0]}–{g.range_A[1]} Å")
        if g.dispersion_A_per_pix is not None:
            lines.append(f"Dispersion: {g.dispersion_A_per_pix} Å/pix")
        if g.fwhm_A_at_slit_arcsec:
            lines.append("")
            lines.append("FWHM by slit")
            for slit, fwhm in sorted(
                g.fwhm_A_at_slit_arcsec.items(), key=lambda kv: kv[0]
            ):
                lines.append(f'  {slit}": {fwhm} Å')
        if g.notes:
            lines.append("")
            lines.append(f"Notes: {g.notes}")

        self.details.setPlainText("\n".join(lines))