from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from scorpio_pipe.instrument_db import load_instrument_db


class InstrumentBrowserDialog(QtWidgets.QDialog):
    def __init__(self, *, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Instrument database")
        self.resize(950, 650)

        db = load_instrument_db()

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        top = QtWidgets.QLabel(
            "Read-only reference: instrument/detector and grism tables used for UI hints and sanity checks.\n"
            "If you have exact manual values, update resources/instruments/scorpio_instruments.yaml."
        )
        top.setWordWrap(True)
        outer.addWidget(top)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 1)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        split.addWidget(self.tree)

        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        split.addWidget(self.details)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        self.tree.setMinimumWidth(300)

        # Populate
        for inst in db.instruments.values():
            it_inst = QtWidgets.QTreeWidgetItem([inst.name])
            it_inst.setData(0, QtCore.Qt.UserRole, ("instrument", inst.name, None))
            self.tree.addTopLevelItem(it_inst)
            if inst.grisms:
                for g in sorted(inst.grisms.values(), key=lambda x: x.id):
                    it_g = QtWidgets.QTreeWidgetItem([g.id])
                    it_g.setData(0, QtCore.Qt.UserRole, ("grism", inst.name, g.id))
                    it_inst.addChild(it_g)
            it_inst.setExpanded(True)

        self.tree.currentItemChanged.connect(self._on_pick)
        if self.tree.topLevelItemCount() > 0:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(self.accept)
        outer.addWidget(btn, alignment=QtCore.Qt.AlignRight)

    def _on_pick(self, cur: QtWidgets.QTreeWidgetItem | None, _prev: QtWidgets.QTreeWidgetItem | None) -> None:
        if cur is None:
            return
        kind, inst_name, grism_id = cur.data(0, QtCore.Qt.UserRole)
        db = load_instrument_db()
        inst = db.instruments.get(inst_name)
        if inst is None:
            self.details.setPlainText("—")
            return

        if kind == "instrument":
            lines: list[str] = [f"Instrument: {inst.name}", ""]
            if inst.notes:
                lines.append(f"Notes: {inst.notes}")
                lines.append("")
            if inst.detector:
                lines.append("Detector")
                for k, v in inst.detector.items():
                    lines.append(f"  {k}: {v}")
                lines.append("")
            if inst.slit_length_arcsec:
                lines.append(f"Slit length: {inst.slit_length_arcsec} arcsec")
            if inst.plate_scale_arcsec_per_pix:
                lines.append(f"Plate scale: {inst.plate_scale_arcsec_per_pix} arcsec/pix")
            if inst.grisms:
                lines.append("")
                lines.append(f"Grisms: {len(inst.grisms)}")
            self.details.setPlainText("\n".join(lines))
            return

        g = inst.grisms.get(grism_id or "") if inst.grisms else None
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
        if g.resolution_fwhm_A is not None:
            lines.append(f"FWHM: {g.resolution_fwhm_A} Å")
        if g.fwhm_by_slit_arcsec:
            lines.append("")
            lines.append("FWHM by slit")
            for slit, fwhm in sorted(g.fwhm_by_slit_arcsec.items()):
                lines.append(f"  {slit}\": {fwhm} Å")
        if g.notes:
            lines.append("")
            lines.append(f"Notes: {g.notes}")
        self.details.setPlainText("\n".join(lines))
