from __future__ import annotations

from typing import Any, Mapping

from scorpio_pipe.metadata import FrameMeta
from scorpio_pipe.ports.instrument import InstrumentAdapter, ModeKey, ModeProfile

from scorpio_pipe.instruments import scorpio2 as sc2


class Scorpio2Adapter:
    key = "scorpio2"
    display_name = "SCORPIO-2"

    def parse_header(
        self,
        header: Mapping[str, Any],
        *,
        strict: bool = True,
        policy: Any | None = None,
        fallback_sources: Any | None = None,
    ) -> FrameMeta:
        return sc2.parse_header(header, strict=strict, policy=policy, fallback_sources=fallback_sources)

    def mode_from_meta(self, meta: FrameMeta) -> ModeKey:
        return ModeKey(
            instrument=self.display_name,
            mode=str(meta.mode or ""),
            disperser=str(meta.disperser or ""),
            slit=str(meta.slit_width_key),
            binning=str(meta.binning_key),
            shape=f"{int(meta.naxis1)}x{int(meta.naxis2)}",
            readout=f"{meta.readout_key.node}|{meta.readout_key.rate:g}|{meta.readout_key.gain:g}",
        )

    def defaults_for_mode(self, mode: ModeKey) -> ModeProfile:
        return {
            "instrument": {"name": mode.instrument},
            "setup": {
                "mode": mode.mode,
                "disperser": mode.disperser,
                "slit": mode.slit,
                "binning": mode.binning,
                "shape": mode.shape,
            },
        }
