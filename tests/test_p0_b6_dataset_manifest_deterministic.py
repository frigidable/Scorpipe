import json

from scorpio_pipe.dataset.builder import FrameRecord, build_dataset_manifest_from_records
from scorpio_pipe.dataset.classify import FrameClass
from scorpio_pipe.dataset.manifest import DatasetManifest
from scorpio_pipe.metadata.frame_meta import parse_frame_meta


BASE_HDR = {
    "INSTRUME": "SCORPIO-2",
    "DATE-OBS": "2018-02-15",
    "TIME-OBS": "18:04:32.16",
    "EXPTIME": 5.0,
    "OBJECT": "Target",
    "IMAGETYP": "OBJECT",
    "MODE": "SPECTRA",
    "DISPERSE": "VPHG1200@540",
    "SLITWID": 1.0,
    "GAIN": 1.0,
    "RDNOISE": 3.0,
    # Geometry / binning
    "NAXIS1": 2048,
    "NAXIS2": 2048,
    "BINNING": "1 1",
    # Readout keys
    "READMODE": "1",
        "NODE": "0",
    "RATE": 100.0,
    "READRATE": "100",
    "READGAIN": "1.0",
}


def _rec(frame_id: str, hdr: dict, cls: FrameClass) -> FrameRecord:
    meta = parse_frame_meta(hdr, strict=True)
    return FrameRecord(frame_id=frame_id, path=f"{frame_id}.fits", meta=meta, frame_class=cls)


def test_dataset_manifest_json_is_deterministic(monkeypatch):
    # Freeze timestamp to make the JSON fully reproducible.
    monkeypatch.setattr(DatasetManifest, "now_utc_iso", staticmethod(lambda: "2000-01-01T00:00:00Z"))

    bias_hdr = dict(BASE_HDR)
    bias_hdr["IMAGETYP"] = "BIAS"

    flat_hdr = dict(BASE_HDR)
    flat_hdr["IMAGETYP"] = "FLAT"
    flat_hdr["OBJECT"] = "Flat"

    sci_hdr = dict(BASE_HDR)
    sci_hdr["IMAGETYP"] = "OBJECT"
    sci_hdr["OBJECT"] = "Science"

    records_a = [
        _rec("sci_0002", sci_hdr, FrameClass.SCIENCE),
        _rec("bias_0001", bias_hdr, FrameClass.BIAS),
        _rec("flat_0003", flat_hdr, FrameClass.FLAT),
    ]
    records_b = list(reversed(records_a))

    m1 = build_dataset_manifest_from_records(records_a, pipeline_version="6.0.25", night_id="15_02_2018")
    m2 = build_dataset_manifest_from_records(records_b, pipeline_version="6.0.25", night_id="15_02_2018")

    j1 = json.loads(m1.to_json_text(indent=2))
    j2 = json.loads(m2.to_json_text(indent=2))

    assert j1 == j2
