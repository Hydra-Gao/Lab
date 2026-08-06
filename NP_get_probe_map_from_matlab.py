"""

python NP_get_probe_map_from_matlab.py --data-folder "D:\Lab\Raw_data\TG963_Cb\Cb_2026_07_14_site2_1_g0\Cb_2026_07_14_site2_1_g0_imec0"

"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from kilosort.io import save_probe


@dataclass(frozen=True)
class Geometry:
    n_shank: int
    shank_width: float
    shank_pitch: float
    even_x_offset: float
    odd_x_offset: float
    horizontal_pitch: float
    vertical_pitch: float
    rows_per_shank: int
    electrodes_per_shank: int


# Direct Python translation of SGLXMetaToCoords.m -> getGeomParams/makeTypeMap.
_GEOMETRY_TYPES: Dict[str, Geometry] = {
    "np1_stag_70um": Geometry(1, 70, 0, 27, 11, 32, 20, 480, 960),
    "nhp_lin_70um": Geometry(1, 70, 0, 27, 27, 32, 20, 480, 960),
    "nhp_stag_125um_med": Geometry(1, 125, 0, 27, 11, 87, 20, 1368, 2496),
    "nhp_stag_125um_long": Geometry(1, 125, 0, 27, 11, 87, 20, 2208, 4416),
    "nhp_lin_125um_med": Geometry(1, 125, 0, 11, 11, 103, 20, 1368, 2496),
    "nhp_lin_125um_long": Geometry(1, 125, 0, 11, 11, 103, 20, 2208, 4416),
    "uhd_8col_1bank": Geometry(1, 70, 0, 14, 14, 6, 6, 48, 384),
    "uhd_8col_16bank": Geometry(1, 70, 0, 14, 14, 6, 6, 768, 6144),
    "np2_ss": Geometry(1, 70, 0, 27, 27, 32, 15, 640, 1280),
    "np2_4s": Geometry(4, 70, 250, 27, 27, 32, 15, 640, 1280),
    "NP1120": Geometry(1, 70, 0, 6.75, 6.75, 4.5, 4.5, 192, 384),
    "NP1121": Geometry(1, 70, 0, 6.25, 6.25, 3, 3, 384, 384),
    "NP1122": Geometry(1, 70, 0, 12.5, 12.5, 3, 3, 24, 384),
    "NP1123": Geometry(1, 70, 0, 10.25, 10.25, 4.5, 4.5, 32, 384),
    "NP1300": Geometry(1, 70, 0, 11, 11, 48, 20, 480, 960),
    "NP1200": Geometry(1, 70, 0, 27, 11, 32, 20, 64, 128),
    "NXT3000": Geometry(1, 70, 0, 53, 53, 0, 15, 128, 128),
}

_PART_TO_GEOMETRY = {
    "3A": "np1_stag_70um",
    "PRB_1_4_0480_1": "np1_stag_70um",
    "PRB_1_4_0480_1_C": "np1_stag_70um",
    "NP1010": "np1_stag_70um",
    "NP1011": "np1_stag_70um",
    "NP1012": "np1_stag_70um",
    "NP1013": "np1_stag_70um",
    "NP1015": "nhp_lin_70um",
    "NP1016": "nhp_lin_70um",
    "NP1017": "nhp_lin_70um",
    "NP1020": "nhp_stag_125um_med",
    "NP1021": "nhp_stag_125um_med",
    "NP1030": "nhp_stag_125um_long",
    "NP1031": "nhp_stag_125um_long",
    "NP1022": "nhp_lin_125um_med",
    "NP1032": "nhp_lin_125um_long",
    "NP1100": "uhd_8col_1bank",
    "NP1110": "uhd_8col_16bank",
    "PRB2_1_2_0640_0": "np2_ss",
    "PRB2_1_4_0480_1": "np2_ss",
    "NP2000": "np2_ss",
    "NP2003": "np2_ss",
    "NP2004": "np2_ss",
    "PRB2_4_2_0640_0": "np2_4s",
    "PRB2_4_4_0480_1": "np2_4s",
    "NP2010": "np2_4s",
    "NP2013": "np2_4s",
    "NP2014": "np2_4s",
    "NP1120": "NP1120",
    "NP1121": "NP1121",
    "NP1122": "NP1122",
    "NP1123": "NP1123",
    "NP1300": "NP1300",
    "NP1200": "NP1200",
    "NXT3000": "NXT3000",
}


def read_meta(meta_path: Path) -> Dict[str, str]:
    """Read SpikeGLX metadata exactly as key=value text fields."""
    meta: Dict[str, str] = {}
    with meta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lstrip("~")
            meta[key] = value.strip()
    if not meta:
        raise ValueError(f"No metadata fields were read from: {meta_path}")
    return meta


def parse_parenthesized_entries(text: str) -> Iterable[str]:
    return re.findall(r"\(([^()]*)\)", text)


def parse_geom_map(meta: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Python equivalent of MATLAB geomMapToGeom()."""
    raw = meta["snsGeomMap"]
    entries = list(parse_parenthesized_entries(raw))
    if len(entries) < 2:
        raise ValueError("snsGeomMap does not contain a header plus channel entries.")

    header = [part.strip() for part in entries[0].split(",")]
    if len(header) < 4:
        raise ValueError(f"Unexpected snsGeomMap header: {entries[0]!r}")

    n_shank = int(float(header[1]))
    shank_pitch = float(header[2])
    shank_width = float(header[3])

    rows = []
    for entry in entries[1:]:
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 4:
            raise ValueError(f"Unexpected snsGeomMap entry: {entry!r}")
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))

    arr = np.asarray(rows, dtype=np.float64)
    shank_ind = arr[:, 0].astype(np.int32)
    local_x = arr[:, 1]
    y = arr[:, 2]
    connected = arr[:, 3].astype(bool)

    if np.any(shank_ind < 0) or np.any(shank_ind >= n_shank):
        raise ValueError("snsGeomMap contains an out-of-range shank index.")

    _ = shank_width  # Parsed for exact MATLAB equivalence/documentation.
    return shank_ind, local_x, y, connected, shank_pitch


def channel_counts_im(meta: Dict[str, str]) -> Tuple[int, int, int]:
    values = [int(x) for x in re.split(r"[,\s]+", meta["snsApLfSy"].strip()) if x]
    if len(values) < 3:
        raise ValueError(f"Unexpected snsApLfSy value: {meta['snsApLfSy']!r}")
    return values[0], values[1], values[2]


def geometry_for_meta(meta: Dict[str, str]) -> Geometry:
    part_number = meta.get("imDatPrb_pn", "3A")
    geom_type = _PART_TO_GEOMETRY.get(part_number)
    if geom_type is None:
        raise ValueError(
            f"Unsupported probe part number {part_number!r}. "
            "Use a .meta file containing snsGeomMap, or add its geometry definition."
        )
    return _GEOMETRY_TYPES[geom_type]


def parse_shank_map(meta: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Python equivalent of MATLAB shankMapToGeom()."""
    ap_count, _, _ = channel_counts_im(meta)
    raw = meta["snsShankMap"]
    entries = list(parse_parenthesized_entries(raw))
    if len(entries) < 2:
        raise ValueError("snsShankMap does not contain a header plus channel entries.")

    rows = []
    for entry in entries[1:]:
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 4:
            raise ValueError(f"Unexpected snsShankMap entry: {entry!r}")
        rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))

    if len(rows) < ap_count:
        raise ValueError(
            f"snsShankMap has only {len(rows)} entries but snsApLfSy reports {ap_count} AP channels."
        )

    arr = np.asarray(rows[:ap_count], dtype=np.int64)
    shank_ind = arr[:, 0].astype(np.int32)
    col_ind = arr[:, 1]
    row_ind = arr[:, 2]
    connected = arr[:, 3].astype(bool)

    geom = geometry_for_meta(meta)
    local_x = col_ind.astype(np.float64) * geom.horizontal_pitch
    even_rows = (row_ind % 2) == 0
    local_x[even_rows] += geom.even_x_offset
    local_x[~even_rows] += geom.odd_x_offset
    y = row_ind.astype(np.float64) * geom.vertical_pitch

    return shank_ind, local_x, y, connected, geom.shank_pitch


def build_matlab_equivalent_probe(meta_path: Path) -> dict:
    """
    Build the same Kilosort map variables as SGLXMetaToCoords.m outType=1.

    Important:
      * snsGeomMap is preferred over snsShankMap.
      * chanMap stays in saved binary-channel order: 0..N-1.
      * xc includes physical shank offset.
      * kcoords is physical shank index + 1, exactly like MATLAB.
    """
    meta = read_meta(meta_path)

    if "snsGeomMap" in meta:
        source = "snsGeomMap"
        shank_ind, local_x, y, connected, shank_pitch = parse_geom_map(meta)
    elif "snsShankMap" in meta:
        source = "snsShankMap + hard-coded probe geometry"
        shank_ind, local_x, y, connected, shank_pitch = parse_shank_map(meta)
    else:
        raise ValueError("Metadata contains neither snsGeomMap nor snsShankMap.")

    n_chan = len(local_x)
    if not (len(shank_ind) == len(y) == len(connected) == n_chan):
        raise ValueError("Probe coordinate arrays have inconsistent lengths.")

    # Exact MATLAB coordinate convention:
    # xcoords = shankInd * shankPitch + xCoord
    xc = shank_ind.astype(np.float64) * shank_pitch + local_x

    # Exact MATLAB indexing convention:
    # chanMap0ind = 0..N-1; kcoords = shankInd + 1
    probe = {
        "chanMap": np.arange(n_chan, dtype=np.int32),
        "xc": xc.astype(np.float32),
        "yc": y.astype(np.float32),
        "kcoords": (shank_ind + 1).astype(np.int32),
        "n_chan": int(n_chan),
    }

    # save_probe JSON does not use MATLAB's 'connected' field. We validate and report it.
    disconnected = np.flatnonzero(~connected)
    print(f"Geometry source: {source}")
    print(f"Probe part number: {meta.get('imDatPrb_pn', '3A')}")
    print(f"Saved channels in map: {n_chan}")
    print(f"Physical shanks present: {np.unique(shank_ind).tolist()}")
    print(f"Kilosort kcoords: {np.unique(probe['kcoords']).tolist()}")
    print(f"Disconnected/use=0 entries: {disconnected.size}")
    if disconnected.size:
        print("WARNING: disconnected entries are retained, matching the MATLAB chanMap construction.")
        print(f"First disconnected map rows: {disconnected[:20].tolist()}")

    return probe


def find_ap_meta(data_folder: Path) -> Path:
    candidates = sorted(data_folder.glob("*.ap.meta"))
    if not candidates:
        candidates = sorted(data_folder.glob("*.meta"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .meta file found in {data_folder}")
    if len(candidates) > 1:
        names = "\n  ".join(str(p) for p in candidates)
        raise RuntimeError(
            "Multiple metadata files found. Pass --meta explicitly:\n  " + names
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a Kilosort probe JSON using SGLXMetaToCoords.m-equivalent geometry logic."
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=None,
        help="SpikeGLX recording folder. Used only to locate the AP .meta file and default output path.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Explicit AP .meta path. Overrides --data-folder auto-detection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Kilosort probe JSON path.",
    )
    args = parser.parse_args()

    if args.meta is None and args.data_folder is None:
        parser.error("Provide --meta or --data-folder.")

    meta_path = args.meta if args.meta is not None else find_ap_meta(args.data_folder)
    meta_path = meta_path.expanduser().resolve()
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)

    output_path = args.output
    if output_path is None:
        output_path = meta_path.parent / f"{meta_path.stem}_matlab_equivalent_probe.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    probe = build_matlab_equivalent_probe(meta_path)
    save_probe(probe, output_path)

    print(f"Saved Kilosort probe file: {output_path}")
    print("First 10 rows:")
    for i in range(min(10, probe["n_chan"])):
        print(
            f"row={i:3d} binary_ch={int(probe['chanMap'][i]):3d} "
            f"x={float(probe['xc'][i]):7.2f} y={float(probe['yc'][i]):7.2f} "
            f"kcoord={int(probe['kcoords'][i])}"
        )


if __name__ == "__main__":
    main()
