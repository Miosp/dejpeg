"""JPEG metadata parser.

Walks JPEG markers to extract the DQT (quantization) tables, SOF component
sampling, and baseline-vs-progressive flag. This is the source of the q_table
conditioning signal (spec section 1, signal 1).

Two parsers exist with one truth: this module and
``packages/inference-core/src/codec/jpegMeta.ts``. They must produce
byte-identical output on every fixture (cross-language test). If the browser
derives different conditioning than training did, the model is fed something it
never saw.

Output is a canonical, JSON-serializable dict. Quant-table values are returned
in natural (raster) order via de-zigzag, not the zigzag order stored in the
file, because the canonical meaning of a quantization table is the natural-order
matrix and both parsers apply the identical inverse-zigzag.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

# JPEG zigzag scan order: zigzag[i] is the natural (raster) position of the
# i-th coefficient as stored in the DQT marker / entropy stream.
ZIGZAG = (
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
)

SOI = 0xD8
RST0, RST7 = 0xD0, 0xD7
EOI = 0xD9
SOS = 0xDA
DQT = 0xDB
DHT = 0xC4
COM = 0xFE
# SOF markers (start of frame). C0 baseline, C1 extended sequential,
# C2 progressive, C3 lossless; C5/C6/C9/C10/C11 the differential variants.
SOF_PROGRESSIVE = 0xC2
SOF_MARKERS = frozenset(
    (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)
)
# Standalone markers carry no length field.
STANDALONE = frozenset((SOI, EOI, COM)) | set(range(RST0, RST7 + 1))


def _dezigzag(raw: list[int]) -> list[int]:
    natural = [0] * 64
    for i, v in enumerate(raw):
        natural[ZIGZAG[i]] = v
    return natural


@dataclass
class QuantTable:
    table_id: int
    precision: int  # 0 = 8-bit, 1 = 16-bit
    values: list[int]  # 64 values, natural (raster) order

    def canonical(self) -> dict[str, Any]:
        return {"precision": self.precision, "values": list(self.values)}


@dataclass
class Component:
    component_id: int
    h: int  # horizontal sampling factor
    v: int  # vertical sampling factor
    qt_id: int  # quantization table id used by this component

    def canonical(self) -> dict[str, Any]:
        return {"id": self.component_id, "h": self.h, "v": self.v, "qt_id": self.qt_id}


@dataclass
class JpegMeta:
    sof_marker: int
    progressive: bool
    precision: int
    height: int
    width: int
    components: list[Component] = field(default_factory=list)
    quant_tables: dict[int, QuantTable] = field(default_factory=dict)

    def canonical(self) -> dict[str, Any]:
        return {
            "sof_marker": self.sof_marker,
            "progressive": self.progressive,
            "precision": self.precision,
            "height": self.height,
            "width": self.width,
            "num_components": len(self.components),
            "components": [c.canonical() for c in self.components],
            "quant_tables": {
                str(tid): qt.canonical()
                for tid, qt in sorted(self.quant_tables.items())
            },
        }

    def to_canonical_json(self) -> str:
        """Canonical, byte-stable JSON for cross-language comparison."""
        return json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))


class JpegParseError(ValueError):
    pass


def _read_u16be(buf: bytes, off: int) -> int:
    return struct.unpack_from(">H", buf, off)[0]


def parse_jpeg(data: bytes) -> JpegMeta:
    """Parse JPEG metadata from raw bytes.

    Stops at the first SOS (start of scan); all DQT/SOF markers precede it.
    Raises ``JpegParseError`` on malformed input.
    """
    if len(data) < 4 or data[0] != 0xFF or data[1] != SOI:
        raise JpegParseError("not a JPEG (missing SOI marker)")

    sof_marker: int | None = None
    progressive = False
    precision = 0
    height = 0
    width = 0
    components: list[Component] = []
    quant_tables: dict[int, QuantTable] = {}

    off = 2
    n = len(data)
    while off + 1 < n:
        if data[off] != 0xFF:
            # Not at a marker boundary; skip a byte. Should not happen between
            # segments, but be defensive against padding.
            off += 1
            continue
        marker = data[off + 1]
        # Fill bytes (0xFF padding) are allowed; skip them.
        if marker == 0x00 or marker == 0xFF:
            off += 1
            continue
        off += 2
        if marker == SOS:
            break
        if marker in STANDALONE or marker in (0xD0,):
            continue

        if off + 2 > n:
            raise JpegParseError("truncated segment length")
        seg_len = _read_u16be(data, off)
        seg_end = off + seg_len
        if seg_len < 2 or seg_end > n:
            raise JpegParseError(f"invalid segment length {seg_len} at {off}")

        payload = data[off + 2 : seg_end]

        if marker == DQT:
            _parse_dqt(payload, quant_tables)
        elif marker in SOF_MARKERS:
            if sof_marker is not None:
                raise JpegParseError("multiple SOF markers")
            sof_marker = marker
            progressive = marker == SOF_PROGRESSIVE
            precision, height, width, components = _parse_sof(payload)

        off = seg_end

    if sof_marker is None:
        raise JpegParseError("no SOF marker found")

    return JpegMeta(
        sof_marker=sof_marker,
        progressive=progressive,
        precision=precision,
        height=height,
        width=width,
        components=components,
        quant_tables=quant_tables,
    )


def _parse_dqt(payload: bytes, quant_tables: dict[int, QuantTable]) -> None:
    i = 0
    pn = len(payload)
    while i < pn:
        if i + 1 > pn:
            raise JpegParseError("truncated DQT table header")
        pq_tq = payload[i]
        i += 1
        precision = pq_tq >> 4
        table_id = pq_tq & 0x0F
        if precision not in (0, 1):
            raise JpegParseError(f"invalid DQT precision {precision}")
        count = 64 * (2 if precision == 1 else 1)
        if i + count > pn:
            raise JpegParseError("truncated DQT table values")
        raw: list[int] = []
        for k in range(64):
            if precision == 0:
                raw.append(payload[i + k])
            else:
                raw.append(_read_u16be(payload, i + 2 * k))
        i += count
        quant_tables[table_id] = QuantTable(
            table_id=table_id,
            precision=precision,
            values=_dezigzag(raw),
        )


def _parse_sof(
    payload: bytes,
) -> tuple[int, int, int, list[Component]]:
    if len(payload) < 6:
        raise JpegParseError("truncated SOF")
    precision = payload[0]
    height = _read_u16be(payload, 1)
    width = _read_u16be(payload, 3)
    num_components = payload[5]
    comps: list[Component] = []
    off = 6
    for _ in range(num_components):
        if off + 3 > len(payload):
            raise JpegParseError("truncated SOF component")
        cid = payload[off]
        hv = payload[off + 1]
        qt = payload[off + 2]
        comps.append(Component(cid, hv >> 4, hv & 0x0F, qt))
        off += 3
    return precision, height, width, comps


def parse_file(path: str) -> JpegMeta:
    with open(path, "rb") as f:
        return parse_jpeg(f.read())
