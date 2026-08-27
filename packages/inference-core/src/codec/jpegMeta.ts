/**
 * JPEG metadata parser — TypeScript twin of
 * `packages/dejpeg-train/src/dejpeg_train/data/jpegmeta.ts`.
 *
 * Two parsers, one truth: this module and the Python parser must produce
 * byte-identical canonical JSON on every fixture (cross-language test). If the
 * browser derives different conditioning than training did, the model is fed
 * something it never saw.
 *
 * Quant-table values are returned in natural (raster) order via de-zigzag;
 * both parsers apply the identical inverse-zigzag. Output keys are emitted in
 * sorted order to match Python's `json.dumps(sort_keys=True)`.
 */

// JPEG zigzag scan order: zigzag[i] = natural (raster) position of the i-th
// coefficient as stored in the DQT marker / entropy stream.
const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40,
  48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29,
  22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
  47, 55, 62, 63,
];

const SOI = 0xd8;
const RST0 = 0xd0;
const RST7 = 0xd7;
const EOI = 0xd9;
const SOS = 0xda;
const DQT = 0xdb;
const SOF_PROGRESSIVE = 0xc2;
const SOF_MARKERS = new Set([
  0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
]);
const STANDALONE = new Set([SOI, EOI, 0xfe, ...range(RST0, RST7)]);

function range(a: number, b: number): number[] {
  const out: number[] = [];
  for (let i = a; i <= b; i++) out.push(i);
  return out;
}

function dezigzag(raw: number[]): number[] {
  const natural = new Array<number>(64).fill(0);
  for (let i = 0; i < 64; i++) natural[ZIGZAG[i]] = raw[i];
  return natural;
}

export interface QuantTable {
  precision: number;
  values: number[];
}

export interface ComponentMeta {
  id: number;
  h: number;
  v: number;
  qt_id: number;
}

export interface JpegMeta {
  sof_marker: number;
  progressive: boolean;
  precision: number;
  height: number;
  width: number;
  num_components: number;
  components: ComponentMeta[];
  quant_tables: Record<string, QuantTable>;
}

export class JpegParseError extends Error {}

function u16be(buf: Uint8Array, off: number): number {
  return (buf[off] << 8) | buf[off + 1];
}

/** Parse JPEG metadata from raw bytes. Stops at the first SOS marker. */
export function parseJpeg(data: Uint8Array): JpegMeta {
  if (data.length < 4 || data[0] !== 0xff || data[1] !== SOI) {
    throw new JpegParseError("not a JPEG (missing SOI marker)");
  }

  let sofMarker = -1;
  let progressive = false;
  let precision = 0;
  let height = 0;
  let width = 0;
  let components: ComponentMeta[] = [];
  const quantTables = new Map<number, QuantTable>();

  let off = 2;
  const n = data.length;
  while (off + 1 < n) {
    if (data[off] !== 0xff) {
      off += 1;
      continue;
    }
    const marker = data[off + 1];
    if (marker === 0x00 || marker === 0xff) {
      off += 1;
      continue;
    }
    off += 2;
    if (marker === SOS) break;
    if (STANDALONE.has(marker)) continue;

    if (off + 2 > n) throw new JpegParseError("truncated segment length");
    const segLen = u16be(data, off);
    const segEnd = off + segLen;
    if (segLen < 2 || segEnd > n) {
      throw new JpegParseError(`invalid segment length ${segLen} at ${off}`);
    }

    const payload = data.subarray(off + 2, segEnd);

    if (marker === DQT) {
      parseDqt(payload, quantTables);
    } else if (SOF_MARKERS.has(marker)) {
      if (sofMarker !== -1) throw new JpegParseError("multiple SOF markers");
      sofMarker = marker;
      progressive = marker === SOF_PROGRESSIVE;
      [precision, height, width, components] = parseSof(payload);
    }

    off = segEnd;
  }

  if (sofMarker === -1) throw new JpegParseError("no SOF marker found");

  // Build quant_tables as a record keyed by string id, sorted by numeric id to
  // match the Python parser's `sorted(items())`.
  const qtRecord: Record<string, QuantTable> = {};
  for (const id of [...quantTables.keys()].sort((a, b) => a - b)) {
    qtRecord[String(id)] = quantTables.get(id)!;
  }

  return {
    sof_marker: sofMarker,
    progressive,
    precision,
    height,
    width,
    num_components: components.length,
    components,
    quant_tables: qtRecord,
  };
}

function parseDqt(payload: Uint8Array, quantTables: Map<number, QuantTable>): void {
  let i = 0;
  const pn = payload.length;
  while (i < pn) {
    if (i + 1 > pn) throw new JpegParseError("truncated DQT table header");
    const pqTq = payload[i];
    i += 1;
    const precision = pqTq >> 4;
    const tableId = pqTq & 0x0f;
    if (precision !== 0 && precision !== 1) {
      throw new JpegParseError(`invalid DQT precision ${precision}`);
    }
    const raw: number[] = new Array(64);
    for (let k = 0; k < 64; k++) {
      if (precision === 0) {
        raw[k] = payload[i + k];
      } else {
        raw[k] = u16be(payload, i + 2 * k);
      }
    }
    i += 64 * (precision === 1 ? 2 : 1);
    quantTables.set(tableId, { precision, values: dezigzag(raw) });
  }
}

function parseSof(payload: Uint8Array): [number, number, number, ComponentMeta[]] {
  if (payload.length < 6) throw new JpegParseError("truncated SOF");
  const precision = payload[0];
  const height = u16be(payload, 1);
  const width = u16be(payload, 3);
  const numComponents = payload[5];
  const comps: ComponentMeta[] = [];
  let off = 6;
  for (let c = 0; c < numComponents; c++) {
    if (off + 3 > payload.length) throw new JpegParseError("truncated SOF component");
    const id = payload[off];
    const hv = payload[off + 1];
    const qt = payload[off + 2];
    comps.push({ id, h: hv >> 4, v: hv & 0x0f, qt_id: qt });
    off += 3;
  }
  return [precision, height, width, comps];
}

/**
 * Canonical, byte-stable JSON for cross-language comparison. Keys are inserted
 * in sorted order to match Python's `json.dumps(sort_keys=True)`.
 */
export function toCanonicalJson(meta: JpegMeta): string {
  const components = meta.components.map((c) => ({
    h: c.h,
    id: c.id,
    qt_id: c.qt_id,
    v: c.v,
  }));
  const quantTables: Record<string, QuantTable> = {};
  for (const key of Object.keys(meta.quant_tables).sort()) {
    quantTables[key] = meta.quant_tables[key];
  }
  // Top-level keys in sorted order: components, height, num_components,
  // precision, progressive, quant_tables, sof_marker, width.
  const obj = {
    components,
    height: meta.height,
    num_components: meta.num_components,
    precision: meta.precision,
    progressive: meta.progressive,
    quant_tables: quantTables,
    sof_marker: meta.sof_marker,
    width: meta.width,
  };
  return JSON.stringify(obj);
}
