/** Conditioning transform: JPEG luma quant table -> 65-D model input vector.
 *
 * Layout: [log-normalized 64 quant values, validity flag].
 *   log(x + 1) / log(256) maps the [1, 255] quant range into [0, 1].
 *   validity = 1 when a real table was parsed, 0 when dropped or unrecoverable.
 *
 * Byte-identical twin of packages/dejpeg-train/src/dejpeg_train/model/conditioning.py.
 * The Phase-0.5 cross-language parity test enforces this.
 */
import { parseJpeg, type JpegMeta } from "./jpegMeta";

const LOG256 = Math.log(256);

export function quantTableToCondition(
  table: number[] | Float32Array,
  validity = 1,
): Float32Array {
  const out = new Float32Array(65);
  for (let i = 0; i < 64; i++) {
    const v = Number(table[i]);
    out[i] = Math.log(v + 1) / LOG256;
  }
  out[64] = validity;
  return out;
}

export function buildConditionFromJpeg(bytes: Uint8Array): Float32Array {
  let meta: JpegMeta;
  try {
    meta = parseJpeg(bytes);
  } catch {
    return quantTableToCondition(new Array(64).fill(0), 0);
  }
  const luma = meta.quant_tables["0"]?.values;
  if (!luma || luma.length !== 64) {
    return quantTableToCondition(new Array(64).fill(0), 0);
  }
  return quantTableToCondition(luma, 1);
}
