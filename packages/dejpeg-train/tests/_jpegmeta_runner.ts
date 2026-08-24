// Test runner: prints canonical JSON for one JPEG fixture path.
// Invoked by tests/test_jpegmeta.py as: bun <this> <fixture_path>
// File IO lives here (node:fs) so the parser module stays browser-pure.
import { readFileSync } from "node:fs";
import { parseJpeg, toCanonicalJson } from "../../inference-core/src/codec/jpegMeta.ts";

const path = process.argv[2];
if (!path) {
  console.error("usage: bun _jpegmeta_runner.ts <jpeg_path>");
  process.exit(2);
}
const meta = parseJpeg(new Uint8Array(readFileSync(path)));
console.log(toCanonicalJson(meta));
