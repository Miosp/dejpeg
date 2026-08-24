import type { AstroIntegration } from "astro";
import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import path from "node:path";

const SW_SOURCE = "./src/service-worker/index.ts";

// Models are cached at runtime by inference-core's ModelCache (Cache API),
// never by the SW precache. See Task 6 spec + task-6-review.md Critical finding.
const PRECACHE_EXCLUDE_PREFIXES = ["models/"];

// Defense-in-depth: skip any single file larger than Cloudflare Pages' 25MB
// asset limit so a future large binary cannot regress into the precache list.
const MAX_PRECACHE_FILE_BYTES = 25 * 1024 * 1024;

async function collectPrecacheEntries(outDir: string): Promise<string[]> {
  const entries: string[] = [];
  const skip = new Set(["sw.js", "sw.js.map"]);

  async function walk(rel: string) {
    const abs = path.join(outDir, rel);
    const stat = await fs.stat(abs);
    if (stat.isDirectory()) {
      for (const child of await fs.readdir(abs)) {
        await walk(rel ? `${rel}/${child}` : child);
      }
    } else if (
      !skip.has(rel) &&
      !rel.endsWith(".map") &&
      !PRECACHE_EXCLUDE_PREFIXES.some((p) => rel.startsWith(p)) &&
      stat.size <= MAX_PRECACHE_FILE_BYTES
    ) {
      entries.push(`/${rel}`);
    }
  }

  await walk("");
  return entries;
}

export function serwist(): AstroIntegration {
  return {
    name: "serwist",
    hooks: {
      "astro:build:done": async ({ dir }) => {
        const { build } = await import("vite");
        const outDir = typeof dir === "string" ? dir : fileURLToPath(dir);
        const cwd = process.cwd();

        await build({
          configFile: false,
          root: cwd,
          logLevel: "warn",
          build: {
            lib: {
              entry: path.resolve(cwd, SW_SOURCE),
              formats: ["es"],
              fileName: () => "sw.js",
            },
            outDir: path.relative(cwd, outDir),
            emptyOutDir: false,
            rollupOptions: {
              output: { entryFileNames: "sw.js" },
            },
          },
        });

        const entries = await collectPrecacheEntries(outDir);
        const swPath = path.join(outDir, "sw.js");
        let sw = await fs.readFile(swPath, "utf8");
        sw = sw.replace(
          /self\.__SW_MANIFEST/g,
          JSON.stringify(entries),
        );
        await fs.writeFile(swPath, sw);

        // eslint-disable-next-line no-console
        console.log(`[serwist] precache entries injected: ${entries.length}`);
      },
    },
  };
}
