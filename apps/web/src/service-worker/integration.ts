import type { AstroIntegration } from "astro";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const SW_SOURCE = "./src/service-worker/index.ts";

// Models are cached at runtime by inference-core's ModelCache (Cache API),
// never by the SW precache. See Task 6 spec + task-6-review.md Critical finding.
const PRECACHE_EXCLUDE_PREFIXES = ["models/"];

// Cloudflare Workers Static Assets consumes these at deploy time and never
// serves them; precaching them fails install with bad-precaching-response.
const PRECACHE_EXCLUDE_FILES = new Set(["sw.js", "sw.js.map", "_headers", "_redirects"]);

// Defense-in-depth: skip any single file larger than Cloudflare Pages' 25MB
// asset limit so a future large binary cannot regress into the precache list.
const MAX_PRECACHE_FILE_BYTES = 25 * 1024 * 1024;

interface PrecacheEntry {
  url: string;
  revision: string;
}

async function collectPrecacheEntries(outDir: string): Promise<PrecacheEntry[]> {
  const entries: PrecacheEntry[] = [];

  async function walk(rel: string) {
    const abs = path.join(outDir, rel);
    const stat = await fs.stat(abs);
    if (stat.isDirectory()) {
      for (const child of await fs.readdir(abs)) {
        await walk(rel ? `${rel}/${child}` : child);
      }
    } else if (
      !PRECACHE_EXCLUDE_FILES.has(rel) &&
      !rel.endsWith(".map") &&
      !PRECACHE_EXCLUDE_PREFIXES.some((p) => rel.startsWith(p)) &&
      stat.size <= MAX_PRECACHE_FILE_BYTES
    ) {
      const content = await fs.readFile(abs);
      entries.push({
        url: `/${rel}`,
        revision: createHash("md5").update(content).digest("hex"),
      });
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
          // Library-mode builds skip Vite's client defines, which leaves raw
          // `process.env.NODE_ENV` references in the serwist runtime and
          // crashes the SW at evaluation time in the browser.
          define: {
            "process.env.NODE_ENV": JSON.stringify("production"),
          },
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
