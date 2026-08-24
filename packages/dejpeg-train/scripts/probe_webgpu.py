"""Probe whether headless Chromium exposes WebGPU under WSL2.

Determines if the Phase-0.5 browser check can run here, or whether we need the
Windows host Chrome. Prints: navigator.gpu present, adapter available, adapter
info. Exit 0 on WebGPU available, 1 otherwise.
"""
import sys
from playwright.sync_api import sync_playwright

WEBGPU_FLAGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,WebGPU",
    "--use-vulkan=swiftshader",
    "--disable-vulkan-surface",
    "--ignore-gpu-blocklist",
    "--enable-unsafe-swiftshader",
]


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=WEBGPU_FLAGS,
        )
        ctx = browser.new_context()
        page = ctx.new_page()
        msgs = []
        page.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: msgs.append(f"[pageerror] {e}"))
        page.set_content("<html><body></body></html>")
        result = page.evaluate(
            """async () => {
                const out = { gpu: typeof navigator.gpu !== 'undefined' };
                if (!out.gpu) return out;
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    out.adapter = !!adapter;
                    if (adapter) {
                        const info = adapter.info || {};
                        out.vendor = info.vendor || info.architecture || 'unknown';
                        out.desc = (adapter.info && adapter.info.description) || 'n/a';
                    }
                } catch (e) { out.adapter = false; out.err = String(e); }
                return out;
            }"""
        )
        browser.close()
    print("navigator.gpu:", result.get("gpu"))
    print("adapter:", result.get("adapter"))
    if "vendor" in result:
        print("adapter vendor:", result.get("vendor"))
        print("adapter desc:", result.get("desc"))
    if "err" in result:
        print("adapter error:", result.get("err"))
    if msgs:
        print("--- console (first 10) ---")
        for m in msgs[:10]:
            print(m)
    ok = bool(result.get("gpu") and result.get("adapter"))
    print("WEBSGPU_OK:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
