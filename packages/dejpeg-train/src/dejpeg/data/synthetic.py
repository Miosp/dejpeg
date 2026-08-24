"""Synthetic non-photographic generator (spec §2.2).

Closes the gap no photographic corpus closes: JPEG ringing is worst on text and
graphics, and DF2K/LIU4K contain none. Emits 640x640 lossless RGB uint8 arrays.

Generators (matplotlib + PIL + numpy, fully deterministic under seed):
  * chart       -- line/bar/scatter plots, light/dark, varied colors
  * text_block  -- text across fonts/sizes/weights + supersample-AA vs aliased
  * gradient    -- hard-stepped gradients (banding failure mode)
  * flat        -- flat-colour regions with sharp boundaries
  * line_art    -- thin vector lines / polygons
  * ui_mockup   -- cards/buttons/dividers/labels

Optional HTML generator (`allow_html=True`) renders real web layout via Playwright
if Chromium is installed; falls back to ui_mockup otherwise. Install Chromium with
`uv run playwright install chromium` to enable the real HTML path.
"""
from __future__ import annotations

import io
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

SIZE = 640
_WORDS = (
    "the quick brown fox jumps over the lazy dog 0123456789 "
    "DeJPEGNet artifact block ringing banding contrast edge sharp "
    "AAAAA BBBBB CCCCC DDDDD EEEEE mistflame shadow horizon pixel "
    "function gradient quantization subsampling luminance chroma"
)


def _dejavu_paths() -> tuple[str, str]:
    regular = findfont(FontProperties(family="DejaVu Sans", weight="normal"))
    bold = findfont(FontProperties(family="DejaVu Sans", weight="bold"))
    return regular, bold


def _mpl_to_array(fig) -> np.ndarray:
    canvas = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return np.ascontiguousarray(buf[:, :, :3])


class SyntheticGenerator:
    def __init__(self, size: int = SIZE, allow_html: bool = False):
        self.size = size
        self.allow_html = allow_html
        self.reg_font, self.bold_font = _dejavu_paths()
        self._html_ok = allow_html and self._check_html()
        table = [
            ("chart", 0.18),
            ("text_block", 0.22),
            ("gradient", 0.15),
            ("flat", 0.15),
            ("line_art", 0.15),
            ("ui_mockup", 0.15),
        ]
        if self._html_ok:
            table.append(("html", 0.10))
            table = [(n, w * (1 - 0.10) if n != "html" else w) for n, w in table]
        self.names = [n for n, _ in table]
        self.weights = np.array([w for _, w in table], dtype=float)
        self.weights /= self.weights.sum()
        self._dispatch: dict[str, Callable[[np.random.RandomState], np.ndarray]] = {
            "chart": self._gen_chart,
            "text_block": self._gen_text_block,
            "gradient": self._gen_gradient,
            "flat": self._gen_flat,
            "line_art": self._gen_line_art,
            "ui_mockup": self._gen_ui_mockup,
            "html": self._gen_html if self._html_ok else self._gen_ui_mockup,
        }

    @staticmethod
    def _check_html() -> bool:
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return False
        try:
            with sync_playwright() as p:
                p.chromium.launch()
            return True
        except Exception:
            return False

    def generate(self, seed: int) -> np.ndarray:
        rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
        name = rng.choice(self.names, p=self.weights)
        arr = self._dispatch[name](rng)
        arr = self._finalize(arr)
        assert arr.shape == (self.size, self.size, 3) and arr.dtype == np.uint8
        return arr

    def _finalize(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] != self.size or arr.shape[1] != self.size:
            img = Image.fromarray(arr).resize((self.size, self.size), Image.NEAREST)
            arr = np.array(img)
        return np.ascontiguousarray(arr.astype(np.uint8))

    # ---- dark/light background picker ----
    def _bg(self, rng: np.random.RandomState) -> tuple[int, bool]:
        dark = rng.rand() < 0.5
        base = 255 if not dark else 0
        return base, dark

    # ---------------------------------------------------------------- charts
    def _gen_chart(self, rng: np.random.RandomState) -> np.ndarray:
        dark = rng.rand() < 0.5
        bg = "#0a0a0a" if dark else "#ffffff"
        fg = "#dddddd" if dark else "#111111"
        fig = plt.figure(figsize=(6.4, 6.4), dpi=100, facecolor=bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        kind = rng.choice(["line", "bar", "scatter"])
        n = rng.randint(5, 12)
        x = np.arange(n)
        y = rng.randn(n).cumsum()
        color = tuple(rng.rand(3))
        if kind == "line":
            ax.plot(x, y, color=color, lw=rng.choice([1, 2, 3]))
        elif kind == "bar":
            ax.bar(x, y, color=color)
        else:
            ax.scatter(x, y, c=[color], s=rng.randint(20, 120))
        ax.tick_params(colors=fg)
        for s in ax.spines.values():
            s.set_color(fg)
        ax.set_title("chart", color=fg)
        fig.tight_layout()
        return _mpl_to_array(fig)

    # ---------------------------------------------------------------- text
    def _gen_text_block(self, rng: np.random.RandomState) -> np.ndarray:
        base, dark = self._bg(rng)
        img = Image.new("RGB", (self.size, self.size), (base, base, base))
        draw = ImageDraw.Draw(img)
        supersample = rng.rand() < 0.5  # two AA modes: supersample vs 1x aliased
        scale = 2 if supersample else 1
        fg = (0, 0, 0) if not dark else (255, 255, 255)
        font_path = self.reg_font if rng.rand() < 0.5 else self.bold_font
        size = rng.choice([14, 18, 24, 32])
        font = ImageFont.truetype(font_path, size * scale)
        words = _WORDS.split()
        y = rng.randint(10, 40) * scale
        max_w = self.size * scale
        line_words: list[str] = []
        line = ""
        wi = 0
        while y < self.size * scale - size * scale and wi < len(words) * 3:
            w = words[rng.randint(0, len(words))]
            sep = " " if line else ""
            test = line + sep + w
            if draw.textlength(test, font=font) > max_w:
                draw.text((20 * scale, y), line, font=font, fill=fg)
                y += int(size * 1.5 * scale)
                line = w
            else:
                line = test
            wi += 1
        if line:
            draw.text((20 * scale, y), line, font=font, fill=fg)
        arr = np.array(img)
        if scale != 1:
            arr = np.array(Image.fromarray(arr).resize((self.size, self.size), Image.LANCZOS))
        return arr

    # ---------------------------------------------------------------- gradient
    def _gen_gradient(self, rng: np.random.RandomState) -> np.ndarray:
        steps = rng.randint(2, 8)  # few discrete bands -> banding
        axis = rng.choice([0, 1])
        band = self.size // steps
        c1 = rng.randint(0, 256, size=3)
        c2 = rng.randint(0, 256, size=3)
        bands = [((steps - i) * c1 + i * c2) // steps for i in range(steps)]
        row = np.concatenate(
            [np.tile(bands[i].astype(np.uint8), (band, self.size, 1)) for i in range(steps)],
            axis=0,
        )
        if row.shape[0] < self.size:
            pad = np.tile(row[-1:], (self.size - row.shape[0], 1, 1))
            row = np.concatenate([row, pad], axis=0)
        return row.transpose(1, 0, 2) if axis == 1 else row

    # ---------------------------------------------------------------- flat
    def _gen_flat(self, rng: np.random.RandomState) -> np.ndarray:
        base, _ = self._bg(rng)
        img = Image.new("RGB", (self.size, self.size), (base, base, base))
        draw = ImageDraw.Draw(img)
        n = rng.randint(3, 8)
        for _ in range(n):
            x0, y0 = rng.randint(0, self.size - 50, size=2)
            x1, y1 = rng.randint(x0 + 20, self.size), rng.randint(y0 + 20, self.size)
            color = tuple(int(c) for c in rng.randint(0, 256, size=3))
            draw.rectangle([x0, y0, x1, y1], fill=color)
        return np.array(img)

    # ---------------------------------------------------------------- line art
    def _gen_line_art(self, rng: np.random.RandomState) -> np.ndarray:
        dark = rng.rand() < 0.5
        bg = "#000000" if dark else "#ffffff"
        fg = "#ffffff" if dark else "#000000"
        fig = plt.figure(figsize=(6.4, 6.4), dpi=100, facecolor=bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        for _ in range(rng.randint(3, 9)):
            npts = rng.randint(2, 6)
            xs = rng.rand(npts)
            ys = rng.rand(npts)
            ax.plot(xs, ys, color=tuple(rng.rand(3)), lw=rng.choice([0.5, 1, 2]))
        return _mpl_to_array(fig)

    # ---------------------------------------------------------------- UI mockup
    def _gen_ui_mockup(self, rng: np.random.RandomState) -> np.ndarray:
        dark = rng.rand() < 0.5
        bg = (24, 24, 27) if dark else (250, 250, 250)
        card = (39, 39, 42) if dark else (255, 255, 255)
        accent = tuple(int(c) for c in rng.randint(60, 256, size=3))
        text = (229, 229, 229) if dark else (24, 24, 27)
        img = Image.new("RGB", (self.size, self.size), bg)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.reg_font, 16)
        for _ in range(rng.randint(2, 5)):
            x0 = rng.randint(0, self.size - 220)
            y0 = rng.randint(0, self.size - 160)
            w = rng.randint(180, 320)
            h = rng.randint(80, 160)
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=card, outline=accent)
            draw.text((x0 + 12, y0 + 10), "Label", font=font, fill=text)
            draw.rectangle([x0 + 12, y0 + h - 34, x0 + 80, y0 + h - 14], fill=accent)
        return np.array(img)

    # ---------------------------------------------------------------- HTML (optional)
    def _gen_html(self, rng: np.random.RandomState) -> np.ndarray:
        from playwright.sync_api import sync_playwright

        zoom = rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])
        dark = rng.rand() < 0.5
        bg = "#0b0b0b" if dark else "#ffffff"
        fg = "#eeeeee" if dark else "#111111"
        accent = "#3b82f6"
        html = f"""<html><head><meta charset='utf-8'>
        <style>body{{margin:0;background:{bg};color:{fg};font-family:sans-serif}}
        h1{{font-size:{rng.choice([24,32,40])}px;padding:16px}}
        p{{padding:0 16px;font-size:{rng.choice([12,14,16])}px;line-height:1.6}}
        .c{{border:1px solid {accent};margin:16px;padding:12px}}</style></head>
        <body><h1>DeJPEGNet</h1><div class='c'><p>{_WORDS}</p></div></body></html>"""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": self.size, "height": self.size}, device_scale_factor=1)
            page.set_content(html)
            png = page.screenshot(omit_background=False)
            browser.close()
        return np.array(Image.open(io.BytesIO(png)).convert("RGB"))
