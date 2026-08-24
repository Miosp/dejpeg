// apps/web/src/lib/state/theme.svelte.ts

export type ThemeName = "dark" | "light";

function getSystemTheme(): ThemeName {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function getStoredTheme(): ThemeName | null {
  try {
    const raw = localStorage.getItem("dejpeg:theme");
    if (raw === null) return null;
    const parsed = JSON.parse(raw);
    return parsed === "dark" || parsed === "light" ? parsed : null;
  } catch {
    return null;
  }
}

class ThemeState {
  #current: ThemeName = $state<ThemeName>("light");

  constructor() {
    if (typeof window !== "undefined") {
      this.#current = getStoredTheme() ?? getSystemTheme();
      this.#applyToDom();
    }
  }

  get current(): ThemeName {
    return this.#current;
  }

  set current(v: ThemeName) {
    this.#current = v;
    this.#persist();
    this.#applyToDom();
  }

  toggle() {
    this.current = this.#current === "dark" ? "light" : "dark";
  }

  #persist() {
    try {
      localStorage.setItem("dejpeg:theme", JSON.stringify(this.#current));
    } catch {
      // ignore quota / SSR errors
    }
  }

  #applyToDom() {
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", this.#current);
    }
  }
}

export const theme = new ThemeState();
