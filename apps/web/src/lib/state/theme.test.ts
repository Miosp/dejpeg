import { test, expect, beforeEach } from "bun:test";
import { theme } from "./theme.svelte";

// Reset before each test
beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

// Read through a thunk so TS does not narrow `theme.current` to the literal
// assigned on the previous line — `toggle()` mutates it at runtime.
const readCurrent = () => theme.current;

test("defaults to system preference", () => {
  // matchMedia returns false in bun (no prefers-color-scheme), so default is "light"
  expect(theme.current).toBe("light");
});

test("toggle switches dark <-> light", () => {
  theme.current = "light";
  theme.toggle();
  expect(readCurrent()).toBe("dark");
  theme.toggle();
  expect(readCurrent()).toBe("light");
});

test("persists to localStorage", () => {
  theme.current = "dark";
  expect(localStorage.getItem("dejpeg:theme")).toBe('"dark"');
});

test("sets data-theme attribute on documentElement", () => {
  theme.current = "dark";
  expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
});
