export interface KeyboardShortcutOpts {
  onFit: () => void;
  onActualSize: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onToggleOriginal: (down: boolean) => void;
  onPrevItem: () => void;
  onNextItem: () => void;
  onProcessSelected: () => void;
  onRemoveItem: () => void;
  onToggleSettingsBasic: () => void;
  onToggleSettingsAdvanced: () => void;
}

export function setupKeyboardShortcuts(opts: KeyboardShortcutOpts): () => void {
  function onKeyDown(e: KeyboardEvent) {
    const target = e.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA") return;

    switch (e.code) {
      case "Digit0": e.preventDefault(); opts.onFit(); break;
      case "Digit1": e.preventDefault(); opts.onActualSize(); break;
      case "Equal":
      case "NumpadAdd": e.preventDefault(); opts.onZoomIn(); break;
      case "Minus":
      case "NumpadSubtract": e.preventDefault(); opts.onZoomOut(); break;
      case "Space":
        e.preventDefault();
        opts.onToggleOriginal(true);
        break;
      case "ArrowLeft": e.preventDefault(); opts.onPrevItem(); break;
      case "ArrowRight": e.preventDefault(); opts.onNextItem(); break;
      case "Enter": e.preventDefault(); opts.onProcessSelected(); break;
      case "Delete":
      case "Backspace": e.preventDefault(); opts.onRemoveItem(); break;
      case "KeyB": opts.onToggleSettingsBasic(); break;
      case "KeyA": opts.onToggleSettingsAdvanced(); break;
    }
  }

  function onKeyUp(e: KeyboardEvent) {
    if (e.code === "Space") {
      opts.onToggleOriginal(false);
    }
  }

  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);

  return () => {
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
  };
}
