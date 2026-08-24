// localStorage-backed reactive state. Returns a box with a `.value` property
// so reactivity is preserved for primitives as well as objects (Svelte 5 $state
// on a primitive is bound to the source identifier and cannot be returned from
// a function; wrapping in an object sidesteps that). Writes persist on change.

export interface Persisted<T> {
  value: T;
}

export function persisted<T>(key: string, initial: T): Persisted<T> {
  let stored: T;
  try {
    const raw = localStorage.getItem(key);
    stored = raw === null ? initial : (JSON.parse(raw) as T);
  } catch {
    stored = initial;
  }
  const box = $state<Persisted<T>>({ value: stored });
  $effect.root(() => {
    $effect(() => {
      try {
        localStorage.setItem(key, JSON.stringify(box.value));
      } catch {
        // ignore quota / SSR errors
      }
    });
  });
  return box;
}
