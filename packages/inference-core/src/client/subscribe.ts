export type Listener<T> = (value: T) => void;

export interface Subscription {
  unsubscribe(): void;
}

export class Subscribable<T> {
  private listeners = new Set<Listener<T>>();

  subscribe(listener: Listener<T>): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  protected emit(value: T): void {
    for (const listener of this.listeners) listener(value);
  }
}
