import { Subscribable } from "../../src/client/subscribe.js";
import { describe, it, expect } from "bun:test";

describe("Subscribable", () => {
  it("delivers emitted values to subscribers", () => {
    const subject = new (class extends Subscribable<number> {
      push(v: number) {
        this.emit(v);
      }
    })();
    const received: number[] = [];
    subject.subscribe((v) => received.push(v));
    subject.push(1);
    subject.push(2);
    expect(received).toEqual([1, 2]);
  });

  it("unsubscribe stops delivery", () => {
    const subject = new (class extends Subscribable<number> {
      push(v: number) {
        this.emit(v);
      }
    })();
    const received: number[] = [];
    const unsubscribe = subject.subscribe((v) => received.push(v));
    subject.push(1);
    unsubscribe();
    subject.push(2);
    expect(received).toEqual([1]);
  });

  it("delivers to multiple subscribers independently", () => {
    const subject = new (class extends Subscribable<string> {
      push(v: string) {
        this.emit(v);
      }
    })();
    const a: string[] = [];
    const b: string[] = [];
    subject.subscribe((v) => a.push(v));
    subject.subscribe((v) => b.push(v));
    subject.push("x");
    expect(a).toEqual(["x"]);
    expect(b).toEqual(["x"]);
  });
});
