import type { ClientInbound, HostOutbound } from "../host/protocol.js";

export interface HostTransport {
  postMessage(msg: ClientInbound, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<HostOutbound>) => void) | null;
  onerror: ((e: ErrorEvent) => void) | null;
  terminate(): void;
}

export class WorkerTransport implements HostTransport {
  constructor(private readonly worker: Worker) {}

  postMessage(msg: ClientInbound, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer ?? []);
  }

  get onmessage(): ((e: MessageEvent<HostOutbound>) => void) | null {
    return this.worker.onmessage as ((e: MessageEvent<HostOutbound>) => void) | null;
  }
  set onmessage(value: ((e: MessageEvent<HostOutbound>) => void) | null) {
    this.worker.onmessage = value as ((e: MessageEvent) => void) | null;
  }

  get onerror(): ((e: ErrorEvent) => void) | null {
    return this.worker.onerror;
  }
  set onerror(value: ((e: ErrorEvent) => void) | null) {
    this.worker.onerror = value;
  }

  terminate(): void {
    this.worker.terminate();
  }
}
