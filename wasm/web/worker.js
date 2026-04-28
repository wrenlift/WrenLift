// Web Worker host for the wlift_wasm interpreter.
//
// The main page used to import wlift_wasm directly and call
// `run(source)` on the UI thread. That works for short scripts
// but freezes the page on:
//
//   * heavy compute (fib(35) under interpreter, big binary_trees
//     loops, etc.) — wasm can't yield to the JS event loop while
//     we're inside `run()`,
//
//   * await loops with no immediate resolution — `Browser.fetch`
//     in particular can stall the entire UI for the round-trip
//     duration since the awaiting fiber's `Fiber.yield` returns
//     to the calling fiber, not to the JS event loop.
//
// Moving the interpreter into a Worker means the UI stays
// responsive regardless. The protocol is intentionally tiny:
//
//   main → worker:  { cmd: "run", id, source }
//   worker → main:  { cmd: "ready", version }
//   worker → main:  { cmd: "result", id, output, ok, errorKind, elapsedMs }
//
// `setTimeout` / `fetch` / `WebSocket` all exist in worker scope,
// so the wasm-side bridges work the same way as on the main
// thread — we just install the shims on the worker's `globalThis`
// instead of `window`. The JS shim list mirrors `index.html`
// exactly; if a new bridge lands, it has to be wired in both
// places (the worker is the runtime side, the page is the UI).
//
// DOM-touching bridges (Phase 1.2+):
// ─────────────────────────────────
// Workers don't have `document` / `window`. Bridges that need
// the DOM (querySelector, addEventListener on real elements,
// localStorage, navigator, etc.) won't work by direct call here.
// The pattern for those: the foreign method posts a message
// describing the operation to the main thread; the main thread
// runs it and posts back a `dom-result` with the original
// handle; we then call `resolve_future(handle, value)` from the
// worker's `message` listener. Same handle store, one extra
// postMessage hop. Implementation lands when the first DOM-
// dependent foreign class shows up.
//
// Canvas / WebGL / WebGPU: use `OffscreenCanvas` —
// `canvas.transferControlToOffscreen()` from main hands the
// rendering surface to the worker, which can then draw without
// touching the DOM.

import init, {
  version,
  run,
  resolve_future,
  reject_future,
  ws_open,
  ws_message,
  ws_close,
} from "../pkg/wlift_wasm.js";

// Async-bridge shims, identical surface to the inline page setup.
// The worker has its own `globalThis`, so we install a fresh copy
// — the page's shims aren't inherited.
globalThis._wlift_set_timeout = (handle, ms) => {
  setTimeout(() => resolve_future(handle, ""), ms);
};
globalThis._wlift_fetch = (handle, url) => {
  fetch(url)
    .then((r) => r.text())
    .then((t) => resolve_future(handle, t))
    .catch((e) => reject_future(handle, String(e)));
};

globalThis.__wlift_sockets ??= new Map();
globalThis._wlift_ws_open = (handle, url) => {
  const ws = new WebSocket(url);
  ws.addEventListener("open",    () => ws_open(handle));
  ws.addEventListener("message", (e) => ws_message(handle, String(e.data)));
  ws.addEventListener("close",   () => ws_close(handle));
  ws.addEventListener("error",   () => ws_close(handle));
  globalThis.__wlift_sockets.set(handle, ws);
};
globalThis._wlift_ws_send = (handle, text) => {
  globalThis.__wlift_sockets.get(handle)?.send(text);
};
globalThis._wlift_ws_close = (handle) => {
  const ws = globalThis.__wlift_sockets.get(handle);
  if (ws) {
    ws.close();
    globalThis.__wlift_sockets.delete(handle);
  }
};

// DOM bridges round-trip through the main thread — workers have
// no `document`. Each foreign method posts a `dom-op` describing
// the call; the page's `WorkerWlift` adapter executes it and
// posts back a `dom-result`/`dom-error` carrying the same handle.
// Adding a new DOM op here means: (a) a new `_wlift_dom_*` shim
// in this file that posts the op, (b) a matching `case` in
// `wlift.js`'s main-thread dispatcher.
globalThis._wlift_dom_text = (handle, selector) => {
  self.postMessage({ cmd: "dom-op", handle, op: "text", args: [selector] });
};
globalThis._wlift_dom_set_text = (handle, selector, value) => {
  self.postMessage({ cmd: "dom-op", handle, op: "setText", args: [selector, value] });
};
globalThis._wlift_dom_get_attribute = (handle, selector, name) => {
  self.postMessage({ cmd: "dom-op", handle, op: "getAttribute", args: [selector, name] });
};
globalThis._wlift_dom_set_attribute = (handle, selector, name, value) => {
  self.postMessage({ cmd: "dom-op", handle, op: "setAttribute", args: [selector, name, value] });
};
globalThis._wlift_dom_add_class = (handle, selector, name) => {
  self.postMessage({ cmd: "dom-op", handle, op: "addClass", args: [selector, name] });
};
globalThis._wlift_dom_remove_class = (handle, selector, name) => {
  self.postMessage({ cmd: "dom-op", handle, op: "removeClass", args: [selector, name] });
};
globalThis._wlift_dom_query_all = (handle, selector) => {
  self.postMessage({ cmd: "dom-op", handle, op: "queryAll", args: [selector] });
};

await init();
self.postMessage({ cmd: "ready", version: version() });

self.addEventListener("message", (e) => {
  const msg = e.data;
  if (msg.cmd === "run") {
    const { id, source } = msg;
    const t0 = performance.now();
    let result;
    try {
      result = run(source);
    } catch (err) {
      self.postMessage({
        cmd: "result",
        id,
        output: String(err),
        ok: false,
        errorKind: -1,
        elapsedMs: performance.now() - t0,
      });
      return;
    }
    self.postMessage({
      cmd: "result",
      id,
      output: result.output,
      ok: result.ok,
      errorKind: result.errorKind,
      elapsedMs: performance.now() - t0,
    });
  } else if (msg.cmd === "dom-result") {
    // Main thread completed a `dom-op`; settle the future from
    // here so the awaiting Wren fiber wakes on its next yield.
    resolve_future(msg.handle, msg.value ?? "");
  } else if (msg.cmd === "dom-error") {
    reject_future(msg.handle, msg.error ?? "DOM op failed");
  }
});
