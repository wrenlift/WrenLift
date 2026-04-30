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

// Use a namespace import so the JIT shim below can reach the
// `wren_*` runtime helpers without enumerating each name in the
// `import` list. The wlift_wasm wasm exports are returned by
// `init()` and stashed below for use as JIT-module imports.
import { createGpuBridge } from "./gpu-bridge.js";
import * as wlift_wasm from "./wlift_wasm.js";
const {
  default: init,
  version,
  run,
  run_hatch,
  run_with_hatches,
  resolve_future,
  resolve_future_bytes,
  reject_future,
  ws_open,
  ws_message,
  ws_close,
  wlift_extract_wasm_plugins,
  wlift_register_plugin_dynamic_export,
} = wlift_wasm;

// Async-bridge shims, identical surface to the inline page setup.
// The worker has its own `globalThis`, so we install a fresh copy
// — the page's shims aren't inherited.
globalThis._wlift_set_timeout = (handle, ms) => {
  setTimeout(() => resolve_future(handle, ""), ms);
};
// Workers can't requestAnimationFrame directly — it's a main-thread
// API. Fall back to a 16ms timer, which approximates 60 Hz without
// vsync alignment. Upgrade path: postMessage to main, main calls
// rAF, postMessages back, worker resolves the future. That's a
// follow-up tied to the OffscreenCanvas + DOM-event-shuttle work.
globalThis._wlift_next_frame = (handle) => {
  setTimeout(() => resolve_future(handle, ""), 16);
};
globalThis._wlift_fetch = (handle, url) => {
  fetch(url)
    .then((r) => r.text())
    .then((t) => resolve_future(handle, t))
    .catch((e) => reject_future(handle, String(e)));
};
// Worker-side binary fetch — see wlift.js MainWlift.init for the
// main-mode pair. `resolve_future_bytes` is destructured below.
globalThis._wlift_fetch_bytes = (handle, url) => {
  fetch(url)
    .then((r) => r.arrayBuffer())
    .then((ab) => resolve_future_bytes(handle, new Uint8Array(ab)))
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

// Storage shims — `localStorage` / `sessionStorage` exist on
// `WorkerGlobalScope`, so unlike DOM we can call them inline and
// resolve the future synchronously. No `dom-op` round-trip
// needed; main mode uses an identical install via wlift.js.
function pickStorage(scope) {
  return scope === "session" ? sessionStorage : localStorage;
}
globalThis._wlift_storage_get = (handle, scope, key) => {
  try {
    const v = pickStorage(scope).getItem(key);
    resolve_future(handle, v ?? "");
  } catch (e) { reject_future(handle, String(e)); }
};
globalThis._wlift_storage_set = (handle, scope, key, value) => {
  try {
    pickStorage(scope).setItem(key, value);
    resolve_future(handle, "");
  } catch (e) { reject_future(handle, String(e)); }
};
globalThis._wlift_storage_remove = (handle, scope, key) => {
  try {
    pickStorage(scope).removeItem(key);
    resolve_future(handle, "");
  } catch (e) { reject_future(handle, String(e)); }
};
globalThis._wlift_storage_clear = (handle, scope) => {
  try {
    pickStorage(scope).clear();
    resolve_future(handle, "");
  } catch (e) { reject_future(handle, String(e)); }
};

// Scheduler yield helper. The Rust `run()` async fn awaits this
// each tick to give the JS event loop a chance to drain BOTH
// microtasks (resolved Promises) AND macrotasks (setTimeout
// callbacks, fetch responses, message events). `setTimeout(0)`
// is the cheapest way to hop a macrotask boundary; clamped to
// ~4 ms by browsers under nested calls.
globalThis._wlift_yield_to_event_loop = () =>
  new Promise((resolve) => setTimeout(resolve, 0));

// Perf logger — `run()` brackets each phase and calls this so
// we can see where wall time goes. Gated by a global flag so the
// console isn't spammed by default; flip on via the page or
// devtools: `globalThis.__wlift_perf_enabled = true`.
globalThis._wlift_perf_log = (label, ms) => {
  if (globalThis.__wlift_perf_enabled) {
    console.log(`[wlift perf] ${label}: ${ms.toFixed(2)} ms`);
  }
};

// Tier-up Phase 2b — JIT instantiate + call shims. Mirror of
// `wlift.js`'s install for worker mode, since the worker has its
// own `globalThis`. We reach for the wlift_wasm namespace's
// `wren_*` JS wrappers as imports for the JIT'd module. Phase 5
// can switch to raw wasm exports for ~zero-overhead inter-module
// calls.
const __wliftJitInstances = [];
// Phase 5 — see `wlift.js` for the design notes; this is the
// worker-side mirror. Each JIT'd module imports the table and
// `call_indirect`s through it for inter-fn calls.
const __wliftJitTable = new WebAssembly.Table({ initial: 0, element: "anyfunc" });
// Bound at instantiation below — captured from `await init()`.
// Keep the binding declared up-front so the closures around it
// can be defined before init resolves.
let __wliftWrenImports = null;
globalThis._wlift_jit_instantiate = (bytes) => {
  const module = new WebAssembly.Module(bytes);
  const instance = new WebAssembly.Instance(module, { wren: __wliftWrenImports });
  // Find the single `fn_*` export and stash it directly — see
  // `wlift.js` for the rationale.
  let fn = null;
  for (const key of Object.keys(instance.exports)) {
    if (key.startsWith("fn_")) { fn = instance.exports[key]; break; }
  }
  const slot = __wliftJitInstances.length;
  __wliftJitInstances.push(fn);
  // Phase 5 — publish for inter-module `call_indirect`.
  __wliftJitTable.grow(1);
  __wliftJitTable.set(slot, fn);
  return slot;
};
// Defensive call shims — see `wlift.js` for the rationale.
const __wliftSafeCall = (slot, invoke, label) => {
  const target = __wliftJitInstances[slot];
  if (typeof target !== "function") {
    console.warn(`[wlift jit] ${label}: slot ${slot} is not a function (${typeof target})`);
    return 0x7FFC000000000000n;
  }
  const r = invoke(target);
  if (typeof r !== "bigint") {
    console.warn(`[wlift jit] ${label}: slot ${slot} returned non-BigInt (${typeof r}, ${r})`);
    return 0x7FFC000000000000n;
  }
  return r;
};
globalThis._wlift_jit_call_0 = (slot)       => __wliftSafeCall(slot, (fn) => fn(),   "call_0");
globalThis._wlift_jit_call_1 = (slot, a)    => __wliftSafeCall(slot, (fn) => fn(a),  "call_1");
globalThis._wlift_jit_call_2 = (slot, a, b) => __wliftSafeCall(slot, (fn) => fn(a, b), "call_2");

// Per-run reset — see wlift.js for rationale. Worker-mode
// install lives here.
globalThis._wlift_jit_reset = () => {
  for (let i = 0; i < __wliftJitInstances.length; i++) {
    try { __wliftJitTable.set(i, null); } catch (_) { /* table ref types vary by browser */ }
  }
  __wliftJitInstances.length = 0;
};

// `init()` returns the raw wasm exports object. Use those
// (NOT the wasm-bindgen JS wrappers from the namespace import)
// for the JIT-side imports — the wrappers do BigInt↔i64
// marshaling on every call, which the JIT'd code pays per
// cross-module call. With ~22k recursive calls in fib(20) that
// adds up; raw exports are direct wasm-to-wasm.
const wasm = await init();
__wliftWrenImports = {
  wren_num_add:   wasm.wren_num_add,
  wren_num_sub:   wasm.wren_num_sub,
  wren_num_mul:   wasm.wren_num_mul,
  wren_num_div:   wasm.wren_num_div,
  wren_num_mod:   wasm.wren_num_mod,
  wren_num_neg:   wasm.wren_num_neg,
  wren_cmp_lt:    wasm.wren_cmp_lt,
  wren_cmp_gt:    wasm.wren_cmp_gt,
  wren_cmp_le:    wasm.wren_cmp_le,
  wren_cmp_ge:    wasm.wren_cmp_ge,
  wren_cmp_eq:    wasm.wren_cmp_eq,
  wren_cmp_ne:    wasm.wren_cmp_ne,
  wren_not:       wasm.wren_not,
  wren_is_truthy: wasm.wren_is_truthy,
  wren_call_1_slow: wasm.wren_call_1,
  wren_jit_slot_plus_one: wasm.wren_jit_slot_plus_one,
  wren_jit_slot_for_module_var: wasm.wren_jit_slot_for_module_var,
  wren_get_module_var: wasm.wren_get_module_var,
  __wlift_jit_table: __wliftJitTable,
};
// Stash the namespace on the worker's globalThis so the worker-
// side `_wlift_jit_*` shims and JIT counters can be poked at
// during debugging. Main-thread console can't reach this one
// (different scope) — see `wlift.js` for the page-side mirror.
globalThis.wlift_wasm = wlift_wasm;

// ---------------------------------------------------------------
// Canvas + DOM bridge state (worker side)
// ---------------------------------------------------------------
// Main thread `transferControlToOffscreen()`s each `<canvas>`
// element with a known id and postMessages the OffscreenCanvas
// here. We register them keyed by both element id (for
// `wlift_dom_attach_canvas("stage-canvas")` lookups) and an
// integer handle (the value Wren-side code holds onto and the
// GPU bridge consumes).
//
// Events flow the other direction: main thread listens on the
// original `<canvas>` (which becomes a placeholder once
// transferred) and postMessages each event here, where it
// queues onto the matching canvas entry. `wlift_dom_drain_events_into_list`
// drains the queue into a Wren list.
const canvases = new Map();          // handle -> { element, queue, closeRequested, idAlias }
const canvasIdToHandle = new Map();  // string id -> handle
let nextCanvasHandle = 1;

function registerOffscreenCanvas(idAlias, offscreen) {
  // Re-attach replaces the previous entry — main thread may
  // re-transfer (mode switch / fresh boot) and the worker should
  // forget the old handle so it doesn't reuse a detached canvas.
  const prev = canvasIdToHandle.get(idAlias);
  if (prev !== undefined) canvases.delete(prev);
  const handle = nextCanvasHandle++;
  canvases.set(handle, {
    element: offscreen,
    queue: [],
    closeRequested: false,
    idAlias,
  });
  canvasIdToHandle.set(idAlias, handle);
  return handle;
}

// Build the GPU bridge once per worker. The factory closes over
// the canvases registry above so `host_gpu_attach_canvas(handle)`
// resolves through the same Map the DOM bridge populates.
const { bridge: gpuBridge, ensureGpuDevice } = createGpuBridge({
  wasm,
  canvases,
});

// Slot helpers — read plugin memory by (ptr, len). Mirrors the
// main-side `readPluginUtf8`. Plugin's WebAssembly.Instance lives
// in this worker too, so we get its memory via the closed-over
// `instance` ref each `installWasmPlugin` set up.
function makeWorkerDomBridge(getInstance) {
  const readPluginUtf8 = (ptr, len) => {
    if (len <= 0) return "";
    const view = new Uint8Array(getInstance().exports.memory.buffer, ptr, len);
    return new TextDecoder().decode(view);
  };

  return {
    // Worker can't `document.createElement` — it has no DOM.
    // Plugin code that expects to create a fresh canvas must
    // either use the BYO-canvas attach path
    // (`Window.create({"canvas": "<id>"})`) or pin the example
    // to main mode. The throw names the bridge so the failure
    // is self-explanatory.
    wlift_dom_create_canvas: () => {
      throw new Error(
        "[wlift dom/worker] wlift_dom_create_canvas: workers can't create DOM elements. " +
        "Use Window.create({\"canvas\": \"<id>\"}) against a <canvas> the page already owns, " +
        "or run this example in main mode.",
      );
    },

    // Worker-local lookup against the OffscreenCanvases the main
    // thread already transferred at run-with-hatches time.
    wlift_dom_attach_canvas: (id_ptr, id_len) => {
      const id = readPluginUtf8(id_ptr, id_len);
      if (!id) return 0;
      return canvasIdToHandle.get(id) ?? 0;
    },

    wlift_dom_destroy_canvas: (handle) => {
      const c = canvases.get(handle);
      if (!c) return;
      canvases.delete(handle);
      canvasIdToHandle.delete(c.idAlias);
      // OffscreenCanvas has no removeChild; the placeholder DOM
      // element on main remains for re-transfer on the next run.
    },

    // OffscreenCanvas exposes width/height directly — same shape
    // as HTMLCanvasElement.
    wlift_dom_canvas_width:  (handle) => canvases.get(handle)?.element.width  ?? 0,
    wlift_dom_canvas_height: (handle) => canvases.get(handle)?.element.height ?? 0,

    wlift_dom_close_requested: (handle) =>
      canvases.get(handle)?.closeRequested ? 1 : 0,

    // No DOM to set attributes on. No-op in worker (the canvas
    // has whatever attributes the page set before transfer).
    wlift_dom_set_canvas_attribute: () => {},

    // Drain the worker-local event queue into a Wren list. The
    // queue is fed by `dom-event` postMessages from main. Mirrors
    // `makeDomBridge`'s drain on the page side, which is the
    // shape `wlift_window_drain_events` expects: write a fresh
    // list into the receiver slot regardless of queue size, then
    // populate it with one map per event. Skipping the
    // wrenSetSlotNewList on an empty queue would leave the
    // receiver class in slot 0 — the Wren caller then reads
    // `events.count` on the class and gets "Class does not
    // implement 'count'".
    wlift_dom_drain_events_into_list: (vm, list_slot, handle) => {
      const c = canvases.get(handle);
      // Plugin already called `wrenEnsureSlots(vm, 6)` before
      // delegating; slots 0..5 are available.
      wasm.wrenSetSlotNewList(vm, list_slot);
      if (!c) return;
      const events = c.queue.splice(0, c.queue.length);
      const evSlot  = 4;
      const keySlot = 5;
      const valSlot = 3;
      for (const ev of events) {
        wasm.wrenSetSlotNewMap(vm, evSlot);
        for (const k of Object.keys(ev)) {
          writeStringSlot(vm, keySlot, k);
          const v = ev[k];
          if (typeof v === "string") writeStringSlot(vm, valSlot, v);
          else if (typeof v === "number") wasm.wrenSetSlotDouble(vm, valSlot, v);
          else if (typeof v === "boolean") wasm.wrenSetSlotBool(vm, valSlot, v);
          else wasm.wrenSetSlotNull(vm, valSlot);
          wasm.wrenSetMapValue(vm, evSlot, keySlot, valSlot);
        }
        wasm.wrenInsertInList(vm, list_slot, -1, evSlot);
      }
    },
  };
}

// Allocate a NUL-terminated copy of `str` in host memory and call
// wrenSetSlotString. wrenSetSlotString copies the bytes into Wren's
// heap so the host allocation can be released once the call returns.
// Mirrors `writeStringSlot` on the main side.
function writeStringSlot(vm, slot, str) {
  const bytes = new TextEncoder().encode(str);
  const len = bytes.length;
  const ptr = wasm.wlift_host_alloc(len + 1);
  const view = new Uint8Array(wasm.memory.buffer, ptr, len + 1);
  view.set(bytes);
  view[len] = 0;
  wasm.wrenSetSlotString(vm, slot, ptr);
  wasm.wlift_host_free(ptr, len + 1);
}

// ---------------------------------------------------------------
// Plugin loader — worker mirror of the main-thread plumbing in
// `wlift.js::MainWlift.init()`. The runtime (this worker) and
// every cdylib plugin live in the same wasm process, so plugin
// instances have to be created here. Symbol registration must
// happen in the same process too — `wlift_register_plugin_dynamic_export`
// stores a u32 index that the dispatcher then looks up to call
// the right plugin export.
//
// Layers 1 + 2 (passthrough wlift_wasm exports + slot bridges)
// are entirely worker-side and identical to the main-thread
// version. Layers 3 + 4 (DOM + GPU bridges) need browser-API
// state that lives on the main thread. Until a postMessage +
// SharedArrayBuffer proxy lands, the worker stubs throw a clear
// error if a plugin actually invokes those — pure-compute
// plugins (image, hashing, physics-on-CPU) load and run; window
// / GPU plugins still need `mode: "main"` for now.
// ---------------------------------------------------------------
const pluginInstances = new Map();
const dispatchTable   = [];

globalThis.wliftDynamicPluginDispatch = (idx, vm) => {
  const entry = dispatchTable[idx];
  if (!entry) {
    console.warn(`[wlift plugin/worker] dispatch ${idx}: no entry`);
    return;
  }
  try {
    entry.instance.exports[entry.exportName](vm);
  } catch (err) {
    console.error(
      `[wlift plugin/worker] ${entry.libName}.${entry.exportName} threw:`,
      err,
    );
  }
};

// Stub for any browser-API bridge a plugin imports but the
// worker can't service. The thrown message names the exact
// bridge so the failure mode is self-explanatory ("use main
// mode"). Pure-compute plugins import none of these and never
// trip the throw.
function makeBridgeStub(name) {
  return (...args) => {
    throw new Error(
      `[wlift plugin/worker] '${name}' is a DOM/GPU bridge that ` +
      `requires the main thread. Run this example in 'main' mode ` +
      `(the playground's mode picker). Args: ${JSON.stringify(args)}`,
    );
  };
}

function buildPluginEnv(libName, getInstance) {
  // Layer 3 (DOM bridge subset) needs `getInstance` for plugin-
  // memory utf8 reads; layer 4 (GPU bridge) is stateless wrt the
  // plugin instance. Compose them into one env object alongside
  // layers 1 + 2.
  const dom = makeWorkerDomBridge(getInstance);
  return {
    // Layer 1: passthrough of every wlift_wasm export. Plugin
    // calls into `wrenGetSlotDouble` etc. land directly in the
    // host wasm with no JS hop.
    ...wasm,
    // Layer 2: slot bridges. Both the host (this worker's
    // wlift_wasm) and the plugin live in this worker, so the
    // memcpy is direct — no postMessage round-trip needed.
    wlift_get_slot_str: (vm, slot, out_ptr, max) => {
      const host_ptr = wasm.wrenGetSlotString(vm, slot);
      if (!host_ptr) return -1;
      const host_view = new Uint8Array(wasm.memory.buffer);
      let len = 0;
      while (len < max && host_view[host_ptr + len] !== 0) len++;
      const inst = getInstance();
      const plugin_view = new Uint8Array(inst.exports.memory.buffer, out_ptr, max);
      plugin_view.set(host_view.subarray(host_ptr, host_ptr + len));
      return len;
    },
    wlift_set_slot_str: (vm, slot, ptr, len) => {
      const inst = getInstance();
      const plugin_view = new Uint8Array(inst.exports.memory.buffer, ptr, len);
      const host_ptr = wasm.wlift_host_alloc(len + 1);
      const host_view = new Uint8Array(wasm.memory.buffer, host_ptr, len + 1);
      host_view.set(plugin_view);
      host_view[len] = 0;
      wasm.wrenSetSlotString(vm, slot, host_ptr);
      wasm.wlift_host_free(host_ptr, len + 1);
    },
    // Layer 3: DOM bridge subset. Worker can't touch the real
    // DOM, but BYO-canvas attach + size queries + event drain
    // are all worker-local against the OffscreenCanvas the main
    // thread transfers in.
    ...dom,
    // Layer 4: full GPU bridge — same imports object MainWlift
    // uses, sourced from the shared `gpu-bridge.js` factory.
    // Runs against `OffscreenCanvas.getContext("webgpu")` in
    // worker scope; identical surface to main.
    ...gpuBridge,
    // Anything the plugin imports that isn't covered above (e.g.
    // unhandled `host_*` bridges) gets stubbed dynamically in
    // `installWasmPlugin` so the failure mode is a self-named
    // throw rather than a wasm validation error.
  };
}

async function installWasmPlugin(libName, wasmBytes) {
  if (pluginInstances.has(libName)) return;

  // GPU plugins need a `navigator.gpu` adapter+device pair
  // resolved before the plugin's first synchronous host_gpu_*
  // call. The Wren-side foreign methods can't await; pre-resolve
  // here while we're already in an async context. Cheap if
  // WebGPU is unavailable (returns null and `host_gpu_ready`
  // reports false).
  if (libName === "wlift_gpu" || libName === "wlift_gpu_web") {
    await ensureGpuDevice();
  }

  // Compile first so we can read the import list and stub every
  // missing `env.*` import dynamically. Hardcoded lists drift
  // every time a bridge is added on main; reading the module's
  // own imports keeps the worker side honest.
  const module = await WebAssembly.compile(wasmBytes);
  let instance;
  const env = buildPluginEnv(libName, () => instance);
  for (const desc of WebAssembly.Module.imports(module)) {
    if (desc.module !== "env" || desc.kind !== "function") continue;
    if (env[desc.name] !== undefined) continue;
    env[desc.name] = makeBridgeStub(desc.name);
  }

  instance = await WebAssembly.instantiate(module, { env });
  pluginInstances.set(libName, instance);

  // Walk plugin exports → register each `wlift_*` function in the
  // host's foreign-method registry. The host returns a u32 index
  // per registration; the dispatcher consults it on each foreign
  // call to route back to the right plugin instance.
  for (const exportName of Object.keys(instance.exports)) {
    if (!exportName.startsWith("wlift_")) continue;
    if (typeof instance.exports[exportName] !== "function") continue;
    const idx = wlift_register_plugin_dynamic_export(libName, exportName);
    dispatchTable[idx] = { instance, exportName, libName };
  }
}

// Drains every dep's `NativeLib` sections, instantiates each
// plugin once, registers its `wlift_*` exports. Idempotent on
// repeated bundles thanks to the `pluginInstances` guard.
async function installPluginsForDeps(deps) {
  for (const buf of deps) {
    const plugins = wlift_extract_wasm_plugins(buf);
    for (const p of plugins) {
      await installWasmPlugin(p.lib, p.bytes);
    }
  }
}

self.postMessage({ cmd: "ready", version: version() });

self.addEventListener("message", (e) => {
  const msg = e.data;
  if (msg.cmd === "transfer-canvases") {
    // Main side transferred one or more OffscreenCanvases via the
    // structured-clone transfer list. Each entry is
    // `{ id: <elementId>, offscreen: <OffscreenCanvas> }`.
    // Register them so the next `wlift_dom_attach_canvas("<id>")`
    // call from inside a plugin resolves locally.
    for (const c of (msg.canvases || [])) {
      registerOffscreenCanvas(c.id, c.offscreen);
    }
    return;
  }
  if (msg.cmd === "dom-event") {
    // Main thread captured a DOM event (keydown/keyup/mouse*) on
    // the page-side canvas placeholder and forwarded it. The
    // worker queues it on the matching canvas entry; the next
    // `wlift_dom_drain_events_into_list(handle)` call drains it.
    // `handle` is resolved by id alias rather than element ref
    // because the actual element ref doesn't cross the worker
    // boundary.
    const handle = canvasIdToHandle.get(msg.canvasId);
    const c = handle !== undefined ? canvases.get(handle) : null;
    if (c) c.queue.push(msg.event);
    return;
  }
  if (msg.cmd === "run-with-hatches") {
    const { id, source, deps } = msg;
    // Pre-installs each `@hatch:*` dep, then runs user source.
    // Plugin extraction has to happen *before* `run_with_hatches`
    // — `#!native` / `#!symbol` resolution at install time looks
    // the symbols up in the foreign-method registry that the
    // plugin's exports must already have populated.
    //
    // Don't bill plugin compile/instantiate time to the runtime
    // pill — `WebAssembly.compile` + `instantiate` for each
    // bundled plugin is build machinery, not user code. Start
    // the clock *after* `installPluginsForDeps` resolves so
    // the displayed elapsed-ms reflects only `run_with_hatches`.
    let t0;
    installPluginsForDeps(deps)
      .then(() => {
        t0 = performance.now();
        return run_with_hatches(source, deps);
      })
      .then(
        (result) => {
          self.postMessage({
            cmd: "result",
            id,
            output: result.output,
            ok: result.ok,
            errorKind: result.errorKind,
            elapsedMs: t0 != null ? performance.now() - t0 : 0,
          });
        },
        (err) => {
          self.postMessage({
            cmd: "result",
            id,
            output: String(err),
            ok: false,
            errorKind: -1,
            elapsedMs: t0 != null ? performance.now() - t0 : 0,
          });
        },
      );
  } else if (msg.cmd === "run-hatch") {
    const { id, bytes } = msg;
    // Same shape as `run`: drive `run_hatch` through the
    // scheduler. Async-bridge support inside hatch-bundled
    // modules works the same way as in plain source mode.
    const t0 = performance.now();
    run_hatch(bytes).then(
      (result) => {
        self.postMessage({
          cmd: "result",
          id,
          output: result.output,
          ok: result.ok,
          errorKind: result.errorKind,
          elapsedMs: performance.now() - t0,
        });
      },
      (err) => {
        self.postMessage({
          cmd: "result",
          id,
          output: String(err),
          ok: false,
          errorKind: -1,
          elapsedMs: performance.now() - t0,
        });
      },
    );
  } else if (msg.cmd === "run") {
    const { id, source } = msg;
    // `run` is async — it returns a Promise that resolves
    // once the scheduler loop has drained any parked fibers
    // (i.e. once Browser.fetch / setTimeout / WebSocket / etc.
    // have had a chance to settle).
    const t0 = performance.now();
    run(source).then(
      (result) => {
        self.postMessage({
          cmd: "result",
          id,
          output: result.output,
          ok: result.ok,
          errorKind: result.errorKind,
          elapsedMs: performance.now() - t0,
        });
      },
      (err) => {
        self.postMessage({
          cmd: "result",
          id,
          output: String(err),
          ok: false,
          errorKind: -1,
          elapsedMs: performance.now() - t0,
        });
      },
    );
  } else if (msg.cmd === "dom-result") {
    // Main thread completed a `dom-op`; settle the future from
    // here so the awaiting Wren fiber wakes on its next yield.
    resolve_future(msg.handle, msg.value ?? "");
  } else if (msg.cmd === "dom-error") {
    reject_future(msg.handle, msg.error ?? "DOM op failed");
  } else if (msg.cmd === "memory") {
    // Page asked for our wasm Memory. Only structurally
    // cloneable across the worker boundary when the buffer is
    // a SharedArrayBuffer — i.e. the wasm module was built with
    // threading. Without that, post `null` so the page falls
    // back to the plain worker boundary semantics.
    const isShared =
      typeof SharedArrayBuffer !== "undefined" &&
      wasm.memory?.buffer instanceof SharedArrayBuffer;
    self.postMessage({ cmd: "memory", memory: isShared ? wasm.memory : null });
  }
});
