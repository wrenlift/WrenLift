// Façade over the wlift_wasm wasm-bindgen module.
//
// Two execution modes share one async API:
//
//   import { createWlift } from "./wlift.js";
//   const wlift = await createWlift();              // worker (default)
//   const wlift = await createWlift({ mode: "main" }); // main thread
//
//   wlift.version           // string
//   await wlift.run(source) // { output, ok, errorKind, elapsedMs }
//   wlift.memory            // WebAssembly.Memory | null
//   wlift.terminate()
//
// `mode: "worker"` (default for compute-only workloads) —
// wlift_wasm runs inside `worker.js`. The page never sees the
// wasm linear memory directly; everything crosses the boundary
// as structured-cloned messages. UI thread stays responsive
// during heavy compute.
//
// **Async-bridge caveat:** today's worker mode does NOT have a
// top-level await scheduler. Foreign methods that resolve
// asynchronously — `Browser.fetch`, `Browser.setTimeout`,
// `Browser.connect`, and worker-mode `Dom.*` — settle their
// future *after* the awaiting fiber's `Fiber.yield` returned to
// the script's outer fiber, and nothing wakes that outer fiber
// back up. The first `await` returns `null`. Sync compute, plus
// any foreign method whose JS shim resolves its future inline,
// still works fine. For now: pick `mode: "main"` to demo the
// async bridges (the JS shims are inline-resolving in main mode
// for sync ops; truly-async ones still need the scheduler).
// The scheduler lands in a future phase.
//
// `mode: "main"` (opt-in) — wlift_wasm runs inline on the same
// thread as the page. `wlift.memory` returns the live
// `WebAssembly.Memory` so the page can read Wren-heap bytes
// directly (`new Uint8Array(wlift.memory.buffer, ptr, len)`)
// without the postMessage marshal cost. Tradeoff: long-running
// scripts freeze the UI; this mode is for pages that prioritise
// memory-throughput over responsiveness — e.g. an in-page tool
// streaming many KB/frame to a canvas, or a debugger that
// inspects Wren strings without copying them out.
//
// `mode: "worker", shared: true` (opt-in, gated on environment) —
// when `crossOriginIsolated` is true and SAB is available, asks
// the worker to expose its `WebAssembly.Memory` (now backed by
// `SharedArrayBuffer`) to the page via a one-shot
// `postMessage({cmd: "memory"})`. The page can then read worker-
// side wasm bytes directly with no marshal cost — useful for
// streaming large outputs. Falls back to plain worker mode if
// the page isn't crossOriginIsolated. Page must be served with
// COOP/COEP headers; `serve.js` next to this file sets both.
//
// Threaded wasm-pack build (required for true SAB-backed wasm):
//   RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
//     wasm-pack build wasm --target web --release \
//                          --no-default-features -- -Z build-std=std,panic_abort
// Needs nightly Rust. Without the threaded build, the wasm
// module's memory isn't shared even if SAB is allowed; the
// `shared: true` flag falls through to plain worker mode.

export async function createWlift(opts = {}) {
  const { mode = "worker", shared = false } = opts;
  if (mode === "worker") return new WorkerWlift({ shared }).init();
  if (mode === "main")   return new MainWlift().init();
  throw new Error(`createWlift: unknown mode '${mode}', expected 'worker' or 'main'`);
}

// Returns true iff SAB is available *and* the page is
// crossOriginIsolated (i.e. served with COOP/COEP). The wasm
// module still needs to be built with threading for the wasm
// memory itself to be SAB-backed; `shared: true` opts the page
// in to *requesting* it from the worker.
export function sharedMemorySupported() {
  return typeof SharedArrayBuffer !== "undefined"
      && typeof crossOriginIsolated !== "undefined"
      && crossOriginIsolated === true;
}

// ---------------------------------------------------------------------------
// Worker mode (default)
// ---------------------------------------------------------------------------

class WorkerWlift {
  constructor({ shared = false } = {}) {
    this.mode    = "worker";
    this.version = null;
    // SAB requires both crossOriginIsolated AND a SAB-aware
    // wasm build. We probe the page side (cheap); the worker
    // side resolves on `cmd: "memory"` reply — if the wasm
    // module isn't SAB-built, the memory comes back as a
    // regular ArrayBuffer-backed `WebAssembly.Memory` which
    // can't cross worker → page, so the request is a no-op.
    this._sharedRequested = shared && sharedMemorySupported();
    this._memory = null;
    this.worker  = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.nextId  = 1;
    this._ready  = new Promise((resolve) => { this._resolveReady = resolve; });

    this.worker.addEventListener("message", (e) => {
      const m = e.data;
      if (m.cmd === "ready") {
        this.version = m.version;
        if (this._sharedRequested) {
          this.worker.postMessage({ cmd: "memory" });
        } else {
          this._resolveReady();
        }
      } else if (m.cmd === "memory") {
        // Worker handed back its `WebAssembly.Memory` (only
        // structurally cloneable when its buffer is a SAB).
        // If the worker's wasm isn't SAB-built we get null
        // here; surface as `wlift.memory === null` (same as
        // plain worker mode).
        this._memory = m.memory ?? null;
        this._resolveReady();
      } else if (m.cmd === "result") {
        const resolve = this.pending.get(m.id);
        if (resolve) {
          this.pending.delete(m.id);
          resolve(m);
        }
      } else if (m.cmd === "dom-op") {
        // Worker can't reach `document`; we run the op on the
        // page's behalf and post the answer back. Each `op` is
        // matched by a `_wlift_dom_*` shim in `worker.js`.
        try {
          const value = dispatchDomOp(m.op, m.args);
          this.worker.postMessage({ cmd: "dom-result", handle: m.handle, value });
        } catch (err) {
          this.worker.postMessage({ cmd: "dom-error", handle: m.handle, error: String(err) });
        }
      }
    });
  }

  async init() {
    await this._ready;
    return this;
  }

  run(source) {
    return new Promise((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve);
      this.worker.postMessage({ cmd: "run", id, source });
    });
  }

  // Run a `.hatch` byte stream — multi-module bundle produced by
  // `wlift src/ --bundle out.hatch --bundle-target wasm32-unknown-unknown`.
  // The bundle's `target` manifest field is checked against the
  // wasm runtime's family marker; cross-family bundles fail
  // load with a `WrongTarget` message in the result.
  runHatch(bytes) {
    return new Promise((resolve) => {
      const id = this.nextId++;
      this.pending.set(id, resolve);
      // Transfer the underlying buffer so we don't pay a copy on
      // postMessage. `bytes` should be a Uint8Array.
      const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      this.worker.postMessage(
        { cmd: "run-hatch", id, bytes: buf },
        [buf.buffer],
      );
    });
  }

  // SAB-mode: the page can read worker-side wasm bytes directly
  // through `memory.buffer`. Plain worker mode: returns null
  // (the boundary is the point). Feature-probe with `if (wlift.memory)`.
  get memory() { return this._memory; }

  terminate() { this.worker.terminate(); }
}

// ---------------------------------------------------------------------------
// Main-thread mode (opt-in)
// ---------------------------------------------------------------------------

class MainWlift {
  constructor() {
    this.mode    = "main";
    this.version = null;
    this._mod    = null;
    this._memory = null;
  }

  async init() {
    const mod = await import("../pkg/wlift_wasm.js");

    // Same async-bridge shims `worker.js` installs, but on the
    // page's `globalThis` instead. The bridges are workload-side
    // (setTimeout / fetch / WebSocket exist on both window and
    // WorkerGlobalScope), so the install is a verbatim copy.
    globalThis._wlift_set_timeout = (handle, ms) => {
      setTimeout(() => mod.resolve_future(handle, ""), ms);
    };
    globalThis._wlift_fetch = (handle, url) => {
      fetch(url)
        .then((r) => r.text())
        .then((t) => mod.resolve_future(handle, t))
        .catch((e) => mod.reject_future(handle, String(e)));
    };
    globalThis.__wlift_sockets ??= new Map();
    globalThis._wlift_ws_open = (handle, url) => {
      const ws = new WebSocket(url);
      ws.addEventListener("open",    () => mod.ws_open(handle));
      ws.addEventListener("message", (e) => mod.ws_message(handle, String(e.data)));
      ws.addEventListener("close",   () => mod.ws_close(handle));
      ws.addEventListener("error",   () => mod.ws_close(handle));
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

    // DOM shims — main mode runs `document.*` directly (we're on
    // the page thread; no postMessage hop). The worker mode's
    // round-trip uses the same `dispatchDomOp` table below, so
    // both paths share the implementation of each op.
    // Each main-mode shim is a thin wrapper around the same
    // `dispatchDomOp` table the worker mode uses on the page
    // side. One implementation, two callers.
    const settleSync = (handle, op, args) => {
      try {
        mod.resolve_future(handle, dispatchDomOp(op, args));
      } catch (err) {
        mod.reject_future(handle, String(err));
      }
    };
    globalThis._wlift_dom_text          = (h, s)        => settleSync(h, "text", [s]);
    globalThis._wlift_dom_set_text      = (h, s, v)     => settleSync(h, "setText", [s, v]);
    globalThis._wlift_dom_get_attribute = (h, s, n)     => settleSync(h, "getAttribute", [s, n]);
    globalThis._wlift_dom_set_attribute = (h, s, n, v)  => settleSync(h, "setAttribute", [s, n, v]);
    globalThis._wlift_dom_add_class     = (h, s, n)     => settleSync(h, "addClass", [s, n]);
    globalThis._wlift_dom_remove_class  = (h, s, n)     => settleSync(h, "removeClass", [s, n]);
    globalThis._wlift_dom_query_all     = (h, s)        => settleSync(h, "queryAll", [s]);

    // Storage shims — same install in both modes since
    // localStorage/sessionStorage are sync and worker-safe.
    const pickStorage = (scope) => (scope === "session" ? sessionStorage : localStorage);
    globalThis._wlift_storage_get = (handle, scope, key) => {
      try {
        const v = pickStorage(scope).getItem(key);
        mod.resolve_future(handle, v ?? "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };
    globalThis._wlift_storage_set = (handle, scope, key, value) => {
      try {
        pickStorage(scope).setItem(key, value);
        mod.resolve_future(handle, "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };
    globalThis._wlift_storage_remove = (handle, scope, key) => {
      try {
        pickStorage(scope).removeItem(key);
        mod.resolve_future(handle, "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };
    globalThis._wlift_storage_clear = (handle, scope) => {
      try {
        pickStorage(scope).clear();
        mod.resolve_future(handle, "");
      } catch (e) { mod.reject_future(handle, String(e)); }
    };

    // Scheduler yield helper — see `worker.js` for rationale.
    globalThis._wlift_yield_to_event_loop = () =>
      new Promise((resolve) => setTimeout(resolve, 0));

    // Perf logger — gated by `globalThis.__wlift_perf_enabled`.
    // Same install in worker.js so worker-mode runs are
    // diagnosable too.
    globalThis._wlift_perf_log = (label, ms) => {
      if (globalThis.__wlift_perf_enabled) {
        console.log(`[wlift perf] ${label}: ${ms.toFixed(2)} ms`);
      }
    };

    // wasm-bindgen `--target web` returns the wasm exports object
    // from the default `init()`. `exports.memory` is the live
    // `WebAssembly.Memory` instance — that's what main-mode
    // callers want for direct buffer access.
    const wasm = await mod.default();
    this._mod    = mod;
    this._memory = wasm.memory;
    this.version = mod.version();

    // Expose the module namespace to `globalThis` so the dev
    // console can poke at JIT counters / debug exports in main
    // mode without an extra `import` step. Worker mode stashes
    // its own copy on the worker's globalThis (see worker.js)
    // — the page-side `wlift_wasm` only reflects main-mode
    // state. Two separate instances; that's working as
    // designed.
    globalThis.wlift_wasm = mod;

    // Tier-up Phase 2b — JIT instantiate + call shims. The Rust
    // side calls `wlift_wasm::tier_up::emit_*` to produce wasm
    // bytes; these shims compile + instantiate them with the
    // `wren_*` runtime helpers as imports, store the instance
    // in a per-page table, and expose `call_0` / `call_1` /
    // `call_2` arities for invocation. Must run AFTER
    // `await mod.default()` so the `wasm` exports object is
    // populated for the import bindings. Phase 4 generalises
    // the call shape; for now arity caps at 2 (fib's signature).
    const __wliftJitInstances = [];
    // Phase 5 — shared funcref table. Each JIT'd module imports
    // `wren.__wlift_jit_table` and emits `call_indirect` through
    // it for inter-fn calls (skipping the JS hop). We grow the
    // table by one entry per `_wlift_jit_instantiate` and
    // `set(slot, fn)` so all modules see every JIT'd function at
    // its host-side slot. No upper bound declared on the table
    // type, so growth is unbounded by the wasm spec — the runtime
    // caps via `wasm_jit_slot()` lookup, not the table.
    const __wliftJitTable = new WebAssembly.Table({ initial: 0, element: "anyfunc" });
    const __wliftWrenImports = {
      wren_num_add:  wasm.wren_num_add,
      wren_num_sub:  wasm.wren_num_sub,
      wren_num_mul:  wasm.wren_num_mul,
      wren_num_div:  wasm.wren_num_div,
      wren_num_mod:  wasm.wren_num_mod,
      wren_num_neg:  wasm.wren_num_neg,
      wren_cmp_lt:   wasm.wren_cmp_lt,
      wren_cmp_gt:   wasm.wren_cmp_gt,
      wren_cmp_le:   wasm.wren_cmp_le,
      wren_cmp_ge:   wasm.wren_cmp_ge,
      wren_cmp_eq:   wasm.wren_cmp_eq,
      wren_cmp_ne:   wasm.wren_cmp_ne,
      wren_not:      wasm.wren_not,
      wren_is_truthy:wasm.wren_is_truthy,
      wren_call_1_slow: wasm.wren_call_1,
      wren_jit_slot_plus_one: wasm.wren_jit_slot_plus_one,
      wren_jit_slot_for_module_var: wasm.wren_jit_slot_for_module_var,
      wren_get_module_var: wasm.wren_get_module_var,
      __wlift_jit_table: __wliftJitTable,
    };
    globalThis._wlift_jit_instantiate = (bytes) => {
      const module = new WebAssembly.Module(bytes);
      const instance = new WebAssembly.Instance(module, {
        wren: __wliftWrenImports,
      });
      // `emit_mir` exports exactly one `fn_<symbol_index>` per
      // module — find it once and stash the function reference
      // directly so subsequent calls don't pay name-lookup cost.
      let fn = null;
      for (const key of Object.keys(instance.exports)) {
        if (key.startsWith("fn_")) { fn = instance.exports[key]; break; }
      }
      const slot = __wliftJitInstances.length;
      __wliftJitInstances.push(fn);
      // Phase 5 — publish the funcref so other JIT'd modules can
      // `call_indirect` to us. Order matters: the slot returned
      // here must match `wasm_jit_slot()` on the Rust side, since
      // `wren_jit_slot_plus_one` resolves callees against that
      // index.
      __wliftJitTable.grow(1);
      __wliftJitTable.set(slot, fn);
      return slot;
    };
    // Defensive shim — if the wasm function returns undefined
    // (mismatched signature, unbound import filled with stub,
    // wrong slot), surface the failure as `null`-bits rather
    // than letting wasm-bindgen throw "Cannot convert undefined
    // to BigInt" deep in the dispatch path. The console.warn
    // makes the misuse visible without crashing the whole run.
    const __wliftSafeCall = (slot, fn, label) => {
      const target = __wliftJitInstances[slot];
      if (typeof target !== "function") {
        console.warn(`[wlift jit] ${label}: slot ${slot} is not a function (${typeof target})`);
        return 0x7FFC000000000000n; // Value::null() — TAG_NULL = QNAN
      }
      const r = fn(target);
      if (typeof r !== "bigint") {
        console.warn(`[wlift jit] ${label}: slot ${slot} returned non-BigInt (${typeof r}, ${r}); falling back to null`);
        return 0x7FFC000000000000n;
      }
      return r;
    };
    globalThis._wlift_jit_call_0 = (slot)       => __wliftSafeCall(slot, (fn) => fn(),   "call_0");
    globalThis._wlift_jit_call_1 = (slot, a)    => __wliftSafeCall(slot, (fn) => fn(a),  "call_1");
    globalThis._wlift_jit_call_2 = (slot, a, b) => __wliftSafeCall(slot, (fn) => fn(a, b), "call_2");

    // Per-run reset — wlift_wasm calls this at the top of each
    // `run()` so JIT'd modules from prior VMs become
    // garbage-collectable. We can't shrink the funcref Table
    // (no API), so we null-out every entry and reset the
    // companion array. The JIT'd wasm modules themselves drop
    // out of scope once the table releases its references.
    globalThis._wlift_jit_reset = () => {
      for (let i = 0; i < __wliftJitInstances.length; i++) {
        try { __wliftJitTable.set(i, null); } catch (_) { /* table ref types vary by browser */ }
      }
      __wliftJitInstances.length = 0;
    };

    return this;
  }

  async runHatch(bytes) {
    // Same shape as `run`, but the user code is a `.hatch`
    // bundle. `_mod.run_hatch` is the wasm-bindgen export added
    // alongside `run` — both go through the same VM-setup +
    // scheduler-loop machinery in `wlift_wasm::lib::run_inner`.
    const t0 = performance.now();
    let result;
    try {
      const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      result = await this._mod.run_hatch(buf);
    } catch (err) {
      return {
        cmd: "result",
        output: String(err),
        ok: false,
        errorKind: -1,
        elapsedMs: performance.now() - t0,
      };
    }
    return {
      cmd: "result",
      output: result.output,
      ok: result.ok,
      errorKind: result.errorKind,
      elapsedMs: performance.now() - t0,
    };
  }

  async run(source) {
    // `_mod.run` is `pub async fn` on the Rust side, so it
    // returns a Promise. `await` it so the scheduler loop
    // (which drains parked fibers across `await` points) gets
    // to drive async bridges to completion before we hand back
    // a result.
    const t0 = performance.now();
    let result;
    try {
      result = await this._mod.run(source);
    } catch (err) {
      return {
        cmd: "result",
        output: String(err),
        ok: false,
        errorKind: -1,
        elapsedMs: performance.now() - t0,
      };
    }
    // Same shape as the worker reply so the UI doesn't have to
    // care which mode it's talking to.
    return {
      cmd: "result",
      output: result.output,
      ok: result.ok,
      errorKind: result.errorKind,
      elapsedMs: performance.now() - t0,
    };
  }

  get memory() { return this._memory; }

  terminate() {
    // No worker to kill; nothing to do. Provided for API parity.
  }
}

// ---------------------------------------------------------------------------
// DOM op dispatch table — shared by worker mode (main runs the
// op on the worker's behalf) and main mode (we run inline).
// Each entry returns a String (or throws). Adding a DOM op needs
// three pieces:
//
//   1. A `dom_*` foreign method in `wasm/src/dom.rs`.
//   2. A `_wlift_dom_*` shim in `worker.js` (posts a `dom-op`)
//      and another in `MainWlift.init()` (calls dispatch directly).
//   3. A `case` here.
//
// Returning a string keeps the protocol JSON-serialisable
// without bringing in a richer wire format yet — the existing
// future store flows everything as `String`s anyway.
// ---------------------------------------------------------------------------

function dispatchDomOp(op, args) {
  switch (op) {
    case "text": {
      const [selector] = args;
      const el = document.querySelector(selector);
      return el ? el.textContent ?? "" : "";
    }
    case "setText": {
      const [selector, value] = args;
      const el = document.querySelector(selector);
      if (el) el.textContent = value;
      return "";
    }
    case "getAttribute": {
      const [selector, name] = args;
      const el = document.querySelector(selector);
      return el ? (el.getAttribute(name) ?? "") : "";
    }
    case "setAttribute": {
      const [selector, name, value] = args;
      const el = document.querySelector(selector);
      if (el) el.setAttribute(name, value);
      return "";
    }
    case "addClass": {
      const [selector, name] = args;
      const el = document.querySelector(selector);
      if (el) el.classList.add(name);
      return "";
    }
    case "removeClass": {
      const [selector, name] = args;
      const el = document.querySelector(selector);
      if (el) el.classList.remove(name);
      return "";
    }
    case "queryAll": {
      const [selector] = args;
      // Encode count as a decimal string so it rides the
      // `Future<String>` transport. Future variant: return a
      // JSON-encoded list of element ids/handles.
      return String(document.querySelectorAll(selector).length);
    }
    default:
      throw new Error(`unknown DOM op '${op}'`);
  }
}
