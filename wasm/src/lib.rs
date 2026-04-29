//! wasm-bindgen entry shim for the WrenLift interpreter.
//!
//! Three things exported to JS / wasi hosts:
//!
//!   * `version()` — returns a `&str` describing the wlift build,
//!     useful as a smoke test that the wasm module loaded at all.
//!   * `run(source: &str) -> RunResult` — compiles the source as a
//!     `<wasm>` module, drives the BC interpreter, and returns the
//!     captured `System.print` output plus an `ok` flag.
//!   * `resolve_future` / `reject_future` — host-side handle store
//!     for the Promise ↔ `Fiber.yield` bridge. JS Promise callbacks
//!     flip a handle's state; the awaiting fiber wakes on its next
//!     yield-loop iteration. See
//!     `project_wasm_promise_fiber_bridge.md` for the full design.
//!
//! No JIT, no plugin loading, no fs / sockets. Future phases grow
//! the surface (incremental compile, multi-module install,
//! browser-API foreign classes); this is the minimal demo.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use wren_lift::runtime::engine::{ExecutionMode, InterpretResult};
use wren_lift::runtime::object::ObjFiber;
use wren_lift::runtime::vm::{VMConfig, VM};

pub mod browser;
pub mod dom;
pub mod storage;
pub mod tier_up;

// `js` carries the `extern "C"` block that imports browser-side
// shims (today: `_wlift_set_timeout`). Compiled only on the
// `wasm32-unknown-unknown` target, which is the only one where
// wasm-bindgen's JS interop applies. Wasi (the smoke target)
// short-circuits the same paths to a synchronous resolution.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod js {
    use wasm_bindgen::prelude::*;

    // Shims live on `globalThis` so the embedding page (or worker)
    // provides them by simple assignment, e.g.
    //
    //   globalThis._wlift_fetch = (h, url) => { fetch(url)... };
    //
    // before `init()`. No `module = "..."` means we don't have to
    // ship a separate JS file alongside `pkg/`; the host wires
    // these up however it likes (inline `<script>`, an external
    // module, etc.).
    #[wasm_bindgen]
    extern "C" {
        /// JS-provided shim:
        ///
        ///   globalThis._wlift_set_timeout = (handle, ms) => {
        ///       setTimeout(() => resolve_future(handle, ""), ms);
        ///   };
        ///
        /// `resolve_future` is exported from this same wasm module
        /// (`pkg/wlift_wasm.js`); the host imports it and closes
        /// over it before installing the shim.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_set_timeout)]
        pub fn wlift_set_timeout(handle: u32, ms: f64);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_next_frame = (handle) => {
        ///       requestAnimationFrame(() => resolve_future(handle, ""));
        ///   };
        ///
        /// Worker mode falls back to `setTimeout(..., 16)` because
        /// `requestAnimationFrame` is a main-thread API. The
        /// fallback is roughly 60 Hz; not vsync-locked, but better
        /// than `setTimeout(0)` (unbounded fire rate).
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_next_frame)]
        pub fn wlift_next_frame(handle: u32);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_fetch = (handle, url) => {
        ///       fetch(url)
        ///         .then(r => r.text())
        ///         .then(t => resolve_future(handle, t))
        ///         .catch(e => reject_future(handle, String(e)));
        ///   };
        ///
        /// Same wiring as `_wlift_set_timeout`.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_fetch)]
        pub fn wlift_fetch(handle: u32, url: &str);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_ws_open = (handle, url) => {
        ///       const ws = new WebSocket(url);
        ///       ws.addEventListener("open",    () => ws_open(handle));
        ///       ws.addEventListener("message", (e) => ws_message(handle, String(e.data)));
        ///       ws.addEventListener("close",   () => ws_close(handle));
        ///       ws.addEventListener("error",   () => ws_close(handle));
        ///       globalThis.__wlift_sockets ??= new Map();
        ///       globalThis.__wlift_sockets.set(handle, ws);
        ///   };
        ///
        /// The page keeps a `Map<handle, WebSocket>` so subsequent
        /// `_wlift_ws_send` / `_wlift_ws_close` calls can find the
        /// underlying socket.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_ws_open)]
        pub fn wlift_ws_open(handle: u32, url: &str);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_ws_send = (handle, text) => {
        ///       globalThis.__wlift_sockets?.get(handle)?.send(text);
        ///   };
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_ws_send)]
        pub fn wlift_ws_send(handle: u32, text: &str);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_ws_close = (handle) => {
        ///       const ws = globalThis.__wlift_sockets?.get(handle);
        ///       if (ws) { ws.close(); globalThis.__wlift_sockets.delete(handle); }
        ///   };
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_ws_close)]
        pub fn wlift_ws_close(handle: u32);

        // ---------- DOM bridge ------------------------------------------
        // Main-thread page or `WorkerWlift` adapter installs these.
        // Worker-mode shims `postMessage` to the main thread; main-mode
        // shims call `document.*` directly. See `wlift.js` and the
        // `dom` Rust module for the full story.

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_dom_text = (handle, selector) => {
        ///       const el = document.querySelector(selector);
        ///       resolve_future(handle, el ? el.textContent : "");
        ///   };
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_dom_text)]
        pub fn wlift_dom_text(handle: u32, selector: &str);

        /// JS-provided shim:
        ///
        ///   globalThis._wlift_dom_set_text = (handle, selector, value) => {
        ///       const el = document.querySelector(selector);
        ///       if (el) el.textContent = value;
        ///       resolve_future(handle, "");
        ///   };
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_dom_set_text)]
        pub fn wlift_dom_set_text(handle: u32, selector: &str, value: &str);

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_dom_get_attribute)]
        pub fn wlift_dom_get_attribute(handle: u32, selector: &str, name: &str);

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_dom_set_attribute)]
        pub fn wlift_dom_set_attribute(handle: u32, selector: &str, name: &str, value: &str);

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_dom_add_class)]
        pub fn wlift_dom_add_class(handle: u32, selector: &str, name: &str);

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_dom_remove_class)]
        pub fn wlift_dom_remove_class(handle: u32, selector: &str, name: &str);

        /// `Dom.queryAll(selector)` — resolves to the count of
        /// matching elements as a decimal string (e.g. `"3"`).
        /// Single result type (String) keeps it on the existing
        /// `Future<String>` transport.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_dom_query_all)]
        pub fn wlift_dom_query_all(handle: u32, selector: &str);

        /// JS-provided yield helper used by the scheduler loop.
        /// Returns a Promise that resolves on `setTimeout(0)` so
        /// the JS event loop drains BOTH microtasks AND
        /// macrotasks (including `setTimeout` callbacks). A
        /// `Promise.resolve()` await would only flush microtasks
        /// and leave `setTimeout(150).await` permanently
        /// pending. The page wires this in `worker.js` and
        /// `MainWlift.init` next to the other shims:
        ///
        ///   globalThis._wlift_yield_to_event_loop =
        ///     () => new Promise(r => setTimeout(r, 0));
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_yield_to_event_loop)]
        pub fn wlift_yield_to_event_loop() -> js_sys::Promise;

        /// Perf logger — host installs as a `console.log`-style
        /// shim. `_wlift_perf_log("phase", ms)` is called at each
        /// phase boundary inside `run()` so we can see where the
        /// time actually goes. Hosts that don't care can install
        /// a no-op.
        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_perf_log)]
        pub fn wlift_perf_log(label: &str, ms: f64);

        /// `performance.now()` in milliseconds (high-resolution
        /// monotonic clock). Available on both Window and
        /// WorkerGlobalScope. Used by the perf logger.
        #[wasm_bindgen(js_namespace = ["performance"], js_name = now)]
        pub fn perf_now() -> f64;

        // ---------- Storage bridge --------------------------------------
        // localStorage / sessionStorage exist in both Window and
        // WorkerGlobalScope, so the JS shim can hit them inline in
        // either mode — no postMessage round-trip needed. Foreign
        // methods take a `scope` arg ("local" | "session") so we
        // share one method table between the two backing stores.

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_storage_get)]
        pub fn wlift_storage_get(handle: u32, scope: &str, key: &str);

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_storage_set)]
        pub fn wlift_storage_set(handle: u32, scope: &str, key: &str, value: &str);

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_storage_remove)]
        pub fn wlift_storage_remove(handle: u32, scope: &str, key: &str);

        #[wasm_bindgen(js_namespace = globalThis, js_name = _wlift_storage_clear)]
        pub fn wlift_storage_clear(handle: u32, scope: &str);
    }
}

// ---------------------------------------------------------------------------
// Promise ↔ Fiber.yield handle store (Phase 1.1 scaffolding)
// ---------------------------------------------------------------------------
//
// Foreign methods that call browser async APIs (`fetch`,
// `setTimeout`, `WebSocket.send`, ...) mint a handle here, kick
// off the JS Promise, and return the handle to Wren wrapped in a
// `Future` instance. JS resolves/rejects via the
// `resolve_future` / `reject_future` exports below; the Wren
// `Future.await` method polls `peek_state` in a `Fiber.yield`
// loop until the handle flips out of `Pending`.
//
// The store is a global Mutex<HashMap> for now. Multi-VM hosts
// (server-side wasi running multiple wlift instances) eventually
// want a per-VM store, but that's a refactor with no consumers
// in Phase 1.1 — defer until at least one host ships multiple
// concurrent VMs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub enum FutureState {
    Pending = 0,
    Resolved = 1,
    Rejected = 2,
}

#[derive(Debug)]
struct FutureSlot {
    state: FutureState,
    /// Resolved value, currently encoded as a UTF-8 string (JSON
    /// for fetch payloads, plain text for echo APIs). Switching
    /// to a richer encoding (e.g. wasm-bindgen `JsValue` round-
    /// trip via the wlift heap) lands when there's a second
    /// consumer that actually needs it.
    value: Option<String>,
    /// Error message on `Rejected`. Surfaced to Wren via
    /// `Fiber.abort` inside `Future.await`.
    error: Option<String>,
}

fn future_store() -> &'static Mutex<HashMap<u32, FutureSlot>> {
    static STORE: std::sync::OnceLock<Mutex<HashMap<u32, FutureSlot>>> = std::sync::OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_future_handle() -> u32 {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Internal — Phase 1.1+ foreign methods call this to mint a
/// handle and pre-stash a `Pending` slot. Public so the future
/// `core::browser` foreign-method registry (lives in this crate
/// for now) can call it directly without going through the JS
/// host.
pub fn create_pending_future() -> u32 {
    let handle = next_future_handle();
    let mut store = future_store().lock().expect("future store poisoned");
    store.insert(
        handle,
        FutureSlot {
            state: FutureState::Pending,
            value: None,
            error: None,
        },
    );
    handle
}

// ---------------------------------------------------------------------------
// Parked fiber store (Phase 1.3 — top-level await scheduler)
// ---------------------------------------------------------------------------
//
// `Future.await` calls `BrowserCore.parkSelf(handle)` before each
// `Fiber.yield`. The foreign method captures `vm.fiber` and
// pushes `(fiber_ptr, handle)` here. After `vm.interpret`
// returns, the async `run()` loop snapshots this list, picks the
// entries whose handles have settled, and resumes them via
// `vm_interp::run_fiber`. Between drains it `await`s a microtask
// so the JS event loop can fire pending Promise callbacks.
//
// **GC rooting:** the raw `*mut ObjFiber` here is *not* a GC
// root. Callers rely on the Wren-side `var` reference the user
// holds (`var f = Fiber.new {…}`) keeping the fiber alive across
// the parked window. If a fiber drops out of all Wren scopes
// while still parked the resume is undefined behaviour. In
// practice, fibers ARE held by `var` declarations because the
// host needs the handle to call `.try()` in the first place.
//
// `*mut ObjFiber` isn't `Send` by default; we wrap it in a
// `ParkedFiber` and add an `unsafe impl Send` since wasm32 is
// single-threaded — the `Mutex` exists for `OnceLock` interior
// mutability, not actual cross-thread synchronisation.

// Fields are only read inside the scheduler loop, which is
// gated to `wasm32 + unknown`. The host build still pushes
// entries (via `park_fiber`) so the type stays present, but
// clippy flags the fields as dead — silence it instead of
// gating the whole module surface on wasm32.
#[allow(dead_code)]
struct ParkedFiber {
    fiber: *mut ObjFiber,
    handle: u32,
}
// SAFETY: wasm32 is single-threaded. The pointer is dereferenced
// only on the same thread that pushed it, inside the
// scheduler's resume call. Real cross-thread Send would need GC
// rooting + atomic state.
unsafe impl Send for ParkedFiber {}

fn parked_fibers() -> &'static Mutex<Vec<ParkedFiber>> {
    static PARKED: std::sync::OnceLock<Mutex<Vec<ParkedFiber>>> = std::sync::OnceLock::new();
    PARKED.get_or_init(|| Mutex::new(Vec::new()))
}

/// Internal — `browser_park_self` calls this. Pushes the current
/// fiber + the handle it's awaiting onto the parked list.
pub fn park_fiber(fiber: *mut ObjFiber, handle: u32) {
    parked_fibers()
        .lock()
        .expect("parked fiber list poisoned")
        .push(ParkedFiber { fiber, handle });
}

// ---------------------------------------------------------------------------
// Scheduler wake hook
// ---------------------------------------------------------------------------
//
// Without this, the scheduler's `await` loop could only poll —
// `setTimeout(0)` between iterations to give JS a chance to run.
// Each poll costs ~4 ms (browser nested-setTimeout clamp) plus
// the postMessage round-trip in worker mode, so a fiber awaiting
// many DOM ops paid 50–60 ms latency *per op*.
//
// Reactive wakeup: `resolve_future` / `reject_future` settle a
// handle, then immediately call the JS resolver stashed here.
// That resolves the Promise the scheduler is awaiting, which
// re-enters the scheduler in the next microtask — no polling
// cost. The `setTimeout(0)` path still exists as a fallback
// when no resolver has been registered (e.g. the very first
// scheduler tick before parking any fiber).

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
struct SchedulerWaker(js_sys::Function);
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
// SAFETY: wasm32 is single-threaded.
unsafe impl Send for SchedulerWaker {}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn scheduler_waker() -> &'static Mutex<Option<SchedulerWaker>> {
    static W: std::sync::OnceLock<Mutex<Option<SchedulerWaker>>> = std::sync::OnceLock::new();
    W.get_or_init(|| Mutex::new(None))
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn wake_scheduler_if_armed() {
    let waker = scheduler_waker()
        .lock()
        .expect("scheduler waker poisoned")
        .take();
    if let Some(SchedulerWaker(fun)) = waker {
        let _ = fun.call0(&JsValue::UNDEFINED);
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn wake_scheduler_if_armed() {}

/// JS-callable: flip a handle to `Resolved` and stash the value.
/// Idempotent — resolving an already-resolved/rejected handle is
/// a no-op (matches Promise semantics; the first settlement wins).
#[wasm_bindgen]
pub fn resolve_future(handle: u32, value: String) {
    let settled = {
        let mut store = future_store().lock().expect("future store poisoned");
        match store.get_mut(&handle) {
            Some(slot) if slot.state == FutureState::Pending => {
                slot.state = FutureState::Resolved;
                slot.value = Some(value);
                true
            }
            _ => false,
        }
    };
    if settled {
        wake_scheduler_if_armed();
    }
}

/// JS-callable: flip a handle to `Rejected` with an error message.
#[wasm_bindgen]
pub fn reject_future(handle: u32, error: String) {
    let settled = {
        let mut store = future_store().lock().expect("future store poisoned");
        match store.get_mut(&handle) {
            Some(slot) if slot.state == FutureState::Pending => {
                slot.state = FutureState::Rejected;
                slot.error = Some(error);
                true
            }
            _ => false,
        }
    };
    if settled {
        wake_scheduler_if_armed();
    }
}

/// JS-callable: read the current state of a handle. Used by
/// JS-side debugging / introspection tooling AND by the
/// `browser::browser_peek_state` foreign method that Wren's
/// `Future.await` polls in its yield loop.
#[wasm_bindgen]
pub fn future_state(handle: u32) -> FutureState {
    let store = future_store().lock().expect("future store poisoned");
    store
        .get(&handle)
        .map(|s| s.state)
        .unwrap_or(FutureState::Rejected)
}

/// Take the resolved value (or rejection error) out of a slot
/// and drop the slot. Called from `browser::browser_take_value`
/// after `peekState` reports non-pending — the slot is one-shot
/// from the Wren side, matching Promise's "settled once" rule.
///
/// Returns `None` if the handle was never registered, was
/// already taken, or hasn't settled yet (the await loop should
/// only call this after `peekState != 0`).
pub fn future_store_take_value(handle: u32) -> Option<String> {
    let mut store = future_store().lock().expect("future store poisoned");
    let slot = store.remove(&handle)?;
    match slot.state {
        FutureState::Pending => None,
        FutureState::Resolved => slot.value,
        FutureState::Rejected => slot.error,
    }
}

// ---------------------------------------------------------------------------
// WebSocket handle store (Phase 1.2)
// ---------------------------------------------------------------------------
//
// `Browser.connect(url)` mints a *connection* handle here (not a
// future), then returns it wrapped in a Wren `WebSocket` instance.
// Subsequent `send` / `recv` / `close` foreign methods drive the
// connection through this store.
//
// The contract differs from `FutureSlot` because a socket is
// multi-shot — many messages flow over one handle, each delivered
// to the next pending `recv()` future or buffered for a later
// caller. The store therefore holds:
//
//   * `state` — Connecting → Open → Closed lifecycle.
//   * `inbox` — messages that arrived with no waiter ready.
//   * `waiters` — future handles parked by `recv()`. The next
//                 inbound message wakes the head waiter.
//
// Browser-host JS wires its `WebSocket` listeners to call
// `ws_open` / `ws_message` / `ws_close` below. Wasi has no real
// socket; the wasi smoke runs a loopback (send → inbox) so the
// bridge can be exercised without a network stack.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub enum WebSocketState {
    Connecting = 0,
    Open = 1,
    Closed = 2,
}

#[derive(Debug)]
struct WebSocketSlot {
    state: WebSocketState,
    inbox: VecDeque<String>,
    waiters: VecDeque<u32>,
}

fn websocket_store() -> &'static Mutex<HashMap<u32, WebSocketSlot>> {
    static STORE: std::sync::OnceLock<Mutex<HashMap<u32, WebSocketSlot>>> =
        std::sync::OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_websocket_handle() -> u32 {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Mint a fresh socket handle in `Connecting` state. Foreign
/// methods call this before kicking off the JS-side `WebSocket`
/// constructor (browser) or flipping the state synchronously
/// (wasi loopback).
pub fn create_websocket() -> u32 {
    let handle = next_websocket_handle();
    let mut store = websocket_store().lock().expect("websocket store poisoned");
    store.insert(
        handle,
        WebSocketSlot {
            state: WebSocketState::Connecting,
            inbox: VecDeque::new(),
            waiters: VecDeque::new(),
        },
    );
    handle
}

/// JS-callable: flip a socket to `Open`. Browser hosts call this
/// from the `WebSocket`'s `open` listener so subsequent `send`
/// calls aren't queued in JS-land for nothing.
#[wasm_bindgen]
pub fn ws_open(handle: u32) {
    let mut store = websocket_store().lock().expect("websocket store poisoned");
    if let Some(slot) = store.get_mut(&handle) {
        if slot.state == WebSocketState::Connecting {
            slot.state = WebSocketState::Open;
        }
    }
}

/// JS-callable: deliver an incoming message. Wakes the head
/// `recv()` future if one is parked; otherwise buffers the
/// message for a later caller. Drops the message silently if the
/// handle is unknown — late arrivals on a closed/torn-down socket
/// shouldn't crash the page.
#[wasm_bindgen]
pub fn ws_message(handle: u32, msg: String) {
    let waiter = {
        let mut store = websocket_store().lock().expect("websocket store poisoned");
        let Some(slot) = store.get_mut(&handle) else {
            return;
        };
        if let Some(w) = slot.waiters.pop_front() {
            Some(w)
        } else {
            slot.inbox.push_back(msg.clone());
            None
        }
    };
    if let Some(w) = waiter {
        resolve_future(w, msg);
    }
}

/// JS-callable: flip a socket to `Closed` and reject any parked
/// `recv()` futures. Idempotent — closing twice is a no-op.
#[wasm_bindgen]
pub fn ws_close(handle: u32) {
    // Drain the waiter list under the store lock, then settle the
    // futures outside it — `reject_future` reaches into the future
    // store, and re-entering the websocket lock isn't necessary.
    let waiters: Vec<u32> = {
        let mut store = websocket_store().lock().expect("websocket store poisoned");
        match store.get_mut(&handle) {
            Some(slot) if slot.state != WebSocketState::Closed => {
                slot.state = WebSocketState::Closed;
                slot.waiters.drain(..).collect()
            }
            _ => Vec::new(),
        }
    };
    for w in waiters {
        reject_future(w, "WebSocket closed".to_string());
    }
}

/// JS-callable: read the current state of a socket. Used by JS
/// debug tooling and by `browser::browser_ws_state` if a Wren-
/// side state probe ends up wanted.
#[wasm_bindgen]
pub fn websocket_state(handle: u32) -> WebSocketState {
    let store = websocket_store().lock().expect("websocket store poisoned");
    store
        .get(&handle)
        .map(|s| s.state)
        .unwrap_or(WebSocketState::Closed)
}

/// Internal: enqueue a message into a socket's inbox without going
/// through the JS event path. The wasi mock uses this to loopback
/// `send()` straight back to `recv()` — there's no real socket on
/// wasi, but the bridge wiring (handle → future → fiber resume)
/// still gets exercised end-to-end.
pub fn websocket_inbox_push(handle: u32, msg: String) {
    ws_message(handle, msg);
}

/// Internal: park a future handle on a socket's waiter queue, or
/// resolve it immediately if the inbox already has a message.
/// Returns the future handle; callers wrap it in a Wren `Future`.
pub fn websocket_recv(handle: u32) -> u32 {
    let future = create_pending_future();
    let immediate = {
        let mut store = websocket_store().lock().expect("websocket store poisoned");
        match store.get_mut(&handle) {
            Some(slot) if slot.state == WebSocketState::Closed => Some(Err(())),
            Some(slot) => match slot.inbox.pop_front() {
                Some(msg) => Some(Ok(msg)),
                None => {
                    slot.waiters.push_back(future);
                    None
                }
            },
            None => Some(Err(())),
        }
    };
    match immediate {
        Some(Ok(msg)) => resolve_future(future, msg),
        Some(Err(())) => reject_future(future, "WebSocket closed".to_string()),
        None => {}
    }
    future
}

// ---------------------------------------------------------------------------
// Init + version
// ---------------------------------------------------------------------------

/// One-time setup. The wasm-bindgen `start` attribute makes this
/// run on module instantiation, before any JS-side `run()` /
/// `version()` call.
///
/// Two jobs:
///
///   * Install `console_error_panic_hook` so panics surface as a
///     readable JS `console.error` instead of the bare
///     "unreachable executed" trap.
///
///   * Publish every statically-linked plugin's foreign-method
///     symbols to the runtime's foreign-method registry. This is
///     the wasm replacement for the host's `libloading::dlsym`
///     pass — Wren `foreign class` declarations annotated with
///     `#!native = "wlift_image"` look up here at install time.
#[wasm_bindgen(start)]
pub fn _wasm_init() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        register_static_plugins();
        // Plug the wasm-tier-up compile + dispatch hooks into
        // the runtime's broker. Has to happen before any
        // `record_call` that might trigger tier-up — i.e.
        // before any user code runs.
        tier_up::register_tier_up_callbacks();
        // Warm the prelude bytecode cache so the *first* user
        // `run()` doesn't pay ~30 ms of parse / sema / MIR build
        // for `BROWSER_PRELUDE`. Subsequent runs already hit the
        // cache; this just shifts the work from "first user click"
        // to module instantiation, where it overlaps with the
        // page's `await init()` and is invisible.
        let _ = prelude_blob();
    }
}

/// Static-link plugin registry. Wasi smoke binaries call this from
/// `main` before they spin up a VM. Browser builds get it via the
/// `_wasm_init` shim above.
#[cfg(target_arch = "wasm32")]
pub fn register_static_plugins() {
    wlift_image::register_static_symbols();
    browser::register_static_symbols();
    dom::register_static_symbols();
    storage::register_static_symbols();
}

/// Host-side stub so non-wasm callers (the smoke `_start` binary
/// running on a native target via `cargo run -p wlift_wasm --bin
/// smoke` for local debugging — outside the wasi target) still
/// compile. The host wlift loads `libwlift_image.dylib` /
/// `.so` / `.dll` via dlsym; this no-ops there.
#[cfg(not(target_arch = "wasm32"))]
pub fn register_static_plugins() {}

// ---------------------------------------------------------------------------
// On-demand wasm plugin loader — Rust-side bridge surface
// ---------------------------------------------------------------------------
//
// `wren_lift::runtime::foreign_wasm::dispatch_dynamic` declares
// `env::wlift_dispatch_dynamic_plugin(idx, vm)` as a wasm import
// (via `#[link(wasm_import_module = "env")]`). JS provides that
// import at instantiation — the loader owns the registry of
// plugin instances + which export to call by index. This file
// only exposes the *registration* side of the bridge; the
// dispatch path is JS-only by design (cross-wasm-module calls
// have no in-Rust shortcut once you skip wasm-bindgen on the
// plugin side).

/// Wasm-bindgen export. JS calls this once per foreign-method
/// export discovered in a plugin module. The returned `idx` is
/// what JS keys its `(plugin_instance, export_name)` map with;
/// the same idx ends up in `Method::ForeignCDynamic(idx)` and
/// rides back through `env::wlift_dispatch_dynamic_plugin` at
/// call time.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wlift_register_plugin_dynamic_export(library: &str, symbol: &str) -> u32 {
    wren_lift::runtime::foreign::register_plugin_dynamic(library, symbol)
}

/// Wasm-bindgen export. Decode a `.hatch` bundle and surface every
/// wasm-targeted NativeLib section as `{lib, bytes}` so JS can
/// instantiate each plugin module before installing the bundle's
/// Wlbc / Source sections (foreign-class declarations need the
/// plugin's symbols already registered when they bind).
///
/// Returns an empty array on parse failure rather than throwing —
/// the playground falls back to the no-plugin path naturally.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wlift_extract_wasm_plugins(bytes: &[u8]) -> js_sys::Array {
    let out = js_sys::Array::new();
    let Ok(hatch) = wren_lift::hatch::load(bytes) else {
        return out;
    };
    // Plugins ride in NativeLib sections. On a wasm-targeted
    // bundle the only NativeLibs are .wasm artefacts (the bundle
    // builder filters by target — see `pack_bundled_native_libs`),
    // so any NativeLib here is a candidate.
    for s in &hatch.sections {
        if !matches!(s.kind, wren_lift::hatch::SectionKind::NativeLib) {
            continue;
        }
        // Quick wasm-magic sanity check: `\0asm\x01\0\0\0`. Skip
        // anything that doesn't look like a wasm module so we don't
        // hand JS a `.dylib` to instantiate.
        if s.data.len() < 8 || &s.data[..4] != b"\0asm" {
            continue;
        }
        let entry = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&entry, &"lib".into(), &s.name.clone().into());
        let bytes_view = js_sys::Uint8Array::from(s.data.as_slice());
        let _ = js_sys::Reflect::set(&entry, &"bytes".into(), &bytes_view);
        out.push(&entry);
    }
    out
}

/// Allocate `len` bytes inside the host's linear memory and
/// return a pointer. Used by the JS-side bridge fns
/// (`wlift_set_slot_str` etc.) to land plugin-supplied bytes in
/// the host's address space before calling Wren-API functions
/// that read from there. The companion `wlift_host_free` returns
/// the bytes when the caller is done.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wlift_host_alloc(len: u32) -> u32 {
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    // SAFETY: `with_capacity(n)` reserves `n` bytes; setting
    // length to `n` exposes uninitialised bytes which the caller
    // immediately overwrites via JS-side memory copy. The Vec
    // is intentionally leaked — `wlift_host_free` reconstructs
    // and drops it.
    unsafe {
        buf.set_len(len as usize);
    }
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr as u32
}

/// Counterpart to [`wlift_host_alloc`]. Reconstructs the `Vec`
/// from the raw pointer + length so it drops normally.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wlift_host_free(ptr: u32, len: u32) {
    if ptr == 0 || len == 0 {
        return;
    }
    // SAFETY: ptr/len pair were produced by `wlift_host_alloc`
    // above; matching alloc/free contract.
    unsafe {
        let _ = Vec::<u8>::from_raw_parts(ptr as *mut u8, len as usize, len as usize);
    }
}

/// Build identifier — hard-coded for the moment so JS can sanity-
/// check the loaded wasm matches what its bundler thought it was
/// importing.
#[wasm_bindgen]
pub fn version() -> String {
    format!(
        "wlift_wasm {} (interpreter-only; wasm32 build)",
        env!("CARGO_PKG_VERSION")
    )
}

// ---------------------------------------------------------------------------
// Hatchfile introspection (drives JS-side dep resolution)
// ---------------------------------------------------------------------------
//
// The browser playground can't `git clone` and can't follow
// arbitrary path references — but it *can* read the consumer's
// hatchfile, walk `[dependencies]` to figure out what to fetch,
// and ask `fetch()` for each. The shape of each dep entry
// (version string vs `{ git = ..., rev = ... }` vs `{ path = ...
// }`) is what tells the JS side how to fetch — that's the
// metadata we surface here.
//
// `parse_hatchfile_toml` returns the manifest as a JS object so
// the JS host doesn't need to ship a TOML parser. `peek_manifest`
// extracts the manifest from an already-fetched `.hatch` byte
// stream so transitive deps can be discovered without
// instantiating the bundle.

/// Parse a hatchfile (TOML text) and return the manifest as a
/// JS-side object. Errors surface as a JS exception — JS callers
/// `try { ... } catch { ... }` to recover.
///
/// JS shape of the returned object (subset — full mirror of
/// `crate::hatch::Manifest`):
/// ```json
/// {
///   "name": "@hatch:my-package",
///   "version": "0.1.0",
///   "entry": "main",
///   "modules": [...],
///   "dependencies": {
///     "@hatch:assert": "0.2.0",
///     "@user:lib": { "git": "https://example.com/repo", "rev": "abc" },
///     "local": { "path": "../local" }
///   },
///   "target": "wasm32-unknown-unknown"
/// }
/// ```
/// Serialize the manifest as a *plain JS object* so the
/// playground's `Object.entries(manifest.dependencies)` works.
/// `serde_wasm_bindgen`'s default round-trips `BTreeMap` /
/// `HashMap` as `js_sys::Map`, which silently no-ops under
/// `Object.entries()` — leading to "no deps to install" with
/// no error, then the user code's `import "@hatch:..."` looks
/// like a missing module. `json_compatible()` flips
/// `serialize_maps_as_objects(true)` (alongside a couple of
/// other knobs); behaviour now matches `JSON.parse` shape.
fn json_serializer() -> serde_wasm_bindgen::Serializer {
    serde_wasm_bindgen::Serializer::json_compatible()
}

#[wasm_bindgen]
pub fn parse_hatchfile_toml(text: &str) -> Result<JsValue, JsValue> {
    let manifest: wren_lift::hatch::Manifest =
        toml::from_str(text).map_err(|e| JsValue::from_str(&format!("hatchfile parse: {e}")))?;
    use serde::Serialize;
    manifest
        .serialize(&json_serializer())
        .map_err(|e| JsValue::from_str(&format!("manifest → JsValue: {e}")))
}

/// Inspect a `.hatch` byte stream and return its manifest as a
/// JS object — same shape as `parse_hatchfile_toml`. Used by
/// the JS-side dep walker to discover transitive `[dependencies]`
/// without installing the bundle.
#[wasm_bindgen]
pub fn peek_manifest(bytes: &[u8]) -> Result<JsValue, JsValue> {
    let hatch = wren_lift::hatch::load(bytes)
        .map_err(|e| JsValue::from_str(&format!("hatch load: {e}")))?;
    use serde::Serialize;
    hatch
        .manifest
        .serialize(&json_serializer())
        .map_err(|e| JsValue::from_str(&format!("manifest → JsValue: {e}")))
}

/// Result of a single `run` call, exported to JS as a structural
/// object via `wasm_bindgen`'s getter convention.
#[wasm_bindgen]
pub struct RunResult {
    output: String,
    ok: bool,
    error_kind: ErrorKind,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum ErrorKind {
    None,
    CompileError,
    RuntimeError,
}

#[wasm_bindgen]
impl RunResult {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> String {
        self.output.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ok(&self) -> bool {
        self.ok
    }

    #[wasm_bindgen(getter, js_name = errorKind)]
    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }
}

/// Wren-side prelude prepended to every `run()` call. Defines the
/// `Future`, `Browser`, `WebSocket`, and `Dom` classes that wrap
/// the foreign-method handles into a pleasant Wren API. Without
/// this, every page would have to inline ~40 lines of Wren shim
/// boilerplate before getting to its own code.
///
/// One source of truth — both worker and main-thread modes pick
/// it up via `run()`. Trade-off: error line numbers shift by the
/// prelude's height. We reserve the first chunk of lines and
/// offer a `prelude_line_count()` export so tooling can subtract
/// it back out when reporting positions.
const BROWSER_PRELUDE: &str = r##"
#!native = "browser"
foreign class BrowserCore {
    #!symbol = "browser_set_timeout"
    foreign static setTimeoutHandle(ms)
    #!symbol = "browser_next_frame"
    foreign static nextFrameHandle()
    #!symbol = "browser_fetch"
    foreign static fetchHandle(url)
    #!symbol = "browser_ws_open"
    foreign static wsOpen(url)
    #!symbol = "browser_ws_send"
    foreign static wsSend(handle, text)
    #!symbol = "browser_ws_recv"
    foreign static wsRecv(handle)
    #!symbol = "browser_ws_close"
    foreign static wsClose(handle)
    #!symbol = "browser_peek_state"
    foreign static peekState(handle)
    #!symbol = "browser_take_value"
    foreign static takeValue(handle)
    #!symbol = "browser_park_self"
    foreign static parkSelf(handle)
}
#!native = "dom"
foreign class DomCore {
    #!symbol = "dom_text"
    foreign static textHandle(selector)
    #!symbol = "dom_set_text"
    foreign static setTextHandle(selector, value)
    #!symbol = "dom_get_attribute"
    foreign static getAttributeHandle(selector, name)
    #!symbol = "dom_set_attribute"
    foreign static setAttributeHandle(selector, name, value)
    #!symbol = "dom_add_class"
    foreign static addClassHandle(selector, name)
    #!symbol = "dom_remove_class"
    foreign static removeClassHandle(selector, name)
    #!symbol = "dom_query_all"
    foreign static queryAllHandle(selector)
}
class Future {
    construct new_(handle) { _h = handle }
    handle { _h }
    state  { BrowserCore.peekState(_h) }
    await {
        // 0 = pending, 1 = resolved, 2 = rejected. Cache the
        // settled state BEFORE `takeValue` removes the slot —
        // a subsequent `peekState` on a missing handle reports
        // Rejected, which would mis-fire `Fiber.abort` on a
        // resolved Future.
        //
        // `parkSelf` registers the current fiber + handle with
        // the host's scheduler so it can resume us once the
        // future settles. Has to come BEFORE `Fiber.yield` —
        // after the yield, control's already with the host and
        // we'd never get a chance to register.
        while (state == 0) {
            BrowserCore.parkSelf(_h)
            Fiber.yield()
        }
        var settled = state
        var v = BrowserCore.takeValue(_h)
        if (settled == 2) Fiber.abort(v == null ? "Future rejected" : v)
        return v
    }
}
class WebSocket {
    construct new_(handle) { _h = handle }
    handle      { _h }
    send(text)  { BrowserCore.wsSend(_h, text) }
    recv        { Future.new_(BrowserCore.wsRecv(_h)) }
    close       { BrowserCore.wsClose(_h) }
}
class Browser {
    static setTimeout(ms) { Future.new_(BrowserCore.setTimeoutHandle(ms)) }
    // Vsync-paced yield. Use this in render loops instead of
    // setTimeout(0) — setTimeout runs as fast as macrotasks drain
    // (no upper bound) and keeps firing in backgrounded tabs;
    // rAF naturally caps at the display refresh rate and pauses
    // when the tab is hidden.
    static nextFrame      { Future.new_(BrowserCore.nextFrameHandle()) }
    static fetch(url)     { Future.new_(BrowserCore.fetchHandle(url)) }
    static connect(url)   { WebSocket.new_(BrowserCore.wsOpen(url)) }
}
class Dom {
    static text(selector)                  { Future.new_(DomCore.textHandle(selector)) }
    static setText(selector, value)        { Future.new_(DomCore.setTextHandle(selector, value)) }
    static getAttribute(selector, name)    { Future.new_(DomCore.getAttributeHandle(selector, name)) }
    static setAttribute(selector, n, v)    { Future.new_(DomCore.setAttributeHandle(selector, n, v)) }
    static addClass(selector, name)        { Future.new_(DomCore.addClassHandle(selector, name)) }
    static removeClass(selector, name)     { Future.new_(DomCore.removeClassHandle(selector, name)) }
    static queryAll(selector)              { Future.new_(DomCore.queryAllHandle(selector)) }
}
#!native = "storage"
foreign class StorageCore {
    #!symbol = "storage_get"
    foreign static getHandle(scope, key)
    #!symbol = "storage_set"
    foreign static setHandle(scope, key, value)
    #!symbol = "storage_remove"
    foreign static removeHandle(scope, key)
    #!symbol = "storage_clear"
    foreign static clearHandle(scope)
}
class LocalStorage {
    static get(key)         { Future.new_(StorageCore.getHandle("local", key)) }
    static set(key, value)  { Future.new_(StorageCore.setHandle("local", key, value)) }
    static remove(key)      { Future.new_(StorageCore.removeHandle("local", key)) }
    static clear            { Future.new_(StorageCore.clearHandle("local")) }
}
class SessionStorage {
    static get(key)         { Future.new_(StorageCore.getHandle("session", key)) }
    static set(key, value)  { Future.new_(StorageCore.setHandle("session", key, value)) }
    static remove(key)      { Future.new_(StorageCore.removeHandle("session", key)) }
    static clear            { Future.new_(StorageCore.clearHandle("session")) }
}
"##;

/// Number of lines `run()` prepends to the user source. Today
/// it's just the one-line `PRELUDE_IMPORT`. Exposed so callers
/// can shift error spans back to user-source line numbers — the
/// big `BROWSER_PRELUDE` is loaded as a separate module and no
/// longer inflates the user module's line count.
#[wasm_bindgen]
pub fn prelude_line_count() -> u32 {
    // `format!("{}\n{}", PRELUDE_IMPORT, source)` — one prefix
    // line + the newline separator.
    1
}

/// One-line `import` that pulls every prelude class into the
/// user's module. With the prelude moved out of the per-`run()`
/// compile path (cached as a wlbc blob — see `prelude_blob`),
/// this is what the user code pays parse / sema / codegen cost
/// for instead of the full ~100 lines.
const PRELUDE_IMPORT: &str =
    r#"import "wlift_prelude" for Future, Browser, WebSocket, Dom, LocalStorage, SessionStorage"#;

/// Compile `BROWSER_PRELUDE` to a `.wlbc`-style bytecode blob the
/// first time it's needed, then cache it forever. Each `run()`
/// loads this blob via `vm.interpret_bytecode` rather than
/// re-parsing + re-semating + re-codegenning ~100 lines of
/// foreign-class declarations on every invocation. The first
/// call still pays the compile cost (one-time, on a throwaway
/// VM); subsequent calls just deserialize the blob, which is
/// roughly an order of magnitude cheaper.
fn prelude_blob() -> &'static Vec<u8> {
    static B: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let mut tmp = VM::new(VMConfig {
            execution_mode: ExecutionMode::Interpreter,
            ..VMConfig::default()
        });
        tmp.compile_source_to_blob(BROWSER_PRELUDE)
            .expect("BROWSER_PRELUDE must compile — check the const for a typo")
    })
}

/// Compile + interpret a Wren source string. Output captured into
/// the returned `RunResult`. The VM lives only for the duration
/// of this call — every `run` is a fresh interpreter, no shared
/// state across calls.
///
/// `run` is **async** so async foreign methods (`Browser.fetch`,
/// `Browser.setTimeout`, `Browser.connect`, cross-thread `Dom.*`)
/// actually settle their futures: after the user's source runs
/// once, any fiber parked on a still-pending future gets
/// resumed by the scheduler loop below. Each loop iteration
/// `await`s a JS microtask so pending Promises can fire their
/// `.then` callbacks (which call `resolve_future` / `reject_future`),
/// then resumes every parked fiber whose handle has settled.
///
/// The browser prelude (`Future` / `Browser` / `WebSocket` /
/// `Dom` / `LocalStorage` / …) is prepended to every source so
/// callers don't have to inline the shim boilerplate. See
/// `BROWSER_PRELUDE` above.
/// Wall-clock milliseconds since module load, via JS
/// `performance.now()`. Wraps the import so the rest of `run()`
/// can use `let t = perf_mark();` without a cfg every time.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn perf_mark() -> f64 {
    js::perf_now()
}
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn perf_mark() -> f64 {
    0.0
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn perf_log(label: &str, start: f64) {
    js::wlift_perf_log(label, js::perf_now() - start);
}
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn perf_log(_label: &str, _start: f64) {}

/// What to feed into the VM after the prelude is in place. The
/// `Source` arm runs through the parser + sema + lower path the
/// playground has always used; the `Hatch` arm bypasses that
/// pipeline by installing a pre-compiled `.hatch` bundle —
/// multiple modules at once, plus the `target` compatibility
/// check (cf. `crate::hatch::check_target_compat`). The
/// `SourceWithHatches` arm pre-installs a vec of `@hatch:*`
/// dependency bundles, then runs `Source` against the now-loaded
/// modules — same shape the host CLI's `HatchRunner` uses for
/// pulling external packages, but here the JS host has fetched
/// the bytes itself (wasm can't synchronously fetch).
enum RunInput<'a> {
    Source(&'a str),
    Hatch(&'a [u8]),
    SourceWithHatches(&'a str, Vec<Vec<u8>>),
}

/// Run a `.hatch` bundle in the playground. Same setup as `run`
/// (fresh VM, prelude install, scheduler loop) but feeds the
/// user-code phase a hatch byte stream instead of source. Modules
/// inside the hatch install in their declared order — so a hatch
/// produced via `wlift src/ --bundle out.hatch --bundle-target
/// wasm32-unknown-unknown` runs end-to-end here, including any
/// in-bundle `import` between modules (cross-module resolution
/// in wasm).
///
/// `@hatch:*` external package resolution is *not* implemented
/// here — the wasm runtime can't synchronously fetch packages
/// during compilation. Use the host's `hatch` CLI (or a
/// JS-side prefetch + multiple `install_hatch` calls, when that
/// API lands) to assemble the closure of needed packages, then
/// pass the result here as a single bundle.
#[wasm_bindgen]
pub async fn run_hatch(bytes: &[u8]) -> RunResult {
    run_inner(RunInput::Hatch(bytes)).await
}

/// Run user source after pre-installing a list of `@hatch:*`
/// dependency bundles. The JS host is responsible for fetching
/// each dep's bytes (typically by scanning `import "@hatch:..."`
/// patterns in the source and pulling matching `.hatch` files
/// from a CDN); this entry point installs them before the user
/// source's parser hits an `import "@hatch:foo"` line, so the
/// import resolves against an already-loaded module.
///
/// `deps` is a JS `Array` of `Uint8Array`. Each element is one
/// `.hatch` byte stream. Order matters only for transitive
/// dependencies — a hatch that imports another hatch must come
/// after its dep in the array. The JS-side helper that does the
/// fetch is responsible for that ordering (topological sort over
/// the dep graph).
#[wasm_bindgen]
pub async fn run_with_hatches(source: &str, deps: js_sys::Array) -> RunResult {
    // Decode each `Uint8Array` into a `Vec<u8>` once, on this
    // side of the bindgen boundary. Holding `Uint8Array`
    // references across the upcoming `await` would land in a
    // long-running async context — copying out is cheaper and
    // sidesteps the lifetime question.
    let mut dep_bytes: Vec<Vec<u8>> = Vec::with_capacity(deps.length() as usize);
    for entry in deps.iter() {
        match entry.dyn_into::<js_sys::Uint8Array>() {
            Ok(arr) => dep_bytes.push(arr.to_vec()),
            Err(_) => {
                return RunResult {
                    output: "run_with_hatches: every `deps` element must be a Uint8Array"
                        .to_string(),
                    ok: false,
                    error_kind: ErrorKind::CompileError,
                };
            }
        }
    }
    run_inner(RunInput::SourceWithHatches(source, dep_bytes)).await
}

#[wasm_bindgen]
pub async fn run(source: &str) -> RunResult {
    run_inner(RunInput::Source(source)).await
}

async fn run_inner(input: RunInput<'_>) -> RunResult {
    use std::sync::{Arc, Mutex as StdMutex};

    let t_total = perf_mark();

    // Capture compile + runtime diagnostics into the same stream
    // we hand back as `output`. Without this, errors land on the
    // wasm `eprintln!` path — which in a browser is the devtools
    // console, *not* the page's output pane. Caller couldn't see
    // why a script failed.
    let diagnostics = Arc::new(StdMutex::new(String::new()));
    let diag_capture = diagnostics.clone();

    let mut config = VMConfig {
        execution_mode: ExecutionMode::Interpreter,
        ..VMConfig::default()
    };
    config.error_fn = Some(Box::new(move |_kind, _module, _line, msg| {
        let mut buf = diag_capture.lock().expect("diagnostic buffer poisoned");
        buf.push_str(msg);
        if !msg.ends_with('\n') {
            buf.push('\n');
        }
    }));

    // Wipe per-run JIT caches (main-module pointer etc.) before
    // the new VM is allocated. The allocator may hand the new VM
    // the same address as the previous one, which would make the
    // cached `vm_ptr` check pass but the cached `module_ptr`
    // dangling — the prologue would deref freed memory and trap
    // ("memory access out of bounds" inside
    // `wren_jit_slot_for_module_var`). Running the reset *before*
    // VM::new is the simplest insurance.
    tier_up::reset_jit_runtime_caches();

    let t_vm_new = perf_mark();
    let mut vm = VM::new(config);
    vm.output_buffer = Some(String::new());
    perf_log("vm_new", t_vm_new);

    // Wipe any leftover parked fibers from a prior `run()` —
    // those fibers belong to a now-dropped VM and the pointers
    // are invalid.
    parked_fibers()
        .lock()
        .expect("parked fiber list poisoned")
        .clear();

    // Install the precompiled prelude into module
    // `wlift_prelude`. Cheap: bytecode deserialise, no parse /
    // sema / codegen. The user source then `import`s those
    // names, and the import path short-circuits because the
    // module's already loaded.
    let t_blob = perf_mark();
    let _ = prelude_blob(); // ensure cache populated; first run pays the compile
    perf_log("prelude_compile_or_cache_hit", t_blob);

    let t_prelude_load = perf_mark();
    let prelude_load = vm.interpret_bytecode("wlift_prelude", prelude_blob());
    perf_log("prelude_bytecode_install", t_prelude_load);
    if !matches!(prelude_load, InterpretResult::Success) {
        // Cached blob failed to load — should never happen since
        // the const is fixed at build time. Fall back to a clean
        // error so the page surfaces it instead of silently
        // running without a prelude.
        return RunResult {
            output: "wlift internal error: prelude bytecode failed to load".to_string(),
            ok: false,
            error_kind: ErrorKind::CompileError,
        };
    }
    let t_user = perf_mark();
    let result = match input {
        RunInput::Source(source) => {
            let combined = format!("{}\n{}", PRELUDE_IMPORT, source);
            vm.interpret("main", &combined)
        }
        RunInput::Hatch(bytes) => {
            // `interpret_hatch` runs the manifest's check_target_compat
            // gate first, then installs every Wlbc / Source section
            // in declared order. Cross-module imports inside the
            // bundle resolve against already-installed peers — the
            // same path the host CLI uses, just without
            // native-lib extraction (gated to `feature = "host"`).
            vm.interpret_hatch(bytes)
        }
        RunInput::SourceWithHatches(source, deps) => {
            // Install each prefetched `@hatch:*` dependency bundle
            // first. `install_hatch_modules` runs each module's
            // top-level (class definitions, registrations) so the
            // user source's `import "@hatch:foo" for Bar` resolves
            // against an already-loaded module by the time it's
            // parsed. Stop on the first failure — a missing or
            // broken dep is unrecoverable for the consumer.
            let mut install_result = InterpretResult::Success;
            for dep_bytes in &deps {
                let r = vm.install_hatch_modules(dep_bytes);
                if !matches!(r, InterpretResult::Success) {
                    install_result = r;
                    break;
                }
            }
            if matches!(install_result, InterpretResult::Success) {
                let combined = format!("{}\n{}", PRELUDE_IMPORT, source);
                vm.interpret("main", &combined)
            } else {
                install_result
            }
        }
    };
    perf_log("user_compile_and_run", t_user);

    // Scheduler loop. Drives parked fibers across JS event-loop
    // turns until either nothing's parked or the tick budget
    // runs out.
    //
    // Each tick yields via `setTimeout(0)` rather than
    // `Promise.resolve()` — that's the critical detail. A
    // microtask-only await would never give
    // `Browser.setTimeout(150).await` a chance to fire, since
    // `setTimeout` callbacks ride the macrotask queue. Browsers
    // clamp nested `setTimeout(_, 0)` to ~4 ms, so 2500 ticks
    // ≈ 10 s wall clock — covers a reasonable awaited timer
    // chain and bails on actually-stuck programs.
    let t_sched = perf_mark();
    // Scheduler loop is wasm-only — the host build (clippy /
    // unit tests) has no JS event loop, so any parked fiber is
    // definitionally unreachable. Gating the whole block keeps
    // host clippy clean (no unreachable_code / never_loop /
    // unused_assignments lints to silence one by one).
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    if matches!(result, InterpretResult::Success) {
        // Idle-budget instead of total-tick budget — the previous
        // 2500-tick cap killed productive long-running programs
        // (game loop awaiting `Browser.nextFrame` once per frame
        // hit the cap at ~2500 frames ≈ 10 s wall clock and the
        // run silently completed mid-flight). Now: we only
        // decrement the budget when a tick wakes up to find
        // nothing settled. Each frame the game advances counts
        // as progress and resets the budget; the bail still
        // triggers if the program is *actually* stuck (every
        // parked fiber awaiting a handle that nobody resolves).
        const IDLE_TICK_BUDGET: u32 = 250; // ~1 s of pure idle
        let mut idle_budget: u32 = IDLE_TICK_BUDGET;
        loop {
            // If nothing's parked, the user code already finished
            // synchronously — no scheduler work to do.
            if parked_fibers()
                .lock()
                .expect("parked fiber list poisoned")
                .is_empty()
            {
                break;
            }

            if idle_budget == 0 {
                break;
            }

            // Sleep until *either* (a) a future settles and
            // calls `wake_scheduler_if_armed`, or (b) the
            // `setTimeout(0)` macrotask backstop fires. The
            // wake path is what eliminates polling latency: the
            // moment `resolve_future` runs from a JS callback,
            // it triggers our resolver and the scheduler is
            // back in business one microtask later. The
            // setTimeout backstop is there for cases where the
            // first tick has nothing parked yet, or a future
            // settled while no waker was armed (rare race).
            let promise = js_sys::Promise::new(&mut |resolve, _reject| {
                *scheduler_waker().lock().expect("scheduler waker poisoned") =
                    Some(SchedulerWaker(resolve));
            });
            // Race the wake against a setTimeout backstop so
            // we can't deadlock if no future settles. The
            // backstop fires every ~4 ms (browser clamp),
            // which is the worst-case latency on a future
            // that settled without arming the waker.
            let backstop = js::wlift_yield_to_event_loop();
            let race = js_sys::Promise::race(&js_sys::Array::of2(&promise, &backstop));
            let _ = wasm_bindgen_futures::JsFuture::from(race).await;
            // Drop any waker we left armed — `Promise::race`
            // resolved through the backstop side, so the
            // waker won't be called. Leaving it set would
            // make the next tick fire immediately on the
            // next `resolve_future` call regardless of which
            // tick that future was actually parked from.
            scheduler_waker()
                .lock()
                .expect("scheduler waker poisoned")
                .take();

            // Snapshot the parked list, partition into ready vs.
            // still-pending, and resume the ready ones. New
            // entries pushed during resume (a fiber that awaits
            // again) accumulate into the live list.
            let parked =
                std::mem::take(&mut *parked_fibers().lock().expect("parked fiber list poisoned"));

            let mut still_parked = Vec::new();
            let mut resumed_this_tick = 0u32;
            for entry in parked {
                let state = future_store()
                    .lock()
                    .expect("future store poisoned")
                    .get(&entry.handle)
                    .map(|s| s.state)
                    .unwrap_or(FutureState::Rejected);

                if state == FutureState::Pending {
                    still_parked.push(entry);
                    continue;
                }
                resumed_this_tick += 1;

                // Resume the parked fiber. `vm_interp::run_fiber`
                // executes whatever fiber is currently in
                // `vm.fiber` until it yields or returns. Errors
                // get surfaced through the diagnostic buffer
                // (which lands in `result.output`) — without this,
                // an abort inside a scheduler-resumed fiber dies
                // silently because there's no caller fiber to
                // route it through.
                vm.fiber = entry.fiber;
                let res = wren_lift::runtime::vm_interp::run_fiber(&mut vm);
                if let Err(e) = res {
                    vm.report_error(&format!("scheduler-resumed fiber aborted: {}", e));
                    continue;
                }
                // Even when `run_fiber` returns Ok, an abort under
                // `is_try` lands in the fiber's `.error` slot
                // instead of an Err return — the bytecode catch
                // path marks the fiber Done with the message
                // stashed there, and run_fiber drops out cleanly.
                // Without surfacing it, the abort is silent in
                // the page UI. Pull the message out and route it
                // through `error_fn` so the diagnostic buffer
                // picks it up alongside `System.print` output.
                let err_str = unsafe {
                    let f = entry.fiber;
                    if !f.is_null() {
                        let err = (*f).error;
                        if err.is_string_object() {
                            let ptr = err.as_object().expect("string object pointer")
                                as *const wren_lift::runtime::object::ObjString;
                            Some((*ptr).as_str().to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                if let Some(s) = err_str {
                    vm.report_error(&format!("fiber error: {}", s));
                }
            }

            // Re-park anything still pending. New fibers pushed
            // by the resumes above (re-await on a different
            // handle) are already in the live list.
            parked_fibers()
                .lock()
                .expect("parked fiber list poisoned")
                .extend(still_parked);

            // Reset the idle budget on any progress; otherwise
            // count this tick against it. Productive game loops
            // (rAF tick → resume → re-park → wait) reset the
            // budget every frame and run forever; truly stuck
            // programs (every parked fiber awaits a handle no
            // one will resolve) tick the budget down and bail
            // after IDLE_TICK_BUDGET wakes (~1 s).
            if resumed_this_tick > 0 {
                idle_budget = IDLE_TICK_BUDGET;
            } else {
                idle_budget = idle_budget.saturating_sub(1);
            }
        }
    }
    perf_log("scheduler_loop", t_sched);

    let t_output = perf_mark();
    let mut output = vm.take_output();

    // Drain the diagnostic stream, strip ANSI colour codes, and
    // append it to whatever `System.print` already wrote. The
    // page renders plain text in a `<pre>`; ANSI escapes would
    // show up literally as `^[[31m…`.
    let diag = diagnostics.lock().expect("diagnostic buffer poisoned");
    if !diag.is_empty() {
        let plain = strip_ansi(&diag);
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&plain);
    }

    let (ok, error_kind) = match result {
        InterpretResult::Success => (true, ErrorKind::None),
        InterpretResult::CompileError => (false, ErrorKind::CompileError),
        InterpretResult::RuntimeError => (false, ErrorKind::RuntimeError),
    };
    perf_log("output_capture", t_output);
    perf_log("run_total", t_total);

    RunResult {
        output,
        ok,
        error_kind,
    }
}

/// Strip ANSI escape sequences from a diagnostic string. Handles
/// the CSI form (`ESC [ ... letter`) and the simpler reset (`ESC
/// [ m`) used by ariadne. Good enough for the colour codes the
/// runtime emits — not a general ANSI parser.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
            // Walk to the terminating letter (ASCII 0x40..=0x7E).
            i += 2;
            while i < bytes.len() && !(0x40..=0x7e).contains(&bytes[i]) {
                i += 1;
            }
            // Skip the terminator itself.
            if i < bytes.len() {
                i += 1;
            }
            continue;
        }
        // Append the next char without splitting UTF-8 code points.
        let next = s[i..]
            .chars()
            .next()
            .expect("char boundary preserved by parser");
        out.push(next);
        i += next.len_utf8();
    }
    out
}
