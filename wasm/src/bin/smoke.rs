//! WASI smoke test for the wasm-built WrenLift runtime.
//!
//! Doesn't go through `wasm-bindgen` — just compiles to
//! `wasm32-wasip1` so `wasmtime smoke.wasm` runs the Wren program
//! and stdout flows through normally. This proves the runtime
//! itself works under wasm; the `wasm-bindgen` lib in
//! `wasm/src/lib.rs` is the browser-facing variant of the same
//! interpreter.
//!
//! Run with:
//!
//!     cargo build -p wlift_wasm --bin smoke --target wasm32-wasip1 --release
//!     wasmtime target/wasm32-wasip1/release/smoke.wasm

use wren_lift::runtime::engine::{ExecutionMode, InterpretResult};
use wren_lift::runtime::vm::{VMConfig, VM};

// `r##"..."##` (rather than the usual `r#"..."#`) because the
// Wren source includes selectors like `"#title"` whose `"#`
// would otherwise terminate the raw string early.
const SOURCE: &str = r##"
import "time" for TimeCore

System.print("hello from wasm!")
var i = 0
var sum = 0
while (i < 10) {
    sum = sum + i
    i = i + 1
}
System.print("0+1+...+9 = %(sum)")
System.print("List ops:")
var nums = [1, 2, 3, 4, 5]
var doubled = nums.map {|x| x * 2}.toList
System.print(doubled)
// Probe `web-time`-backed clocks: confirm Instant + SystemTime
// don't panic at runtime under wasm32-unknown-unknown / wasi.
var t0 = TimeCore.mono
var t1 = TimeCore.mono
System.print("time ok: mono delta >= 0 = %((t1 - t0) >= 0)")
var unix = TimeCore.unix
System.print("unix ok: nonzero = %(unix > 0)")

// Foreign-method registry probe: dispatch into the statically-
// linked `wlift_image` plugin. We don't ship a real PNG in the
// smoke binary — calling decode with garbage bytes is enough to
// confirm the symbol resolved (the plugin's own runtime_error
// path runs instead of "Class does not implement 'decode_(_)'",
// which is what we'd see if the foreign loader returned an
// error).
#!native = "wlift_image"
foreign class ImageCore {
    #!symbol = "wlift_image_decode"
    foreign static decode(bytes)
}
var probe = Fiber.new { ImageCore.decode([1, 2, 3]) }
var err = probe.try()
if (err is String && err.contains("does not implement")) {
    System.print("foreign: BAD (%(err))")
} else {
    System.print("foreign: ok (decode dispatched; err=%(err))")
}

// Promise ↔ Fiber.yield bridge probe. `Browser.setTimeout(0)`
// returns a Future; under wasi this resolves synchronously, so
// `await`'s yield loop sees `state == 1` on the very next
// iteration and the fiber returns. The browser host wires the
// same path to `setTimeout` via wasm-bindgen.
#!native = "browser"
foreign class BrowserCore {
    #!symbol = "browser_set_timeout"
    foreign static setTimeoutHandle(ms)
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
}
class Future {
    construct new_(handle) { _h = handle }
    state { BrowserCore.peekState(_h) }
    await {
        // 0 = pending, 1 = resolved, 2 = rejected. Cache the
        // settled state BEFORE `takeValue` removes the slot —
        // a subsequent `peekState` on a missing handle reports
        // Rejected, which would mis-fire `Fiber.abort` on a
        // resolved Future.
        while (state == 0) Fiber.yield()
        var settled = state
        var v = BrowserCore.takeValue(_h)
        if (settled == 2) Fiber.abort(v == null ? "Future rejected" : v)
        return v
    }
}
class WebSocket {
    construct new_(handle) { _h = handle }
    send(text)  { BrowserCore.wsSend(_h, text) }
    recv        { Future.new_(BrowserCore.wsRecv(_h)) }
    close       { BrowserCore.wsClose(_h) }
}
class Browser {
    static setTimeout(ms) { Future.new_(BrowserCore.setTimeoutHandle(ms)) }
    static fetch(url)     { Future.new_(BrowserCore.fetchHandle(url)) }
    static connect(url)   { WebSocket.new_(BrowserCore.wsOpen(url)) }
}

var awaitFiber = Fiber.new {
    Browser.setTimeout(0).await
    System.print("future: awaited a 0ms timeout")
    return "awaited"
}
var awaitResult = awaitFiber.try()
if (awaitResult == "awaited") {
    System.print("future: ok")
} else {
    System.print("future: BAD (%(awaitResult))")
}

// fetch bridge probe. Under wasi the foreign method resolves
// synchronously with `MOCK:<url>`; under a real browser host the
// JS `_wlift_fetch` shim does an actual `fetch(url)` and feeds
// the response body back through the same handle store. The Wren
// code is identical either way.
var fetchFiber = Fiber.new {
    return Browser.fetch("https://example.invalid/api").await
}
var fetchResult = fetchFiber.try()
if (fetchResult == "MOCK:https://example.invalid/api") {
    System.print("fetch: ok")
} else {
    System.print("fetch: BAD (%(fetchResult))")
}

// WebSocket bridge probe. Under wasi the connect/send/recv
// loopback through a single handle's inbox queue — no real
// network. A two-message round trip exercises both the immediate-
// inbox path (recv finds a buffered message) and the parked-
// waiter path (recv blocks, send wakes the waiter).
var wsFiber = Fiber.new {
    var ws = Browser.connect("ws://localhost/echo")
    ws.send("hello")
    ws.send("world")
    var first  = ws.recv.await
    var second = ws.recv.await
    ws.close
    return first + " " + second
}
var wsResult = wsFiber.try()
if (wsResult == "hello world") {
    System.print("ws: ok")
} else {
    System.print("ws: BAD (%(wsResult))")
}

// DOM bridge probe. There's no DOM under wasi — `dom_*` foreign
// methods resolve synchronously with `MOCK_DOM_*:<selector>`, so
// the smoke test verifies symbol resolution + the read/write
// future shape without needing a browser. Browser hosts route
// the same call through the page's `dispatchDomOp` table.
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
class Dom {
    static text(selector)                  { Future.new_(DomCore.textHandle(selector)) }
    static setText(selector, value)        { Future.new_(DomCore.setTextHandle(selector, value)) }
    static getAttribute(selector, name)    { Future.new_(DomCore.getAttributeHandle(selector, name)) }
    static setAttribute(selector, n, v)    { Future.new_(DomCore.setAttributeHandle(selector, n, v)) }
    static addClass(selector, name)        { Future.new_(DomCore.addClassHandle(selector, name)) }
    static removeClass(selector, name)     { Future.new_(DomCore.removeClassHandle(selector, name)) }
    static queryAll(selector)              { Future.new_(DomCore.queryAllHandle(selector)) }
}
var domFiber = Fiber.new {
    var read    = Dom.text("#title").await
    var written = Dom.setText("#title", "hello dom").await
    return read + "|" + written
}
var domResult = domFiber.try()
if (domResult == "MOCK_DOM_TEXT:#title|MOCK_DOM_SET_TEXT:#title") {
    System.print("dom: ok")
} else {
    System.print("dom: BAD (%(domResult))")
}

// New Dom ops — attributes, classes, queryAll.
var domExtFiber = Fiber.new {
    var attrSet  = Dom.setAttribute("#x", "data-id", "42").await
    var attrGet  = Dom.getAttribute("#x", "data-id").await
    var addCls   = Dom.addClass("#x", "active").await
    var rmCls    = Dom.removeClass("#x", "active").await
    var qAll     = Dom.queryAll(".item").await
    return [attrSet, attrGet, addCls, rmCls, qAll].join(",")
}
var domExtResult = domExtFiber.try()
var expected = "MOCK_DOM_SET_ATTR:#x,MOCK_DOM_GET_ATTR:#x,MOCK_DOM_ADD_CLASS:#x,MOCK_DOM_REMOVE_CLASS:#x,MOCK_DOM_QUERY_ALL:.item"
if (domExtResult == expected) {
    System.print("dom-ext: ok")
} else {
    System.print("dom-ext: BAD (%(domExtResult))")
}

// Storage bridge — sync in both modes (localStorage/
// sessionStorage exist on WorkerGlobalScope), so the wasi mock
// just verifies symbol resolution via the same MOCK_* payload
// pattern the DOM ops use.
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
var storageFiber = Fiber.new {
    var setR    = LocalStorage.set("k", "v").await
    var getR    = LocalStorage.get("k").await
    var removeR = LocalStorage.remove("k").await
    var clearR  = LocalStorage.clear.await
    return [setR, getR, removeR, clearR].join(",")
}
var storageResult = storageFiber.try()
var storageExpect = "MOCK_STORAGE_SET:local:k,MOCK_STORAGE_GET:local:k,MOCK_STORAGE_REMOVE:local:k,MOCK_STORAGE_CLEAR:local"
if (storageResult == storageExpect) {
    System.print("storage: ok")
} else {
    System.print("storage: BAD (%(storageResult))")
}
"##;

fn main() -> std::process::ExitCode {
    // Static plugin registration. On `wasm32-*` targets this
    // populates `runtime::foreign`'s registry with every linked
    // plugin's `wlift_*` exports; on a non-wasm host it's a
    // no-op (the host loader uses dlsym instead).
    wlift_wasm::register_static_plugins();

    let mut vm = VM::new(VMConfig {
        execution_mode: ExecutionMode::Interpreter,
        ..VMConfig::default()
    });

    println!("=== wlift_wasm smoke (wasi) ===");
    let result = vm.interpret("main", SOURCE);
    match result {
        InterpretResult::Success => {
            println!("--- ok ---");
            std::process::ExitCode::SUCCESS
        }
        InterpretResult::CompileError => {
            eprintln!("--- compile error ---");
            std::process::ExitCode::from(65)
        }
        InterpretResult::RuntimeError => {
            eprintln!("--- runtime error ---");
            std::process::ExitCode::from(70)
        }
    }
}
