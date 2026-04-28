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

const SOURCE: &str = r#"
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
class Browser {
    static setTimeout(ms) { Future.new_(BrowserCore.setTimeoutHandle(ms)) }
    static fetch(url)     { Future.new_(BrowserCore.fetchHandle(url)) }
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
"#;

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
