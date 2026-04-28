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
"#;

fn main() -> std::process::ExitCode {
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
