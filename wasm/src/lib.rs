//! wasm-bindgen entry shim for the WrenLift interpreter.
//!
//! Two functions exposed to JS / wasi hosts so far:
//!
//!   * `version()` — returns a `&str` describing the wlift build,
//!     useful as a smoke test that the wasm module loaded at all.
//!   * `run(source: &str) -> RunResult` — compiles the source as a
//!     `<wasm>` module, drives the BC interpreter, and returns the
//!     captured `System.print` output plus an `ok` flag.
//!
//! No JIT, no plugin loading, no fs / sockets. Future phases grow
//! the surface (incremental compile, multi-module install,
//! browser-API foreign classes); this is the minimal demo.

use wasm_bindgen::prelude::*;

use wren_lift::runtime::engine::{InterpretResult, ExecutionMode};
use wren_lift::runtime::vm::{VM, VMConfig};

/// Optional one-time setup. Browser callers should invoke this
/// from JS before the first `run` so that any panic deeper in
/// the runtime surfaces as a console error rather than the
/// generic `unreachable executed` string.
#[wasm_bindgen(start)]
pub fn _wasm_init() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
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

/// Compile + interpret a Wren source string. Output captured into
/// the returned `RunResult`. The VM lives only for the duration
/// of this call — every `run` is a fresh interpreter, no shared
/// state across calls. (A multi-call API that holds onto a `VM`
/// instance lands in Phase 1.1.)
#[wasm_bindgen]
pub fn run(source: &str) -> RunResult {
    let mut vm = VM::new(VMConfig {
        // No JIT on wasm — gated out at compile time, but the mode
        // gate also short-circuits all the tier-up bookkeeping the
        // BC interpreter would otherwise pay per call.
        execution_mode: ExecutionMode::Interpreter,
        ..VMConfig::default()
    });
    vm.output_buffer = Some(String::new());

    let result = vm.interpret("main", source);
    let output = vm.take_output();

    let (ok, error_kind) = match result {
        InterpretResult::Success => (true, ErrorKind::None),
        InterpretResult::CompileError => (false, ErrorKind::CompileError),
        InterpretResult::RuntimeError => (false, ErrorKind::RuntimeError),
    };

    RunResult {
        output,
        ok,
        error_kind,
    }
}
