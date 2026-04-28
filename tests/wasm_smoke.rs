//! WASM smoke test: build the `smoke` binary for `wasm32-wasip1`,
//! run it under the embedded `wasmtime` runtime, and assert the
//! Wren program's stdout matches the expected lines.
//!
//! Run with `cargo test --test wasm_smoke -- --nocapture` (the
//! `wasm32-wasip1` rustup target must be installed; the test
//! `cargo build`s the `wlift_wasm` smoke binary on demand).

use std::io::Read;
use std::process::Command;

#[test]
fn smoke_runs_under_wasmtime() {
    // Build the wasi smoke binary. Skipping a stale/cached check on
    // purpose — cargo's incremental cache makes this cheap once
    // the runtime sub-crate is compiled, and we'd rather rebuild
    // than silently test a stale artifact.
    let status = Command::new("cargo")
        .args([
            "build",
            "-p",
            "wlift_wasm",
            "--bin",
            "smoke",
            "--target",
            "wasm32-wasip1",
            "--release",
            "--quiet",
        ])
        .status()
        .expect("invoke cargo to build smoke.wasm");
    assert!(
        status.success(),
        "cargo build for wlift_wasm smoke binary failed"
    );

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("wasm32-wasip1")
        .join("release")
        .join("smoke.wasm");
    assert!(
        wasm_path.exists(),
        "smoke.wasm missing at {}",
        wasm_path.display()
    );

    // Run via the embedded `wasmtime` crate (a dev-dep already
    // pinned for the JIT codegen WAT inspector) instead of the
    // `wasmtime` CLI — works on any host without an installed
    // wasmtime binary, and the output capture is cleaner.
    use wasmtime::{Config, Engine, Linker, Module, Store};
    use wasmtime_wasi::preview1;
    let mut config = Config::new();
    config.consume_fuel(false);
    let engine = Engine::new(&config).expect("wasmtime engine");

    let module = Module::from_file(&engine, &wasm_path).expect("load smoke.wasm");

    // Wasi preview1 stdout/stderr need a backing host pipe; the
    // wasi-cap-std sync impl handles that. We pipe stdout through
    // a writer the test can read after `_start` returns.
    let stdout = wasmtime_wasi::pipe::MemoryOutputPipe::new(64 * 1024);
    let stdout_clone = stdout.clone();
    let wasi = wasmtime_wasi::WasiCtxBuilder::new()
        .stdout(stdout_clone)
        .build_p1();
    let mut store = Store::new(&engine, wasi);

    let mut linker: Linker<preview1::WasiP1Ctx> = Linker::new(&engine);
    preview1::add_to_linker_sync(&mut linker, |s| s).expect("wasi-preview1 imports");

    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiate smoke.wasm");
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .expect("smoke.wasm exports _start");

    // WASI's `_start` ends by invoking `proc_exit`, which wasmtime
    // surfaces as an `I32Exit` "trap" carrying the exit code. We
    // accept code 0 as a clean run; any other code (or a real
    // trap) is a failure.
    if let Err(err) = start.call(&mut store, ()) {
        if let Some(exit) = err.downcast_ref::<wasmtime_wasi::I32Exit>() {
            assert_eq!(
                exit.0, 0,
                "smoke.wasm exited with non-zero code {}",
                exit.0
            );
        } else {
            panic!("smoke.wasm trapped: {:?}", err);
        }
    }

    let captured = stdout.contents();
    let captured = std::str::from_utf8(&captured).expect("smoke output is utf-8");
    eprintln!("{}", captured);

    assert!(
        captured.contains("hello from wasm!"),
        "missing greeting; captured:\n{}",
        captured
    );
    assert!(
        captured.contains("0+1+...+9 = 45"),
        "missing arithmetic; captured:\n{}",
        captured
    );
    assert!(
        captured.contains("[2, 4, 6, 8, 10]"),
        "missing list ops; captured:\n{}",
        captured
    );
    assert!(
        captured.contains("time ok: mono delta >= 0 = true"),
        "monotonic Instant didn't survive in wasm; captured:\n{}",
        captured
    );
    assert!(
        captured.contains("unix ok: nonzero = true"),
        "SystemTime returned 0/epoch on wasm; captured:\n{}",
        captured
    );

    // Quiet `unused` warning on the `Read` import — kept around
    // because earlier iterations of this test piped through a
    // separate reader thread; if we revive that path we don't
    // want to dig back through diff history.
    let _ = std::io::empty().read(&mut [0u8; 0]).is_ok();
}
