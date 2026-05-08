//! WASM smoke test: build the `smoke` binary for `wasm32-wasip1`,
//! run it under the embedded `wasmtime` runtime, and assert the
//! Wren program's stdout matches the expected lines.
//!
//! Two variants run sequentially in the same `#[test]` body so they
//! can't race for the cached `target/wasm32-wasip1/release/smoke.wasm`
//! artifact:
//!
//!   1. **baseline** — default flags. Exercises the scalar fallback
//!      in `simd_kernels`, which is what every browser without v128
//!      support runs.
//!   2. **simd128** — `RUSTFLAGS="-C target-feature=+simd128"`.
//!      Exercises the wasm SIMD fast path; the assertion below also
//!      walks the code section and confirms wasm v128 opcodes are
//!      present, so a future regression that drops the intrinsic
//!      paths back to scalar (and slows the modern-browser path)
//!      can't sneak through quietly.
//!
//! Run with `cargo test --test wasm_smoke -- --nocapture` (the
//! `wasm32-wasip1` rustup target must be installed).

use std::io::Read;
use std::process::Command;

#[test]
fn smoke_runs_under_wasmtime() {
    let baseline_wasm = build_smoke(false);
    let baseline_simd = count_simd_opcodes(&baseline_wasm);
    eprintln!(
        "[smoke] baseline build: {} bytes, {} SIMD opcodes",
        baseline_wasm.len(),
        baseline_simd,
    );
    assert_eq!(
        baseline_simd, 0,
        "baseline build (no -C target-feature=+simd128) must not emit \
         any wasm SIMD opcodes — older browsers reject v128 modules \
         outright. Found {} v128 ops in the wasm.",
        baseline_simd
    );
    run_and_assert(&baseline_wasm);

    let simd128_wasm = build_smoke(true);
    let simd128_count = count_simd_opcodes(&simd128_wasm);
    eprintln!(
        "[smoke] simd128 build: {} bytes, {} SIMD opcodes",
        simd128_wasm.len(),
        simd128_count,
    );
    // The kernels in `simd_kernels.rs` emit explicit `v128_load` /
    // `f32x4_*` / `i32x4_*` calls for every SIMD method; even after
    // dead-code elimination the smoke program touches enough of them
    // (`+`, `-`, `>`, `bitmask`, `splat`, `load`, `store`) that we
    // expect well over 100 opcodes in the artifact.
    assert!(
        simd128_count > 100,
        "simd128 build emitted only {} v128 opcodes — expected the \
         explicit `core::arch::wasm32` paths in `simd_kernels.rs` to \
         produce many more. Has the kernel module regressed back to \
         the scalar fallback under +simd128?",
        simd128_count
    );
    run_and_assert(&simd128_wasm);

    // Quiet `unused` warning on the `Read` import.
    let _ = std::io::empty().read(&mut [0u8; 0]).is_ok();
}

/// Build the `smoke` binary, optionally with wasm SIMD enabled.
/// Returns the wasm bytes from `target/wasm32-wasip1/release/smoke.wasm`
/// after the cargo invocation completes.
fn build_smoke(simd128: bool) -> Vec<u8> {
    let mut cmd = Command::new("cargo");
    cmd.args([
        "build",
        "-p",
        "wlift_wasm",
        "--bin",
        "smoke",
        "--target",
        "wasm32-wasip1",
        "--release",
        "--quiet",
    ]);
    if simd128 {
        // RUSTFLAGS overrides any value already in the env so the
        // test stays deterministic regardless of how the harness
        // was invoked.
        cmd.env("RUSTFLAGS", "-C target-feature=+simd128");
    } else {
        // Force the absence of any prior `+simd128` so the baseline
        // run actually emits a v128-free module even if the user
        // had RUSTFLAGS set in their shell.
        cmd.env_remove("RUSTFLAGS");
    }
    let status = cmd.status().expect("invoke cargo to build smoke.wasm");
    let flavour = if simd128 { "simd128" } else { "baseline" };
    assert!(
        status.success(),
        "cargo build for wlift_wasm smoke ({}) failed",
        flavour
    );

    let wasm_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("wasm32-wasip1")
        .join("release")
        .join("smoke.wasm");
    assert!(
        wasm_path.exists(),
        "smoke.wasm missing at {} after {} build",
        wasm_path.display(),
        flavour,
    );
    std::fs::read(&wasm_path).expect("read smoke.wasm")
}

/// Count wasm SIMD (v128) opcodes in `bytes` by walking each function
/// in the code section. Uses `wasmparser` so we don't false-positive
/// on `0xFD` bytes that show up inside LEB128 encodings, immediates,
/// data segments, etc.
fn count_simd_opcodes(bytes: &[u8]) -> usize {
    use wasmparser::{Parser, Payload};
    let mut count = 0usize;
    for payload in Parser::new(0).parse_all(bytes) {
        let payload = payload.expect("wasmparser walks smoke.wasm");
        if let Payload::CodeSectionEntry(body) = payload {
            let mut reader = body.get_operators_reader().expect("operators reader");
            while !reader.eof() {
                let op = reader.read().expect("read operator");
                if is_simd_op(&op) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Heuristic: any operator whose textual name starts with one of the
/// wasm SIMD lane-width prefixes counts. Keeps the test independent
/// of `wasmparser`'s `Operator` variant naming churn — we just turn
/// the operator into its debug spelling and pattern-match the head.
fn is_simd_op(op: &wasmparser::Operator<'_>) -> bool {
    let s = format!("{:?}", op);
    s.starts_with("V128")
        || s.starts_with("I8x16")
        || s.starts_with("I16x8")
        || s.starts_with("I32x4")
        || s.starts_with("I64x2")
        || s.starts_with("F32x4")
        || s.starts_with("F64x2")
}

/// Run `wasm` under wasmtime + WASI preview1 and assert the captured
/// stdout matches every expected line.
fn run_and_assert(wasm: &[u8]) {
    use wasmtime::{Config, Engine, Linker, Module, Store};
    use wasmtime_wasi::preview1;
    let mut config = Config::new();
    config.consume_fuel(false);
    let engine = Engine::new(&config).expect("wasmtime engine");

    let module = Module::from_binary(&engine, wasm).expect("load smoke.wasm");

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

    if let Err(err) = start.call(&mut store, ()) {
        if let Some(exit) = err.downcast_ref::<wasmtime_wasi::I32Exit>() {
            assert_eq!(exit.0, 0, "smoke.wasm exited with non-zero code {}", exit.0);
        } else {
            panic!("smoke.wasm trapped: {:?}", err);
        }
    }

    let captured = stdout.contents();
    let captured = std::str::from_utf8(&captured).expect("smoke output is utf-8");
    eprintln!("{}", captured);

    for needle in [
        "hello from wasm!",
        "0+1+...+9 = 45",
        "[2, 4, 6, 8, 10]",
        "simd: ok",
        "time ok: mono delta >= 0 = true",
        "unix ok: nonzero = true",
        "foreign: ok",
        "future: ok",
        "fetch: ok",
        "ws: ok",
        "dom: ok",
        "dom-ext: ok",
        "storage: ok",
    ] {
        assert!(
            captured.contains(needle),
            "missing `{}` in smoke output:\n{}",
            needle,
            captured,
        );
    }
}
