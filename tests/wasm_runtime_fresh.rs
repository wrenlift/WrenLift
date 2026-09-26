//! The prelinked wasm runtime must define what an AOT program calls.
//!
//! `wlift_runtime.o` is wren_lift built with `aot_runtime` for
//! wasm32-wasip1 and joined with wasi-libc by
//! `tools/build_wasm_runtime.sh`. Nothing rebuilds it when a helper is
//! added, so a stale object would let a program link and then fail at
//! instantiate with `unknown import`. This reads the object's symbol
//! table and fails naming the first entry point or helper it lacks.
//!
//! A second test links the object into a module with rust-lld and runs
//! a program through its entry points under wasmtime.
//!
//! Skipped when the object is absent: a checkout that never built the
//! wasm runtime is not broken. `WLIFT_RUNTIME` names one explicitly.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use wasmparser::{KnownCustom, Linking, Parser, Payload, SymbolFlags, SymbolInfo};
use wren_lift::capi::AOT_ENTRY_NAMES;
use wren_lift::codegen::runtime_fns::RUNTIME_FN_NAMES;

fn runtime_object() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("WLIFT_RUNTIME") {
        let p = PathBuf::from(explicit);
        return p.is_file().then_some(p);
    }
    // The test binary sits under target/<profile>/deps; the object sits
    // under target/<profile>/wasm32-wasip1.
    let exe = std::env::current_exe().ok()?;
    exe.ancestors()
        .map(|dir| dir.join("wasm32-wasip1").join("wlift_runtime.o"))
        .find(|p| p.is_file())
}

/// The global symbols the object defines.
fn defined_symbols(bytes: &[u8]) -> HashSet<String> {
    let mut out = HashSet::new();
    for payload in Parser::new(0).parse_all(bytes) {
        let Payload::CustomSection(section) = payload.expect("parsing the runtime object") else {
            continue;
        };
        let KnownCustom::Linking(linking) = section.as_known() else {
            continue;
        };
        for sub in linking {
            let Linking::SymbolTable(symbols) = sub.expect("reading the linking section") else {
                continue;
            };
            for symbol in symbols {
                let (flags, name) = match symbol.expect("reading a symbol") {
                    SymbolInfo::Func { flags, name, .. }
                    | SymbolInfo::Global { flags, name, .. } => (flags, name),
                    SymbolInfo::Data { flags, name, .. } => (flags, Some(name)),
                    _ => continue,
                };
                if flags.intersects(SymbolFlags::UNDEFINED | SymbolFlags::BINDING_LOCAL) {
                    continue;
                }
                if let Some(name) = name {
                    out.insert(name.to_string());
                }
            }
        }
    }
    out
}

#[test]
fn the_wasm_runtime_defines_what_aot_code_calls() {
    let Some(path) = runtime_object() else {
        eprintln!("no wasm32-wasip1/wlift_runtime.o beside the test binary; skipping");
        return;
    };
    let bytes = std::fs::read(&path).expect("reading the runtime object");
    let defined = defined_symbols(&bytes);
    let missing: Vec<&str> = AOT_ENTRY_NAMES
        .iter()
        .chain(RUNTIME_FN_NAMES)
        .copied()
        .filter(|n| !defined.contains(*n))
        .collect();
    assert!(
        missing.is_empty(),
        "{} is stale or incomplete: it does not define {}.\n\
         A wasm program would link and then fail at instantiate with\n\
         `unknown import: env::{}`. Rebuild it with tools/build_wasm_runtime.sh.",
        path.display(),
        missing.join(", "),
        missing[0],
    );
}

/// rust-lld from the toolchain building this test.
fn rust_lld() -> Option<PathBuf> {
    let out = |args: &[&str]| {
        let o = Command::new("rustc").args(args).output().ok()?;
        String::from_utf8(o.stdout).ok()
    };
    let sysroot = out(&["--print", "sysroot"])?;
    let host = out(&["-vV"])?
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_string))?;
    let lld = Path::new(sysroot.trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin/rust-lld");
    lld.is_file().then_some(lld)
}

/// The wasi-libc library directory the object was prelinked against.
fn wasi_lib_dir() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = std::env::var_os("WASI_SYSROOT")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    roots.extend(
        [
            "/opt/homebrew/opt/wasi-libc/share/wasi-sysroot",
            "/usr/local/opt/wasi-libc/share/wasi-sysroot",
            "/opt/wasi-sdk/share/wasi-sysroot",
            "/usr/share/wasi-sysroot",
        ]
        .map(PathBuf::from),
    );
    roots
        .into_iter()
        .map(|r| r.join("lib/wasm32-wasip1"))
        .find(|d| d.join("crt1-reactor.o").is_file())
}

#[test]
fn the_wasm_runtime_runs_a_program_under_wasmtime() {
    use wasmtime::{Engine, Linker, Module, Store};
    use wasmtime_wasi::preview1::{self, WasiP1Ctx};

    let (Some(object), Some(lld), Some(libs)) = (runtime_object(), rust_lld(), wasi_lib_dir())
    else {
        eprintln!("no runtime object, rust-lld or wasi-libc; skipping");
        return;
    };
    let wasm = object.with_file_name("wlift_runtime_check.wasm");
    let status = Command::new(&lld)
        .args(["-flavor", "wasm", "--no-entry"])
        .arg(libs.join("crt1-reactor.o"))
        .arg(&object)
        .arg(format!("-L{}", libs.display()))
        .args(["-lc", "--export=malloc", "--export=_initialize"])
        .args(["--export=wlift_aot_new_vm", "--export=wrenInterpret"])
        .arg("-o")
        .arg(&wasm)
        .status()
        .expect("running rust-lld");
    assert!(status.success(), "linking the runtime object failed");

    let engine = Engine::default();
    let module = Module::from_file(&engine, &wasm).expect("loading the linked module");
    let stdout = wasmtime_wasi::pipe::MemoryOutputPipe::new(64 * 1024);
    let wasi = wasmtime_wasi::WasiCtxBuilder::new()
        .stdout(stdout.clone())
        .build_p1();
    let mut store = Store::new(&engine, wasi);
    let mut linker: Linker<WasiP1Ctx> = Linker::new(&engine);
    preview1::add_to_linker_sync(&mut linker, |s| s).expect("wasi imports");
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiating the linked module");

    instance
        .get_typed_func::<(), ()>(&mut store, "_initialize")
        .and_then(|f| f.call(&mut store, ()))
        .expect("_initialize");
    let malloc = instance
        .get_typed_func::<i32, i32>(&mut store, "malloc")
        .expect("malloc");
    let memory = instance.get_memory(&mut store, "memory").expect("memory");
    let c_string = |store: &mut Store<WasiP1Ctx>, text: &str| {
        let bytes = [text.as_bytes(), &[0]].concat();
        let at = malloc
            .call(&mut *store, bytes.len() as i32)
            .expect("malloc");
        memory
            .write(&mut *store, at as usize, &bytes)
            .expect("writing a string");
        at
    };
    let name = c_string(&mut store, "main");
    let source = c_string(
        &mut store,
        "class P {\n  construct new(x) { _x = x }\n  x { _x }\n}\n\
         var s = 0\nfor (i in 0...1000) s = s + P.new(i).x\nSystem.print(s)\n\
         var f = Fiber.new {\n  Fiber.yield(1)\n  return 2\n}\n\
         System.print([f.call(), f.call()])\n",
    );

    let vm = instance
        .get_typed_func::<(), i32>(&mut store, "wlift_aot_new_vm")
        .and_then(|f| f.call(&mut store, ()))
        .expect("wlift_aot_new_vm");
    let result = instance
        .get_typed_func::<(i32, i32, i32), i32>(&mut store, "wrenInterpret")
        .and_then(|f| f.call(&mut store, (vm, name, source)))
        .expect("wrenInterpret");

    let out = String::from_utf8(stdout.contents().to_vec()).expect("utf-8 output");
    assert_eq!(result, 0, "the program failed:\n{out}");
    assert_eq!(out, "499500\n[1, 2]\n");
}
