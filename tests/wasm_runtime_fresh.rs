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
use std::process::Command;

mod common;
use common::{runtime_object, rust_lld, wasi_lib_dir};

use wasmparser::{KnownCustom, Linking, Parser, Payload, SymbolFlags, SymbolInfo};
use wren_lift::capi::AOT_ENTRY_NAMES;
use wren_lift::codegen::runtime_fns::RUNTIME_FN_NAMES;
use wren_lift::runtime::object_layout::{Layout, layout_mismatches};

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

/// The runtime object linked alone into a reactor module, instantiated
/// and initialised under wasmtime, with its stdout captured. None when
/// the object or the tools to link it are absent.
struct Linked {
    store: wasmtime::Store<wasmtime_wasi::preview1::WasiP1Ctx>,
    instance: wasmtime::Instance,
    stdout: wasmtime_wasi::pipe::MemoryOutputPipe,
}

fn link_runtime(exports: &[&str]) -> Option<Linked> {
    use wasmtime::{Engine, Linker, Module, Store};
    use wasmtime_wasi::preview1::{self, WasiP1Ctx};

    let (Some(object), Some(lld), Some(libs)) = (runtime_object(), rust_lld(), wasi_lib_dir())
    else {
        eprintln!("no runtime object, rust-lld or wasi-libc; skipping");
        return None;
    };
    let name = exports.join("-");
    let wasm = object.with_file_name(format!("wlift_runtime_check_{name}.wasm"));
    let status = Command::new(&lld)
        .args(["-flavor", "wasm", "--no-entry"])
        .arg(libs.join("crt1-reactor.o"))
        .arg(&object)
        .arg(format!("-L{}", libs.display()))
        .args(["-lc", "--export=malloc", "--export=_initialize"])
        .args(exports.iter().map(|e| format!("--export={e}")))
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
    Some(Linked {
        store,
        instance,
        stdout,
    })
}

impl Linked {
    fn malloc(&mut self, len: usize) -> i32 {
        self.instance
            .get_typed_func::<i32, i32>(&mut self.store, "malloc")
            .and_then(|f| f.call(&mut self.store, len as i32))
            .expect("malloc")
    }

    fn memory(&mut self) -> wasmtime::Memory {
        self.instance
            .get_memory(&mut self.store, "memory")
            .expect("memory")
    }

    fn c_string(&mut self, text: &str) -> i32 {
        let bytes = [text.as_bytes(), &[0]].concat();
        let at = self.malloc(bytes.len());
        let memory = self.memory();
        memory
            .write(&mut self.store, at as usize, &bytes)
            .expect("writing a string");
        at
    }
}

#[test]
fn the_wasm_runtime_runs_a_program_under_wasmtime() {
    let Some(mut m) = link_runtime(&["wlift_aot_new_vm", "wrenInterpret"]) else {
        return;
    };
    let name = m.c_string("main");
    let source = m.c_string(
        "class P {\n  construct new(x) { _x = x }\n  x { _x }\n}\n\
         var s = 0\nfor (i in 0...1000) s = s + P.new(i).x\nSystem.print(s)\n\
         var f = Fiber.new {\n  Fiber.yield(1)\n  return 2\n}\n\
         System.print([f.call(), f.call()])\n",
    );
    let vm = m
        .instance
        .get_typed_func::<(), i32>(&mut m.store, "wlift_aot_new_vm")
        .and_then(|f| f.call(&mut m.store, ()))
        .expect("wlift_aot_new_vm");
    let result = m
        .instance
        .get_typed_func::<(i32, i32, i32), i32>(&mut m.store, "wrenInterpret")
        .and_then(|f| f.call(&mut m.store, (vm, name, source)))
        .expect("wrenInterpret");

    let out = String::from_utf8(m.stdout.contents().to_vec()).expect("utf-8 output");
    assert_eq!(result, 0, "the program failed:\n{out}");
    assert_eq!(out, "499500\n[1, 2]\n");
}

/// The wasm32 layout table is what a wasm32 build of the runtime has.
#[test]
fn the_wasm32_layout_table_is_the_layout_wasm32_has() {
    let Some(mut m) = link_runtime(&["wlift_layout_probe"]) else {
        return;
    };
    let cap = Layout::ILP32.entries().len();
    let out = m.malloc(cap * 4);
    let n = m
        .instance
        .get_typed_func::<(i32, i32), i32>(&mut m.store, "wlift_layout_probe")
        .and_then(|f| f.call(&mut m.store, (out, cap as i32)))
        .expect("wlift_layout_probe") as usize;
    let mut bytes = vec![0u8; cap * 4];
    let memory = m.memory();
    memory
        .read(&m.store, out as usize, &mut bytes)
        .expect("reading the probe");
    let values: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    assert_eq!(
        n, cap,
        "the probe and this test disagree on the entry count"
    );
    let actual = Layout::from_values(&values).expect("a full layout");
    assert!(
        Layout::ILP32 == actual,
        "object_layout's wasm32 table is stale:\n{}",
        layout_mismatches(&Layout::ILP32, &actual)
    );
}
