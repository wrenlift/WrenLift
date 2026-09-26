//! The prelinked wasm runtime must define what an AOT program calls.
//!
//! `wlift_runtime.o` is wren_lift built with `aot_runtime` for
//! wasm32-wasip1 and joined with wasi-libc by
//! `tools/build_wasm_runtime.sh`. Nothing rebuilds it when a helper is
//! added, so a stale object would let a program link and then fail at
//! instantiate with `unknown import`. This reads the object's symbol
//! table and fails naming the first entry point or helper it lacks.
//!
//! Skipped when the object is absent: a checkout that never built the
//! wasm runtime is not broken. `WLIFT_RUNTIME` names one explicitly.

use std::collections::HashSet;
use std::path::PathBuf;

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
