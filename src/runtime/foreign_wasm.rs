//! Wasm-side foreign-library loader.
//!
//! `wasm32-unknown-unknown` has no `dlopen`/`dlsym`, so plugins
//! ship as Rust crates that the runtime statically links in. Each
//! plugin calls [`register_plugin_symbol`] at host-init time
//! (`wlift_wasm` drives this from its `_wasm_init` shim) to add
//! its `(library_name, symbol, fn_ptr)` triples to a process-wide
//! registry. Then [`load_library`] and [`resolve_symbol`] —
//! invoked by the runtime exactly the same way they are on host
//! when a Wren `foreign class` declaration carries
//! `#!native = "<name>"` / `#!symbol = "<name>"` — consult the
//! registry instead of the OS loader.
//!
//! The host build's `foreign.rs` re-exports `libloading::Library`
//! and uses `dlopen`. This file mirrors that public surface:
//! `vm.rs` / `engine.rs` / `vm_interp.rs` reference
//! `runtime::foreign::*` portably across both targets.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::runtime::object::ForeignCFn;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

pub const WLIFT_PLUGIN_ABI_VERSION: u32 = 1;
pub const WLIFT_PLUGIN_ABI_SYMBOL: &str = "wlift_plugin_abi_version";

// ---------------------------------------------------------------------------
// Static plugin registry
// ---------------------------------------------------------------------------

/// Process-wide map of `library_name -> symbol -> fn_ptr`. Populated
/// by [`register_plugin_symbol`] at startup, consulted by
/// [`load_library`] / [`resolve_symbol`] at every foreign-class
/// install.
fn registry() -> &'static Mutex<HashMap<String, HashMap<String, ForeignCFn>>> {
    static REG: OnceLock<Mutex<HashMap<String, HashMap<String, ForeignCFn>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a single foreign-method symbol against a library name.
/// Plugin crates expose a `pub fn register_static_symbols()` that
/// calls this for every `#[no_mangle]` `wlift_*` export they want
/// the runtime to bind. Idempotent — re-registering the same
/// `(library, symbol)` pair just overwrites; convenient for hot
/// reload tooling that wants to swap a function without bouncing
/// the whole VM.
pub fn register_plugin_symbol(library: &str, symbol: &str, func: ForeignCFn) {
    let mut reg = registry().lock().expect("foreign registry poisoned");
    reg.entry(library.to_string())
        .or_default()
        .insert(symbol.to_string(), func);
}

/// Convenience for plugins whose exports are declared
/// `unsafe extern "C" fn(*mut VM)` (the standard
/// `#[no_mangle] pub unsafe extern "C" fn ...` shape). The host
/// loader achieves the same cast implicitly through
/// `libloading::Symbol<ForeignCFn>` — every Wren foreign-method
/// dispatch site already operates on the assumption that the
/// callee respects the safe-FFI contract documented by
/// `NativeContext`. We transmute through the same hole here so
/// plugins don't need a per-symbol cast boilerplate.
///
/// # Safety
///
/// The caller is asserting that `func` honours the
/// `extern "C" fn(*mut VM)` ABI: the function reads slot args off
/// the VM, writes a return value via `set_return`, and never
/// unwinds. Every wlift plugin's `wlift_<name>` exports satisfy
/// this — they're written explicitly to the same shape the host
/// build invokes after `dlsym`.
pub fn register_plugin_symbol_unsafe(
    library: &str,
    symbol: &str,
    func: unsafe extern "C" fn(*mut VM),
) {
    // SAFETY: identical ABI; only the surface-level `unsafe`
    // marker differs. `ForeignCFn` is `extern "C" fn(*mut VM)`,
    // matching `func`'s calling convention bit-for-bit.
    let safe: ForeignCFn = unsafe { std::mem::transmute(func) };
    register_plugin_symbol(library, symbol, safe);
}

/// Snapshot of every `(library, symbol)` pair currently registered.
/// Hosts use this for diagnostics / a "what's available?" REPL
/// command. Cheap because it clones strings, not function
/// pointers; expected callers measure in seconds, not frames.
pub fn registered_symbols() -> Vec<(String, String)> {
    let reg = registry().lock().expect("foreign registry poisoned");
    let mut out = Vec::new();
    for (lib, syms) in reg.iter() {
        for sym in syms.keys() {
            out.push((lib.clone(), sym.clone()));
        }
    }
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// Public types — same shape as `foreign.rs`
// ---------------------------------------------------------------------------

/// Library handle. The host uses `libloading::Library` here; the
/// wasm build only carries the name for diagnostics — the actual
/// symbol lookup goes through the static registry, not through
/// any owned dylib state.
pub struct Library {
    name: String,
}

#[derive(Debug)]
pub enum ForeignLoadError {
    LibraryNotFound {
        name: String,
        tried: Vec<String>,
    },
    SymbolNotFound {
        library: String,
        symbol: String,
    },
    AbiMismatch {
        library: String,
        expected: u32,
        found: u32,
    },
    Unsupported,
}

impl std::fmt::Display for ForeignLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForeignLoadError::LibraryNotFound { name, tried } => {
                if tried.is_empty() {
                    write!(
                        f,
                        "native library '{}' is not registered in the wasm runtime.",
                        name
                    )
                } else {
                    write!(
                        f,
                        "native library '{}' is not registered (tried: {}).",
                        name,
                        tried.join(", ")
                    )
                }
            }
            ForeignLoadError::SymbolNotFound { library, symbol } => {
                write!(
                    f,
                    "symbol '{}' not registered under '{}' (wasm build).",
                    symbol, library
                )
            }
            ForeignLoadError::AbiMismatch {
                library,
                expected,
                found,
            } => {
                write!(
                    f,
                    "plugin '{}' ABI version {} does not match host expected {}.",
                    library, found, expected
                )
            }
            ForeignLoadError::Unsupported => {
                write!(f, "foreign plugin loading is not supported on wasm.")
            }
        }
    }
}

impl std::error::Error for ForeignLoadError {}

// ---------------------------------------------------------------------------
// Loader API — mirrors `foreign.rs`
// ---------------------------------------------------------------------------

pub fn library_candidates(name: &str) -> Vec<String> {
    vec![name.to_string()]
}

pub fn verify_plugin_abi(_library: &Library, _name: &str) -> Result<(), ForeignLoadError> {
    // Statically linked plugins inherit the host's ABI version by
    // construction — there's no way to mix versions across the
    // wasm boundary because both are baked into the same artefact.
    // The host loader checks this for dlopen'd plugins; we don't
    // need to here.
    Ok(())
}

pub fn load_library(
    name: &str,
    _search_paths: &[PathBuf],
    _name_overrides: &HashMap<String, PathBuf>,
) -> Result<Library, ForeignLoadError> {
    let reg = registry().lock().expect("foreign registry poisoned");
    if reg.contains_key(name) {
        Ok(Library {
            name: name.to_string(),
        })
    } else {
        Err(ForeignLoadError::LibraryNotFound {
            name: name.to_string(),
            tried: vec![format!("<wasm static registry: {} libs>", reg.len())],
        })
    }
}

pub fn resolve_symbol(
    library: &Library,
    library_name: &str,
    symbol: &str,
) -> Result<ForeignCFn, ForeignLoadError> {
    let reg = registry().lock().expect("foreign registry poisoned");
    // Trust the caller's `library_name` over the handle's stored
    // name — the runtime passes both because the host loader's
    // diagnostics need the user-facing name even when the
    // Library itself is only addressable by its OS handle. For us
    // the two should match, but we accept either.
    let key = if reg.contains_key(library_name) {
        library_name
    } else {
        library.name.as_str()
    };
    reg.get(key)
        .and_then(|syms| syms.get(symbol).copied())
        .ok_or_else(|| ForeignLoadError::SymbolNotFound {
            library: library_name.to_string(),
            symbol: symbol.to_string(),
        })
}

pub fn default_native_lib_filename(raw: &str) -> String {
    // No host-specific extension on wasm — same string back so any
    // diagnostics that thread through this code path stay readable.
    raw.to_string()
}

pub fn base_name_of_signature(sig: &str) -> &str {
    // Match the host implementation: strip a trailing `(_,_,...)`
    // arity suffix, keep the bare base name.
    sig.split_once('(').map(|(base, _)| base).unwrap_or(sig)
}

pub fn dispatch_foreign_c(vm: &mut VM, func: ForeignCFn, args: &[Value]) -> Value {
    // Mirror the host build's dispatch shape: stage args into the
    // VM's slot stack so the foreign function (which is written
    // against the slot-based `NativeContext` API) finds them at
    // slot 0..args.len(), invoke the function, then hand back
    // whatever it stashed in slot 0 via `set_return`.
    let vm_ptr = vm as *mut VM;
    vm.api_stack.clear();
    vm.api_stack.extend_from_slice(args);
    func(vm_ptr);
    // The function writes its result via `set_return`, which goes
    // into slot 0. If it called `runtime_error` instead, slot 0
    // is null but `vm.has_error` is set — the BC interpreter
    // checks that on the call site.
    vm.api_stack.first().copied().unwrap_or_else(Value::null)
}
