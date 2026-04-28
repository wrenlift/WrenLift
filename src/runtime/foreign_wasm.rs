//! Wasm-side stub for the foreign-library loader.
//!
//! Plugin dlopen / dlsym makes no sense on `wasm32-unknown-unknown`
//! — there's no shared-library loader and every native plugin is a
//! `cdylib` built for a host triple. This stub keeps the same
//! external API as `foreign.rs` so the rest of the runtime (vm.rs,
//! vm_interp.rs, engine.rs) can stay platform-agnostic; every load
//! attempt fails with `LibraryNotFound`, which the call sites
//! already surface as a runtime error to Wren code.
//!
//! When wasm builds gain a `core::browser` module that exposes
//! Web APIs as foreign classes, this file is the place to grow
//! a "register a wasm-bindgen-backed native fn under symbol X"
//! registry — same shape, different backing store.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::runtime::object::ForeignCFn;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

pub const WLIFT_PLUGIN_ABI_VERSION: u32 = 1;
pub const WLIFT_PLUGIN_ABI_SYMBOL: &str = "wlift_plugin_abi_version";

/// Opaque handle that takes the place of `libloading::Library`. The
/// host build owns a real dylib here; the wasm build owns nothing
/// and exists only so `Vec<Library>` field types in `VM` resolve.
pub struct Library;

#[derive(Debug)]
pub enum ForeignLoadError {
    LibraryNotFound { name: String, tried: Vec<String> },
    SymbolNotFound { library: String, symbol: String },
    AbiMismatch { library: String, expected: u32, found: u32 },
    Unsupported,
}

impl std::fmt::Display for ForeignLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForeignLoadError::LibraryNotFound { name, .. } => {
                write!(f, "native library '{}' is not available in the wasm build.", name)
            }
            ForeignLoadError::SymbolNotFound { library, symbol } => {
                write!(f, "symbol '{}' not found in '{}' (wasm build).", symbol, library)
            }
            ForeignLoadError::AbiMismatch { library, expected, found } => {
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

pub fn library_candidates(name: &str) -> Vec<String> {
    vec![name.to_string()]
}

pub fn verify_plugin_abi(_library: &Library, _name: &str) -> Result<(), ForeignLoadError> {
    Err(ForeignLoadError::Unsupported)
}

pub fn load_library(
    name: &str,
    _search_paths: &[PathBuf],
    _name_overrides: &HashMap<String, PathBuf>,
) -> Result<Library, ForeignLoadError> {
    Err(ForeignLoadError::LibraryNotFound {
        name: name.to_string(),
        tried: vec!["<wasm: no native loader>".to_string()],
    })
}

pub fn resolve_symbol(
    _library: &Library,
    library_name: &str,
    symbol: &str,
) -> Result<ForeignCFn, ForeignLoadError> {
    Err(ForeignLoadError::SymbolNotFound {
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

pub fn dispatch_foreign_c(vm: &mut VM, _func: ForeignCFn, _args: &[Value]) -> Value {
    // No symbol could have been resolved, but if one is somehow
    // dispatched, surface a runtime error rather than calling a
    // null pointer.
    use crate::runtime::object::NativeContext;
    vm.runtime_error("foreign methods are not supported in the wasm build.".to_string());
    Value::null()
}
