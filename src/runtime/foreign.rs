//! Native-library binding for Wren `foreign` classes/methods.
//!
//! Wires compile-time `#!native = "libname"` (on a foreign class) and
//! `#!symbol = "fn"` (on a foreign method) to `dlopen` / `dlsym` via
//! `libloading`. Resolved symbols are cast to the standard Wren
//! embedding ABI `extern "C" fn(vm: *mut VM)` and dispatched through
//! the existing slot-based C API.
//!
//! Loaded libraries are owned by the `VM` in `native_libs` so their
//! symbols stay valid for the VM's lifetime; dropping the VM unloads
//! everything in one pass.

use std::path::PathBuf;

use libloading::Library;

use crate::runtime::object::ForeignCFn;
use crate::runtime::value::Value;
use crate::runtime::vm::VM;

/// Errors surfaced by the foreign loader. Emitted as runtime errors at
/// class install time so the failure is visible to Wren code rather
/// than aborting the process.
#[derive(Debug)]
pub enum ForeignLoadError {
    LibraryNotFound { name: String, tried: Vec<String> },
    SymbolNotFound { library: String, symbol: String },
}

impl std::fmt::Display for ForeignLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForeignLoadError::LibraryNotFound { name, tried } => {
                write!(
                    f,
                    "could not load native library '{}' (tried: {})",
                    name,
                    tried.join(", ")
                )
            }
            ForeignLoadError::SymbolNotFound { library, symbol } => {
                write!(f, "symbol '{}' not found in '{}'", symbol, library)
            }
        }
    }
}

/// Expand a bare library name into platform-specific candidates the OS
/// loader is likely to find. A bare `"sqlite3"` becomes:
///
/// * macOS: `libsqlite3.dylib`, `sqlite3.dylib`, `libsqlite3`
/// * Linux: `libsqlite3.so`, `sqlite3.so`, `libsqlite3`
/// * Windows: `sqlite3.dll`, `libsqlite3.dll`
///
/// An input containing `/`, `\`, or a `.dylib` / `.so` / `.dll` suffix
/// is treated as an explicit path and returned unchanged so callers can
/// pass absolute paths verbatim (for explicit system locations).
pub fn library_candidates(name: &str) -> Vec<String> {
    let looks_explicit = name.contains('/')
        || name.contains('\\')
        || name.ends_with(".dylib")
        || name.ends_with(".so")
        || name.ends_with(".dll");
    if looks_explicit {
        return vec![name.to_string()];
    }

    #[cfg(target_os = "macos")]
    {
        vec![
            format!("lib{}.dylib", name),
            format!("{}.dylib", name),
            format!("lib{}", name),
        ]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        vec![
            format!("lib{}.so", name),
            format!("{}.so", name),
            format!("lib{}", name),
        ]
    }
    #[cfg(windows)]
    {
        vec![format!("{}.dll", name), format!("lib{}.dll", name)]
    }
}

/// Open a library, trying each candidate filename produced by
/// [`library_candidates`]. If `search_paths` is non-empty, each
/// candidate is also tried prefixed with every search path before
/// falling back to the OS loader's default search (which already
/// honors `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, `/usr/lib`,
/// `/opt/homebrew/lib`, etc.).
///
/// The name `"self"` is a sentinel that resolves to the current
/// process image (equivalent to `dlopen(NULL)`). Useful for tests and
/// for wiring foreign methods against symbols linked into the host
/// executable.
pub fn load_library(name: &str, search_paths: &[PathBuf]) -> Result<Library, ForeignLoadError> {
    if name == "self" {
        // Open the current process image so exported symbols linked
        // into the host executable are discoverable via `dlsym`. This
        // is `dlopen(NULL, ...)` on unix and `GetModuleHandle(NULL)` on
        // Windows; libloading exposes it behind a platform-specific
        // constructor.
        #[cfg(unix)]
        {
            let inner = libloading::os::unix::Library::this();
            return Ok(Library::from(inner));
        }
        #[cfg(windows)]
        {
            return unsafe { libloading::os::windows::Library::this() }
                .map(Library::from)
                .map_err(|_| ForeignLoadError::LibraryNotFound {
                    name: name.to_string(),
                    tried: vec!["<process image>".to_string()],
                });
        }
    }

    let candidates = library_candidates(name);
    let mut tried = Vec::new();

    // Try each search path first (explicit locations win over ambient ones).
    for dir in search_paths {
        for cand in &candidates {
            let full = dir.join(cand);
            tried.push(full.display().to_string());
            if let Ok(lib) = unsafe { Library::new(&full) } {
                return Ok(lib);
            }
        }
    }

    // Then let the OS loader resolve bare names from its own search
    // paths. On unix this means the dyld / ld.so defaults.
    for cand in &candidates {
        tried.push(cand.clone());
        if let Ok(lib) = unsafe { Library::new(cand) } {
            return Ok(lib);
        }
    }

    Err(ForeignLoadError::LibraryNotFound {
        name: name.to_string(),
        tried,
    })
}

/// Look up `symbol` inside `library` and cast it to the Wren foreign
/// method ABI. The returned function pointer is tied to the library's
/// lifetime — the VM must outlive the caller's use of the pointer.
pub fn resolve_symbol(
    library: &Library,
    library_name: &str,
    symbol: &str,
) -> Result<ForeignCFn, ForeignLoadError> {
    unsafe {
        let sym: libloading::Symbol<ForeignCFn> =
            library.get(symbol.as_bytes()).map_err(|_| {
                ForeignLoadError::SymbolNotFound {
                    library: library_name.to_string(),
                    symbol: symbol.to_string(),
                }
            })?;
        Ok(*sym)
    }
}

/// Base name of a Wren method signature (e.g. `"open(_)"` → `"open"`).
/// Used when a `foreign` method omits `#!symbol` — we default to the
/// Wren method name.
pub fn base_name_of_signature(sig: &str) -> &str {
    sig.split_once(['(', '[', '='])
        .map(|(head, _)| head)
        .unwrap_or(sig)
        .trim_start_matches("static ")
}

/// Dispatch helper invoked at every `Method::ForeignC` call site.
/// Copies `args` (receiver + method arguments) into the VM's
/// `api_stack` so the standard Wren C API (`wrenGetSlotDouble`,
/// `wrenSetSlotString`, …) sees them at the expected indices, invokes
/// the extern fn, and returns whatever the fn left in slot 0.
///
/// Slot convention (Wren standard):
///
/// * slot 0 — receiver on entry, return value on exit
/// * slot 1..=argc — method arguments
#[inline]
pub fn dispatch_foreign_c(vm: &mut VM, func: ForeignCFn, args: &[Value]) -> Value {
    if vm.api_stack.len() < args.len() {
        vm.api_stack.resize(args.len(), Value::null());
    }
    for (i, &arg) in args.iter().enumerate() {
        vm.api_stack[i] = arg;
    }
    func(vm as *mut VM);
    vm.api_stack.first().copied().unwrap_or(Value::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::vm::VM;

    #[test]
    fn base_name_strips_signature_suffix() {
        assert_eq!(base_name_of_signature("open(_)"), "open");
        assert_eq!(base_name_of_signature("close()"), "close");
        assert_eq!(base_name_of_signature("bar"), "bar");
        assert_eq!(base_name_of_signature("[_]"), "");
        assert_eq!(base_name_of_signature("name=(_)"), "name");
    }

    #[test]
    fn candidates_handle_bare_and_explicit_names() {
        let bare = library_candidates("sqlite3");
        assert!(!bare.is_empty());
        // Bare names expand to platform-specific filenames.
        assert!(bare.iter().any(|c| c.contains("sqlite3")));

        // Absolute paths pass through unchanged.
        let explicit = library_candidates("/opt/homebrew/lib/libsqlite3.dylib");
        assert_eq!(explicit, vec!["/opt/homebrew/lib/libsqlite3.dylib"]);

        // Suffixed names pass through unchanged.
        let suffixed = library_candidates("mylib.so");
        assert_eq!(suffixed, vec!["mylib.so"]);
    }

    #[test]
    fn load_library_missing_returns_error() {
        let result = load_library("__wrenlift_does_not_exist__", &[]);
        assert!(matches!(result, Err(ForeignLoadError::LibraryNotFound { .. })));
    }

    extern "C" fn test_doubler(vm: *mut VM) {
        unsafe {
            let slots = &mut (*vm).api_stack;
            let n = slots[1].as_num().unwrap_or(0.0);
            slots[0] = Value::num(n * 2.0);
        }
    }

    #[test]
    fn dispatch_foreign_c_round_trips_slots() {
        let mut vm = VM::new_default();
        let args = [Value::num(0.0), Value::num(21.0)];
        let result = dispatch_foreign_c(&mut vm, test_doubler, &args);
        assert_eq!(result.as_num(), Some(42.0));
    }

    #[test]
    fn dispatch_foreign_c_grows_api_stack_as_needed() {
        // Starting with an undersized api_stack, the bridge must grow
        // it to fit receiver + args before invoking the extern fn.
        let mut vm = VM::new_default();
        vm.api_stack.clear();
        let args = [Value::num(0.0), Value::num(7.0)];
        let result = dispatch_foreign_c(&mut vm, test_doubler, &args);
        assert_eq!(result.as_num(), Some(14.0));
        assert!(vm.api_stack.len() >= args.len());
    }
}
