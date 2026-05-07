#![allow(clippy::not_unsafe_ptr_arg_deref)]
/// C embedding API for WrenLift.
///
/// Provides a wren.h-compatible interface for embedding WrenLift in C/C++
/// applications. All functions use `extern "C"` linkage and `#[no_mangle]`
/// for direct FFI consumption.
///
/// # Usage from C
///
/// ```c
/// #include "wrenlift.h"
///
/// static void write_fn(WrenVM* vm, const char* text) {
///     printf("%s", text);
/// }
///
/// int main() {
///     WrenConfiguration config;
///     wrenInitConfiguration(&config);
///     config.writeFn = write_fn;
///
///     WrenVM* vm = wrenNewVM(&config);
///     wrenInterpret(vm, "main", "System.print(\"Hello!\")");
///     wrenFreeVM(vm);
/// }
/// ```
use std::ffi::{c_char, c_double, c_int, c_void, CStr, CString};
use std::ptr;

use crate::runtime::object::{
    NativeContext, ObjHeader, ObjList, ObjMap, ObjString, ObjType, ObjTypedArray,
};
use crate::runtime::value::Value;
use crate::runtime::vm::{VMConfig, VM};

// ---------------------------------------------------------------------------
// Opaque types
// ---------------------------------------------------------------------------

/// Opaque VM handle for C consumers.
pub type WrenVM = VM;

/// A persistent handle to a Wren value (prevents GC collection).
#[repr(C)]
pub struct WrenHandle {
    index: usize,
    /// Method signature for call handles (empty for value handles).
    signature: String,
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrenErrorType {
    CompileError = 0,
    RuntimeError = 1,
    StackTrace = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrenInterpretResult {
    Success = 0,
    CompileError = 1,
    RuntimeError = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrenType {
    Bool = 0,
    Num = 1,
    Foreign = 2,
    List = 3,
    Map = 4,
    Null = 5,
    String = 6,
    Unknown = 7,
}

// ---------------------------------------------------------------------------
// Callback types (C function pointers)
// ---------------------------------------------------------------------------

pub type WrenWriteFn = extern "C" fn(vm: *mut WrenVM, text: *const c_char);
pub type WrenErrorFn = extern "C" fn(
    vm: *mut WrenVM,
    error_type: WrenErrorType,
    module: *const c_char,
    line: c_int,
    message: *const c_char,
);
pub type WrenForeignMethodFn = extern "C" fn(vm: *mut WrenVM);
pub type WrenFinalizerFn = extern "C" fn(data: *mut c_void);
pub type WrenResolveModuleFn =
    extern "C" fn(vm: *mut WrenVM, importer: *const c_char, name: *const c_char) -> *const c_char;
pub type WrenLoadModuleFn =
    extern "C" fn(vm: *mut WrenVM, name: *const c_char) -> WrenLoadModuleResult;
pub type WrenBindForeignMethodFn = extern "C" fn(
    vm: *mut WrenVM,
    module: *const c_char,
    class_name: *const c_char,
    is_static: bool,
    signature: *const c_char,
) -> Option<WrenForeignMethodFn>;
pub type WrenBindForeignClassFn = extern "C" fn(
    vm: *mut WrenVM,
    module: *const c_char,
    class_name: *const c_char,
) -> WrenForeignClassMethods;

// ---------------------------------------------------------------------------
// Configuration and result structs
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct WrenLoadModuleResult {
    pub source: *const c_char,
    pub on_complete: Option<extern "C" fn(*mut WrenVM, *const c_char, WrenLoadModuleResult)>,
    pub user_data: *mut c_void,
}

#[repr(C)]
pub struct WrenForeignClassMethods {
    pub allocate: Option<WrenForeignMethodFn>,
    pub finalize: Option<WrenFinalizerFn>,
}

#[repr(C)]
pub struct WrenConfiguration {
    pub reallocate_fn: *mut c_void, // Not used, kept for ABI compat
    pub resolve_module_fn: Option<WrenResolveModuleFn>,
    pub load_module_fn: Option<WrenLoadModuleFn>,
    pub bind_foreign_method_fn: Option<WrenBindForeignMethodFn>,
    pub bind_foreign_class_fn: Option<WrenBindForeignClassFn>,
    pub write_fn: Option<WrenWriteFn>,
    pub error_fn: Option<WrenErrorFn>,
    pub initial_heap_size: usize,
    pub min_heap_size: usize,
    pub heap_growth_percent: c_int,
    pub user_data: *mut c_void,
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

pub const WREN_VERSION_MAJOR: c_int = 0;
pub const WREN_VERSION_MINOR: c_int = 5;
pub const WREN_VERSION_PATCH: c_int = 0;

#[no_mangle]
pub extern "C" fn wrenGetVersionNumber() -> c_int {
    WREN_VERSION_MAJOR * 1_000_000 + WREN_VERSION_MINOR * 1_000 + WREN_VERSION_PATCH
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenInitConfiguration(config: *mut WrenConfiguration) {
    if config.is_null() {
        return;
    }
    unsafe {
        *config = WrenConfiguration {
            reallocate_fn: ptr::null_mut(),
            resolve_module_fn: None,
            load_module_fn: None,
            bind_foreign_method_fn: None,
            bind_foreign_class_fn: None,
            write_fn: None,
            error_fn: None,
            initial_heap_size: 10 * 1024 * 1024,
            min_heap_size: 1024 * 1024,
            heap_growth_percent: 50,
            user_data: ptr::null_mut(),
        };
    }
}

// ---------------------------------------------------------------------------
// VM lifecycle
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenNewVM(config: *const WrenConfiguration) -> *mut WrenVM {
    let vm_config = if config.is_null() {
        VMConfig::default()
    } else {
        unsafe { build_vm_config(&*config) }
    };

    let vm = Box::new(VM::new(vm_config));
    Box::into_raw(vm)
}

#[no_mangle]
pub extern "C" fn wrenFreeVM(vm: *mut WrenVM) {
    if !vm.is_null() {
        unsafe {
            drop(Box::from_raw(vm));
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenCollectGarbage(vm: *mut WrenVM) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).collect_garbage();
    }
}

// ---------------------------------------------------------------------------
// Interpret
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenInterpret(
    vm: *mut WrenVM,
    module: *const c_char,
    source: *const c_char,
) -> WrenInterpretResult {
    if vm.is_null() || module.is_null() || source.is_null() {
        return WrenInterpretResult::RuntimeError;
    }
    let module_name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("main");
    let source_str = unsafe { CStr::from_ptr(source) }.to_str().unwrap_or("");
    let vm_ref = unsafe { &mut *vm };
    use crate::runtime::engine::InterpretResult;
    match vm_ref.interpret(module_name, source_str) {
        InterpretResult::Success => WrenInterpretResult::Success,
        InterpretResult::CompileError => WrenInterpretResult::CompileError,
        InterpretResult::RuntimeError => WrenInterpretResult::RuntimeError,
    }
}

// ---------------------------------------------------------------------------
// AOT bootstrap helper
// ---------------------------------------------------------------------------

/// Single-call entry used by the AOT bootstrap shim.
///
/// `wlift --aot` generates a tiny C `main()` that embeds every
/// reachable module's source (in dependency-first order, entry
/// last) and hands the bundle to this function. The runtime
/// installs each dependency via `vm.interpret(name, source)` —
/// which mirrors how the JIT path loads imports from disk — and
/// then runs the entry. Modules are installed under the exact
/// `import "..."` path the source uses, so the entry's import
/// resolves by name match without a custom resolver.
///
/// Arrays are parallel: `extras_names[i]` is the module name
/// `extras_sources[i]` registers under. Each pointer must be a
/// valid null-terminated UTF-8 C string. `extras_count == 0`
/// degenerates to "interpret entry only" — same shape the
/// pre-bundle bootstrap had.
///
/// Returns a Unix-style exit code: `0` on success, `65` on
/// compile error, `70` on runtime error.
///
/// # Safety
///
/// Every non-null pointer must point at a null-terminated UTF-8
/// C string that lives for the duration of the call. The two
/// `extras_*` arrays must each contain at least `extras_count`
/// pointer-sized elements. Null pointers are tolerated (entry
/// nulls map to runtime/compile error; extras nulls return
/// compile error); other invalid pointers produce undefined
/// behaviour. Generated by `wlift --aot`'s bootstrap, where the
/// arrays are static `const char*` tables with C-string-literal
/// contents — those contracts hold by construction.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_run_aot_program(
    entry_name: *const c_char,
    entry_source: *const c_char,
    extras_count: usize,
    extras_names: *const *const c_char,
    extras_sources: *const *const c_char,
) -> c_int {
    if entry_source.is_null() {
        return 70;
    }
    let entry_n = if entry_name.is_null() {
        "main"
    } else {
        CStr::from_ptr(entry_name).to_str().unwrap_or("main")
    };
    let entry_s = match CStr::from_ptr(entry_source).to_str() {
        Ok(s) => s,
        Err(_) => return 65,
    };

    use crate::runtime::engine::InterpretResult;
    let config = VMConfig::default();
    let mut vm = VM::new(config);

    // Install every bundled dependency before running the entry.
    // Dependency-first walker order keeps each module's own
    // imports satisfied by the time it gets installed.
    if extras_count > 0 && !extras_names.is_null() && !extras_sources.is_null() {
        for i in 0..extras_count {
            let name_ptr = *extras_names.add(i);
            let src_ptr = *extras_sources.add(i);
            if name_ptr.is_null() || src_ptr.is_null() {
                return 65;
            }
            let name = match CStr::from_ptr(name_ptr).to_str() {
                Ok(s) => s,
                Err(_) => return 65,
            };
            let src = match CStr::from_ptr(src_ptr).to_str() {
                Ok(s) => s,
                Err(_) => return 65,
            };
            match vm.interpret(name, src) {
                InterpretResult::Success => {}
                InterpretResult::CompileError => return 65,
                InterpretResult::RuntimeError => return 70,
            }
        }
    }

    match vm.interpret(entry_n, entry_s) {
        InterpretResult::Success => 0,
        InterpretResult::CompileError => 65,
        InterpretResult::RuntimeError => 70,
    }
}

// ---------------------------------------------------------------------------
// AOT runtime entries — the bootstrap shim wires these together to
// drive an AOT-compiled program directly, no `vm.interpret` round-trip.
// ---------------------------------------------------------------------------

/// Populate the first slots of a per-module `wlift_modvars_<n>`
/// array with the VM's core class values, in the same order
/// `sema::PRELUDE_NAMES` lists. Each module's MIR was compiled
/// against that list, so the slot-to-class mapping is stable.
///
/// `modvars_count` is the slot capacity from the AOT codegen
/// (`AotModule::module_var_count`); the function fills up to
/// `min(prelude.len(), modvars_count)` slots and leaves the rest
/// untouched (top-level body initialises user vars itself).
///
/// # Safety
///
/// `modvars` must point at a writable `[u64; modvars_count]` array
/// living for at least the lifetime of `vm`. The bootstrap embeds
/// the symbol via `extern void* wlift_modvars_<n>[]` so this
/// contract holds by construction.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_init_prelude(
    vm: *mut WrenVM,
    modvars: *mut u64,
    modvars_count: usize,
) -> c_int {
    if vm.is_null() || modvars.is_null() {
        return 70;
    }
    let vm_ref = unsafe { &*vm };
    for (i, name) in crate::sema::PRELUDE_NAMES.iter().enumerate() {
        if i >= modvars_count {
            break;
        }
        if let Some(value) = vm_ref.core_class_value(name) {
            unsafe {
                *modvars.add(i) = value.to_bits();
            }
        }
    }
    0
}

/// Allocate a fresh `ObjString` carrying the given UTF-8 byte
/// range and return its NaN-boxed `Value` representation. The
/// bootstrap calls this once per string constant during init,
/// stashing the result in a `wlift_consts_<n>[i]` slot for the
/// AOT body to load via direct addressing — no per-use-site
/// helper call.
///
/// Returns `Value::null()` on null inputs; runtime errors during
/// allocation surface as panics from inside the GC, matching the
/// existing `vm.new_string` contract.
///
/// # Safety
///
/// `text` must be a valid pointer to `len` UTF-8 bytes. The
/// bootstrap embeds these via `static const char[]` literals so
/// the constraint holds by construction.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_alloc_const_string(
    vm: *mut WrenVM,
    text: *const c_char,
    len: usize,
) -> u64 {
    if vm.is_null() || text.is_null() {
        return crate::runtime::value::Value::null().to_bits();
    }
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, len) };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => return crate::runtime::value::Value::null().to_bits(),
    };
    let vm_ref = unsafe { &mut *vm };
    vm_ref.new_string(s).to_bits()
}

/// Set the per-thread `JitContext` to point at this module's
/// `vm` + `modvars` + `name` before calling an AOT-emitted body.
/// Returns the previous context bytes packed into an opaque
/// `[u64; 6]` that `wlift_aot_exit` restores — same save/restore
/// shape the JIT uses when one helper recurses into another.
///
/// The runtime helpers that survived the AOT lowering rewrite
/// (`wren_call_*`, `wren_make_*`, `wren_write_barrier`, etc.)
/// consult this context via `with_context` — without setting it
/// up first, they'd see a null `vm` and short-circuit to a no-op
/// or null return, manifesting as silent failures inside AOT'd
/// code paths.
///
/// # Safety
///
/// All pointers must remain valid for the duration of the AOT
/// body's execution. `name` is a UTF-8 C string; `out_saved`
/// must point at a writable 48-byte buffer.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_enter(
    vm: *mut WrenVM,
    modvars: *mut u64,
    modvars_count: usize,
    name: *const c_char,
    name_len: usize,
    out_saved: *mut u64,
) {
    use crate::codegen::runtime_fns::{read_jit_ctx, set_jit_context, JitContext};

    if !out_saved.is_null() {
        // Stash the current context — same shape used by
        // `restore_rooted_jit_context` so re-entrance works if
        // an AOT body calls another AOT body via runtime
        // dispatch.
        let prev = read_jit_ctx();
        let ptr = out_saved as *mut JitContext;
        unsafe {
            *ptr = prev;
        }
    }

    let new_ctx = JitContext {
        module_vars: modvars,
        module_var_count: modvars_count as u32,
        vm: vm as *mut u8,
        module_name: name as *const u8,
        module_name_len: name_len as u32,
        current_func_id: u32::MAX as u64,
        closure: std::ptr::null_mut(),
        defining_class: std::ptr::null_mut(),
        jit_code_base: std::ptr::null(),
        jit_code_len: 0,
    };
    set_jit_context(new_ctx);
    // Flip on the helper-driven GC trigger. AOT-emitted bodies
    // never re-enter the bytecode interpreter loop where the
    // standard safepoint runs, so without this every alloc helper
    // is a no-op for GC and the heap grows unboundedly. JIT /
    // interpreter modes leave the flag at its default `false` so
    // they don't fire GC at points Cranelift's stack-map metadata
    // doesn't cover.
    crate::codegen::runtime_fns::set_aot_gc_enabled(
        std::env::var_os("WLIFT_AOT_GC").is_some(),
    );
}

/// One method's worth of class-install descriptor: `(sig, fn_ptr,
/// arity, flags)`. Flags bit-0 = `is_static`, bit-1 = `is_constructor`,
/// bit-2 = `is_state_machine` (method body uses Fiber.yield
/// transitively, lowered with `(fiber, resume_v) -> i64`
/// signature). The bootstrap emits one of these per AOT-lowered
/// method as a `Linkage::Local` `Data` blob, then hands the
/// pointer + count to `wlift_aot_install_class`.
#[cfg(feature = "aot")]
#[repr(C)]
pub struct WliftAotMethodDesc {
    pub sig: *const c_char,
    pub sig_len: usize,
    pub fn_ptr: *const u8,
    pub arity: u8,
    pub flags: u8,
}

/// Install a user-defined class in the VM's class table at startup,
/// mirroring the path the JIT install loop takes inside
/// `install_module_mir_and_run_with_sources`. Each AOT-compiled
/// method is registered as a closure pointing at a stub MIR whose
/// `jit_code[fn_id]` is patched to the AOT'd body — that way the
/// existing JIT-dispatch fast path drives the call with no extra
/// runtime branching, and dispatch through `wren_call_*` walks
/// the same `class.methods[sym]` table the interpreter uses.
///
/// Stores the resulting `*mut ObjClass` in `modvars[slot]` so the
/// AOT body's `GetModuleVar` reads it directly. Resolves
/// `parent_name` first against the VM's core classes (Object,
/// Num, …) and then against `modvars[*]` for already-installed
/// user classes — the bootstrap drives modules in dependency
/// order so a parent is installed before its subclass.
///
/// # Safety
///
/// All pointers must be valid for the duration of the call;
/// `methods` must point at `methods_count` consecutive
/// `WliftAotMethodDesc` records. The bootstrap satisfies these
/// via `Linkage::Local` `Data` blobs sized + initialised at AOT
/// build time.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_install_class(
    vm: *mut WrenVM,
    modvars: *mut u64,
    slot: usize,
    name: *const c_char,
    name_len: usize,
    parent_slot: usize,
    num_fields: u16,
    methods: *const WliftAotMethodDesc,
    methods_count: usize,
) -> c_int {
    use crate::runtime::object::{Method, ObjHeader, ObjType};

    if vm.is_null() || modvars.is_null() || name.is_null() {
        return 70;
    }
    let name_bytes = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
    let class_name = match std::str::from_utf8(name_bytes) {
        Ok(s) => s,
        Err(_) => return 65,
    };

    let vm_ref = unsafe { &mut *vm };

    // Resolve superclass via this module's modvars — every class
    // the resolver knew about (core prelude, imported user class,
    // same-module class defined earlier) has a slot, populated by
    // `init_prelude` / the import-copy step / a prior install
    // call. `usize::MAX` means "no superclass declared" → defaults
    // to `Object`, matching Wren's default class hierarchy.
    let mut superclass: *mut crate::runtime::object::ObjClass = std::ptr::null_mut();
    if parent_slot != usize::MAX {
        let parent_bits = unsafe { *modvars.add(parent_slot) };
        let parent_val = crate::runtime::value::Value::from_bits(parent_bits);
        if let Some(p) = parent_val.as_object() {
            let header = p as *const ObjHeader;
            if unsafe { (*header).obj_type } == ObjType::Class {
                superclass = p as *mut crate::runtime::object::ObjClass;
            }
        }
    }
    if superclass.is_null() {
        superclass = vm_ref.object_class;
    }

    let class_name_sym = vm_ref.interner.intern(class_name);
    let class_ptr = vm_ref.gc.alloc_class(class_name_sym, superclass);
    unsafe {
        (*class_ptr).header.class = vm_ref.class_class;
        let inherited = if !superclass.is_null() {
            (*superclass).num_fields
        } else {
            0
        };
        (*class_ptr).num_fields = num_fields + inherited;
    }

    let module_rc = std::rc::Rc::new(String::new());
    for i in 0..methods_count {
        let desc = unsafe { &*methods.add(i) };
        if desc.sig.is_null() || desc.fn_ptr.is_null() {
            continue;
        }
        let sig_bytes = unsafe { std::slice::from_raw_parts(desc.sig as *const u8, desc.sig_len) };
        let sig = match std::str::from_utf8(sig_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let is_static = desc.flags & 1 != 0;
        let is_constructor = desc.flags & 2 != 0;
        let is_state_machine = desc.flags & 4 != 0;

        let sig_sym = vm_ref.interner.intern(sig);
        let bind_sym = if is_static || is_constructor {
            let static_sig = format!("static:{}", sig);
            vm_ref.interner.intern(&static_sig)
        } else {
            sig_sym
        };

        // Register the AOT body as a stub-MIR function with its
        // jit_code slot pre-patched to the native pointer. The
        // closure-dispatch path picks it up like any JIT'd fn.
        let func_id = if is_state_machine {
            vm_ref.engine.register_aot_state_machine_function(
                sig_sym,
                desc.fn_ptr,
                Some(std::rc::Rc::clone(&module_rc)),
            )
        } else {
            vm_ref.engine.register_aot_function(
                sig_sym,
                desc.arity,
                desc.fn_ptr,
                Some(std::rc::Rc::clone(&module_rc)),
            )
        };

        let fn_obj = vm_ref.gc.alloc_fn(sig_sym, 0, 0, func_id.0);
        unsafe {
            (*fn_obj).header.class = vm_ref.fn_class;
        }
        let closure_ptr = vm_ref.gc.alloc_closure(fn_obj);
        unsafe {
            (*closure_ptr).header.class = vm_ref.fn_class;
        }

        unsafe {
            let method = if is_constructor {
                Method::Constructor(closure_ptr)
            } else {
                Method::Closure(closure_ptr)
            };
            let idx = bind_sym.index() as usize;
            let cls = &mut *class_ptr;
            if idx >= cls.methods.len() {
                cls.methods.resize(idx + 1, None);
            }
            cls.methods[idx] = Some(method);
        }
    }

    // Sanity: the GC-allocated class must be a real ObjClass before
    // we hand its pointer back through modvars (catches a botched
    // alloc — observable as a SIGSEGV otherwise).
    let header = class_ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Class {
        return 70;
    }

    let class_value = crate::runtime::value::Value::object(class_ptr as *mut u8);
    unsafe {
        *modvars.add(slot) = class_value.to_bits();
    }
    0
}

/// Register an AOT-compiled closure body with the engine and
/// return its assigned `FuncId` (zero-extended to `u64`). The
/// AOT bootstrap calls this once per closure at startup,
/// stashing the result in `wlift_closures_<n>[i]`. AOT-lowered
/// `MakeClosure` then reads the slot to pass a valid runtime
/// FuncId to `wren_make_closure_*`.
///
/// # Safety
///
/// `fn_ptr` must be a valid AOT-compiled closure body matching
/// the standard `(receiver: u64, ...args: u64) -> u64` calling
/// convention. The bootstrap takes the pointer from a
/// `Linkage::Import` Cranelift FuncId, so the contract holds by
/// construction.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_register_closure(
    vm: *mut WrenVM,
    arity: u8,
    fn_ptr: *const u8,
) -> u64 {
    if vm.is_null() || fn_ptr.is_null() {
        return u64::MAX;
    }
    let vm_ref = unsafe { &mut *vm };
    // Closure bodies don't need a meaningful name in the engine —
    // they're never looked up by name, only by FuncId. Reuse the
    // interner's "<aot-closure>" sentinel.
    let name_sym = vm_ref.interner.intern("<aot-closure>");
    let func_id = vm_ref
        .engine
        .register_aot_function(name_sym, arity, fn_ptr, None);
    func_id.0 as u64
}

/// State-machine variant of [`wlift_aot_register_closure`]. The
/// bootstrap calls this for closures the AOT pipeline lowered
/// with the stackless `(fiber, resume_v) -> i64` shape; the
/// runtime flips `engine.aot_state_machine[id] = true` so
/// `fiber.call`/`Fiber.yield` dispatch picks the poll path
/// instead of `wren_call_*`.
///
/// # Safety
///
/// `fn_ptr` must match the state-machine signature.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_register_state_machine_closure(
    vm: *mut WrenVM,
    fn_ptr: *const u8,
) -> u64 {
    if vm.is_null() || fn_ptr.is_null() {
        return u64::MAX;
    }
    let vm_ref = unsafe { &mut *vm };
    let name_sym = vm_ref.interner.intern("<aot-sm-closure>");
    let func_id = vm_ref
        .engine
        .register_aot_state_machine_function(name_sym, fn_ptr, None);
    func_id.0 as u64
}

/// Per-safepoint descriptor for `wlift_aot_register_code_range`.
/// The bootstrap emits one of these per Cranelift user stack map
/// captured during AOT lowering. `roots_start` + `roots_count`
/// index into the per-function flat root-offsets array.
#[cfg(feature = "aot")]
#[repr(C)]
pub struct WliftAotSafepointDesc {
    /// Bytes from the function's entry to the return address
    /// after the call instruction this safepoint covers. Matches
    /// `(saved_lr - code_range.start)` at GC time.
    pub code_offset: u32,
    /// Index of the first root spill offset in the per-function
    /// roots blob.
    pub roots_start: u32,
    /// Number of consecutive `i32` root spill offsets at
    /// `roots[roots_start..roots_start+roots_count]`.
    pub roots_count: u32,
}

/// Register an AOT-compiled function's code range + safepoint
/// metadata with the engine so the GC stack walker can scan its
/// frames during a collection cycle. AOT bodies are entered
/// directly from the linker's `main()` (no per-call helper to
/// push a `jit_frame_entry`), so without this registration the
/// stack walker has no way to find them and any GC fired from
/// inside an alloc helper sweeps live spill slots.
///
/// Symmetric to the JIT path's `register_code_range` call inside
/// `engine::install_compiled_function`. The runtime synthesises a
/// fresh `FuncId` (we don't need the same id as
/// `register_aot_function` — the GC walker keys metadata off the
/// `code_range`'s `func_id` field, which we control here).
///
/// # Safety
///
/// `fn_ptr` must point at the function's entry; `code_size` must
/// match what Cranelift emitted (`CompiledCode::code_info().total_size`).
/// `safepoints` and `roots` must be valid `[WliftAotSafepointDesc]`
/// and `[i32]` arrays of the given lengths; the bootstrap satisfies
/// these by emitting them as `Linkage::Local` `Data` blobs.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_register_code_range(
    vm: *mut WrenVM,
    fn_ptr: *const u8,
    code_size: u32,
    safepoints: *const WliftAotSafepointDesc,
    safepoints_count: u32,
    roots: *const i32,
) -> c_int {
    use crate::codegen::native_meta::{
        LiveRootMetadata, NativeFrameMetadata, RootLocation, SafepointKind, SafepointMetadata,
    };
    use std::sync::Arc;

    if vm.is_null() || fn_ptr.is_null() {
        return 70;
    }
    let vm_ref = unsafe { &mut *vm };

    let sp_slice: &[WliftAotSafepointDesc] = if safepoints.is_null() || safepoints_count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(safepoints, safepoints_count as usize) }
    };

    let mut safepoints_meta: Vec<SafepointMetadata> = Vec::with_capacity(sp_slice.len());
    for (ord, sp) in sp_slice.iter().enumerate() {
        let mut live_roots: Vec<LiveRootMetadata> = Vec::with_capacity(sp.roots_count as usize);
        if sp.roots_count > 0 && !roots.is_null() {
            let root_slice = unsafe {
                std::slice::from_raw_parts(
                    roots.add(sp.roots_start as usize),
                    sp.roots_count as usize,
                )
            };
            for (slot, &offset) in root_slice.iter().enumerate() {
                live_roots.push(LiveRootMetadata {
                    slot: slot as u16,
                    location: RootLocation::Spill(offset),
                });
            }
        }
        safepoints_meta.push(SafepointMetadata {
            ordinal: ord as u32,
            inst_index: 0,
            code_offset: sp.code_offset,
            kind: SafepointKind::CallRuntime,
            live_roots,
        });
    }

    let metadata = Arc::new(NativeFrameMetadata {
        boxed_values: Vec::new(),
        safepoints: safepoints_meta,
        spill_safe_nonleaf: true,
    });

    // Synthesise a placeholder FuncId for code_range bookkeeping.
    // The GC walker keys metadata off the code_range; the engine's
    // `jit_metadata` slot for that FuncId needs the same Arc so
    // `scan_native_stack_roots` finds it.
    let name_sym = vm_ref.interner.intern("<aot-frame>");
    let func_id = vm_ref
        .engine
        .register_aot_function(name_sym, 0, fn_ptr, None);
    let start = fn_ptr as usize;
    vm_ref
        .engine
        .set_aot_metadata(func_id, Arc::clone(&metadata));
    vm_ref
        .engine
        .register_code_range(func_id, start, start + code_size as usize, metadata);
    0
}

/// One foreign-method install descriptor: signature + the
/// `#!symbol = "..."` override (or null to fall back to the
/// signature's base name) + static/instance flag. Bootstrap
/// emits one per `foreign` method declared inside a `foreign
/// class`. Sister to [`WliftAotMethodDesc`] but for the
/// dlopen/dlsym path.
#[cfg(feature = "aot")]
#[repr(C)]
pub struct WliftAotForeignMethodDesc {
    pub sig: *const c_char,
    pub sig_len: usize,
    /// `#!symbol = "..."` override; `null` means fall back to the
    /// method's base name (Wren convention — `open(_)` looks up
    /// `open` if `#!symbol` isn't set).
    pub symbol: *const c_char,
    pub symbol_len: usize,
    /// Bit 0 = is_static. Constructors aren't allowed to be
    /// `foreign` in Wren — no constructor flag needed.
    pub flags: u8,
}

/// Bind a `foreign class`'s native methods at startup. Loads the
/// library named by `lib_name` via the VM's normal foreign loader
/// (search paths + name overrides set by
/// `vm.apply_hatch_native_manifest_rooted` or
/// `wlift_aot_install_native_lib`), then resolves each method's
/// symbol and binds it into the class method table the way the
/// JIT install loop does in `install_module_mir_and_run_with_sources`.
///
/// `slot` indexes into `modvars` to locate the previously-installed
/// `*mut ObjClass` (so the bootstrap calls this *after*
/// `wlift_aot_install_class` for the same class). Errors during
/// library load or symbol resolution surface as `vm.report_error`
/// and return non-zero — the bootstrap currently ignores the
/// failure code and continues, matching the JIT's "soft" foreign-
/// resolution behaviour (calls to unbound foreign methods surface
/// as runtime "method not found" errors at use site).
///
/// # Safety
///
/// All pointers must be valid for the duration of the call;
/// `methods` must point at `methods_count` consecutive
/// `WliftAotForeignMethodDesc` records. The bootstrap satisfies
/// these via Cranelift `Linkage::Local` `Data` blobs.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_bind_foreign_class(
    vm: *mut WrenVM,
    modvars: *const u64,
    slot: usize,
    lib_name: *const c_char,
    lib_name_len: usize,
    methods: *const WliftAotForeignMethodDesc,
    methods_count: usize,
) -> c_int {
    use crate::runtime::object::{ObjClass, ObjHeader, ObjType};

    if vm.is_null() || modvars.is_null() || lib_name.is_null() {
        return 70;
    }

    let lib_bytes = unsafe { std::slice::from_raw_parts(lib_name as *const u8, lib_name_len) };
    let lib_str = match std::str::from_utf8(lib_bytes) {
        Ok(s) => s,
        Err(_) => return 65,
    };

    // Resolve the receiving class through modvars[slot].
    let class_bits = unsafe { *modvars.add(slot) };
    let class_val = crate::runtime::value::Value::from_bits(class_bits);
    let class_ptr = match class_val.as_object() {
        Some(p) => p as *mut ObjClass,
        None => return 70,
    };
    let header = class_ptr as *const ObjHeader;
    if unsafe { (*header).obj_type } != ObjType::Class {
        return 70;
    }

    let vm_ref = unsafe { &mut *vm };
    let library = match crate::runtime::foreign::load_library(
        lib_str,
        &vm_ref.native_search_paths,
        &vm_ref.native_lib_paths,
    ) {
        Ok(l) => l,
        Err(e) => {
            vm_ref.report_error(&e.to_string());
            return 70;
        }
    };

    // Mark the class foreign so the dispatch path takes the
    // ForeignC branch (matches the JIT install loop).
    unsafe {
        (*class_ptr).is_foreign = true;
    }

    for i in 0..methods_count {
        let desc = unsafe { &*methods.add(i) };
        if desc.sig.is_null() {
            continue;
        }
        let sig_bytes = unsafe { std::slice::from_raw_parts(desc.sig as *const u8, desc.sig_len) };
        let sig = match std::str::from_utf8(sig_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let symbol_owned: String = if desc.symbol.is_null() {
            crate::runtime::foreign::base_name_of_signature(sig).to_string()
        } else {
            let bytes =
                unsafe { std::slice::from_raw_parts(desc.symbol as *const u8, desc.symbol_len) };
            match std::str::from_utf8(bytes) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            }
        };

        let resolved =
            match crate::runtime::foreign::resolve_symbol(&library, lib_str, &symbol_owned) {
                Ok(r) => r,
                Err(e) => {
                    // Per-method miss isn't fatal — the unbound method just errors at
                    // call time. Sibling target-specific wrappers around one dylib
                    // (e.g. gpu_native + gpu_web) routinely miss on the wrong target.
                    if std::env::var_os("WLIFT_TRACE_FOREIGN").is_some() {
                        vm_ref.report_error(&e.to_string());
                    }
                    continue;
                }
            };

        // Foreign methods register under "static:<sig>" or
        // "<sig>" depending on the static flag. Wren calls into
        // `Hello.greet(_)` as `static:greet(_)` on the metaclass,
        // and into `inst.greet(_)` as `greet(_)` on the class —
        // matching the JIT install path's binding shape.
        let is_static = desc.flags & 1 != 0;
        let bind_name = if is_static {
            format!("static:{}", sig)
        } else {
            sig.to_string()
        };
        let bind_sym = vm_ref.interner.intern(&bind_name);

        unsafe {
            match resolved {
                crate::runtime::foreign::ResolvedSymbol::Static(func) => {
                    (*class_ptr).bind_foreign_c(bind_sym, func);
                }
                crate::runtime::foreign::ResolvedSymbol::Dynamic(idx) => {
                    (*class_ptr).bind_foreign_c_dynamic(bind_sym, idx);
                }
            }
        }
    }

    // Keep the library alive for the program's lifetime —
    // `class_ptr` now holds raw fn pointers into its address
    // space.
    vm_ref.native_libs.push(library);
    0
}

/// Push an absolute directory onto the VM's foreign loader
/// search paths so subsequent `wlift_aot_bind_foreign_class`
/// calls find the dylibs that path-linked hatch deps ship in
/// their `libs/` directory. Mirrors what
/// `vm.apply_hatch_native_manifest_rooted` does at install time
/// for the interpreter path.
///
/// # Safety
///
/// `path` must be a valid pointer to `len` UTF-8 bytes for the
/// duration of the call.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_add_native_search_path(
    vm: *mut WrenVM,
    path: *const c_char,
    len: usize,
) -> c_int {
    if vm.is_null() || path.is_null() {
        return 70;
    }
    let bytes = unsafe { std::slice::from_raw_parts(path as *const u8, len) };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return 65,
    };
    let vm_ref = unsafe { &mut *vm };
    vm_ref.native_search_paths.push(std::path::PathBuf::from(s));
    0
}

/// Extract a bundled native library to a temp path and tell the
/// VM's foreign loader to find it there. Mirrors the
/// `install_hatch_modules` NativeLib-section handling: write the
/// bytes to `<tmp>/<name>` (with the platform-appropriate
/// extension), then push `(name, path)` into `vm.native_lib_paths`.
///
/// `wlift_aot_bind_foreign_class` consults that map first, so a
/// `.hatch`-bundled plugin overrides any system-installed library
/// with the same name — same precedence the runtime applies.
///
/// # Safety
///
/// `name`/`bytes` must be valid for the duration of the call.
/// The temp path leaks for the lifetime of the process, matching
/// the runtime's behaviour (libraries can't be unmapped while
/// foreign methods may dispatch into them).
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_install_native_lib(
    vm: *mut WrenVM,
    name: *const c_char,
    name_len: usize,
    bytes: *const u8,
    bytes_len: usize,
) -> c_int {
    if vm.is_null() || name.is_null() || bytes.is_null() {
        return 70;
    }
    let name_bytes = unsafe { std::slice::from_raw_parts(name as *const u8, name_len) };
    let lib_name = match std::str::from_utf8(name_bytes) {
        Ok(s) => s.to_string(),
        Err(_) => return 65,
    };
    let payload = unsafe { std::slice::from_raw_parts(bytes, bytes_len) };

    let filename = crate::runtime::foreign::default_native_lib_filename(&lib_name);
    let tmp_dir = std::env::temp_dir().join("wlift_aot_native_libs");
    if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
        let vm_ref = unsafe { &mut *vm };
        vm_ref.report_error(&format!("creating native-lib temp dir: {}", e));
        return 70;
    }
    let path = tmp_dir.join(&filename);
    if let Err(e) = std::fs::write(&path, payload) {
        let vm_ref = unsafe { &mut *vm };
        vm_ref.report_error(&format!("writing native lib '{}': {}", lib_name, e));
        return 70;
    }

    let vm_ref = unsafe { &mut *vm };
    vm_ref.native_lib_paths.insert(lib_name, path);
    0
}

/// Intern `name` in the VM's interner and return the resulting
/// `SymbolId` (zero-extended to `u64`). The AOT bootstrap calls
/// this once per symbol referenced by the module's body, stashing
/// the result in `wlift_symbols_<n>[slot]`. Lowering issues
/// `load wlift_symbols_<n>[slot]` everywhere it'd otherwise bake a
/// raw source-interner `SymbolId` immediate (Call's method, …) —
/// without this remap the AOT'd code passes the wrong indices to
/// runtime helpers that expect VM-interner indices.
///
/// Returns `u64::MAX` on null/invalid input — the bootstrap can
/// treat that as a hard abort signal if it cared, but typical
/// flow is to trust the descriptor data the codegen emitted.
///
/// # Safety
///
/// `name` must be a valid pointer to `len` UTF-8 bytes. The
/// bootstrap embeds the bytes via Cranelift `Data` symbols so
/// the contract holds by construction.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_intern_symbol(
    vm: *mut WrenVM,
    name: *const c_char,
    len: usize,
) -> u64 {
    if vm.is_null() || name.is_null() {
        return u64::MAX;
    }
    let bytes = unsafe { std::slice::from_raw_parts(name as *const u8, len) };
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return u64::MAX,
    };
    let vm_ref = unsafe { &mut *vm };
    vm_ref.interner.intern(s).index() as u64
}

/// Register an AOT-binary data region (a `wlift_modvars_<n>` or
/// `wlift_consts_<n>` array) with the engine so the GC scans
/// each slot as a `Value` root and writes back forwarded
/// pointers. AOT modules don't enter `engine.modules` (no
/// `interpret()` install pass), so const strings + closure
/// pointers + class pointers stored in these regions would
/// otherwise be invisible to the collector and the first minor
/// GC would sweep them while AOT bodies still reference them.
///
/// Bootstrap calls this once per region per module — modvars
/// first, consts second — right after `init_prelude` so the
/// regions are registered before any AOT body executes.
///
/// # Safety
///
/// `region` must be a writable `[u64; count]` whose lifetime
/// covers the rest of the program. The bootstrap satisfies this
/// by passing addresses of `Linkage::Export` zero-init data
/// blobs — those live in the binary's `.bss` for as long as the
/// process runs.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_register_root_region(
    vm: *mut WrenVM,
    region: *mut u64,
    count: usize,
) -> c_int {
    if vm.is_null() || region.is_null() || count == 0 {
        return 70;
    }
    let vm_ref = unsafe { &mut *vm };
    vm_ref.engine.aot_root_regions.push((region, count));
    0
}

/// Outcome stamped by an AOT state-machine poll function before
/// it returns. The dispatcher calling the poll reads it back via
/// [`wlift_aot_sm_take_poll_kind`] to decide whether the fiber
/// suspended (kind = `Yield`, the returned value is the yield
/// argument) or completed (kind = `Done`, the returned value is
/// the body's final return). Lives in a `thread_local` rather
/// than as a field on `WrenVM` / `ObjFiber` so the
/// already-loaded plugin ABI stays binary-compatible.
#[cfg(feature = "aot")]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AotSmPollKind {
    /// Default — set on entry to each poll so a missing
    /// `wlift_aot_sm_done` would surface as "fiber finished
    /// without a result" rather than a stale Yield from an
    /// earlier invocation.
    None = 0,
    Yield = 1,
    Done = 2,
}

#[cfg(feature = "aot")]
thread_local! {
    static AOT_SM_POLL_KIND: std::cell::Cell<AotSmPollKind> =
        const { std::cell::Cell::new(AotSmPollKind::None) };
}

/// Cached `WLIFT_AOT_TRACE_SM` env-var check. Runs once on
/// first call instead of per-dispatch — the trace path makes
/// `wlift_aot_invoke_sm_method` 100x slower if we re-check the
/// env var on every cross-fn call (refresh loops generate
/// 50K+ dispatches per second).
pub(crate) fn aot_trace_sm_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WLIFT_AOT_TRACE_SM").is_some())
}

/// Read the current state ID of `fiber`'s AOT state machine.
/// Called from the poll fn's dispatch prologue to decide which
/// `br_table` arm to enter on resume. New frames come back as
/// `0`; `wlift_aot_sm_yield` advances it before returning. Reads
/// the topmost frame on `aot_frames` — that's the currently
/// executing state machine.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber`.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_load_state(fiber: *mut crate::runtime::object::ObjFiber) -> u64 {
    if fiber.is_null() {
        return 0;
    }
    unsafe {
        let depth = (*fiber).aot_active_depth;
        let frames = &(*fiber).aot_frames;
        frames.get(depth).map(|f| f.state_id as u64).unwrap_or(0)
    }
}

/// Stamp `kind = Yield`, advance the topmost frame's state ID
/// to `next_state`, and return. The poll fn's epilogue follows
/// this call with a bare `return v` whose value the dispatcher
/// passes back to the `fiber.call` site as the suspension
/// result.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber`. Caller guarantees
/// the fiber has at least one frame on its `aot_frames` stack.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_yield(
    fiber: *mut crate::runtime::object::ObjFiber,
    next_state: u64,
) {
    if !fiber.is_null() {
        unsafe {
            let depth = (*fiber).aot_active_depth;
            let frames = &mut (*fiber).aot_frames;
            if let Some(frame) = frames.get_mut(depth) {
                frame.state_id = next_state as u32;
            }
        }
    }
    AOT_SM_POLL_KIND.with(|c| c.set(AotSmPollKind::Yield));
}

/// Stamp `kind = Done`. The poll fn's bare `return v` carries
/// the body's final return value; the dispatcher transitions the
/// fiber to `Done` and surfaces `v` to the caller.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber`.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_done(_fiber: *mut crate::runtime::object::ObjFiber) {
    AOT_SM_POLL_KIND.with(|c| c.set(AotSmPollKind::Done));
}

/// Read + clear the per-thread state-machine poll kind. The
/// dispatcher calls this immediately after a poll returns to
/// learn whether the fiber suspended or completed. Always
/// resets to `None` so a subsequent poll that forgets to stamp
/// surfaces as a stuck-fiber error instead of inheriting the
/// previous outcome.
#[cfg(feature = "aot")]
#[inline]
pub fn wlift_aot_sm_take_poll_kind() -> AotSmPollKind {
    AOT_SM_POLL_KIND.with(|c| {
        let k = c.get();
        c.set(AotSmPollKind::None);
        k
    })
}

/// Save a Wren value into the topmost frame's saved-slot table
/// at `slot`. Called at every suspension by the AOT-emitted
/// poll function for each value live across the yield, and at
/// each cross-fn call site for caller arg slots. The vec is
/// resized on demand so slots can be assigned in any order
/// without an upfront sizing pass.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber` with at least one
/// frame on `aot_frames`.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_save_value(
    fiber: *mut crate::runtime::object::ObjFiber,
    slot: u64,
    value_bits: u64,
) {
    if fiber.is_null() {
        return;
    }
    let slot = slot as usize;
    unsafe {
        let depth = (*fiber).aot_active_depth;
        let frames = &mut (*fiber).aot_frames;
        let frame = match frames.get_mut(depth) {
            Some(f) => f,
            None => return,
        };
        if slot >= frame.saved_values.len() {
            frame
                .saved_values
                .resize(slot + 1, crate::runtime::value::Value::null());
        }
        frame.saved_values[slot] = crate::runtime::value::Value::from_bits(value_bits);
    }
}

/// Read back a saved value at `slot` from the topmost frame.
/// Called at the resume entry of every state that has live-in
/// saved values, and on first-call entry of a state-machine
/// method when loading args from caller-populated slots.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber`. Returns null bits
/// for out-of-range slots / no-frame so a corrupted state
/// struct surfaces as a runtime null rather than UB.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_load_value(
    fiber: *mut crate::runtime::object::ObjFiber,
    slot: u64,
) -> u64 {
    if fiber.is_null() {
        return crate::runtime::value::Value::null().to_bits();
    }
    let slot = slot as usize;
    unsafe {
        let depth = (*fiber).aot_active_depth;
        let frames = &(*fiber).aot_frames;
        let frame = match frames.get(depth) {
            Some(f) => f,
            None => return crate::runtime::value::Value::null().to_bits(),
        };
        let _ = depth;
        frame
            .saved_values
            .get(slot)
            .copied()
            .unwrap_or_else(crate::runtime::value::Value::null)
            .to_bits()
    }
}

/// Write a value into slot `slot` of the *topmost* frame (the
/// one most recently pushed via [`wlift_aot_sm_push_frame`]).
/// Used by cross-fn callers to populate the child frame's args
/// before invoking its poll fn — distinct from
/// [`wlift_aot_sm_save_value`], which writes to the
/// *active-depth* frame (the caller's), so a caller's
/// live-across save and a child's arg write don't alias.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber` with at least one
/// frame.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_save_arg(
    fiber: *mut crate::runtime::object::ObjFiber,
    slot: u64,
    value_bits: u64,
) {
    if fiber.is_null() {
        return;
    }
    let slot = slot as usize;
    unsafe {
        let frame = match (*fiber).aot_frames.last_mut() {
            Some(f) => f,
            None => return,
        };
        if slot >= frame.saved_values.len() {
            frame
                .saved_values
                .resize(slot + 1, crate::runtime::value::Value::null());
        }
        frame.saved_values[slot] = crate::runtime::value::Value::from_bits(value_bits);
    }
}

/// Diagnostic: log a save_arg event from the SM transform's
/// `CrossFnCallInit` lowering. Called immediately after each
/// `wlift_aot_sm_save_arg` so a trace consumer can correlate
/// what the AOT codegen *meant* to save (the MIR ValueId +
/// resolved value bits) against what `wlift_aot_invoke_sm_method`
/// later reads via `wlift_aot_sm_load_arg` at the matching
/// `CrossFnCallResume`. Gated on `WLIFT_AOT_TRACE_SM`; no-op
/// when unset.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_trace_init_save(
    fiber: *mut crate::runtime::object::ObjFiber,
    slot: u64,
    value_id: u64,
    value_bits: u64,
) {
    if !aot_trace_sm_enabled() {
        return;
    }
    let depth = if fiber.is_null() {
        u64::MAX
    } else {
        unsafe { (*fiber).aot_active_depth as u64 }
    };
    let frames_len = if fiber.is_null() {
        0
    } else {
        unsafe { (*fiber).aot_frames.len() }
    };
    let v = crate::runtime::value::Value::from_bits(value_bits);
    let kind = if v.is_null() {
        "null".to_string()
    } else if v.is_bool() {
        format!("bool({})", v.as_bool().unwrap_or(false))
    } else if v.is_num() {
        format!("num({})", v.as_num().unwrap_or(0.0))
    } else if v.is_object() {
        if let Some(p) = v.as_object() {
            let header = p as *const crate::runtime::object::ObjHeader;
            let ty = unsafe { (*header).obj_type };
            format!("obj({:?}@0x{:x})", ty, p as usize)
        } else {
            "obj(?)".to_string()
        }
    } else {
        format!("0x{:x}", value_bits)
    };
    eprintln!(
        "SM-INIT-SAVE depth={} frames_len={} slot={} v_id=v{} value={}",
        depth, frames_len, slot, value_id, kind
    );
}

/// Diagnostic mirror of [`wlift_aot_trace_init_save`] for the
/// resume_check's `load_arg` reads. The receiver/args were
/// stashed by a matching Init block; this logs what the load
/// actually picked up so a save→load divergence (e.g. a
/// pop_frame between the two, or an unintended frame push)
/// surfaces directly. Gated on `WLIFT_AOT_TRACE_SM`.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_trace_init_load(
    fiber: *mut crate::runtime::object::ObjFiber,
    slot: u64,
    value_id: u64,
    value_bits: u64,
) {
    if !aot_trace_sm_enabled() {
        return;
    }
    let depth = if fiber.is_null() {
        u64::MAX
    } else {
        unsafe { (*fiber).aot_active_depth as u64 }
    };
    let frames_len = if fiber.is_null() {
        0
    } else {
        unsafe { (*fiber).aot_frames.len() }
    };
    let v = crate::runtime::value::Value::from_bits(value_bits);
    let kind = if v.is_null() {
        "null".to_string()
    } else if v.is_bool() {
        format!("bool({})", v.as_bool().unwrap_or(false))
    } else if v.is_num() {
        format!("num({})", v.as_num().unwrap_or(0.0))
    } else if v.is_object() {
        if let Some(p) = v.as_object() {
            let header = p as *const crate::runtime::object::ObjHeader;
            let ty = unsafe { (*header).obj_type };
            format!("obj({:?}@0x{:x})", ty, p as usize)
        } else {
            "obj(?)".to_string()
        }
    } else {
        format!("0x{:x}", value_bits)
    };
    eprintln!(
        "SM-INIT-LOAD depth={} frames_len={} slot={} v_id=v{} value={}",
        depth, frames_len, slot, value_id, kind
    );
}

/// Read a value from slot `slot` of the *topmost* frame (the
/// child frame the parent pushed for a cross-fn call). Used by
/// the parent's call_check block on resume — the receiver and
/// args were stashed there at the matching Init block, and
/// reading them via val_map / Cranelift SSA isn't safe because
/// the dispatcher's `br_table` enters the call_check block
/// without going through Init.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber` with at least one
/// frame.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_load_arg(
    fiber: *mut crate::runtime::object::ObjFiber,
    slot: u64,
) -> u64 {
    if fiber.is_null() {
        return crate::runtime::value::Value::null().to_bits();
    }
    let slot = slot as usize;
    unsafe {
        let frames = &(*fiber).aot_frames;
        let frame = match frames.last() {
            Some(f) => f,
            None => return crate::runtime::value::Value::null().to_bits(),
        };
        frame
            .saved_values
            .get(slot)
            .copied()
            .unwrap_or_else(crate::runtime::value::Value::null)
            .to_bits()
    }
}

/// Push a fresh state-machine frame on `fiber`. Used at every
/// cross-fn call site before invoking a state-machine callee:
/// the caller pushes a frame, writes args into the new
/// frame's slot table via [`wlift_aot_sm_save_value`], and then
/// calls the callee's poll fn. The callee's `wlift_aot_sm_*`
/// helpers all operate on this newly-pushed frame.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber`.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_push_frame(fiber: *mut crate::runtime::object::ObjFiber) {
    if fiber.is_null() {
        return;
    }
    unsafe {
        (*fiber)
            .aot_frames
            .push(crate::runtime::object::AotFrameState::default());
    }
}

/// Pop the topmost state-machine frame off `fiber`. Called
/// after a child state-machine returns `Done`, releasing the
/// child's saved-slot storage so subsequent calls don't carry
/// stale values.
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber`. No-op if the frame
/// stack is already empty (defensive — shouldn't happen in
/// well-formed AOT codegen, but a corrupted call sequence
/// shouldn't UB the runtime).
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_pop_frame(fiber: *mut crate::runtime::object::ObjFiber) {
    if fiber.is_null() {
        return;
    }
    unsafe {
        (*fiber).aot_frames.pop();
    }
}

/// Set the topmost frame's `state_id` to `state` *without*
/// stamping the global poll-kind. Used by cross-fn callers to
/// proactively advance their own resume state before invoking a
/// child state machine — that way, if the child yields, the
/// dispatcher re-enters the caller at this state on the next
/// `fiber.call`. Distinct from [`wlift_aot_sm_yield`] (which
/// also flips kind = Yield, intended for the actual suspending
/// return path).
///
/// # Safety
///
/// `fiber` must be a valid `*mut ObjFiber` with at least one
/// frame on `aot_frames`.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_set_state(
    fiber: *mut crate::runtime::object::ObjFiber,
    state: u64,
) {
    if fiber.is_null() {
        return;
    }
    unsafe {
        let depth = (*fiber).aot_active_depth;
        let frames = &mut (*fiber).aot_frames;
        if let Some(frame) = frames.get_mut(depth) {
            frame.state_id = state as u32;
        }
    }
}

/// Read the current poll-kind without clearing it. Used by
/// cross-fn callers after a child state-machine returns: if the
/// child yielded, the kind is already set to `Yield`, and the
/// caller's propagation path leaves it alone so the outer
/// dispatcher still observes Yield. Distinct from
/// [`wlift_aot_sm_take_poll_kind`], which both reads and resets
/// (used at the dispatcher / outermost layer).
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_peek_poll_kind() -> u32 {
    AOT_SM_POLL_KIND.with(|c| c.get() as u32)
}

/// Reset poll-kind back to `None`. Called by cross-fn callers
/// after observing a child state-machine's `Done` so the
/// caller's own continuation can subsequently set
/// kind = Yield/Done as it pleases without inheriting the
/// child's already-consumed Done stamp.
///
/// Without this reset, a fall-through after a `Done` child call
/// would leave kind = Done globally; the caller's eventual
/// completion (which calls `wlift_aot_sm_done`) would re-stamp
/// fine, but a *yield* later in the caller's body would race
/// with the stale Done in code paths that read kind opportunistically.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_sm_clear_poll_kind() {
    AOT_SM_POLL_KIND.with(|c| c.set(AotSmPollKind::None));
}

/// Cross-function dispatch helper: invokes the state-machine
/// `method_sym` on `recv`'s class with the args already saved
/// into the topmost frame's slots `0..num_args`. Returns the
/// callee's poll result. The caller is expected to:
///
/// 1. Save its own live-across-suspension locals via
///    [`wlift_aot_sm_save_value`].
/// 2. Advance its own state via [`wlift_aot_sm_set_state`] to
///    the state ID it wants the dispatcher to re-enter at.
/// 3. Push a new frame via [`wlift_aot_sm_push_frame`].
/// 4. Save args (receiver into slot 0, user args into 1..N).
/// 5. Call this helper.
/// 6. Peek the poll kind via [`wlift_aot_sm_peek_poll_kind`].
///    On Yield: return immediately with the helper's result.
///    On Done: pop the frame, clear kind, bind result, continue.
///
/// `recv_bits` is the receiver's NaN-boxed value used for
/// class lookup; `method_sym` is the *VM*-interner SymbolId
/// (the AOT bootstrap re-interned via `wlift_aot_intern_symbol`,
/// so the call site loads it from the per-module symbols table).
///
/// # Safety
///
/// All pointers must be valid; `fiber` must have at least one
/// frame already pushed (the caller's), and a freshly-pushed
/// child frame on top.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_invoke_sm_method(
    vm: *mut WrenVM,
    fiber: *mut crate::runtime::object::ObjFiber,
    recv_bits: u64,
    method_sym: u64,
    _num_args: u64,
    resume_v: u64,
) -> u64 {
    use crate::runtime::object::{Method, ObjHeader, ObjType};
    use crate::runtime::value::Value;
    if vm.is_null() || fiber.is_null() {
        return Value::null().to_bits();
    }
    let vm_ref = unsafe { &mut *vm };
    let recv = Value::from_bits(recv_bits);
    let sym_id = crate::intern::SymbolId::from_raw(method_sym as u32);

    // WLIFT_AOT_TRACE_SM=1 — log every cross-fn dispatch's
    // {entry-kind, exit-kind, receiver-class, method, depth} so
    // `Yield` propagation breaks become visible. Off the hot
    // path when unset.
    let trace = aot_trace_sm_enabled();
    let trace_method_name = if trace {
        vm_ref.interner.resolve(sym_id).to_string()
    } else {
        String::new()
    };
    let trace_recv_class = if trace {
        let cls = vm_ref.class_of(recv);
        if cls.is_null() {
            "<no-class>".to_string()
        } else {
            let header = cls as *const ObjHeader;
            if unsafe { (*header).obj_type } == ObjType::Class {
                let name_sym = unsafe { (*cls).name };
                vm_ref.interner.resolve(name_sym).to_string()
            } else {
                "<not-class>".to_string()
            }
        }
    } else {
        String::new()
    };
    let trace_kind_in = if trace {
        AOT_SM_POLL_KIND.with(|c| c.get() as u32)
    } else {
        0
    };
    let trace_depth = if trace {
        unsafe { (*fiber).aot_active_depth }
    } else {
        0
    };
    let trace_log_exit = move |arm: &str, kind_out: u32| {
        if trace {
            eprintln!(
                "SM-INVOKE depth={} kind_in={} {}.{} -> {} kind_out={}",
                trace_depth, trace_kind_in, trace_recv_class, trace_method_name, arm, kind_out
            );
        }
    };

    // Closure receiver + `call(...)` method: this is `_fn.call(
    // user_args)` from inside an SM-tainted body. The receiver is
    // the closure itself, the method symbol is one of the `call(_,
    // ...)` stubs on Fn class. Skip the class-method lookup
    // (it'd resolve to `fn_call_stub`, the native error helper)
    // and dispatch the closure body directly. Without this, every
    // call-via-Fn-call site routed into a CrossFnCallInit returned
    // null, so any SM helper closure stored as a field —
    // `Reader.withTryFn`'s inner `_readFn`, the
    // `MapSequence._fn`, etc. — never ran from inside a tainted
    // caller.
    let closure_ptr_from_recv = if recv.is_object() && vm_ref.is_call_sym(sym_id) {
        recv.as_object().and_then(|p| {
            let header = p as *const ObjHeader;
            if unsafe { (*header).obj_type } == ObjType::Closure {
                Some(p as *mut crate::runtime::object::ObjClosure)
            } else {
                None
            }
        })
    } else {
        None
    };

    let closure_ptr = if let Some(cp) = closure_ptr_from_recv {
        cp
    } else {
        // Resolve the receiver's class — instance, then fall back
        // to "the receiver IS a class" for static dispatch
        // (`Foo.bar(_)` style calls where recv = the class object).
        let class_ptr = vm_ref.class_of(recv);
        if class_ptr.is_null() {
            return Value::null().to_bits();
        }
        let idx = sym_id.index() as usize;
        // Walk class.methods[idx] for an installed Method::Closure
        // pointing at a state-machine FuncId. Polymorphic dispatch
        // works naturally because the lookup uses `class_of(recv)`.
        let mut method_entry = unsafe {
            let cls = class_ptr;
            let methods = &(*cls).methods;
            if idx < methods.len() {
                methods[idx]
            } else {
                None
            }
        };
        // Static methods don't live on the receiver's class
        // (which for a class-object receiver is the metaclass) —
        // they live on the receiver itself, registered under
        // `static:<sig>`. Mirror dispatch_call_rooted's static
        // branch: when the receiver IS a class object and the
        // metaclass lookup missed, build the `static:` symbol
        // and re-look up on the class object directly. Without
        // this, `Http_.readRequest(conn)` from inside an
        // SM-tainted `App.serve_` body misses every dispatch
        // and silently returns null.
        let mut effective_class_ptr = class_ptr;
        if method_entry.is_none() && class_ptr == vm_ref.class_class && recv.is_object() {
            if let Some(recv_class) = recv.as_object().map(|p| p as *mut crate::runtime::object::ObjClass) {
                let header = recv_class as *const ObjHeader;
                if unsafe { (*header).obj_type } == ObjType::Class {
                    let method_str = vm_ref.interner.resolve(sym_id).to_string();
                    let static_str = format!("static:{}", method_str);
                    if let Some(static_sym) = vm_ref.interner.lookup(&static_str) {
                        let s_idx = static_sym.index() as usize;
                        method_entry = unsafe {
                            let methods = &(*recv_class).methods;
                            if s_idx < methods.len() {
                                methods[s_idx]
                            } else {
                                None
                            }
                        };
                        if method_entry.is_some() {
                            effective_class_ptr = recv_class;
                        }
                    }
                }
            }
        }
        let class_ptr = effective_class_ptr;
        let cp = match method_entry {
            Some(Method::Closure(cp)) => cp,
            Some(Method::Constructor(cp)) => cp,
            Some(Method::Native(_))
            | Some(Method::ForeignC(_))
            | Some(Method::ForeignCDynamic(_)) => {
                // Native / foreign methods can't be state-machine
                // (they're not closure-backed bodies), but the
                // tainted caller's CrossFnCallInit pushed a child
                // frame and stashed the receiver+args in slots
                // 0..N anyway. Re-build the args slice from those
                // slots and route through the standard dispatch.
                // Fiber.call / Fiber.try / List.add / etc. all
                // hit this branch when `call(N)` taint expansion
                // brings them under cross-fn dispatch.
                let num_args = _num_args as usize;
                let mut args: Vec<crate::runtime::value::Value> =
                    Vec::with_capacity(num_args + 1);
                args.push(recv);
                for i in 1..=num_args {
                    let v = unsafe {
                        (*fiber)
                            .aot_frames
                            .last()
                            .and_then(|f| f.saved_values.get(i).copied())
                            .unwrap_or_else(crate::runtime::value::Value::null)
                    };
                    args.push(v);
                }
                let saved_ctx = crate::codegen::runtime_fns::read_jit_ctx();
                let result = crate::codegen::runtime_fns::dispatch_method_pub(
                    vm_ref,
                    method_entry.unwrap(),
                    &args,
                    Some(class_ptr),
                );
                crate::codegen::runtime_fns::set_jit_context(saved_ctx);
                let pending = AOT_SM_POLL_KIND.with(|c| c.get() as u32);
                AOT_SM_POLL_KIND.with(|c| c.set(AotSmPollKind::Done));
                trace_log_exit("native-fallback", pending);
                let _ = pending;
                return result;
            }
            None => {
                // Method not installed at all — surface as null;
                // the caller's Done branch binds null and the
                // post-call code continues without crashing.
                trace_log_exit("method-not-found", AotSmPollKind::Done as u32);
                return Value::null().to_bits();
            }
        };
        // Defensive: the receiver's `class_of` must produce a real
        // ObjClass for the dispatch to be meaningful.
        let header = class_ptr as *const ObjHeader;
        if unsafe { (*header).obj_type } != ObjType::Class {
            trace_log_exit("defensive-class-check", AotSmPollKind::None as u32);
            return Value::null().to_bits();
        }
        cp
    };
    let func_id = unsafe {
        let fn_obj = (*closure_ptr).function;
        (*fn_obj).fn_id
    };
    let idx = func_id as usize;
    if trace {
        let fn_obj = unsafe { (*closure_ptr).function };
        let arity = unsafe { (*fn_obj).arity };
        let name_sym = unsafe { (*fn_obj).name };
        let fn_name = vm_ref.interner.resolve(name_sym).to_string();
        eprintln!(
            "SM-DISPATCH depth={} closure=0x{:x} fn_id={} fn_name={:?} arity={}",
            trace_depth,
            closure_ptr as usize,
            idx,
            fn_name,
            arity
        );
    }
    let is_sm = vm_ref
        .engine
        .aot_state_machine
        .get(idx)
        .copied()
        .unwrap_or(false);
    if !is_sm {
        // Non-SM callee in a cross-fn call site. Happens whenever a
        // tainted outer body lowers a Call as `CrossFnCallInit` —
        // every `Fn.call` becomes one of these once `call(_,…)` is
        // in the taint set, but the resolved closure may not itself
        // yield. Dispatch through the regular closure path, pop
        // the child frame the caller pushed, stamp `Done` so the
        // caller's `take_poll_kind` continues instead of yielding,
        // and return the body's result. Without this every
        // non-yielding closure stored as a field (Reader's
        // `setReadFn_` callback, MapSequence's mapping fn, …)
        // returned null when invoked from inside a tainted method.
        let num_args = _num_args as usize;
        let mut args: Vec<crate::runtime::value::Value> =
            Vec::with_capacity(num_args + 1);
        args.push(recv);
        for i in 1..=num_args {
            let v = unsafe {
                (*fiber)
                    .aot_frames
                    .last()
                    .and_then(|f| f.saved_values.get(i).copied())
                    .unwrap_or_else(crate::runtime::value::Value::null)
            };
            args.push(v);
        }
        // Fn.call shape passes user-args only (no receiver as
        // first slot). Method-style cross-fn calls pass the
        // recv-then-user-args slice the body's mir.arity is sized
        // for. `closure_ptr_from_recv.is_some()` discriminates
        // between the two.
        let result_bits = if closure_ptr_from_recv.is_some() {
            crate::codegen::runtime_fns::call_closure_jit_or_sync(
                vm_ref,
                closure_ptr,
                &args[1..],
                None,
            )
        } else {
            crate::codegen::runtime_fns::call_closure_jit_or_sync(
                vm_ref,
                closure_ptr,
                &args,
                None,
            )
        };
        // Don't pop the child frame here — the caller's
        // CrossFnCallResume Done branch (in the cranelift backend)
        // emits its own `wlift_aot_sm_pop_frame` after reading
        // the kind. Popping twice corrupts the active-depth
        // invariant: the caller's own frame disappears, the next
        // `wlift_aot_sm_load_value` reads from a stale frame, and
        // saved-across-suspension values come back null.
        let pending_after_call = AOT_SM_POLL_KIND.with(|c| c.get() as u32);
        AOT_SM_POLL_KIND.with(|c| c.set(AotSmPollKind::Done));
        trace_log_exit("non-sm-closure", pending_after_call);
        let _ = pending_after_call;
        return result_bits;
    }
    let fn_ptr = vm_ref
        .engine
        .jit_code
        .get(idx)
        .copied()
        .unwrap_or(std::ptr::null());
    if fn_ptr.is_null() {
        trace_log_exit("sm-no-jit", AotSmPollKind::None as u32);
        return Value::null().to_bits();
    }
    let poll: extern "C" fn(*mut crate::runtime::object::ObjFiber, u64) -> u64 =
        unsafe { std::mem::transmute(fn_ptr) };
    // Active-depth swap. The CALLEE's frame is exactly one
    // level deeper than the caller's, regardless of the total
    // frame-stack depth. Resuming a body whose nested children
    // are still on the stack (e.g. an outer wrapper closure
    // resuming after a yield from `Probe.run`'s `tinyYield_`)
    // leaves three frames on the stack: wrapper@0, Probe.run@1,
    // tinyYield_@2. When the wrapper's `bb2` re-invokes
    // Probe.run on this resume, the callee's frame is
    // wrapper-depth+1 = 1, NOT `len()-1 = 2` (which is
    // tinyYield_'s leftover frame from the first yield).
    // Computing the swap as `saved_depth + 1` keeps the depth
    // pointing at the right callee even across deeply nested
    // re-entries.
    let saved_depth = unsafe { (*fiber).aot_active_depth };
    let new_depth = saved_depth + 1;
    unsafe {
        (*fiber).aot_active_depth = new_depth;
    }
    // Save and re-point the JIT context's `closure` slot at the
    // poll fn's own closure so its `GetUpvalue` /
    // `wren_load_jit_closure` reads upvalues from the right
    // ObjClosure. Without this, every cross-fn SM dispatch into
    // a closure body inherits the caller's `ctx.closure` and
    // reads upvalues from the wrong array — at best stale, at
    // worst NULL+offset SIGSEGVing on first GetUpvalue. Same
    // shape as `call_closure_jit_or_sync`'s SM branch.
    let saved_ctx = crate::codegen::runtime_fns::read_jit_ctx();
    crate::codegen::runtime_fns::mutate_jit_ctx(|ctx| {
        if ctx.vm.is_null() {
            ctx.vm = vm as *mut u8;
        }
        ctx.current_func_id = idx as u64;
        ctx.closure = closure_ptr as *mut u8;
        ctx.defining_class = std::ptr::null_mut();
    });
    let result = poll(fiber, resume_v);
    let kind_after_poll = AOT_SM_POLL_KIND.with(|c| c.get() as u32);
    crate::codegen::runtime_fns::set_jit_context(saved_ctx);
    unsafe {
        (*fiber).aot_active_depth = saved_depth;
    }
    trace_log_exit("sm-poll", kind_after_poll);
    let _ = kind_after_poll;
    result
}

/// Resolve a bare-builtin import (`import "socket" for SocketCore`,
/// `import "fs" for FS`, ...) into the AOT module's modvars at
/// startup. The walker skips these from the dependency graph
/// (they have no on-disk source) so the bootstrap has to ask
/// the runtime to load the module + look up the binding by name.
///
/// `module_name` is the bare module string (e.g. `"socket"`),
/// `var_name` is the Wren symbol the source binds (e.g.
/// `"SocketCore"`). Both are UTF-8 byte ranges; the bootstrap
/// emits them as `Linkage::Local` `Data` blobs.
///
/// Without this call, every method dispatch through the imported
/// class hits a null receiver — `class_of(null)` is the Null
/// class, the static-method lookup misses, and `wren_call_N`
/// silently returns null. That's why hatch site's `app.listen`
/// printed "listening on ..." but never actually bound a TCP
/// socket: `SocketCore.tcpListen(addr)` never ran.
///
/// # Safety
///
/// Pointers must be valid UTF-8 byte ranges.
#[cfg(feature = "aot")]
#[no_mangle]
pub unsafe extern "C" fn wlift_aot_resolve_runtime_import(
    vm: *mut WrenVM,
    modvars: *mut u64,
    slot: usize,
    module_name: *const c_char,
    module_name_len: usize,
    var_name: *const c_char,
    var_name_len: usize,
) -> c_int {
    if vm.is_null() || modvars.is_null() || module_name.is_null() || var_name.is_null() {
        return 70;
    }
    let mod_bytes = unsafe { std::slice::from_raw_parts(module_name as *const u8, module_name_len) };
    let var_bytes = unsafe { std::slice::from_raw_parts(var_name as *const u8, var_name_len) };
    let mod_str = match std::str::from_utf8(mod_bytes) {
        Ok(s) => s,
        Err(_) => return 65,
    };
    let var_str = match std::str::from_utf8(var_bytes) {
        Ok(s) => s,
        Err(_) => return 65,
    };
    let vm_ref = unsafe { &mut *vm };
    // Lazy-load the built-in module if it hasn't been touched yet.
    // Already-loaded modules are a no-op return-true.
    let _ = vm_ref.try_load_builtin_module(mod_str);
    if let Some(value) = vm_ref.find_imported_var_from(var_str, mod_str) {
        unsafe {
            *modvars.add(slot) = value.to_bits();
        }
        0
    } else {
        // Leave the slot at whatever it was (typically null);
        // first-use will surface as a "X has no method" error
        // matching the JIT path's behaviour for unresolved imports.
        70
    }
}

/// Read a static field off a class without consulting the
/// thread-local `JitContext.defining_class`. The AOT lowering
/// loads the class pointer from this module's modvars at the
/// emitted method's defining-class slot and threads it in
/// here, so the helper is independent of any per-frame TLS
/// setup. Mirrors `wren_get_static_field`'s storage contract
/// (a `HashMap<SymbolId, Value>` keyed by source-side
/// SymbolId — same key on get and set within a class).
///
/// # Safety
///
/// `class_bits` must be the NaN-boxed `Value` of an `ObjClass`,
/// matching what AOT modvars hold post-`wlift_aot_install_class`.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_get_static_field(class_bits: u64, field_sym: u64) -> u64 {
    use crate::runtime::object::ObjClass;
    use crate::runtime::value::Value;
    let class_val = Value::from_bits(class_bits);
    let Some(obj) = class_val.as_object() else {
        return Value::null().to_bits();
    };
    let class = obj as *const ObjClass;
    let sym = crate::intern::SymbolId::from_raw(field_sym as u32);
    unsafe {
        (*class)
            .static_fields
            .get(&sym)
            .copied()
            .unwrap_or(Value::null())
            .to_bits()
    }
}

/// Write a static field on a class. Performs the same write
/// barrier `wren_set_static_field` does — the class is the
/// owner of the slot, and the inserted value may be a young
/// pointer the major collector needs to see.
///
/// # Safety
///
/// `class_bits` must be the NaN-boxed `Value` of an `ObjClass`.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_set_static_field(
    class_bits: u64,
    field_sym: u64,
    value: u64,
) -> u64 {
    use crate::codegen::runtime_fns::read_jit_ctx;
    use crate::runtime::object::{ObjClass, ObjHeader};
    use crate::runtime::value::Value;
    let class_val = Value::from_bits(class_bits);
    let Some(obj) = class_val.as_object() else {
        return value;
    };
    let class = obj as *mut ObjClass;
    let sym = crate::intern::SymbolId::from_raw(field_sym as u32);
    let value = Value::from_bits(value);
    unsafe {
        (*class).static_fields.insert(sym, value);
        let ctx = read_jit_ctx();
        if !ctx.vm.is_null() {
            let vm = &mut *(ctx.vm as *mut crate::runtime::vm::VM);
            vm.gc.write_barrier(class as *mut ObjHeader, value);
        }
    }
    value.to_bits()
}

/// Restore the `JitContext` saved by a matching `wlift_aot_enter`.
/// Pass the same `saved` pointer the enter call wrote into.
///
/// # Safety
///
/// `saved` must point at a 48-byte buffer previously populated
/// by `wlift_aot_enter`.
#[no_mangle]
#[cfg(feature = "aot")]
pub unsafe extern "C" fn wlift_aot_exit(saved: *const u64) {
    use crate::codegen::runtime_fns::{set_jit_context, JitContext};
    if saved.is_null() {
        return;
    }
    let ptr = saved as *const JitContext;
    let prev = unsafe { *ptr };
    set_jit_context(prev);
}

// ---------------------------------------------------------------------------
// Call handles
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenMakeCallHandle(vm: *mut WrenVM, signature: *const c_char) -> *mut WrenHandle {
    if vm.is_null() || signature.is_null() {
        return ptr::null_mut();
    }
    let sig = unsafe { CStr::from_ptr(signature) }
        .to_str()
        .unwrap_or("")
        .to_string();
    Box::into_raw(Box::new(WrenHandle {
        index: 0,
        signature: sig,
    }))
}

#[no_mangle]
pub extern "C" fn wrenCall(vm: *mut WrenVM, method: *mut WrenHandle) -> WrenInterpretResult {
    if vm.is_null() || method.is_null() {
        return WrenInterpretResult::RuntimeError;
    }
    let handle = unsafe { &*method };
    let vm_ref = unsafe { &mut *vm };

    // Receiver is in slot 0, args in slots 1..N
    let receiver = vm_ref.get_slot(0);

    // Count args from signature (number of underscores)
    let num_args = handle.signature.chars().filter(|&c| c == '_').count();
    let mut args = Vec::with_capacity(num_args);
    for i in 1..=num_args {
        args.push(vm_ref.get_slot(i));
    }

    // call_method_on handles class receivers with static: prefix fallback
    let result = vm_ref.call_method_on(receiver, &handle.signature, &args);

    match result {
        Some(value) => {
            vm_ref.set_slot(0, value);
            WrenInterpretResult::Success
        }
        None => WrenInterpretResult::RuntimeError,
    }
}

#[no_mangle]
pub extern "C" fn wrenReleaseHandle(vm: *mut WrenVM, handle: *mut WrenHandle) {
    if vm.is_null() || handle.is_null() {
        return;
    }
    unsafe {
        let h = Box::from_raw(handle);
        (*vm).release_handle(h.index);
    }
}

// ---------------------------------------------------------------------------
// Slot API — reading values
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetSlotCount(vm: *mut WrenVM) -> c_int {
    if vm.is_null() {
        return 0;
    }
    unsafe { (*vm).api_stack.len() as c_int }
}

#[no_mangle]
pub extern "C" fn wrenEnsureSlots(vm: *mut WrenVM, num_slots: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        let vm = &mut *vm;
        while vm.api_stack.len() < num_slots as usize {
            vm.api_stack.push(Value::null());
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotType(vm: *mut WrenVM, slot: c_int) -> WrenType {
    if vm.is_null() {
        return WrenType::Unknown;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if val.is_bool() {
        WrenType::Bool
    } else if val.is_num() {
        WrenType::Num
    } else if val.is_null() {
        WrenType::Null
    } else if val.is_object() {
        let ptr = val.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        unsafe {
            match (*header).obj_type {
                ObjType::String => WrenType::String,
                ObjType::List => WrenType::List,
                ObjType::Map => WrenType::Map,
                ObjType::Foreign => WrenType::Foreign,
                _ => WrenType::Unknown,
            }
        }
    } else {
        WrenType::Unknown
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotBool(vm: *mut WrenVM, slot: c_int) -> bool {
    if vm.is_null() {
        return false;
    }
    unsafe { (*vm).get_slot(slot as usize).as_bool().unwrap_or(false) }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotDouble(vm: *mut WrenVM, slot: c_int) -> c_double {
    if vm.is_null() {
        return 0.0;
    }
    unsafe { (*vm).get_slot(slot as usize).as_num().unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotString(vm: *mut WrenVM, slot: c_int) -> *const c_char {
    if vm.is_null() {
        return ptr::null();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return ptr::null();
    }
    let ptr = val.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe {
        if (*header).obj_type != ObjType::String {
            return ptr::null();
        }
        let obj = &*(ptr as *const ObjString);
        obj.as_c_str()
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotBytes(
    vm: *mut WrenVM,
    slot: c_int,
    length: *mut c_int,
) -> *const c_char {
    if vm.is_null() {
        return ptr::null();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return ptr::null();
    }
    let ptr = val.as_object().unwrap();
    let header = ptr as *const ObjHeader;
    unsafe {
        match (*header).obj_type {
            ObjType::String => {
                let obj = &*(ptr as *const ObjString);
                if !length.is_null() {
                    *length = obj.value.len() as c_int;
                }
                obj.value.as_ptr() as *const c_char
            }
            ObjType::TypedArray => {
                let obj = &*(ptr as *const ObjTypedArray);
                let n = obj.byte_len();
                if !length.is_null() {
                    *length = n as c_int;
                }
                if obj.data.is_null() || n == 0 {
                    ptr::null()
                } else {
                    obj.data as *const c_char
                }
            }
            _ => ptr::null(),
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenGetSlotForeign(vm: *mut WrenVM, slot: c_int) -> *mut c_void {
    if vm.is_null() {
        return ptr::null_mut();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return ptr::null_mut();
    }
    // Return pointer to the foreign object's data.
    val.as_object().unwrap() as *mut c_void
}

#[no_mangle]
pub extern "C" fn wrenGetSlotHandle(vm: *mut WrenVM, slot: c_int) -> *mut WrenHandle {
    if vm.is_null() {
        return ptr::null_mut();
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    let idx = unsafe { (*vm).make_handle(val) };
    Box::into_raw(Box::new(WrenHandle {
        index: idx,
        signature: String::new(),
    }))
}

// ---------------------------------------------------------------------------
// Slot API — writing values
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenSetSlotBool(vm: *mut WrenVM, slot: c_int, value: bool) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).set_slot(slot as usize, Value::bool(value));
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotDouble(vm: *mut WrenVM, slot: c_int, value: c_double) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).set_slot(slot as usize, Value::num(value));
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNull(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).set_slot(slot as usize, Value::null());
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotString(vm: *mut WrenVM, slot: c_int, text: *const c_char) {
    if vm.is_null() || text.is_null() {
        return;
    }
    let s = unsafe { CStr::from_ptr(text) }
        .to_str()
        .unwrap_or("")
        .to_string();
    unsafe {
        let val = (*vm).new_string(s);
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotBytes(
    vm: *mut WrenVM,
    slot: c_int,
    bytes: *const c_char,
    length: usize,
) {
    if vm.is_null() || bytes.is_null() {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(bytes as *const u8, length) };
    let s = String::from_utf8_lossy(slice).into_owned();
    unsafe {
        let val = (*vm).new_string(s);
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotHandle(vm: *mut WrenVM, slot: c_int, handle: *mut WrenHandle) {
    if vm.is_null() || handle.is_null() {
        return;
    }
    unsafe {
        let h = &*handle;
        let vm = &mut *vm;
        if h.index < vm.handles.len() {
            let val = vm.handles[h.index].value;
            vm.set_slot(slot as usize, val);
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNewList(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        let val = (*vm).new_list(vec![]);
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNewMap(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    unsafe {
        let val = (*vm).new_map();
        (*vm).set_slot(slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetSlotNewForeign(
    vm: *mut WrenVM,
    slot: c_int,
    _class_slot: c_int,
    size: usize,
) -> *mut c_void {
    if vm.is_null() {
        return ptr::null_mut();
    }
    let data = vec![0u8; size];
    unsafe {
        let obj = (*vm).gc.alloc_foreign(data);
        let val = Value::object(obj as *mut u8);
        (*vm).set_slot(slot as usize, val);
        (*obj).data.as_mut_ptr() as *mut c_void
    }
}

// ---------------------------------------------------------------------------
// List operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetListCount(vm: *mut WrenVM, slot: c_int) -> c_int {
    if vm.is_null() {
        return 0;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return 0;
    }
    let ptr = val.as_object().unwrap();
    unsafe {
        let list = &*(ptr as *const ObjList);
        list.len() as c_int
    }
}

#[no_mangle]
pub extern "C" fn wrenGetListElement(
    vm: *mut WrenVM,
    list_slot: c_int,
    index: c_int,
    element_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let list_val = unsafe { (*vm).get_slot(list_slot as usize) };
    if !list_val.is_object() {
        return;
    }
    let ptr = list_val.as_object().unwrap();
    unsafe {
        let list = &*(ptr as *const ObjList);
        let idx = if index < 0 {
            (list.len() as i32 + index) as usize
        } else {
            index as usize
        };
        if let Some(elem) = list.get(idx) {
            (*vm).set_slot(element_slot as usize, elem);
        }
    }
}

#[no_mangle]
pub extern "C" fn wrenSetListElement(
    vm: *mut WrenVM,
    list_slot: c_int,
    index: c_int,
    element_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let list_val = unsafe { (*vm).get_slot(list_slot as usize) };
    let elem = unsafe { (*vm).get_slot(element_slot as usize) };
    if !list_val.is_object() {
        return;
    }
    let ptr = list_val.as_object().unwrap();
    unsafe {
        let list = &mut *(ptr as *mut ObjList);
        let idx = if index < 0 {
            (list.len() as i32 + index) as usize
        } else {
            index as usize
        };
        list.set(idx, elem);
    }
}

#[no_mangle]
pub extern "C" fn wrenInsertInList(
    vm: *mut WrenVM,
    list_slot: c_int,
    index: c_int,
    element_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let list_val = unsafe { (*vm).get_slot(list_slot as usize) };
    let elem = unsafe { (*vm).get_slot(element_slot as usize) };
    if !list_val.is_object() {
        return;
    }
    let ptr = list_val.as_object().unwrap();
    unsafe {
        let list = &mut *(ptr as *mut ObjList);
        let idx = if index < 0 {
            let i = list.len() as i32 + 1 + index;
            if i < 0 {
                0usize
            } else {
                i as usize
            }
        } else {
            index as usize
        };
        let idx = idx.min(list.len());
        list.insert(idx, elem);
    }
}

// ---------------------------------------------------------------------------
// Map operations
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetMapCount(vm: *mut WrenVM, slot: c_int) -> c_int {
    if vm.is_null() {
        return 0;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    if !val.is_object() {
        return 0;
    }
    let ptr = val.as_object().unwrap();
    unsafe {
        let map = &*(ptr as *const ObjMap);
        map.entries.len() as c_int
    }
}

#[no_mangle]
pub extern "C" fn wrenGetMapContainsKey(vm: *mut WrenVM, map_slot: c_int, key_slot: c_int) -> bool {
    if vm.is_null() {
        return false;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    if !map_val.is_object() {
        return false;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &*(ptr as *const ObjMap);
        map.contains(key)
    }
}

#[no_mangle]
pub extern "C" fn wrenGetMapValue(
    vm: *mut WrenVM,
    map_slot: c_int,
    key_slot: c_int,
    value_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    if !map_val.is_object() {
        return;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &*(ptr as *const ObjMap);
        let val = map.get(key).unwrap_or(Value::null());
        (*vm).set_slot(value_slot as usize, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenSetMapValue(
    vm: *mut WrenVM,
    map_slot: c_int,
    key_slot: c_int,
    value_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    let val = unsafe { (*vm).get_slot(value_slot as usize) };
    if !map_val.is_object() {
        return;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &mut *(ptr as *mut ObjMap);
        map.set(key, val);
    }
}

#[no_mangle]
pub extern "C" fn wrenRemoveMapValue(
    vm: *mut WrenVM,
    map_slot: c_int,
    key_slot: c_int,
    removed_value_slot: c_int,
) {
    if vm.is_null() {
        return;
    }
    let map_val = unsafe { (*vm).get_slot(map_slot as usize) };
    let key = unsafe { (*vm).get_slot(key_slot as usize) };
    if !map_val.is_object() {
        return;
    }
    let ptr = map_val.as_object().unwrap();
    unsafe {
        let map = &mut *(ptr as *mut ObjMap);
        let removed = map.remove(key).unwrap_or(Value::null());
        (*vm).set_slot(removed_value_slot as usize, removed);
    }
}

// ---------------------------------------------------------------------------
// Module / variable lookup
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenHasModule(vm: *mut WrenVM, module: *const c_char) -> bool {
    if vm.is_null() || module.is_null() {
        return false;
    }
    let name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("");
    unsafe { (*vm).engine.modules.contains_key(name) }
}

#[no_mangle]
pub extern "C" fn wrenHasVariable(
    vm: *mut WrenVM,
    module: *const c_char,
    name: *const c_char,
) -> bool {
    if vm.is_null() || module.is_null() || name.is_null() {
        return false;
    }
    let mod_name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("");
    let var_name = unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("");
    let vm_ref = unsafe { &*vm };
    if let Some(entry) = vm_ref.engine.modules.get(mod_name) {
        entry.var_names.iter().any(|n| n == var_name)
    } else {
        false
    }
}

#[no_mangle]
pub extern "C" fn wrenGetVariable(
    vm: *mut WrenVM,
    module: *const c_char,
    name: *const c_char,
    slot: c_int,
) {
    if vm.is_null() || module.is_null() || name.is_null() {
        return;
    }
    let mod_name = unsafe { CStr::from_ptr(module) }.to_str().unwrap_or("");
    let var_name = unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("");
    let vm_ref = unsafe { &mut *vm };
    if let Some(entry) = vm_ref.engine.modules.get(mod_name) {
        if let Some(idx) = entry.var_names.iter().position(|n| n == var_name) {
            let value = entry.vars[idx];
            vm_ref.set_slot(slot as usize, value);
        }
    }
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenAbortFiber(vm: *mut WrenVM, slot: c_int) {
    if vm.is_null() {
        return;
    }
    let val = unsafe { (*vm).get_slot(slot as usize) };
    let msg = if val.is_object() {
        let ptr = val.as_object().unwrap();
        let header = ptr as *const ObjHeader;
        unsafe {
            if (*header).obj_type == ObjType::String {
                let obj = &*(ptr as *const ObjString);
                obj.value.clone()
            } else {
                "Runtime error.".to_string()
            }
        }
    } else {
        "Runtime error.".to_string()
    };
    unsafe {
        (*vm).runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// User data
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn wrenGetUserData(vm: *mut WrenVM) -> *mut c_void {
    if vm.is_null() {
        return ptr::null_mut();
    }
    unsafe { (*vm).user_data }
}

#[no_mangle]
pub extern "C" fn wrenSetUserData(vm: *mut WrenVM, user_data: *mut c_void) {
    if vm.is_null() {
        return;
    }
    unsafe {
        (*vm).user_data = user_data;
    }
}

// ---------------------------------------------------------------------------
// Hatch package loading
//
// Mirrors the Rust-side `hatch_runner::HatchRunner` API, but keeps
// every function as a plain `extern "C"` so C / C++ / Swift / FFI
// consumers can call them directly. Return values are 0 on success,
// non-zero on error — matching the Unix-style idiom and letting C
// callers chain with `if (wrenInstallHatchFile(...)) return 1;`.
// ---------------------------------------------------------------------------

/// Install a pre-loaded `.hatch` byte buffer into the VM. Returns
/// 0 on success, 1 on compile error, 2 on runtime error, -1 on
/// null-pointer inputs, -2 on builds without hatch packaging
/// (wasm). Useful for embedders that ship packages baked into the
/// binary via `include_bytes!` / a resource file.
#[no_mangle]
pub extern "C" fn wrenInstallHatchBytes(vm: *mut WrenVM, bytes: *const u8, len: usize) -> c_int {
    if vm.is_null() || bytes.is_null() {
        return -1;
    }
    #[cfg(feature = "host")]
    {
        let vm_ref = unsafe { &mut *vm };
        let slice = unsafe { std::slice::from_raw_parts(bytes, len) };
        use crate::runtime::engine::InterpretResult;
        match vm_ref.install_hatch_modules(slice) {
            InterpretResult::Success => 0,
            InterpretResult::CompileError => 1,
            InterpretResult::RuntimeError => 2,
        }
    }
    #[cfg(not(feature = "host"))]
    {
        let _ = (vm, bytes, len);
        -2
    }
}

/// Install a `.hatch` file from disk. Returns 0 on success, 1 on
/// compile error, 2 on runtime error, 3 if the file can't be read,
/// -1 on null-pointer inputs.
#[no_mangle]
pub extern "C" fn wrenInstallHatchFile(vm: *mut WrenVM, path: *const c_char) -> c_int {
    if vm.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let bytes = match std::fs::read(path_str) {
        Ok(b) => b,
        Err(_) => return 3,
    };
    wrenInstallHatchBytes(vm, bytes.as_ptr(), bytes.len())
}

/// Append a directory to the native-library search path. The VM
/// consults this when resolving `#!native` dylib references in
/// imported classes. Per-hatch `native_search_paths` are already
/// picked up automatically; this is for embedder overrides.
/// Returns 0 on success, -1 on null-pointer inputs.
#[no_mangle]
pub extern "C" fn wrenAddNativeSearchPath(vm: *mut WrenVM, path: *const c_char) -> c_int {
    if vm.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let vm_ref = unsafe { &mut *vm };
    let pb = std::path::PathBuf::from(path_str);
    if !vm_ref.native_search_paths.contains(&pb) {
        vm_ref.native_search_paths.push(pb);
    }
    0
}

/// Scan a directory for `.hatch` files and install every one of
/// them. The name embedded in each hatchfile is what later code
/// imports as. Returns the number of artifacts installed on
/// success, or a negative error code (-1 null / -2 read-dir /
/// -3 install failed partway through).
#[no_mangle]
pub extern "C" fn wrenInstallHatchDir(vm: *mut WrenVM, path: *const c_char) -> c_int {
    if vm.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let entries = match std::fs::read_dir(path_str) {
        Ok(e) => e,
        Err(_) => return -2,
    };
    let mut installed = 0;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) != Some("hatch") {
            continue;
        }
        let bytes = match std::fs::read(&p) {
            Ok(b) => b,
            Err(_) => return -3,
        };
        let rc = wrenInstallHatchBytes(vm, bytes.as_ptr(), bytes.len());
        if rc != 0 {
            return -3;
        }
        installed += 1;
    }
    installed as c_int
}

// ---------------------------------------------------------------------------
// Internal: convert C config to Rust config
// ---------------------------------------------------------------------------

unsafe fn build_vm_config(c_config: &WrenConfiguration) -> VMConfig {
    let mut config = VMConfig::default();

    if let Some(write_fn) = c_config.write_fn {
        // Store the C function pointer and wrap it.
        // We need a stable VM pointer for the callback — this is set post-creation.
        config.write_fn = Some(Box::new(move |text: &str| {
            if let Ok(cstr) = CString::new(text) {
                write_fn(ptr::null_mut(), cstr.as_ptr());
            }
        }));
    }

    if c_config.initial_heap_size > 0 {
        config.initial_heap_size = c_config.initial_heap_size;
    }
    if c_config.min_heap_size > 0 {
        config.min_heap_size = c_config.min_heap_size;
    }
    if c_config.heap_growth_percent > 0 {
        config.heap_growth_percent = c_config.heap_growth_percent as u32;
    }

    config
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = wrenGetVersionNumber();
        assert!(v > 0);
        assert_eq!(v, 5 * 1_000);
    }

    #[test]
    fn test_vm_lifecycle() {
        let mut config = std::mem::MaybeUninit::<WrenConfiguration>::uninit();
        wrenInitConfiguration(config.as_mut_ptr());
        let config = unsafe { config.assume_init() };

        let vm = wrenNewVM(&config as *const WrenConfiguration);
        assert!(!vm.is_null());

        wrenCollectGarbage(vm);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_bool() {
        let vm = wrenNewVM(ptr::null());
        assert!(!vm.is_null());

        wrenEnsureSlots(vm, 3);
        assert!(wrenGetSlotCount(vm) >= 3);

        wrenSetSlotBool(vm, 0, true);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Bool);
        assert!(wrenGetSlotBool(vm, 0));

        wrenSetSlotBool(vm, 0, false);
        assert!(!wrenGetSlotBool(vm, 0));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_double() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        wrenSetSlotDouble(vm, 0, 1.234);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Num);
        assert!((wrenGetSlotDouble(vm, 0) - 1.234).abs() < 1e-10);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_null() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        wrenSetSlotNull(vm, 0);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Null);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_string() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        let text = CString::new("hello").unwrap();
        wrenSetSlotString(vm, 0, text.as_ptr());
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::String);

        let got = wrenGetSlotString(vm, 0);
        assert!(!got.is_null());
        let got_str = unsafe { CStr::from_ptr(got) }.to_str().unwrap();
        assert_eq!(got_str, "hello");

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_list() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 3);

        wrenSetSlotNewList(vm, 0);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::List);
        assert_eq!(wrenGetListCount(vm, 0), 0);

        // Insert element
        wrenSetSlotDouble(vm, 1, 42.0);
        wrenInsertInList(vm, 0, -1, 1);
        assert_eq!(wrenGetListCount(vm, 0), 1);

        // Read element
        wrenGetListElement(vm, 0, 0, 2);
        assert_eq!(wrenGetSlotDouble(vm, 2), 42.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_slot_api_map() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 4);

        wrenSetSlotNewMap(vm, 0);
        assert_eq!(wrenGetSlotType(vm, 0), WrenType::Map);
        assert_eq!(wrenGetMapCount(vm, 0), 0);

        // Set key-value
        let key = CString::new("x").unwrap();
        wrenSetSlotString(vm, 1, key.as_ptr());
        wrenSetSlotDouble(vm, 2, 99.0);
        wrenSetMapValue(vm, 0, 1, 2);
        assert_eq!(wrenGetMapCount(vm, 0), 1);

        // Check containsKey
        assert!(wrenGetMapContainsKey(vm, 0, 1));

        // Get value
        wrenGetMapValue(vm, 0, 1, 3);
        assert_eq!(wrenGetSlotDouble(vm, 3), 99.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_handle_lifecycle() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 2);

        wrenSetSlotDouble(vm, 0, 42.0);
        let handle = wrenGetSlotHandle(vm, 0);
        assert!(!handle.is_null());

        // Use handle to restore value
        wrenSetSlotNull(vm, 0);
        wrenSetSlotHandle(vm, 0, handle);
        assert_eq!(wrenGetSlotDouble(vm, 0), 42.0);

        wrenReleaseHandle(vm, handle);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_user_data() {
        let vm = wrenNewVM(ptr::null());

        assert!(wrenGetUserData(vm).is_null());

        let data: u64 = 0xDEADBEEF;
        wrenSetUserData(vm, &data as *const u64 as *mut c_void);
        let got = wrenGetUserData(vm) as *const u64;
        assert_eq!(unsafe { *got }, 0xDEADBEEF);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_interpret_runs_code() {
        let vm = wrenNewVM(ptr::null());
        assert!(!vm.is_null());
        let vm_ref = unsafe { &mut *vm };
        vm_ref.output_buffer = Some(String::new());

        let module = CString::new("test").unwrap();
        let source = CString::new("System.print(\"hello from capi\")").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        let output = vm_ref.take_output();
        assert_eq!(output, "hello from capi\n");

        wrenFreeVM(vm);
    }

    #[test]
    fn test_interpret_compile_error() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        let source = CString::new("class {").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::CompileError);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_has_variable() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        let source = CString::new("var Foo = 42").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        let var_name = CString::new("Foo").unwrap();
        assert!(wrenHasVariable(vm, module.as_ptr(), var_name.as_ptr()));

        let missing = CString::new("Bar").unwrap();
        assert!(!wrenHasVariable(vm, module.as_ptr(), missing.as_ptr()));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_get_variable() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        let source = CString::new("var MyVal = 99").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        wrenEnsureSlots(vm, 1);
        let var_name = CString::new("MyVal").unwrap();
        wrenGetVariable(vm, module.as_ptr(), var_name.as_ptr(), 0);
        assert_eq!(wrenGetSlotDouble(vm, 0), 99.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_call_method() {
        let vm = wrenNewVM(ptr::null());
        let vm_ref = unsafe { &mut *vm };
        vm_ref.output_buffer = Some(String::new());

        // Define a class with static methods
        let module = CString::new("test").unwrap();
        let source = CString::new(
            "class Greeter {\n  static double(x) { return x * 2 }\n  static greet(name) { return \"hello \" + name }\n}",
        )
        .unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::Success);

        // Test 1: numeric static method
        wrenEnsureSlots(vm, 2);
        let class_name = CString::new("Greeter").unwrap();
        wrenGetVariable(vm, module.as_ptr(), class_name.as_ptr(), 0);
        wrenSetSlotDouble(vm, 1, 21.0);

        let sig = CString::new("double(_)").unwrap();
        let handle = wrenMakeCallHandle(vm, sig.as_ptr());
        assert!(!handle.is_null());

        let call_result = wrenCall(vm, handle);
        assert_eq!(call_result, WrenInterpretResult::Success);
        assert_eq!(wrenGetSlotDouble(vm, 0), 42.0);
        wrenReleaseHandle(vm, handle);

        // Test 2: string static method — slots must survive across calls
        wrenEnsureSlots(vm, 2);
        wrenGetVariable(vm, module.as_ptr(), class_name.as_ptr(), 0);
        let arg = CString::new("world").unwrap();
        wrenSetSlotString(vm, 1, arg.as_ptr());

        let sig2 = CString::new("greet(_)").unwrap();
        let handle2 = wrenMakeCallHandle(vm, sig2.as_ptr());
        assert!(!handle2.is_null());

        let call_result2 = wrenCall(vm, handle2);
        assert_eq!(call_result2, WrenInterpretResult::Success);

        let result_str = wrenGetSlotString(vm, 0);
        assert!(!result_str.is_null());
        let got = unsafe { CStr::from_ptr(result_str) }.to_str().unwrap();
        assert_eq!(got, "hello world");

        wrenReleaseHandle(vm, handle2);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_has_module() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("mymod").unwrap();
        let source = CString::new("var X = 1").unwrap();

        // Module doesn't exist yet
        assert!(!wrenHasModule(vm, module.as_ptr()));

        wrenInterpret(vm, module.as_ptr(), source.as_ptr());

        // Now it should exist
        assert!(wrenHasModule(vm, module.as_ptr()));

        let bogus = CString::new("nonexistent").unwrap();
        assert!(!wrenHasModule(vm, bogus.as_ptr()));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_list_set_element() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 3);

        wrenSetSlotNewList(vm, 0);
        wrenSetSlotDouble(vm, 1, 10.0);
        wrenInsertInList(vm, 0, -1, 1);
        wrenSetSlotDouble(vm, 1, 20.0);
        wrenInsertInList(vm, 0, -1, 1);
        assert_eq!(wrenGetListCount(vm, 0), 2);

        // Overwrite first element
        wrenSetSlotDouble(vm, 1, 99.0);
        wrenSetListElement(vm, 0, 0, 1);

        wrenGetListElement(vm, 0, 0, 2);
        assert_eq!(wrenGetSlotDouble(vm, 2), 99.0);

        // Second element unchanged
        wrenGetListElement(vm, 0, 1, 2);
        assert_eq!(wrenGetSlotDouble(vm, 2), 20.0);

        wrenFreeVM(vm);
    }

    #[test]
    fn test_map_remove() {
        let vm = wrenNewVM(ptr::null());
        wrenEnsureSlots(vm, 4);

        wrenSetSlotNewMap(vm, 0);

        let key = CString::new("k").unwrap();
        wrenSetSlotString(vm, 1, key.as_ptr());
        wrenSetSlotDouble(vm, 2, 7.0);
        wrenSetMapValue(vm, 0, 1, 2);
        assert_eq!(wrenGetMapCount(vm, 0), 1);

        // Remove by key
        wrenRemoveMapValue(vm, 0, 1, 3);
        assert_eq!(wrenGetMapCount(vm, 0), 0);
        assert_eq!(wrenGetSlotDouble(vm, 3), 7.0); // removed value returned

        wrenFreeVM(vm);
    }

    #[test]
    fn test_interpret_runtime_error() {
        let vm = wrenNewVM(ptr::null());
        let module = CString::new("test").unwrap();
        // Fiber.abort triggers a runtime error
        let source = CString::new("Fiber.abort(\"boom\")").unwrap();
        let result = wrenInterpret(vm, module.as_ptr(), source.as_ptr());
        assert_eq!(result, WrenInterpretResult::RuntimeError);
        wrenFreeVM(vm);
    }

    #[test]
    fn test_multiple_modules() {
        let vm = wrenNewVM(ptr::null());
        let vm_ref = unsafe { &mut *vm };
        vm_ref.output_buffer = Some(String::new());

        // First module
        let mod_a = CString::new("mod_a").unwrap();
        let src_a = CString::new("var A = 100").unwrap();
        assert_eq!(
            wrenInterpret(vm, mod_a.as_ptr(), src_a.as_ptr()),
            WrenInterpretResult::Success
        );

        // Second module
        let mod_b = CString::new("mod_b").unwrap();
        let src_b = CString::new("var B = 200").unwrap();
        assert_eq!(
            wrenInterpret(vm, mod_b.as_ptr(), src_b.as_ptr()),
            WrenInterpretResult::Success
        );

        // Both modules exist
        assert!(wrenHasModule(vm, mod_a.as_ptr()));
        assert!(wrenHasModule(vm, mod_b.as_ptr()));

        // Variables are isolated per module
        wrenEnsureSlots(vm, 1);
        let var_a = CString::new("A").unwrap();
        let var_b = CString::new("B").unwrap();
        wrenGetVariable(vm, mod_a.as_ptr(), var_a.as_ptr(), 0);
        assert_eq!(wrenGetSlotDouble(vm, 0), 100.0);
        wrenGetVariable(vm, mod_b.as_ptr(), var_b.as_ptr(), 0);
        assert_eq!(wrenGetSlotDouble(vm, 0), 200.0);

        // Cross-module: mod_a doesn't have B
        assert!(!wrenHasVariable(vm, mod_a.as_ptr(), var_b.as_ptr()));

        wrenFreeVM(vm);
    }

    #[test]
    fn test_null_safety() {
        // All functions should handle null pointers gracefully
        assert_eq!(
            wrenInterpret(ptr::null_mut(), ptr::null(), ptr::null()),
            WrenInterpretResult::RuntimeError
        );
        assert!(wrenMakeCallHandle(ptr::null_mut(), ptr::null()).is_null());
        assert_eq!(
            wrenCall(ptr::null_mut(), ptr::null_mut()),
            WrenInterpretResult::RuntimeError
        );
        wrenFreeVM(ptr::null_mut()); // should not crash
        wrenCollectGarbage(ptr::null_mut()); // should not crash
        assert!(!wrenHasModule(ptr::null_mut(), ptr::null()));
        assert!(!wrenHasVariable(ptr::null_mut(), ptr::null(), ptr::null()));
        // Hatch-loader surface: every function returns -1 on null
        // rather than dereferencing, so the bindings are safe to
        // call from a freshly-zeroed C config.
        assert_eq!(wrenInstallHatchBytes(ptr::null_mut(), ptr::null(), 0), -1);
        assert_eq!(wrenInstallHatchFile(ptr::null_mut(), ptr::null()), -1);
        assert_eq!(wrenAddNativeSearchPath(ptr::null_mut(), ptr::null()), -1);
        assert_eq!(wrenInstallHatchDir(ptr::null_mut(), ptr::null()), -1);
    }
}
