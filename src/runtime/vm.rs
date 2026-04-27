/// The Wren virtual machine.
///
/// Owns all runtime state: GC heap, interner, core classes, fibers,
/// module registry, and configuration callbacks.
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Hot reload — SIGUSR1-driven reload-pending flag.
// ---------------------------------------------------------------------------

/// Set asynchronously by the SIGUSR1 handler when an external watcher
/// (typically the `hatch` CLI) signals that one or more user modules
/// have changed on disk. Drained at interpreter safepoints by
/// [`VM::check_pending_reload`].
static RELOAD_PENDING: AtomicBool = AtomicBool::new(false);

extern "C" fn reload_signal_handler(_sig: libc::c_int) {
    // Async-signal-safe: AtomicBool::store with Release ordering is
    // documented to be lock-free on every platform we run on, so
    // setting a flag is safe inside a signal handler. No allocation,
    // no syscalls beyond the implicit memory barrier.
    RELOAD_PENDING.store(true, Ordering::Release);
}

/// Install the SIGUSR1 handler that flips [`RELOAD_PENDING`].
///
/// Idempotent — calling twice rebinds the same handler harmlessly.
/// Intended to be invoked by the `wlift` binary when a `--watch`
/// flag (or `WLIFT_WATCH=1` env var) is present, so the parent
/// `hatch` CLI can drive in-process reload via `kill(pid, SIGUSR1)`.
pub fn install_reload_signal_handler() {
    unsafe {
        libc::signal(
            libc::SIGUSR1,
            reload_signal_handler as *const () as usize as libc::sighandler_t,
        );
    }
}

/// True when the signal handler has flagged a pending reload pass.
/// Used by integration tests / the CLI to assert delivery.
pub fn reload_pending() -> bool {
    RELOAD_PENDING.load(Ordering::Acquire)
}

/// Stat-and-extract the file mtime as fractional seconds since the
/// Unix epoch. Returns `None` for missing files or non-Unix-epoch
/// stat results. Used to baseline modules at install time and decide
/// which ones to reload on a SIGUSR1 pass.
fn file_mtime_secs(path: &str) -> Option<f64> {
    let m = std::fs::metadata(path).ok()?;
    let mt = m.modified().ok()?;
    mt.duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs_f64())
}

use super::engine::{ExecutionEngine, ExecutionMode, InterpretResult};
use super::gc_trait::{GcImpl, GcStrategy};
use super::object::*;
use super::value::Value;
use crate::intern::{Interner, SymbolId};

/// Action requested by a fiber native method, handled by the interpreter loop.
#[derive(Debug)]
pub enum FiberAction {
    /// Switch to target fiber, pass value when it yields/completes.
    Call { target: *mut ObjFiber, value: Value },
    /// Yield from current fiber, return value to caller.
    Yield { value: Value },
    /// Transfer to target fiber (no caller chain).
    Transfer { target: *mut ObjFiber, value: Value },
    /// Suspend the current fiber (no caller to return to).
    Suspend,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Callback for System.print output.
pub type WriteFn = Box<dyn Fn(&str)>;

/// Callback for error reporting.
pub type ErrorFn = Box<dyn Fn(ErrorKind, &str, i32, &str)>;

/// Callback to resolve a module name relative to an importer.
pub type ResolveModuleFn = Box<dyn Fn(&str, &str) -> Option<String>>;

/// Callback to load a module's source code.
///
/// First argument is the requested module name (`./foo`, `@hatch:bar`,
/// `baz`); second is the NAME of the module doing the import, so the
/// loader can resolve relative paths (`./css`) against the importer's
/// on-disk directory instead of the top-level entry file's directory.
/// The second argument is empty for the root file.
pub type LoadModuleFn = Box<dyn Fn(&str, &str) -> Option<String>>;

/// Callback to bind a foreign method.
pub type BindForeignMethodFn = Box<dyn Fn(&str, &str, bool, &str) -> Option<NativeFn>>;

/// Callback to bind a foreign class (allocate + optional finalize).
pub type BindForeignClassFn = Box<dyn Fn(&str, &str) -> Option<ForeignClassMethods>>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErrorKind {
    Compile,
    Runtime,
    StackTrace,
}

/// Foreign class method pair.
pub struct ForeignClassMethods {
    pub allocate: NativeFn,
    pub finalize: Option<fn(*mut u8)>,
}

/// VM configuration. Set before creating the VM.
pub struct VMConfig {
    pub write_fn: Option<WriteFn>,
    pub error_fn: Option<ErrorFn>,
    pub resolve_module_fn: Option<ResolveModuleFn>,
    pub load_module_fn: Option<LoadModuleFn>,
    pub bind_foreign_method_fn: Option<BindForeignMethodFn>,
    pub bind_foreign_class_fn: Option<BindForeignClassFn>,
    pub initial_heap_size: usize,
    pub min_heap_size: usize,
    pub heap_growth_percent: u32,
    /// Execution mode: Interpreter, Tiered (default), or Jit.
    pub execution_mode: ExecutionMode,
    /// Call count threshold before baseline native compilation in Tiered mode.
    pub jit_threshold: u32,
    /// Baseline-native call count threshold before optimize tier-up.
    pub opt_threshold: u32,
    /// Enable fiber stack trace tracking (opt-in, not standard Wren behavior).
    /// When enabled, cross-fiber stack traces include spawn sites and caller chains.
    pub fiber_stack_traces: bool,
    /// Maximum interpreter steps before aborting (default: 1B).
    /// In tiered/JIT mode, JIT-compiled code doesn't count steps so a
    /// higher limit (10B) is recommended for long-running programs.
    pub step_limit: usize,
    /// Maximum call frame depth before aborting (default: 1024).
    pub max_call_depth: usize,
    /// GC strategy to use.
    pub gc_strategy: GcStrategy,
}

impl Default for VMConfig {
    fn default() -> Self {
        Self {
            write_fn: None,
            error_fn: None,
            resolve_module_fn: None,
            load_module_fn: None,
            bind_foreign_method_fn: None,
            bind_foreign_class_fn: None,
            initial_heap_size: 10 * 1024 * 1024,
            min_heap_size: 1024 * 1024,
            heap_growth_percent: 50,
            execution_mode: ExecutionMode::default(),
            jit_threshold: 100,
            opt_threshold: 1000,
            fiber_stack_traces: false,
            step_limit: 1_000_000_000,
            max_call_depth: 1024,
            gc_strategy: GcStrategy::MarkSweep,
        }
    }
}

// ---------------------------------------------------------------------------
// Handle (persistent reference, prevents GC collection)
// ---------------------------------------------------------------------------

/// A persistent handle to a Wren value, preventing GC collection.
pub struct WrenHandle {
    pub value: Value,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HotMethodSymbols {
    pub list_subscript: Option<SymbolId>,
    pub list_subscript_set: Option<SymbolId>,
    pub list_add: Option<SymbolId>,
    pub list_count: Option<SymbolId>,
    pub list_iterate: Option<SymbolId>,
    pub list_iterator_value: Option<SymbolId>,
    pub list_remove_at: Option<SymbolId>,
}

// ---------------------------------------------------------------------------
// VM
// ---------------------------------------------------------------------------

pub struct VM {
    // -- Memory --
    pub gc: GcImpl,
    pub interner: Interner,

    // -- Core classes --
    pub object_class: *mut ObjClass,
    pub class_class: *mut ObjClass,
    pub bool_class: *mut ObjClass,
    pub num_class: *mut ObjClass,
    pub string_class: *mut ObjClass,
    pub list_class: *mut ObjClass,
    pub map_class: *mut ObjClass,
    pub range_class: *mut ObjClass,
    pub null_class: *mut ObjClass,
    pub fn_class: *mut ObjClass,
    pub fiber_class: *mut ObjClass,
    pub system_class: *mut ObjClass,
    pub sequence_class: *mut ObjClass,

    // -- Wrapper / helper classes --
    pub map_sequence_class: *mut ObjClass,
    pub skip_sequence_class: *mut ObjClass,
    pub take_sequence_class: *mut ObjClass,
    pub where_sequence_class: *mut ObjClass,
    pub string_byte_seq_class: *mut ObjClass,
    pub string_code_point_seq_class: *mut ObjClass,
    pub map_entry_class: *mut ObjClass,

    // Typed numeric buffers. Placed at the end of the class-field
    // block so adding them doesn't shift offsets of pre-existing
    // fields that out-of-tree plugins (e.g. @hatch:sqlite) were
    // compiled against.
    pub byte_array_class: *mut ObjClass,
    pub float32_array_class: *mut ObjClass,
    pub float64_array_class: *mut ObjClass,

    // -- Execution state --
    pub fiber: *mut ObjFiber,
    pub modules: HashMap<String, *mut ObjModule>,

    // -- Execution engine (tiered runtime) --
    pub engine: ExecutionEngine,

    // -- API --
    pub api_stack: Vec<Value>,
    pub handles: Vec<WrenHandle>,
    pub user_data: *mut c_void,

    // -- Configuration --
    pub config: VMConfig,

    // -- Error state (set by primitives) --
    pub has_error: bool,

    // -- Output capture (for testing; None = print to stdout) --
    pub output_buffer: Option<String>,

    // -- Fiber switching (set by fiber natives, consumed by interpreter) --
    pub pending_fiber_action: Option<FiberAction>,

    // -- Source code storage (for runtime error reporting with ariadne) --
    pub module_sources: HashMap<String, String>,

    // -- Post-mortem error fiber (saved before restoring prev_fiber) --
    pub error_fiber: *mut ObjFiber,

    /// Last error message from runtime_error (for Fiber.try to retrieve).
    pub last_error: Option<String>,

    /// Modules currently being loaded (for circular import detection).
    loading_modules: HashSet<String>,

    /// Flag set by System.gc() — actual collection happens at next safepoint.
    pub gc_requested: bool,

    /// Pool of reusable register files to avoid per-call heap allocation.
    pub register_pool: Vec<Vec<Value>>,

    /// Pool of reusable temporary fibers for synchronous closure/constructor calls.
    pub sync_fiber_pool: Vec<*mut ObjFiber>,

    /// Global inline method cache for fast monomorphic dispatch.
    pub method_cache: super::vm_interp::MethodCache,

    /// Bitset: `call_sym_flags[sym.index()]` is true for call()/call(_)/... symbols.
    /// Used for O(1) "is this a closure call?" check in the dispatch hot path.
    pub call_sym_flags: Vec<bool>,

    /// Stable core-library symbols used by hot native dispatch fast paths.
    pub hot_method_symbols: HotMethodSymbols,

    /// Dynamic libraries opened on behalf of `foreign` classes via
    /// `#!native`. Kept alive for the VM's lifetime so the function
    /// pointers bound into method tables remain valid.
    pub native_libs: Vec<libloading::Library>,

    /// Extra filesystem directories to search when resolving a
    /// `#!native = "..."` library reference, tried ahead of the OS
    /// loader's ambient search. Populated from a hatchfile's
    /// `native_search_paths` key (or CLI flags in the future).
    pub native_search_paths: Vec<std::path::PathBuf>,

    /// Per-name overrides for `#!native = "key"`. A hit in this map
    /// wins over candidate-filename mangling: the loader uses the
    /// mapped path verbatim. Populated from a hatchfile's
    /// `[native_libs]` section and from `SectionKind::NativeLib`
    /// entries extracted out of a `.hatch` bundle.
    pub native_lib_paths: std::collections::HashMap<String, std::path::PathBuf>,

    /// Temp directories holding bundled native libs extracted from
    /// `.hatch` `NativeLib` sections. Held so the directories (and the
    /// `.dylib` / `.so` files inside) survive as long as the VM does.
    /// Dropped automatically at VM teardown, which reclaims disk.
    native_temp_dirs: Vec<tempfile::TempDir>,

    /// Per-class field layouts, keyed by class name. Value is the
    /// ordered list of field names (`_foo`, `_bar`, ...), including
    /// inherited fields, where the slot index equals the list index.
    /// The compiler consults this when lowering a subclass whose
    /// parent lives in a different module so inherited field slots
    /// line up — without it, the subclass would start its own fields
    /// at slot 0 and clobber the parent's state.
    pub field_layouts: HashMap<String, Vec<String>>,

    /// During `reload_module`, maps class-name → existing `ObjClass`
    /// pointer so the re-install loop can mutate the old class in
    /// place (new methods, updated num_fields) instead of allocating
    /// a fresh one. Keeping identity means instances created before
    /// the reload continue to dispatch against up-to-date methods.
    /// `None` outside of a reload.
    reload_class_table: Option<HashMap<String, *mut ObjClass>>,

    /// Last-seen mtime per absolute-path module, captured at install
    /// time and updated on each reload. Used by
    /// [`Self::check_pending_reload`] to decide which modules a
    /// SIGUSR1 should reload — only modules whose on-disk mtime has
    /// advanced past their captured baseline. Empty until at least
    /// one user module is installed.
    module_mtimes: HashMap<String, f64>,

    /// Wren callbacks registered via `Hatch.onReload(fn)`. Each entry
    /// is an `ObjClosure` value invoked with the just-reloaded
    /// module's canonical name after every successful reload pass.
    /// Traced as a GC root so the closures don't get collected
    /// between registration and the first reload that fires them.
    reload_callbacks: Vec<Value>,

    /// Callbacks registered via `Hatch.beforeReload(fn)`. Fired with
    /// the module's canonical name BEFORE the reload's `interpret`
    /// call re-runs top-level. Frameworks use this to drop state
    /// owned by the about-to-rerun module — e.g. `App` clears its
    /// route table here so direct `app.get/post(...)` calls at
    /// module top-level can re-register cleanly without
    /// accumulating duplicate handlers.
    before_reload_callbacks: Vec<Value>,

    /// File-watch entries registered via `Hatch.watchFile(path, fn)`.
    /// Each is a `(path, callback, last_mtime_secs)` triple; on
    /// every SIGUSR1 pass the runtime stats the path and, if the
    /// mtime advanced, invokes the callback with the path string.
    /// Callbacks held as GC roots for the same reason as the
    /// module reload callbacks.
    file_watches: Vec<FileWatch>,
}

/// One entry in `vm.file_watches`. Held as plain fields so the GC
/// can walk just the `callback` Value, leaving the path / mtime as
/// inert metadata.
struct FileWatch {
    path: String,
    callback: Value,
    last_mtime: f64,
}

impl VM {
    /// Create a new VM with the given configuration.
    pub fn new(config: VMConfig) -> Self {
        // Reset thread-local JIT state to avoid stale data from previous VM instances
        // (critical for test isolation when multiple VMs are created in the same process).
        crate::codegen::runtime_fns::set_jit_context(
            crate::codegen::runtime_fns::JitContext::default(),
        );
        crate::codegen::runtime_fns::clear_jit_roots();

        let mut vm = Self {
            gc: GcImpl::new(config.gc_strategy),
            interner: Interner::new(),

            object_class: ptr::null_mut(),
            class_class: ptr::null_mut(),
            bool_class: ptr::null_mut(),
            num_class: ptr::null_mut(),
            string_class: ptr::null_mut(),
            list_class: ptr::null_mut(),
            map_class: ptr::null_mut(),
            range_class: ptr::null_mut(),
            null_class: ptr::null_mut(),
            fn_class: ptr::null_mut(),
            fiber_class: ptr::null_mut(),
            system_class: ptr::null_mut(),
            sequence_class: ptr::null_mut(),

            map_sequence_class: ptr::null_mut(),
            skip_sequence_class: ptr::null_mut(),
            take_sequence_class: ptr::null_mut(),
            where_sequence_class: ptr::null_mut(),
            string_byte_seq_class: ptr::null_mut(),
            string_code_point_seq_class: ptr::null_mut(),
            map_entry_class: ptr::null_mut(),

            byte_array_class: ptr::null_mut(),
            float32_array_class: ptr::null_mut(),
            float64_array_class: ptr::null_mut(),

            fiber: ptr::null_mut(),
            modules: HashMap::new(),

            engine: {
                let mut e = ExecutionEngine::new(config.execution_mode);
                e.jit_threshold = config.jit_threshold;
                e.opt_threshold = config.opt_threshold;
                e
            },

            api_stack: vec![Value::null(); 16],
            handles: Vec::new(),
            user_data: ptr::null_mut(),

            config,
            has_error: false,
            output_buffer: None,
            pending_fiber_action: None,
            module_sources: HashMap::new(),
            error_fiber: ptr::null_mut(),
            last_error: None,
            loading_modules: HashSet::new(),
            gc_requested: false,
            register_pool: Vec::new(),
            sync_fiber_pool: Vec::new(),
            method_cache: super::vm_interp::MethodCache::new(),
            call_sym_flags: Vec::new(),
            hot_method_symbols: HotMethodSymbols::default(),
            native_libs: Vec::new(),
            native_search_paths: Vec::new(),
            native_lib_paths: HashMap::new(),
            native_temp_dirs: Vec::new(),
            field_layouts: HashMap::new(),
            reload_class_table: None,
            module_mtimes: HashMap::new(),
            reload_callbacks: Vec::new(),
            before_reload_callbacks: Vec::new(),
            file_watches: Vec::new(),
        };

        // Bootstrap core classes.
        super::core::initialize(&mut vm);
        vm.refresh_hot_method_symbols();

        vm
    }

    /// Create a new VM with default configuration.
    pub fn new_default() -> Self {
        Self::new(VMConfig::default())
    }

    /// Compile Wren source to a portable `.wlbc` bytecode cache.
    ///
    /// Runs the full front-end pipeline (lex → parse → sema → lower →
    /// optimize) once and hands the resulting MIR + interner off to
    /// `serialize::emit`. Produces bytes suitable for writing to a
    /// `.wlbc` file and later loading via `interpret_bytecode`.
    ///
    /// Reports all compile-time diagnostics to stderr the same way
    /// `interpret` does, returning `InterpretResult::CompileError` on
    /// any parse / sema failure.
    pub fn compile_source_to_blob(&mut self, source: &str) -> Result<Vec<u8>, InterpretResult> {
        use crate::diagnostics::Severity;
        use crate::mir::opt::{
            self, constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize, licm::Licm,
            sra::Sra, MirPass,
        };
        use crate::parse::parser;
        use crate::sema;

        // 1. Parse
        let parse_result = parser::parse(source);
        if parse_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &parse_result.errors {
                err.eprint(source);
            }
            return Err(InterpretResult::CompileError);
        }

        let mut interner = parse_result.interner;

        // 2. Prelude + sema
        let core_names = [
            "Object",
            "Class",
            "Bool",
            "Num",
            "String",
            "List",
            "Map",
            "Range",
            "Null",
            "Fn",
            "Fiber",
            "System",
            "Sequence",
            "ByteArray",
            "Float32Array",
            "Float64Array",
        ];
        let prelude: Vec<crate::intern::SymbolId> =
            core_names.iter().map(|n| interner.intern(n)).collect();
        let resolve_result =
            sema::resolve::resolve_with_prelude(&parse_result.module, &interner, &prelude);
        if resolve_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &resolve_result.errors {
                err.eprint(source);
            }
            return Err(InterpretResult::CompileError);
        }

        // .wlbc is a single-module artifact by design: imports are
        // AST-level and get resolved into `GetModuleVar` refs during
        // MIR build. The loader satisfies those refs against whatever
        // modules are already installed in the VM — either from a
        // prior `interpret`, a previous `interpret_bytecode`, or (the
        // canonical case) earlier modules in the same `.hatch` that
        // this wlbc is a section of. No diagnostic here: `.hatch` and
        // `hatch-cli` own cross-module concerns; the wlbc emit path
        // just produces a per-module blob.
        let _ = &parse_result.module;

        // 3. Lower to MIR
        let (mut module_mir, new_layouts) = crate::mir::builder::lower_module_with_known_classes(
            &parse_result.module,
            &mut interner,
            &resolve_result,
            &self.field_layouts,
        );
        for (name, layout) in new_layouts {
            self.field_layouts.insert(name, layout);
        }

        // 4. Optimize top-level + method + closure bodies
        let constfold = ConstFold;
        let dce = Dce;
        let cse = Cse::default();
        let type_spec = TypeSpecialize::with_math(&interner);
        let licm = Licm;
        let sra = Sra;
        let passes: Vec<&dyn MirPass> = vec![
            &constfold, &dce, &cse, &type_spec, &constfold, &dce, &licm, &sra, &dce,
        ];
        opt::run_to_fixpoint(&mut module_mir.top_level, &passes, 10);
        for class in &mut module_mir.classes {
            for method in &mut class.methods {
                opt::run_to_fixpoint(&mut method.mir, &passes, 10);
            }
        }
        for closure in &mut module_mir.closures {
            opt::run_to_fixpoint(closure, &passes, 10);
        }

        // 5. Project module var symbols to strings for the snapshot.
        let var_names: Vec<String> = resolve_result
            .module_vars
            .iter()
            .map(|&sym| interner.resolve(sym).to_string())
            .collect();

        // 6. Serialize.
        crate::serialize::emit(&interner, &module_mir, &var_names).map_err(|e| {
            crate::diagnostics::Diagnostic::error(format!("failed to emit .wlbc: {}", e))
                .eprint_no_source();
            InterpretResult::CompileError
        })
    }

    /// Load and execute a module from a `.wlbc` bytecode cache.
    ///
    /// Skips parse / sema / MIR-build / optimize — goes straight to the
    /// shared install path. The snapshot's interner is merged into the
    /// VM interner, closures and classes are registered, and the
    /// top-level fiber runs. Same semantics as `interpret(name, src)`
    /// modulo anything that depended on having source text (e.g. span
    /// diagnostics carry snapshot spans, not fresh source byte ranges).
    pub fn interpret_bytecode(&mut self, module_name: &str, bytes: &[u8]) -> InterpretResult {
        let module_key = module_name.to_string();
        if !self.loading_modules.insert(module_key.clone()) {
            self.report_error(&format!(
                "Circular import detected: module '{}' is already being loaded",
                module_name
            ));
            return InterpretResult::CompileError;
        }

        let blob = match crate::serialize::load(bytes) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("failed to load .wlbc: {}", e);
                self.loading_modules.remove(&module_key);
                return InterpretResult::CompileError;
            }
        };
        self.install_module_mir_and_run(module_name, &blob.interner, blob.module, blob.var_names)
    }

    /// Install every module carried by a `.hatch` — same as
    /// [`Self::interpret_hatch`] but does not require or enforce an
    /// entry module, and still succeeds if the manifest's entry is
    /// missing. Used to preload "library" hatches whose only purpose
    /// is to register classes that a subsequent application hatch
    /// imports.
    ///
    /// Each installed module's top-level runs, the same way a source-
    /// path `import` does. Returns `RuntimeError` if any module's
    /// top-level raises.
    pub fn install_hatch_modules(&mut self, hatch_bytes: &[u8]) -> InterpretResult {
        let hatch = match crate::hatch::load(hatch_bytes) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("failed to load hatch: {}", e);
                return InterpretResult::CompileError;
            }
        };
        self.install_hatch_sections(&hatch)
    }

    /// Fold a hatch manifest's native-library declarations into the
    /// VM's foreign-loader state. Search paths accumulate across
    /// hatches; explicit `[native_libs]` entries with the same key
    /// overwrite (last writer wins) so a consuming hatch can override
    /// a dependency's defaults.
    fn apply_hatch_native_manifest(&mut self, manifest: &crate::hatch::Manifest) {
        for raw in &manifest.native_search_paths {
            let path = std::path::PathBuf::from(raw);
            if !self.native_search_paths.contains(&path) {
                self.native_search_paths.push(path);
            }
        }
        for (name, entry) in &manifest.native_libs {
            if let Some(path) = entry.resolve() {
                self.native_lib_paths
                    .insert(name.clone(), std::path::PathBuf::from(path));
            }
        }
    }

    /// Same as `apply_hatch_native_manifest` but resolves relative
    /// paths against `root` (typically the hatchfile's directory).
    /// Used by spec runs where the file layout on disk — not the
    /// already-extracted temp dir from a `.hatch` bundle — is the
    /// source of truth for native libraries.
    pub fn apply_hatch_native_manifest_rooted(
        &mut self,
        manifest: &crate::hatch::Manifest,
        root: &std::path::Path,
    ) {
        let resolve = |raw: &str| -> std::path::PathBuf {
            let p = std::path::PathBuf::from(raw);
            if p.is_absolute() {
                p
            } else {
                root.join(p)
            }
        };
        for raw in &manifest.native_search_paths {
            let path = resolve(raw);
            if !self.native_search_paths.contains(&path) {
                self.native_search_paths.push(path);
            }
        }
        for (name, entry) in &manifest.native_libs {
            if let Some(raw) = entry.resolve() {
                self.native_lib_paths.insert(name.clone(), resolve(raw));
            }
        }
    }

    /// Extract every `NativeLib` section to a per-hatch temp directory
    /// and register the resulting paths in `native_lib_paths` so
    /// `#!native = "<section name>"` resolves to the extracted file.
    /// The temp directory is prepended to `native_search_paths` so
    /// bare-name lookups without an explicit override also find the
    /// bundled library. Returns `CompileError` on any I/O failure —
    /// partial extraction would leave the hatch in an inconsistent
    /// state.
    fn extract_hatch_native_sections(&mut self, hatch: &crate::hatch::Hatch) -> InterpretResult {
        let has_native = hatch
            .sections
            .iter()
            .any(|s| matches!(s.kind, crate::hatch::SectionKind::NativeLib));
        if !has_native {
            return InterpretResult::Success;
        }

        let temp_dir = match tempfile::Builder::new()
            .prefix(&format!("wrenlift-{}-", hatch.manifest.name))
            .tempdir()
        {
            Ok(d) => d,
            Err(e) => {
                eprintln!("failed to create temp dir for bundled native libs: {}", e);
                return InterpretResult::CompileError;
            }
        };
        let dir_path = temp_dir.path().to_path_buf();

        for section in &hatch.sections {
            if !matches!(section.kind, crate::hatch::SectionKind::NativeLib) {
                continue;
            }
            let filename = crate::runtime::foreign::default_native_lib_filename(&section.name);
            let full = dir_path.join(&filename);
            if let Err(e) = std::fs::write(&full, &section.data) {
                eprintln!(
                    "failed to write bundled native lib '{}' to {}: {}",
                    section.name,
                    full.display(),
                    e
                );
                return InterpretResult::CompileError;
            }
            // Per-name override wins over candidate mangling, so
            // `#!native = "<section name>"` maps directly to the
            // extracted file.
            self.native_lib_paths.insert(section.name.clone(), full);
        }

        // Prepend the extraction dir so bundled libs are reachable by
        // bare name / candidate search even without an explicit
        // override (e.g. a dependency referencing the same lib).
        self.native_search_paths.insert(0, dir_path);
        self.native_temp_dirs.push(temp_dir);
        InterpretResult::Success
    }

    /// Shared body: install every `Wlbc` section named in
    /// `manifest.modules`. Applies the manifest's native-library
    /// declarations (`[native_libs]` overrides + `native_search_paths`)
    /// and extracts any bundled `NativeLib` sections to a temp
    /// directory so `#!native` directives inside the hatch resolve
    /// against both sources at class install time.
    fn install_hatch_sections(&mut self, hatch: &crate::hatch::Hatch) -> InterpretResult {
        self.apply_hatch_native_manifest(&hatch.manifest);
        let result = self.extract_hatch_native_sections(hatch);
        if !matches!(result, InterpretResult::Success) {
            return result;
        }

        // Source sections carry the original .wren text alongside the
        // compiled wlbc. Stash each into `module_sources` so a runtime
        // error raised inside this module renders through ariadne's
        // labelled-span path rather than the bare-prose fallback.
        for section in &hatch.sections {
            if matches!(section.kind, crate::hatch::SectionKind::Source) {
                if let Ok(text) = std::str::from_utf8(&section.data) {
                    self.module_sources
                        .insert(section.name.clone(), text.to_string());
                }
            }
        }

        for module_name in &hatch.manifest.modules {
            // Skip modules already loaded (e.g. by a previously
            // installed hatch) so `--link A --link B` where both carry
            // an overlapping transitive dep doesn't re-run it.
            if self.engine.modules.contains_key(module_name) {
                continue;
            }

            let section = match hatch.sections.iter().find(|s| {
                matches!(s.kind, crate::hatch::SectionKind::Wlbc) && &s.name == module_name
            }) {
                Some(s) => s,
                None => {
                    eprintln!(
                        "hatch manifest lists module '{}' but no wlbc section carries it",
                        module_name
                    );
                    return InterpretResult::CompileError;
                }
            };

            let result = match crate::serialize::load(&section.data) {
                Ok(blob) => self.install_module_mir_and_run(
                    module_name,
                    &blob.interner,
                    blob.module,
                    blob.var_names,
                ),
                Err(e) => {
                    // A `decode:` failure here usually means the
                    // artifact was built against an older wren_lift
                    // whose VERSION constant didn't get bumped before
                    // the format drifted. Surface a hint pointing at
                    // the rebuild remediation alongside the raw error.
                    let mut diag = crate::diagnostics::Diagnostic::error(format!(
                        "hatch module '{}' failed to load its wlbc payload: {}",
                        module_name, e
                    ));
                    if matches!(e, crate::serialize::SerializeError::Decode(_)) {
                        diag = diag.with_note(
                            "this is usually a stale artifact — rebuild it with `hatch build` \
                             against the current wren_lift sources",
                        );
                    }
                    diag.eprint_no_source();
                    return InterpretResult::CompileError;
                }
            };
            if !matches!(result, InterpretResult::Success) {
                return result;
            }
        }
        InterpretResult::Success
    }

    /// Install every module carried by a `.hatch` package, then run
    /// the manifest's entry module. Modules are installed in
    /// `manifest.modules` order, so callers should write that list in
    /// dependency order (imports must be available by the time a
    /// module's `install_module_mir_and_run` fires).
    ///
    /// Rejects any hatch that advertises native-library sections
    /// today — those land in commit 3b. Resource sections are ignored
    /// by this loader; a future `wlift.resource(...)` API will surface
    /// them.
    pub fn interpret_hatch(&mut self, hatch_bytes: &[u8]) -> InterpretResult {
        let hatch = match crate::hatch::load(hatch_bytes) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("failed to load hatch: {}", e);
                return InterpretResult::CompileError;
            }
        };

        // Entry module gets no second run; its top-level executes as
        // part of the install loop. `manifest.entry` is effectively an
        // assertion that the named module exists in the hatch.
        if !hatch.manifest.modules.contains(&hatch.manifest.entry) {
            eprintln!(
                "hatch entry '{}' is not listed in manifest.modules",
                hatch.manifest.entry
            );
            return InterpretResult::CompileError;
        }

        self.install_hatch_sections(&hatch)
    }

    /// Compile and execute Wren source code.
    ///
    /// Pipeline: lex → parse → sema → lower to MIR → optimize → execute.
    /// Execution happens inside a Fiber context via run_fiber().
    pub fn interpret(&mut self, module_name: &str, source: &str) -> InterpretResult {
        use crate::diagnostics::Severity;
        use crate::mir::opt::{
            self, constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize, licm::Licm,
            sra::Sra, MirPass,
        };
        use crate::parse::parser;
        use crate::sema;
        // 0. Cycle detection for imports
        let module_key = module_name.to_string();
        if !self.loading_modules.insert(module_key.clone()) {
            self.report_error(&format!(
                "Circular import detected: module '{}' is already being loaded",
                module_name
            ));
            return InterpretResult::CompileError;
        }

        // 0b. Store source for runtime error reporting
        self.module_sources
            .insert(module_name.to_string(), source.to_string());

        // 1. Parse
        let parse_result = parser::parse(source);
        if parse_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &parse_result.errors {
                err.eprint(source);
            }
            self.loading_modules.remove(&module_key);
            return InterpretResult::CompileError;
        }

        // 2. Use the parse interner, merge afterward
        let mut interner = parse_result.interner;

        // 3. Semantic analysis — register core class names as prelude
        let core_names = [
            "Object",
            "Class",
            "Bool",
            "Num",
            "String",
            "List",
            "Map",
            "Range",
            "Null",
            "Fn",
            "Fiber",
            "System",
            "Sequence",
            "ByteArray",
            "Float32Array",
            "Float64Array",
        ];
        let prelude: Vec<crate::intern::SymbolId> =
            core_names.iter().map(|n| interner.intern(n)).collect();
        let resolve_result =
            sema::resolve::resolve_with_prelude(&parse_result.module, &interner, &prelude);
        if resolve_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &resolve_result.errors {
                err.eprint(source);
            }
            self.loading_modules.remove(&module_key);
            return InterpretResult::CompileError;
        }

        // 3b. Process imports — recursively interpret imported modules
        for stmt in &parse_result.module {
            if let crate::ast::Stmt::Import {
                module: mod_path,
                names: _,
            } = &stmt.0
            {
                let requested = mod_path.0.clone();

                // Resolve the requested name to its canonical form.
                // For relative paths (`./foo`, `../foo`) the loader's
                // resolver returns the absolute on-disk path so that
                // `./foo` and `../foo` referring to the same file
                // dedupe to a single ModuleEntry — without this the
                // engine registers two modules for one file and
                // class-identity (`is`) checks fail across the
                // boundary. Scoped names (`@hatch:foo`) and bare
                // builtins resolve to themselves.
                let importer = module_name.to_string();
                let canonical = self
                    .config
                    .resolve_module_fn
                    .as_ref()
                    .and_then(|fn_| fn_(&requested, &importer))
                    .unwrap_or_else(|| requested.clone());

                // Skip if already loaded
                if !self.engine.modules.contains_key(&canonical) {
                    // Check for built-in optional modules first
                    if self.try_load_builtin_module(&canonical) {
                        // Built-in module registered successfully
                    } else {
                        // Resolve module source via config callback.
                        // Pass the canonical name so the loader's
                        // `dirs` registry stays keyed by canonical
                        // — subsequent imports from inside this
                        // module use the canonical name as `from`,
                        // which has to match what we recorded here.
                        let source_opt = self
                            .config
                            .load_module_fn
                            .as_ref()
                            .and_then(|load_fn| load_fn(&canonical, &importer));
                        if let Some(mod_source) = source_opt {
                            let result = self.interpret(&canonical, &mod_source);
                            if result != InterpretResult::Success {
                                self.loading_modules.remove(&module_key);
                                return result;
                            }
                        } else {
                            self.report_error(&format!("Could not load module '{}'", requested));
                            self.loading_modules.remove(&module_key);
                            return InterpretResult::CompileError;
                        }
                    }
                }

                // Copy imported names from the imported module's vars to this module's
                // resolve_result.module_vars. We'll look them up by name after MIR
                // compilation when building the module var storage (step 8).
                // For now, just ensure the imported module exists.
            }
        }

        // 4. Lower to MIR. Thread `field_layouts` through so subclasses
        // whose parent lives in a previously-loaded module inherit the
        // right slot offsets.
        let (mut module_mir, new_layouts) = crate::mir::builder::lower_module_with_known_classes(
            &parse_result.module,
            &mut interner,
            &resolve_result,
            &self.field_layouts,
        );
        for (name, layout) in new_layouts {
            self.field_layouts.insert(name, layout);
        }

        // 5. Optimize top-level function
        let constfold = ConstFold;
        let dce = Dce;
        let cse = Cse::default();
        let type_spec = TypeSpecialize::with_math(&interner);
        let licm = Licm;
        let sra = Sra;
        let passes: Vec<&dyn MirPass> = vec![
            &constfold, &dce, &cse, &type_spec, &constfold, &dce, &licm, &sra, &dce,
        ];
        opt::run_to_fixpoint(&mut module_mir.top_level, &passes, 10);
        // Also optimize method bodies
        for class in &mut module_mir.classes {
            for method in &mut class.methods {
                opt::run_to_fixpoint(&mut method.mir, &passes, 10);
            }
        }
        // Optimize closure bodies
        for closure in &mut module_mir.closures {
            opt::run_to_fixpoint(closure, &passes, 10);
        }

        // Project resolve_result.module_vars to strings (in the parse
        // interner's frame). The install helper takes ownership of the
        // list and uses it both to pre-populate the module's var slots
        // and to route class definitions to the right slot.
        let var_names: Vec<String> = resolve_result
            .module_vars
            .iter()
            .map(|&sym| interner.resolve(sym).to_string())
            .collect();

        let var_sources = resolve_result.module_var_sources.clone();
        self.install_module_mir_and_run_with_sources(
            module_name,
            &interner,
            module_mir,
            var_names,
            var_sources,
        )
    }

    /// Re-parse and re-install a previously loaded module. Preserves
    /// the identity of classes declared in the module so instances
    /// created before the reload keep working — their `class` pointer
    /// still points at the same `ObjClass`, now carrying the newly
    /// compiled method bodies.
    ///
    /// Side effects:
    /// - Invalidates every beadie bead whose function belongs to
    ///   `name`, so JIT-compiled copies are dropped on the next call.
    /// - Clears every inline cache globally — callers from other
    ///   modules may have cached the pre-reload `FuncId`.
    /// - Re-runs the module's top-level. Top-level side effects
    ///   (`var app = App.new`, handler registration, etc.) run afresh;
    ///   state preservation across reloads is a framework concern.
    pub fn reload_module(&mut self, name: &str) -> Result<(), String> {
        if !self.engine.modules.contains_key(name) {
            return Err(format!("Module '{}' is not loaded.", name));
        }

        // Load fresh source. The stored module name is already the
        // canonical form the loader keys against, so passing an empty
        // importer is enough — the loader reads it directly.
        let source = self
            .config
            .load_module_fn
            .as_ref()
            .and_then(|load_fn| load_fn(name, ""))
            .ok_or_else(|| format!("Could not load source for module '{}'.", name))?;

        // Invalidate beads for functions owned by this module. Other
        // modules' ICs get cleared wholesale below so cross-module
        // cached FuncIds refresh on the next call.
        let name_string = name.to_string();
        let func_count = self.engine.function_count() as u32;
        for i in 0..func_count {
            let fid = super::engine::FuncId(i);
            let belongs = self
                .engine
                .func_module(fid)
                .map(|m| m.as_str() == name_string)
                .unwrap_or(false);
            if belongs {
                self.engine.tier.invalidate(fid);
            }
        }
        self.engine.invalidate_inline_caches();
        // The interpreter's global (class_ptr, method_sym) -> Method
        // cache holds closure pointers that are about to be replaced
        // under the same class_ptr — flush or dispatch keeps returning
        // the old closure.
        self.method_cache.invalidate();

        // Stash existing class pointers so the re-install loop can
        // reuse them instead of allocating fresh ObjClasses.
        let mut preserved: HashMap<String, *mut ObjClass> = HashMap::new();
        if let Some(entry) = self.engine.modules.get(name) {
            for (i, var) in entry.vars.iter().enumerate() {
                if !var.is_object() {
                    continue;
                }
                let ptr = var.as_object().unwrap();
                let header = ptr as *const ObjHeader;
                if unsafe { (*header).obj_type } != ObjType::Class {
                    continue;
                }
                let cls_ptr = ptr as *mut ObjClass;
                // Skip built-in / core classes — their pointer is
                // shared across modules and must not be reset.
                let class_name = entry.var_names.get(i).cloned().unwrap_or_else(|| unsafe {
                    self.interner.resolve((*cls_ptr).name).to_string()
                });
                if self.core_class_value(&class_name).is_some() {
                    continue;
                }
                preserved.insert(class_name, cls_ptr);
            }
        }

        // Drop the module entry + loading guard so `interpret`
        // re-installs cleanly. `self.module_sources` gets overwritten
        // in `interpret` itself.
        self.engine.modules.remove(name);
        self.loading_modules.remove(name);

        if std::env::var_os("WLIFT_RELOAD_TRACE").is_some() {
            eprintln!(
                "wlift: reload '{}' — preserved {} classes, source {} bytes",
                name,
                preserved.len(),
                source.len()
            );
        }
        // Fire before-reload callbacks so frameworks can drop state
        // (route tables, watchers, etc.) the about-to-rerun top-level
        // is going to re-establish. Done AFTER cache invalidation so
        // the callbacks themselves run against the latest dispatch
        // tables.
        self.invoke_before_reload_callbacks(name);
        self.reload_class_table = Some(preserved);
        let result = self.interpret(name, &source);
        self.reload_class_table = None;

        match result {
            InterpretResult::Success => {
                if let Some(mt) = file_mtime_secs(name) {
                    self.module_mtimes.insert(name.to_string(), mt);
                }
                Ok(())
            }
            InterpretResult::CompileError => {
                Err(format!("Reload of '{}' failed: compile error.", name))
            }
            InterpretResult::RuntimeError => {
                Err(format!("Reload of '{}' failed: runtime error.", name))
            }
        }
    }

    /// Drain a pending SIGUSR1 reload signal: scan every absolute-path
    /// module, reload those whose on-disk mtime advanced past the
    /// baseline captured at install time. Cheap when the flag is
    /// clear (one atomic load); only does the stat-walk when an
    /// external watcher has signalled a change.
    ///
    /// Called from the interpreter safepoint check — no-op when the
    /// process didn't opt in to `install_reload_signal_handler`.
    pub fn check_pending_reload(&mut self) {
        if !RELOAD_PENDING.swap(false, Ordering::AcqRel) {
            return;
        }
        let names: Vec<String> = self.engine.modules.keys().cloned().collect();
        let mut reloaded: Vec<String> = Vec::new();
        for name in names {
            if !std::path::Path::new(&name).is_absolute() {
                continue;
            }
            let mtime = match file_mtime_secs(&name) {
                Some(t) => t,
                None => continue,
            };
            let last = self.module_mtimes.get(&name).copied();
            match last {
                Some(prev) if mtime > prev + 0.0001 => {
                    eprintln!("wlift: reloading {}", name);
                    if let Err(e) = self.reload_module(&name) {
                        crate::diagnostics::Diagnostic::error(format!("reload failed: {}", e))
                            .eprint_no_source();
                    } else {
                        reloaded.push(name);
                    }
                }
                None => {
                    // First time we've tracked this path — record
                    // baseline so the next signal compares against it.
                    self.module_mtimes.insert(name, mtime);
                }
                _ => {}
            }
        }
        // Fire registered Hatch.onReload callbacks for each module that
        // actually reloaded. Done after the reload loop so ALL classes
        // are in their post-reload state before the first callback runs.
        for path in reloaded {
            self.invoke_reload_callbacks(&path);
        }

        // After module reloads run, drain file-watch entries:
        // anything registered via Hatch.watchFile fires now if its
        // on-disk mtime advanced. Used by asset systems and other
        // non-Wren-source watchers that piggyback on the same SIGUSR1
        // signal as the module reload pass.
        self.drain_file_watches();
    }

    /// Drive every registered `Hatch.onReload` callback with the just-
    /// reloaded module's path. Each callback is a Wren `Fn`; we
    /// dispatch synchronously through the standard `call(_)` path on
    /// a temporary fiber. Errors are logged but don't propagate — a
    /// broken reload hook shouldn't abort the host fiber.
    fn invoke_reload_callbacks(&mut self, module_path: &str) {
        if self.reload_callbacks.is_empty() {
            return;
        }
        // Snapshot — invoking a callback could (in principle) mutate
        // `reload_callbacks` via re-entrant `Hatch.onReload`. Snapshot
        // before iterating so the loop doesn't observe the mutation.
        let callbacks = self.reload_callbacks.clone();
        let path_value = self.new_string(module_path.to_string());
        for cb in callbacks {
            if let Some(closure_ptr) = cb.as_object() {
                let closure = closure_ptr as *mut ObjClosure;
                if self
                    .call_closure_sync(closure, &[path_value], None)
                    .is_none()
                {
                    crate::diagnostics::Diagnostic::warning("reload callback returned an error")
                        .eprint_no_source();
                }
            }
        }
    }

    /// Walk every registered file watch, refresh its mtime
    /// baseline, and invoke the callback for any path whose mtime
    /// advanced since the last check. The callback receives the
    /// watched path as its single argument.
    fn drain_file_watches(&mut self) {
        if self.file_watches.is_empty() {
            return;
        }
        // Collect (callback, path, new_mtime) for entries to fire,
        // and update the baseline in the same pass. Snapshotting
        // before invoking keeps the loop independent of any
        // re-entrant `Hatch.watchFile` calls the callback might
        // make.
        let mut to_fire: Vec<(Value, String)> = Vec::new();
        for fw in &mut self.file_watches {
            let mt = match file_mtime_secs(&fw.path) {
                Some(t) => t,
                None => continue,
            };
            if mt > fw.last_mtime + 0.0001 {
                fw.last_mtime = mt;
                to_fire.push((fw.callback, fw.path.clone()));
            }
        }
        for (cb, path) in to_fire {
            let path_value = self.new_string(path);
            if let Some(closure_ptr) = cb.as_object() {
                let closure = closure_ptr as *mut ObjClosure;
                if self
                    .call_closure_sync(closure, &[path_value], None)
                    .is_none()
                {
                    crate::diagnostics::Diagnostic::warning(
                        "file-watch callback returned an error",
                    )
                    .eprint_no_source();
                }
            }
        }
    }

    fn invoke_before_reload_callbacks(&mut self, module_path: &str) {
        if self.before_reload_callbacks.is_empty() {
            return;
        }
        let callbacks = self.before_reload_callbacks.clone();
        let path_value = self.new_string(module_path.to_string());
        for cb in callbacks {
            if let Some(closure_ptr) = cb.as_object() {
                let closure = closure_ptr as *mut ObjClosure;
                if self
                    .call_closure_sync(closure, &[path_value], None)
                    .is_none()
                {
                    crate::diagnostics::Diagnostic::warning(
                        "before-reload callback returned an error",
                    )
                    .eprint_no_source();
                }
            }
        }
    }

    /// Install a freshly compiled (or freshly loaded) `ModuleMir` into
    /// the engine, build its classes, and run its top-level fiber.
    ///
    /// This is the shared suffix of the compilation pipeline — steps 6
    /// through 10 of `interpret`, reused by the `.wlbc` load path so
    /// snapshots don't have to pay for parse / sema / MIR-build / opt.
    /// `interner` is in the snapshot's (or parser's) symbol frame and
    /// gets remapped into `self.interner` on entry. `var_names` is the
    /// module's declared top-level var order — normally derived from
    /// `resolve_result.module_vars` or loaded verbatim from a snapshot.
    fn install_module_mir_and_run(
        &mut self,
        module_name: &str,
        interner: &crate::intern::Interner,
        module_mir: crate::mir::ModuleMir,
        var_names: Vec<String>,
    ) -> InterpretResult {
        let sources = vec![None; var_names.len()];
        self.install_module_mir_and_run_with_sources(
            module_name,
            interner,
            module_mir,
            var_names,
            sources,
        )
    }

    fn install_module_mir_and_run_with_sources(
        &mut self,
        module_name: &str,
        interner: &crate::intern::Interner,
        mut module_mir: crate::mir::ModuleMir,
        var_names: Vec<String>,
        var_sources: Vec<Option<String>>,
    ) -> InterpretResult {
        use crate::mir::BlockId;
        // 6. Remap symbols: the source interner and VM interner have different
        // indices for the same strings. Build a mapping and rewrite the MIR.
        let mut sym_map: Vec<crate::intern::SymbolId> = Vec::with_capacity(interner.len());
        for i in 0..interner.len() {
            let old_sym = crate::intern::SymbolId::from_raw(i as u32);
            let s = interner.resolve(old_sym);
            let new_sym = self.interner.intern(s);
            sym_map.push(new_sym);
        }
        module_mir
            .top_level
            .remap_symbols(|old| sym_map[old.index() as usize]);
        for class in &mut module_mir.classes {
            class.name = sym_map[class.name.index() as usize];
            if let Some(ref mut sup) = class.superclass {
                *sup = sym_map[sup.index() as usize];
            }
            for method in &mut class.methods {
                method
                    .mir
                    .remap_symbols(|old| sym_map[old.index() as usize]);
            }
        }
        for closure in &mut module_mir.closures {
            closure.remap_symbols(|old| sym_map[old.index() as usize]);
        }

        // 7a. Resolve closure FuncIds up front so every MIR that
        // contains `MakeClosure` — top-level, class methods, and the
        // closures themselves (a closure can declare a nested closure
        // of its own) — can be patched before any of them are frozen
        // via `register_function_in`. Pre-computing `closure_func_ids`
        // from the engine's current length lets us patch *all* MIR
        // first and register second.
        //
        // Binding each function to its defining module lets
        // `GetModuleVar @idx` resolve against the slots the MIR was
        // compiled against, even when the caller lives in a different
        // module.
        let module_rc = std::rc::Rc::new(module_name.to_string());
        let closure_count = module_mir.closures.len();
        let base_fid = self.engine.functions.len() as u32;
        let closure_func_ids: Vec<u32> = (0..closure_count as u32).map(|i| base_fid + i).collect();

        // Patch every MIR that might emit `MakeClosure`, including
        // nested closures (e.g. a test `describe { ... }` body whose
        // inner `Test.it { ... }` blocks are themselves closures).
        patch_closure_ids(&mut module_mir.top_level, &closure_func_ids);
        for class in &mut module_mir.classes {
            for method in &mut class.methods {
                patch_closure_ids(&mut method.mir, &closure_func_ids);
            }
        }
        for closure in &mut module_mir.closures {
            patch_closure_ids(closure, &closure_func_ids);
        }

        // Register closures now that their MIR is fully patched. The
        // order matches the pre-computed ids.
        for (i, closure) in module_mir.closures.drain(..).enumerate() {
            let fid = self
                .engine
                .register_function_in(closure, Some(std::rc::Rc::clone(&module_rc)));
            debug_assert_eq!(
                fid.0, closure_func_ids[i],
                "closure fid drift: expected {}, got {}",
                closure_func_ids[i], fid.0
            );
        }

        // 7b. Register top-level function with the engine
        let func_id = self
            .engine
            .register_function_in(module_mir.top_level, Some(std::rc::Rc::clone(&module_rc)));

        // 8. Create module var storage, pre-populated with core class values
        // and imported module vars.
        //
        // The installed module's imports can reference builtin optional
        // modules (e.g. a .hatch-bundled `@hatch:io` that does `import
        // "io" for IoCore`). Those only show up in `find_imported_var`
        // once the backing runtime module is registered — which normally
        // happens lazily via the interpret-path's import loop. Under
        // `install_hatch_sections`, imports don't re-run, so we pre-load
        // every builtin whose exported class name appears as one of this
        // module's required vars. Idempotent; a no-op when the module's
        // already registered.
        let module_key = module_name.to_string();
        let builtin_for_var: &[(&str, &str)] = &[
            ("Meta", "meta"),
            ("Random", "random"),
            ("FS", "fs"),
            ("OS", "os"),
            ("TimeCore", "time"),
            ("HashCore", "hash"),
            ("HttpCore", "http"),
            ("ProcCore", "proc"),
            ("RegexCore", "regex"),
            ("UuidCore", "uuid"),
            ("IoCore", "io"),
            ("TomlCore", "toml"),
            ("CryptoCore", "crypto"),
            ("ZipCore", "zip"),
            ("SocketCore", "socket"),
            ("Hatch", "hatch"),
        ];
        for name in &var_names {
            if let Some(&(_, module_id)) = builtin_for_var.iter().find(|(cls, _)| *cls == name) {
                if !self.engine.modules.contains_key(module_id) {
                    let _ = self.try_load_builtin_module(module_id);
                }
            }
        }
        let mut module_vars = Vec::with_capacity(var_names.len());
        for (i, name) in var_names.iter().enumerate() {
            let source = var_sources.get(i).and_then(|s| s.as_ref());
            if let Some(value) = self.core_class_value(name) {
                // Prelude / builtin classes (System, Object, etc.) win
                // over anything else — they're baked in and can't be
                // shadowed by a module-var slot pointing elsewhere.
                module_vars.push(value);
            } else if let Some(import_source) = source {
                // The var came from `import "<import_source>" for <name>`.
                // Resolve against that specific module rather than scanning
                // every loaded module by name — otherwise two modules that
                // both export a class called "Response" leak into each
                // other at runtime via whichever HashMap iteration order
                // happens to visit first.
                //
                // The import_source is the requested name (`./live`); the
                // engine keys ModuleEntry by canonical name (the absolute
                // path for relative imports). Resolve through the same
                // hook the import-loop uses so they line up. Falls back
                // to the raw name for builtin / scoped imports.
                let canonical = self
                    .config
                    .resolve_module_fn
                    .as_ref()
                    .and_then(|fn_| fn_(import_source, module_name))
                    .unwrap_or_else(|| import_source.clone());
                if let Some(value) = self.find_imported_var_from(name, &canonical) {
                    module_vars.push(value);
                } else {
                    // Source module hasn't exported this name yet (e.g.
                    // circular or forward-declared import). Start null;
                    // class registration or post-install fixup will fill
                    // it in when the source module is fully installed.
                    module_vars.push(Value::null());
                }
            } else if let Some(value) = self.find_imported_var(name) {
                // No explicit source (prelude, re-exported, or legacy
                // snapshot path). Fall back to the name-only scan.
                module_vars.push(value);
            } else {
                module_vars.push(Value::null());
            }
        }
        self.engine.modules.insert(
            module_key.clone(),
            super::engine::ModuleEntry {
                top_level: func_id,
                vars: module_vars,
                var_names: var_names.clone(),
            },
        );
        // Baseline mtime for the SIGUSR1 watcher. Only meaningful for
        // user modules whose canonical name is an absolute path.
        if std::path::Path::new(module_name).is_absolute() {
            if let Some(mt) = file_mtime_secs(module_name) {
                self.module_mtimes.insert(module_name.to_string(), mt);
            }
        }

        // 8b. Create user-defined classes and bind their methods.
        for class_mir in module_mir.classes {
            // Find module var slot for this class.
            let class_name_str = self.interner.resolve(class_mir.name).to_string();
            let slot = var_names.iter().position(|name| name == &class_name_str);

            // Resolve superclass (check core classes, then module vars)
            let superclass = class_mir
                .superclass
                .and_then(|sup_sym| {
                    let sup_name = self.interner.resolve(sup_sym).to_string();
                    // Try core classes first
                    if let Some(v) = self.core_class_value(&sup_name) {
                        return v.as_object().map(|p| p as *mut ObjClass);
                    }
                    // Try module vars (user-defined classes created earlier)
                    if let Some(entry) = self.engine.modules.get(&module_key) {
                        for &var_val in &entry.vars {
                            if var_val.is_object() {
                                let ptr = var_val.as_object().unwrap() as *mut ObjClass;
                                let header = ptr as *const ObjHeader;
                                if unsafe { (*header).obj_type } == ObjType::Class {
                                    let name =
                                        unsafe { self.interner.resolve((*ptr).name).to_string() };
                                    if name == sup_name {
                                        return Some(ptr);
                                    }
                                }
                            }
                        }
                    }
                    None
                })
                .unwrap_or(self.object_class);

            // Create the class — or, during `reload_module`, reuse the
            // existing ObjClass pointer and reset its method table so
            // instances that predate the reload continue to dispatch
            // against the new methods.
            let reused = self
                .reload_class_table
                .as_ref()
                .and_then(|t| t.get(&class_name_str).copied());
            let class_ptr = match reused {
                Some(existing) => {
                    unsafe {
                        // Drop stale methods/attributes; the method-
                        // binding loop below repopulates. Seed the
                        // table with whatever the (possibly new)
                        // superclass exposes so inherited methods
                        // still resolve.
                        let parent_methods = if !superclass.is_null() {
                            (*superclass).methods.clone()
                        } else {
                            Vec::new()
                        };
                        (*existing).methods = parent_methods;
                        (*existing).static_fields.clear();
                        (*existing).method_attributes.clear();
                        (*existing).attributes.clear();
                        (*existing).superclass = superclass;
                        (*existing).name = class_mir.name;
                        (*existing).is_foreign = false;
                    }
                    existing
                }
                None => self.gc.alloc_class(class_mir.name, superclass),
            };
            unsafe {
                (*class_ptr).header.class = self.class_class;
                // Total fields = own fields + inherited fields from superclass chain
                let inherited_fields = if !superclass.is_null() {
                    (*superclass).num_fields
                } else {
                    0
                };
                (*class_ptr).num_fields = class_mir.num_fields + inherited_fields;
                (*class_ptr).attributes = class_mir.attributes;
            }

            // Register each method's MIR and bind to the class
            for method_mir in class_mir.methods {
                let method_func_id = self
                    .engine
                    .register_function_in(method_mir.mir, Some(std::rc::Rc::clone(&module_rc)));

                // Create ObjFn + ObjClosure for the method
                let sig_sym = self.interner.intern(&method_mir.signature);
                let fn_ptr = self.gc.alloc_fn(sig_sym, 0, 0, method_func_id.0);
                unsafe {
                    (*fn_ptr).header.class = self.fn_class;
                    (*fn_ptr).trivial_getter_field = self
                        .engine
                        .trivial_getter_fields
                        .get(method_func_id.0 as usize)
                        .copied()
                        .flatten()
                        .unwrap_or(u16::MAX);
                    (*fn_ptr).trivial_setter_field = self
                        .engine
                        .trivial_setter_fields
                        .get(method_func_id.0 as usize)
                        .copied()
                        .flatten()
                        .unwrap_or(u16::MAX);
                }

                let closure_ptr = self.gc.alloc_closure(fn_ptr);
                unsafe {
                    (*closure_ptr).header.class = self.fn_class;
                }

                let sig_sym = self.interner.intern(&method_mir.signature);
                let bind_sym = if method_mir.is_static || method_mir.is_constructor {
                    let static_sig = format!("static:{}", method_mir.signature);
                    self.interner.intern(&static_sig)
                } else {
                    sig_sym
                };

                unsafe {
                    let method = if method_mir.is_constructor {
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
                    if !method_mir.attributes.is_empty() {
                        cls.method_attributes
                            .insert(bind_sym, method_mir.attributes);
                    }
                }
            }

            // Resolve #!native / #!symbol foreign methods via dlopen/dlsym.
            // Unresolved methods remain absent from the method table — calls
            // surface as a normal method-not-found runtime error.
            if let Some(lib_name) = class_mir.native_library.as_ref() {
                let trace = std::env::var_os("WLIFT_TRACE_FOREIGN").is_some();
                match crate::runtime::foreign::load_library(
                    lib_name,
                    &self.native_search_paths,
                    &self.native_lib_paths,
                ) {
                    Ok(lib) => {
                        if trace {
                            crate::diagnostics::Diagnostic::info(format!(
                                "loaded native library '{}'",
                                lib_name
                            ))
                            .eprint_no_source();
                        }
                        unsafe {
                            (*class_ptr).is_foreign = true;
                        }
                        for fm in &class_mir.foreign_methods {
                            let symbol_name: String = fm.symbol.clone().unwrap_or_else(|| {
                                crate::runtime::foreign::base_name_of_signature(&fm.signature)
                                    .to_string()
                            });
                            match crate::runtime::foreign::resolve_symbol(
                                &lib,
                                lib_name,
                                &symbol_name,
                            ) {
                                Ok(func) => {
                                    let sig_sym = self.interner.intern(&fm.signature);
                                    let bind_sym = if fm.is_static {
                                        let static_sig = format!("static:{}", fm.signature);
                                        self.interner.intern(&static_sig)
                                    } else {
                                        sig_sym
                                    };
                                    if trace {
                                        crate::diagnostics::Diagnostic::info(format!(
                                            "bound '{}' → {} (sym={})",
                                            fm.signature,
                                            symbol_name,
                                            bind_sym.index(),
                                        ))
                                        .eprint_no_source();
                                    }
                                    unsafe {
                                        (*class_ptr).bind_foreign_c(bind_sym, func);
                                    }
                                }
                                Err(err) => {
                                    crate::diagnostics::Diagnostic::error(err.to_string())
                                        .eprint_no_source();
                                }
                            }
                        }
                        self.native_libs.push(lib);
                    }
                    Err(err) => {
                        crate::diagnostics::Diagnostic::error(err.to_string()).eprint_no_source();
                    }
                }
            } else if !class_mir.foreign_methods.is_empty() {
                // Phase 3b does not yet route through bind_foreign_method_fn
                // when #!native is absent. Surface a diagnostic so silent
                // "method not found" errors aren't mysterious.
                let class_name = self.interner.resolve(class_mir.name).to_string();
                crate::diagnostics::Diagnostic::warning(format!(
                    "class '{}' has foreign methods but no #!native directive",
                    class_name
                ))
                .eprint_no_source();
            }

            // Store class in module vars
            if let Some(idx) = slot {
                let class_val = Value::object(class_ptr as *mut u8);
                if let Some(entry) = self.engine.modules.get_mut(&module_key) {
                    while entry.vars.len() <= idx {
                        entry.vars.push(Value::null());
                    }
                    entry.vars[idx] = class_val;
                }
            }
        }

        // Eager compilation: compile the module entry before running it.
        // Disabled when Cranelift is the backend — we need 100 interpreter
        // calls first to collect IC + profile data for TypeSpecialize and
        // inline caching. Without this warmup, the JIT code uses full
        // dispatch for every method call (30-40x slower).
        #[cfg(not(feature = "cranelift"))]
        if self.config.execution_mode != super::engine::ExecutionMode::Interpreter
            && self.config.step_limit > 100_000
            && self
                .engine
                .get_mir(func_id)
                .as_deref()
                .map(|mir| should_eager_compile_entry(mir, &self.interner))
                .unwrap_or(false)
        {
            let _ = self.engine.tier_up(func_id, &self.interner);
        }

        // 9. Create a fiber and push the initial call frame
        let fiber = self.gc.alloc_fiber();
        let reg_size = self
            .engine
            .get_mir(func_id)
            .map(|m| m.next_value as usize)
            .unwrap_or(0);
        unsafe {
            (*fiber).header.class = self.fiber_class;
            (*fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                pc: 0,
                values: vec![Value::UNDEFINED; reg_size],
                module_name: std::rc::Rc::new(module_key.clone()),
                return_dst: None,
                closure: None,
                defining_class: None,
                bc_ptr: std::ptr::null(),
            });
        }

        // Set as active fiber (save previous)
        let prev_fiber = self.fiber;
        self.fiber = fiber;

        // 10. Run the fiber
        let result = super::vm_interp::run_fiber(self);

        // Reload fiber pointer: GC may have promoted/moved it during run_fiber.
        // The local `fiber` from before run_fiber is potentially stale.
        let fiber = self.fiber;

        let interpret_result = match result {
            Ok(_) => {
                self.fiber = prev_fiber;
                InterpretResult::Success
            }
            Err(e) => {
                // Save the error fiber for post-mortem inspection before restoring
                self.error_fiber = fiber;
                let loc = self.extract_error_location(fiber);
                self.report_runtime_error(&e, loc.as_ref(), fiber);
                self.fiber = prev_fiber;
                InterpretResult::RuntimeError
            }
        };
        self.loading_modules.remove(&module_key);
        interpret_result
    }

    /// Extract source location from a fiber's current MIR frame for error reporting.
    pub(crate) fn extract_error_location(
        &self,
        fiber: *mut ObjFiber,
    ) -> Option<super::vm_interp::SourceLoc> {
        if fiber.is_null() {
            return None;
        }
        let frame = unsafe { (*fiber).mir_frames.last()? };

        // Try bytecode source map first (pc-based lookup).
        // `frame.pc` is saved by Op handlers AFTER they've finished
        // reading their operands — so on a failed call, pc points
        // at the START of the *next* instruction, not the failing
        // one. `lookup_span` does a binary search by `offset <= pc`,
        // which on an exact match against a `record_span` boundary
        // returns the next instruction's span (e.g. the wrapping
        // `System.print(...)` call instead of the inner failing
        // `x.tryAccept`). Use `pc.saturating_sub(1)` so the lookup
        // lands inside the instruction that actually errored.
        if let Some(bc) = self.engine.peek_bytecode(frame.func_id) {
            let lookup_pc = frame.pc.saturating_sub(1);
            if let Some(span) = bc.lookup_span(lookup_pc) {
                return Some(super::vm_interp::SourceLoc {
                    span: span.clone(),
                    module: frame.module_name.clone(),
                });
            }
        }

        // Fallback to MIR span_map (for non-bytecode paths)
        let mir = self.engine.get_mir(frame.func_id)?;
        let block = mir.blocks.get(frame.current_block.0 as usize)?;
        let ip = frame.ip;
        if ip > 0 {
            let (val_id, _) = block.instructions.get(ip - 1)?;
            let span = mir.span_map.get(val_id)?;
            Some(super::vm_interp::SourceLoc {
                span: span.clone(),
                module: frame.module_name.clone(),
            })
        } else if let Some((val_id, _)) = block.instructions.first() {
            let span = mir.span_map.get(val_id)?;
            Some(super::vm_interp::SourceLoc {
                span: span.clone(),
                module: frame.module_name.clone(),
            })
        } else {
            None
        }
    }

    /// Resolve a line number from a span offset within a module's source.
    fn line_from_span(&self, module: &str, byte_offset: usize) -> Option<usize> {
        self.module_sources.get(module).map(|src| {
            src[..byte_offset.min(src.len())]
                .chars()
                .filter(|c| *c == '\n')
                .count()
                + 1
        })
    }

    /// Extract a StackFrame from a MirCallFrame.
    fn frame_to_stack_frame(&self, frame: &MirCallFrame) -> StackFrame {
        let func_name = self
            .engine
            .get_mir(frame.func_id)
            .map(|mir| self.interner.resolve(mir.name).to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        let line = self.engine.get_mir(frame.func_id).and_then(|mir| {
            let block = mir.blocks.get(frame.current_block.0 as usize)?;
            let idx = if frame.ip > 0 { frame.ip - 1 } else { 0 };
            let (vid, _) = block.instructions.get(idx)?;
            let span = mir.span_map.get(vid)?;
            self.line_from_span(&frame.module_name, span.start)
        });
        StackFrame {
            func_name,
            module: (*frame.module_name).clone(),
            line,
        }
    }

    /// Build a stack trace from a single fiber's MIR call frames.
    fn build_stack_trace(&self, fiber: *mut ObjFiber) -> Vec<String> {
        if fiber.is_null() {
            return Vec::new();
        }
        let frames = unsafe { &(*fiber).mir_frames };
        frames
            .iter()
            .rev()
            .map(|frame| self.frame_to_stack_frame(frame).to_string())
            .collect()
    }

    /// Capture the current fiber's call stack as a Vec<StackFrame> snapshot.
    fn capture_current_trace(&self) -> Vec<StackFrame> {
        if self.fiber.is_null() {
            return Vec::new();
        }
        let frames = unsafe { &(*self.fiber).mir_frames };
        frames
            .iter()
            .rev()
            .map(|frame| self.frame_to_stack_frame(frame))
            .collect()
    }

    /// Build a full cross-fiber stack trace, walking the caller chain.
    /// Each fiber boundary is annotated. Includes spawn traces when available.
    fn build_full_fiber_trace(&self, fiber: *mut ObjFiber) -> Vec<String> {
        if fiber.is_null() {
            return Vec::new();
        }

        let mut trace = self.build_stack_trace(fiber);

        if !self.config.fiber_stack_traces {
            return trace;
        }

        // Walk the caller chain
        let mut current = unsafe { (*fiber).caller };
        while !current.is_null() {
            trace.push("  --- in calling fiber ---".to_string());
            trace.extend(self.build_stack_trace(current));
            current = unsafe { (*current).caller };
        }

        // Append spawn trace if present
        let spawn_trace = unsafe { &(*fiber).spawn_trace };
        if let Some(ref frames) = spawn_trace {
            if !frames.is_empty() {
                trace.push("  --- spawned at ---".to_string());
                for frame in frames {
                    trace.push(frame.to_string());
                }
            }
        }

        trace
    }

    /// Generate a contextual help note for a runtime error.
    fn error_help(error: &super::vm_interp::RuntimeError) -> Option<String> {
        use super::vm_interp::RuntimeError;
        match error {
            RuntimeError::MethodNotFound { class_name, method } => {
                if method.contains("(_") {
                    Some(format!(
                        "the class `{}` does not define `{}`. Check spelling and argument count",
                        class_name, method
                    ))
                } else {
                    Some(format!(
                        "`{}` has no getter or method `{}`. Did you mean to call it with `()`?",
                        class_name, method
                    ))
                }
            }
            RuntimeError::GuardFailed(expected) => {
                Some(format!("expected a value of type {}", expected))
            }
            RuntimeError::Error(msg) if msg.contains("not a function") => {
                Some("only Fn objects and methods can be called".to_string())
            }
            RuntimeError::Error(msg) if msg.contains("out of bounds") => {
                Some("ensure the index is within 0..(count - 1)".to_string())
            }
            _ => None,
        }
    }

    /// Report a runtime error with full ariadne diagnostics, source snippets,
    /// contextual help, and a stack trace.
    fn report_runtime_error(
        &self,
        error: &super::vm_interp::RuntimeError,
        loc: Option<&super::vm_interp::SourceLoc>,
        fiber: *mut ObjFiber,
    ) {
        if let Some(loc) = loc {
            if let Some(source) = self.module_sources.get(loc.module.as_str()) {
                let mut diag = crate::diagnostics::Diagnostic::error(error.to_string())
                    .with_label(loc.span.clone(), "error occurred here");

                // Add contextual help note
                if let Some(help) = Self::error_help(error) {
                    diag = diag.with_note(help);
                }

                // Add stack trace as a note (uses full cross-fiber trace when enabled)
                let trace = self.build_full_fiber_trace(fiber);
                if trace.len() > 1 {
                    let trace_str = format!("stack trace:\n{}", trace.join("\n"));
                    diag = diag.with_note(trace_str);
                }

                if self.output_buffer.is_some() {
                    // In test mode, render to string via error_fn
                    let rendered = diag.render_to_string(source);
                    self.report_error(&rendered);
                } else {
                    diag.eprint(source);
                }
                return;
            }
        }
        // Fallback: no source location available (e.g. error
        // raised inside a hatch-bundled module that didn't ship
        // its source section). Still go through the Diagnostic
        // formatter so the output matches the labelled-span path
        // visually — same colored "Error:" header, same Note rows
        // for context + stack trace.
        let mut diag = crate::diagnostics::Diagnostic::error(error.to_string());
        if let Some(help) = Self::error_help(error) {
            diag = diag.with_note(help);
        }
        let trace = self.build_full_fiber_trace(fiber);
        if !trace.is_empty() {
            let trace_str = format!("stack trace:\n{}", trace.join("\n"));
            diag = diag.with_note(trace_str);
        }
        if self.output_buffer.is_some() {
            // Test harness — render to string so error_fn captures it.
            let rendered = diag.render_to_string("");
            self.report_error(&rendered);
        } else {
            diag.eprint_no_source();
        }
    }

    // -- Helpers for core library primitives --

    /// Allocate a GC-managed string and return it as a Value.
    pub fn new_string(&mut self, s: String) -> Value {
        let obj = self.gc.alloc_string(s);
        unsafe {
            (*obj).header.class = self.string_class;
        }
        Value::object(obj as *mut u8)
    }

    /// Write to output (captured buffer if set, otherwise stdout).
    pub fn vm_write(&mut self, s: &str) {
        if let Some(ref mut buf) = self.output_buffer {
            buf.push_str(s);
        } else {
            print!("{}", s);
        }
    }

    /// Take captured output (for tests).
    pub fn take_output(&mut self) -> String {
        self.output_buffer.take().unwrap_or_default()
    }

    /// Allocate a GC-managed list and return it as a Value.
    pub fn new_list(&mut self, elements: Vec<Value>) -> Value {
        let root_len_before = crate::codegen::runtime_fns::jit_roots_snapshot_len();
        for &elem in &elements {
            crate::codegen::runtime_fns::push_jit_root(elem);
        }

        let obj = self.gc.alloc_list();
        let list_val = Value::object(obj as *mut u8);
        crate::codegen::runtime_fns::push_jit_root(list_val);
        unsafe {
            (*obj).header.class = self.list_class;
            for idx in 0..elements.len() {
                let elem = crate::codegen::runtime_fns::jit_root_at(root_len_before + idx);
                (*obj).add(elem);
            }
        }
        let list_val = crate::codegen::runtime_fns::jit_root_at(root_len_before + elements.len());
        crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
        list_val
    }

    /// Allocate a GC-managed range and return it as a Value.
    pub fn new_range(&mut self, from: f64, to: f64, inclusive: bool) -> Value {
        let obj = self.gc.alloc_range(from, to, inclusive);
        unsafe {
            (*obj).header.class = self.range_class;
        }
        Value::object(obj as *mut u8)
    }

    /// Allocate a GC-managed map and return it as a Value.
    pub fn new_map(&mut self) -> Value {
        let obj = self.gc.alloc_map();
        unsafe {
            (*obj).header.class = self.map_class;
        }
        Value::object(obj as *mut u8)
    }

    /// Allocate a fresh, zero-initialized typed array. The object's
    /// `header.class` is set to the matching ByteArray /
    /// Float32Array / Float64Array ObjClass so method dispatch
    /// picks the right surface.
    pub fn new_typed_array(&mut self, count: u32, kind: TypedArrayKind) -> Value {
        let obj = self.gc.alloc_typed_array(count, kind);
        let cls = match kind {
            TypedArrayKind::U8 => self.byte_array_class,
            TypedArrayKind::F32 => self.float32_array_class,
            TypedArrayKind::F64 => self.float64_array_class,
        };
        unsafe {
            (*obj).header.class = cls;
        }
        Value::object(obj as *mut u8)
    }

    /// Create a core class with the given name and optional superclass.
    pub fn make_class(&mut self, name: &str, superclass: *mut ObjClass) -> *mut ObjClass {
        let sym = self.interner.intern(name);
        let class = self.gc.alloc_class(sym, superclass);
        unsafe {
            (*class).header.class = self.class_class;
            // Set protocol conformance (inherits from superclass + own built-in protocols).
            let builtin = crate::sema::protocol::builtin_protocols_for(name);
            (*class).protocols = (*class).protocols.union(builtin);
        }
        class
    }

    /// Bind a native primitive to a class by method signature string.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn primitive(&mut self, class: *mut ObjClass, signature: &str, func: NativeFn) {
        let sym = self.interner.intern(signature);
        unsafe {
            (*class).bind_native(sym, func);
        }
    }

    /// Bind a native primitive to the class's metaclass (static method).
    /// For simplicity, we store static methods directly on the class.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn primitive_static(&mut self, class: *mut ObjClass, signature: &str, func: NativeFn) {
        // In Wren, static methods are on the metaclass. We store them
        // on the class itself with a "static:" prefix for now.
        let sig = format!("static:{}", signature);
        let sym = self.interner.intern(&sig);
        unsafe {
            (*class).bind_native(sym, func);
        }
    }

    /// Mark a symbol as a call() family symbol for fast dispatch checks.
    pub fn mark_call_symbol(&mut self, sym: SymbolId) {
        let idx = sym.index() as usize;
        if idx >= self.call_sym_flags.len() {
            self.call_sym_flags.resize(idx + 1, false);
        }
        self.call_sym_flags[idx] = true;
    }

    /// Check if a method symbol is a call() family symbol (O(1) array lookup).
    #[inline(always)]
    pub fn is_call_sym(&self, sym: SymbolId) -> bool {
        let idx = sym.index() as usize;
        idx < self.call_sym_flags.len() && unsafe { *self.call_sym_flags.get_unchecked(idx) }
    }

    fn refresh_hot_method_symbols(&mut self) {
        self.hot_method_symbols = HotMethodSymbols {
            list_subscript: self.interner.lookup("[_]"),
            list_subscript_set: self.interner.lookup("[_]=(_)"),
            list_add: self.interner.lookup("add(_)"),
            list_count: self.interner.lookup("count"),
            list_iterate: self.interner.lookup("iterate(_)"),
            list_iterator_value: self.interner.lookup("iteratorValue(_)"),
            list_remove_at: self.interner.lookup("removeAt(_)"),
        };
    }

    /// Look up a static method on a class.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn find_static(&self, class: *mut ObjClass, signature: &str) -> Option<&Method> {
        let sig = format!("static:{}", signature);
        let sym = self.interner.lookup(&sig)?;
        unsafe { (*class).find_method(sym) }
    }

    /// Get the class for a value (using the built-in class hierarchy).
    pub fn class_of(&self, value: Value) -> *mut ObjClass {
        if value.is_num() {
            self.num_class
        } else if value.is_bool() {
            self.bool_class
        } else if value.is_null() {
            self.null_class
        } else if value.is_object() {
            let ptr = value.as_object().unwrap();
            let header = ptr as *const ObjHeader;
            unsafe {
                let cls = (*header).class;
                if cls.is_null() {
                    self.object_class
                } else {
                    cls
                }
            }
        } else {
            self.object_class
        }
    }

    /// Get the class name for a value.
    pub fn class_name_of(&self, value: Value) -> String {
        let class = self.class_of(value);
        if class.is_null() {
            return "Object".to_string();
        }
        unsafe { self.interner.resolve((*class).name).to_string() }
    }

    /// Map a core class name to its Value (pointer to ObjClass).
    /// Search all loaded modules for a variable with the given name.
    /// Used to resolve imported names across module boundaries.
    /// Look up a named export in a specific source module. Used when
    /// the call site is `import "<source>" for <name>` — we honour the
    /// import instead of returning the first class with a matching
    /// name from any module (which was non-deterministic under
    /// HashMap iteration and routinely picked the wrong one when two
    /// modules defined, say, `Response`).
    fn find_imported_var_from(&self, name: &str, source: &str) -> Option<Value> {
        let entry = self.engine.modules.get(source)?;
        let idx = entry.var_names.iter().position(|n| n == name)?;
        let value = entry.vars.get(idx).copied()?;
        // Refuse to bind a null slot — if the source module is still
        // initialising and hasn't written this var yet, the caller
        // falls through to the legacy path or stores null and retries
        // lazily.
        if value.is_null() {
            return None;
        }
        Some(value)
    }

    fn find_imported_var(&self, name: &str) -> Option<Value> {
        let target_sym = self.interner.lookup(name)?;
        for entry in self.engine.modules.values() {
            for &var_val in &entry.vars {
                if var_val.is_object() {
                    if let Some(ptr) = var_val.as_object() {
                        let header = ptr as *const ObjHeader;
                        if unsafe { (*header).obj_type } == ObjType::Class {
                            let cls = ptr as *mut ObjClass;
                            if unsafe { (*cls).name } == target_sym {
                                return Some(var_val);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    pub fn core_class_value(&self, name: &str) -> Option<Value> {
        let class_ptr = match name {
            "Object" => self.object_class,
            "Class" => self.class_class,
            "Bool" => self.bool_class,
            "Num" => self.num_class,
            "String" => self.string_class,
            "List" => self.list_class,
            "ByteArray" => self.byte_array_class,
            "Float32Array" => self.float32_array_class,
            "Float64Array" => self.float64_array_class,
            "Map" => self.map_class,
            "Range" => self.range_class,
            "Null" => self.null_class,
            "Fn" => self.fn_class,
            "Fiber" => self.fiber_class,
            "System" => self.system_class,
            "Sequence" => self.sequence_class,
            _ => return None,
        };
        Some(Value::object(class_ptr as *mut u8))
    }

    /// Write text via the configured write callback (or stdout).
    pub fn write(&self, text: &str) {
        if let Some(ref f) = self.config.write_fn {
            f(text);
        } else {
            print!("{}", text);
        }
    }

    /// Report a runtime error.
    ///
    /// Two callers:
    /// - the normal path through `report_runtime_error` which has
    ///   already pre-rendered a Diagnostic to a String (via
    ///   `render_to_string`) and just needs the host to surface it
    ///   verbatim — the message starts with the ariadne ANSI
    ///   "Error:" header in that case;
    /// - direct callers that pass raw error prose (e.g. "Could not
    ///   load module 'X'"), which we wrap in a Diagnostic so every
    ///   error reaching the user goes through the same formatter.
    ///
    /// Test mode (`config.error_fn` set) forwards the message
    /// verbatim so the test harness can pattern-match on it.
    pub fn report_error(&self, msg: &str) {
        if let Some(ref f) = self.config.error_fn {
            f(ErrorKind::Runtime, "", 0, msg);
            return;
        }
        // If the msg already starts with the rendered ariadne header
        // (it begins with the colored "Error" sequence), pass it
        // through as-is. Otherwise wrap raw text in a Diagnostic so
        // it looks like every other error.
        if msg.contains("Error]") || msg.starts_with("\x1b[") {
            eprintln!("{}", msg.trim_end_matches('\n'));
        } else {
            crate::diagnostics::Diagnostic::error(msg.to_string()).eprint_no_source();
        }
    }

    /// Try to load a built-in optional module ("meta" or "random").
    /// Returns true if the module was recognized and registered.
    fn try_load_builtin_module(&mut self, name: &str) -> bool {
        match name {
            "meta" => {
                let class = super::core::meta::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "meta".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["Meta".to_string()],
                    },
                );
                true
            }
            "random" => {
                let class = super::core::random::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "random".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["Random".to_string()],
                    },
                );
                true
            }
            "fs" => {
                let class = super::core::fs::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "fs".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["FS".to_string()],
                    },
                );
                true
            }
            "os" => {
                let class = super::core::os::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "os".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["OS".to_string()],
                    },
                );
                true
            }
            "time" => {
                let class = super::core::time::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "time".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["TimeCore".to_string()],
                    },
                );
                true
            }
            "hash" => {
                let class = super::core::hash::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "hash".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["HashCore".to_string()],
                    },
                );
                true
            }
            "crypto" => {
                let class = super::core::crypto::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "crypto".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["CryptoCore".to_string()],
                    },
                );
                true
            }
            "zip" => {
                let class = super::core::zip::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "zip".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["ZipCore".to_string()],
                    },
                );
                true
            }
            "socket" => {
                let class = super::core::socket::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "socket".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["SocketCore".to_string()],
                    },
                );
                true
            }
            "io" => {
                let class = super::core::io::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "io".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["IoCore".to_string()],
                    },
                );
                true
            }
            "http" => {
                let class = super::core::http::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "http".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["HttpCore".to_string()],
                    },
                );
                true
            }
            "proc" => {
                let class = super::core::proc::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "proc".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["ProcCore".to_string()],
                    },
                );
                true
            }
            "regex" => {
                let class = super::core::regex::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "regex".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["RegexCore".to_string()],
                    },
                );
                true
            }
            "uuid" => {
                let class = super::core::uuid::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "uuid".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["UuidCore".to_string()],
                    },
                );
                true
            }
            "toml" => {
                let class = super::core::toml::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "toml".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["TomlCore".to_string()],
                    },
                );
                true
            }
            "hatch" => {
                let class = super::core::hatch::register(self);
                let class_value = Value::object(class as *mut u8);
                self.engine.modules.insert(
                    "hatch".to_string(),
                    super::engine::ModuleEntry {
                        top_level: super::engine::FuncId(u32::MAX),
                        vars: vec![class_value],
                        var_names: vec!["Hatch".to_string()],
                    },
                );
                true
            }
            _ => false,
        }
    }

    /// Create a persistent handle that prevents a value from being GC'd.
    pub fn make_handle(&mut self, value: Value) -> usize {
        let idx = self.handles.len();
        self.handles.push(WrenHandle { value });
        idx
    }

    /// Release a handle.
    pub fn release_handle(&mut self, idx: usize) {
        if idx < self.handles.len() {
            self.handles[idx].value = Value::null();
        }
    }

    #[inline]
    fn reset_sync_fiber(&self, fiber: *mut ObjFiber) {
        if fiber.is_null() {
            return;
        }
        unsafe {
            (*fiber).stack.clear();
            (*fiber).frames.clear();
            (*fiber).mir_frames.clear();
            (*fiber).state = FiberState::New;
            (*fiber).caller = std::ptr::null_mut();
            (*fiber).error = Value::null();
            (*fiber).resume_value_dst = None;
            (*fiber).spawn_trace = None;
            (*fiber).is_try = false;
            (*fiber).jit_resume_value = None;
            (*fiber).header.class = self.fiber_class;
        }
    }

    #[inline]
    fn acquire_sync_fiber(&mut self) -> *mut ObjFiber {
        let fiber = self.sync_fiber_pool.pop().unwrap_or_else(|| {
            let fiber = self.gc.alloc_fiber();
            unsafe {
                (*fiber).header.class = self.fiber_class;
            }
            fiber
        });
        self.reset_sync_fiber(fiber);
        fiber
    }

    #[inline]
    fn release_sync_fiber(&mut self, fiber: *mut ObjFiber) {
        if fiber.is_null() {
            return;
        }
        self.reset_sync_fiber(fiber);
        self.sync_fiber_pool.push(fiber);
    }

    /// Scan active JIT frames for GC roots using the explicit JIT frame list
    /// and safepoint metadata (stack maps).
    ///
    /// Each JIT function registers (frame_pointer, func_id) on entry via
    /// `wren_jit_frame_push`. For each registered frame, we look up the
    /// function's NativeFrameMetadata by func_id and conservatively scan
    /// all spill slots that could contain boxed values.
    ///
    /// Returns (values, slot_addresses) for GC write-back.
    fn scan_native_stack_roots(&self) -> (Vec<Value>, Vec<*mut u64>) {
        use crate::codegen::native_meta::RootLocation;
        let mut found_roots = Vec::new();
        let mut slot_addrs = Vec::new();

        let jit_entries = crate::codegen::runtime_fns::jit_frame_entries();
        for &(jit_fp, func_id, ret_addr) in &jit_entries {
            if jit_fp == 0 {
                continue;
            }

            // Look up the function's metadata by func_id.
            let meta = self
                .engine
                .jit_metadata
                .get(func_id as usize)
                .and_then(|m| m.as_ref());
            let Some(meta) = meta else { continue };

            // Find the code range for this function to compute the safepoint offset.
            let code_range = self
                .engine
                .code_ranges
                .iter()
                .find(|r| r.func_id == crate::runtime::engine::FuncId(func_id));

            // Precise safepoint scanning: use the return address to find the
            // ACTIVE safepoint, then only scan roots that are live at that point.
            // Falls back to conservative (all safepoints) if no match found.
            let active_safepoint = code_range.and_then(|cr| {
                let offset = ret_addr.wrapping_sub(cr.start) as u32;
                meta.safepoints.iter().find(|sp| sp.code_offset == offset)
            });

            let roots_to_scan: &[crate::codegen::native_meta::LiveRootMetadata] =
                if let Some(sp) = active_safepoint {
                    &sp.live_roots
                } else {
                    // No precise match — skip (don't conservatively scan,
                    // which would corrupt non-object spill slots).
                    continue;
                };

            for root in roots_to_scan {
                if let RootLocation::Spill(spill_offset) = root.location {
                    let addr = (jit_fp as isize + spill_offset as isize) as *mut u64;
                    let bits = unsafe { *addr };
                    let val = Value::from_bits(bits);
                    if val.is_object() {
                        found_roots.push(val);
                        slot_addrs.push(addr);
                    }
                }
            }
        }

        (found_roots, slot_addrs)
    }

    #[cfg(target_arch = "aarch64")]
    #[allow(dead_code)]
    fn scan_native_stack_roots_debug(&self) -> (Vec<Value>, u32, u32) {
        use crate::codegen::native_meta::RootLocation;
        let mut found_roots = Vec::new();
        let mut frames_walked = 0u32;
        let mut jit_frames = 0u32;

        let mut fp: usize;
        unsafe { core::arch::asm!("mov {}, x29", out(reg) fp, options(nomem, nostack)) };

        let max_frames = 256;

        while fp != 0 && frames_walked < max_frames {
            frames_walked += 1;
            let return_addr = unsafe { *((fp + 8) as *const usize) };
            if return_addr == 0 {
                break;
            }

            if std::env::var_os("WLIFT_TRACE_STACKWALK").is_some() && frames_walked <= 15 {
                let in_range = self
                    .engine
                    .code_ranges
                    .iter()
                    .any(|r| return_addr >= r.start && return_addr < r.end);
                eprintln!(
                    "  frame {}: fp={:#x} ret={:#x} in_jit={}",
                    frames_walked, fp, return_addr, in_range
                );
            }
            if let Some(code_range) = self.engine.find_code_range(return_addr) {
                jit_frames += 1;
                let offset = (return_addr - code_range.start) as u32;
                if let Some(safepoint) = code_range.metadata.find_safepoint(offset) {
                    let jit_fp = unsafe { *(fp as *const usize) };
                    if jit_fp != 0 {
                        for root in &safepoint.live_roots {
                            if let RootLocation::Spill(spill_offset) = root.location {
                                let addr = (jit_fp as isize + spill_offset as isize) as *const u64;
                                let bits = unsafe { *addr };
                                found_roots.push(Value::from_bits(bits));
                            }
                        }
                    }
                }
            }

            fp = unsafe { *(fp as *const usize) };
        }

        (found_roots, frames_walked, jit_frames)
    }

    // Fallback stub kept so the debug signature is callable on any arch.
    // aarch64 has its own implementation further up; other hosts don't
    // unwind native frames today, so the helper just returns empties.
    #[cfg(not(target_arch = "aarch64"))]
    #[allow(dead_code)]
    fn scan_native_stack_roots_debug(&self) -> (Vec<Value>, u32, u32) {
        (Vec::new(), 0, 0)
    }

    // scan_native_stack_roots is now enabled for all platforms (not just aarch64)

    pub fn collect_garbage(&mut self) {
        let mut roots: Vec<Value> = Vec::new();

        // 1. API stack
        let api_len = self.api_stack.len();
        roots.extend_from_slice(&self.api_stack);

        // 2. Handles
        let handles_start = roots.len();
        for h in &self.handles {
            roots.push(h.value);
        }

        // 3. VM-owned class pointers — may be nursery-allocated, so they can
        //    be promoted/forwarded during GC.
        let classes_start = roots.len();
        let core_classes: [*mut ObjClass; 20] = [
            self.object_class,
            self.class_class,
            self.bool_class,
            self.num_class,
            self.string_class,
            self.list_class,
            self.map_class,
            self.range_class,
            self.null_class,
            self.fn_class,
            self.fiber_class,
            self.system_class,
            self.sequence_class,
            self.map_sequence_class,
            self.skip_sequence_class,
            self.take_sequence_class,
            self.where_sequence_class,
            self.string_byte_seq_class,
            self.string_code_point_seq_class,
            self.map_entry_class,
        ];
        for &ptr in &core_classes {
            if !ptr.is_null() {
                roots.push(Value::object(ptr as *mut u8));
            } else {
                roots.push(Value::null());
            }
        }

        // 4. Current fiber (GC traces its mir_frames/caller chain internally)
        let fiber_idx = roots.len();
        if !self.fiber.is_null() {
            roots.push(Value::object(self.fiber as *mut u8));
        }

        // 5. Reusable synchronous temp fibers.
        let sync_fiber_pool_start = roots.len();
        for &fiber in &self.sync_fiber_pool {
            roots.push(Value::object(fiber as *mut u8));
        }

        // 6. JIT roots — values held in native frames invisible to normal root scanning.
        let jit_roots_start = roots.len();
        let mut jit_roots = crate::codegen::runtime_fns::take_jit_roots();
        roots.append(&mut jit_roots);
        let jit_roots_end = roots.len();

        // 7. Native shadow stack roots for active compiled frames.
        let native_shadow_start = roots.len();
        let (native_shadow_lengths, mut native_shadow_roots) =
            crate::codegen::runtime_fns::take_native_shadow_roots();
        roots.append(&mut native_shadow_roots);
        let native_shadow_end = roots.len();

        // 7b. Stack map roots: scan native stack frames for GC roots using
        // frame pointer chain + safepoint metadata. These are added to the
        // root set alongside shadow roots. After GC, updated values are
        // written back directly to the spill slots on the native stack.
        let native_stack_start = roots.len();
        let (stack_roots, stack_slot_addrs) = self.scan_native_stack_roots();
        let native_stack_count = stack_roots.len();
        roots.extend(stack_roots);

        if std::env::var_os("WLIFT_VALIDATE_STACKMAP").is_some() {
            let shadow_count = native_shadow_end - native_shadow_start;
            let jit_entries = crate::codegen::runtime_fns::jit_frame_entries();
            eprintln!(
                "stackmap-validate: shadow={} stack_roots={} jit_frames={} code_ranges={}",
                shadow_count,
                native_stack_count,
                jit_entries.len(),
                self.engine.code_ranges.len(),
            );
        }

        // 8. JitContext GC-managed pointers (closure, defining_class).
        let jit_ctx_start = roots.len();
        let (jit_closure, jit_class) = crate::codegen::runtime_fns::jit_context_roots();
        roots.push(jit_closure);
        roots.push(jit_class);

        // 9. Module variables
        let mut module_ranges: Vec<(String, usize, usize)> = Vec::new();
        for (name, entry) in &self.engine.modules {
            let start = roots.len();
            roots.extend_from_slice(&entry.vars);
            module_ranges.push((name.clone(), start, roots.len()));
        }

        // 10. Reload callbacks — closures registered via Hatch.onReload.
        let reload_cb_start = roots.len();
        roots.extend_from_slice(&self.reload_callbacks);
        let reload_cb_end = roots.len();

        // 10b. Pre-reload callbacks — Hatch.beforeReload.
        let before_reload_cb_start = roots.len();
        roots.extend_from_slice(&self.before_reload_callbacks);
        let before_reload_cb_end = roots.len();

        // 10c. File-watch callbacks — Hatch.watchFile.
        let file_watch_start = roots.len();
        for fw in &self.file_watches {
            roots.push(fw.callback);
        }
        let file_watch_end = roots.len();

        #[cfg(debug_assertions)]
        if std::env::var_os("WLIFT_VALIDATE_BARRIERS").is_some() {
            self.gc.validate_write_barriers();
        }

        // Collect
        self.gc.collect(&mut roots);

        // Write back updated values (GC may have forwarded nursery pointers)
        self.api_stack.copy_from_slice(&roots[..api_len]);
        for (i, h) in self.handles.iter_mut().enumerate() {
            h.value = roots[handles_start + i];
        }

        // Write back core class pointers
        let class_fields: [&mut *mut ObjClass; 20] = [
            &mut self.object_class,
            &mut self.class_class,
            &mut self.bool_class,
            &mut self.num_class,
            &mut self.string_class,
            &mut self.list_class,
            &mut self.map_class,
            &mut self.range_class,
            &mut self.null_class,
            &mut self.fn_class,
            &mut self.fiber_class,
            &mut self.system_class,
            &mut self.sequence_class,
            &mut self.map_sequence_class,
            &mut self.skip_sequence_class,
            &mut self.take_sequence_class,
            &mut self.where_sequence_class,
            &mut self.string_byte_seq_class,
            &mut self.string_code_point_seq_class,
            &mut self.map_entry_class,
        ];
        for (i, field) in class_fields.into_iter().enumerate() {
            let val = roots[classes_start + i];
            if let Some(ptr) = val.as_object() {
                *field = ptr as *mut ObjClass;
            }
        }

        // Write back fiber pointer
        if !self.fiber.is_null() {
            let val = roots[fiber_idx];
            if let Some(ptr) = val.as_object() {
                self.fiber = ptr as *mut ObjFiber;
            }
        }

        // Write back pooled sync fibers (nursery forwarding)
        for (i, fiber) in self.sync_fiber_pool.iter_mut().enumerate() {
            let val = roots[sync_fiber_pool_start + i];
            if let Some(ptr) = val.as_object() {
                *fiber = ptr as *mut ObjFiber;
            }
        }

        // Write back JIT roots (nursery forwarding)
        if jit_roots_end > jit_roots_start {
            crate::codegen::runtime_fns::set_jit_roots(
                roots[jit_roots_start..jit_roots_end].to_vec(),
            );
        }

        if native_shadow_end > native_shadow_start {
            crate::codegen::runtime_fns::set_native_shadow_roots(
                native_shadow_lengths,
                roots[native_shadow_start..native_shadow_end].to_vec(),
            );
        } else if !native_shadow_lengths.is_empty() {
            crate::codegen::runtime_fns::set_native_shadow_roots(native_shadow_lengths, Vec::new());
        }

        // Write back stack map roots directly to native stack spill slots.
        // This ensures GC-forwarded pointers are visible when JIT code resumes.
        for i in 0..native_stack_count {
            let updated = roots[native_stack_start + i];
            unsafe { std::ptr::write(stack_slot_addrs[i], updated.to_bits()) };
        }

        // Write back JitContext pointers (nursery forwarding)
        crate::codegen::runtime_fns::update_jit_context_roots(
            roots[jit_ctx_start],
            roots[jit_ctx_start + 1],
        );

        // Write back module vars (may contain nursery objects that were promoted)
        for (name, start, end) in &module_ranges {
            if let Some(entry) = self.engine.modules.get_mut(name) {
                entry.vars.copy_from_slice(&roots[*start..*end]);
            }
        }

        // Write back reload callbacks (forwarded if their closures were
        // promoted out of the nursery).
        if reload_cb_end > reload_cb_start {
            self.reload_callbacks
                .copy_from_slice(&roots[reload_cb_start..reload_cb_end]);
        }
        if before_reload_cb_end > before_reload_cb_start {
            self.before_reload_callbacks
                .copy_from_slice(&roots[before_reload_cb_start..before_reload_cb_end]);
        }
        if file_watch_end > file_watch_start {
            for (i, fw) in self.file_watches.iter_mut().enumerate() {
                fw.callback = roots[file_watch_start + i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NativeContext implementation for VM
// ---------------------------------------------------------------------------

impl NativeContext for VM {
    fn get_slot(&self, index: usize) -> Value {
        if index < self.api_stack.len() {
            self.api_stack[index]
        } else {
            Value::null()
        }
    }

    fn set_slot(&mut self, index: usize, value: Value) {
        while self.api_stack.len() <= index {
            self.api_stack.push(Value::null());
        }
        self.api_stack[index] = value;
    }

    fn slot_count(&self) -> usize {
        self.api_stack.len()
    }

    fn alloc_string(&mut self, s: String) -> Value {
        self.new_string(s)
    }

    fn alloc_list(&mut self, elements: Vec<Value>) -> Value {
        self.new_list(elements)
    }

    fn alloc_range(&mut self, from: f64, to: f64, inclusive: bool) -> Value {
        self.new_range(from, to, inclusive)
    }

    fn alloc_map(&mut self) -> Value {
        self.new_map()
    }

    fn alloc_typed_array(&mut self, count: u32, kind: TypedArrayKind) -> Value {
        self.new_typed_array(count, kind)
    }

    fn runtime_error(&mut self, msg: String) {
        self.has_error = true;
        self.last_error = Some(msg.clone());
        self.report_error(&msg);
    }

    fn has_error(&self) -> bool {
        self.has_error
    }

    fn get_class_of(&self, value: Value) -> *mut ObjClass {
        self.class_of(value)
    }

    fn get_class_name_of(&self, value: Value) -> String {
        self.class_name_of(value)
    }

    fn intern(&mut self, s: &str) -> SymbolId {
        self.interner.intern(s)
    }

    fn resolve_symbol(&self, id: SymbolId) -> &str {
        self.interner.resolve(id)
    }

    fn write_output(&mut self, s: &str) {
        self.vm_write(s);
    }

    fn alloc_fiber(&mut self) -> *mut ObjFiber {
        self.gc.alloc_fiber()
    }

    fn get_fiber_class(&self) -> *mut ObjClass {
        self.fiber_class
    }

    fn get_fn_class(&self) -> *mut ObjClass {
        self.fn_class
    }

    fn set_fiber_action_call(&mut self, target: *mut ObjFiber, value: Value) {
        self.pending_fiber_action = Some(FiberAction::Call { target, value });
    }

    fn set_fiber_action_yield(&mut self, value: Value) {
        self.pending_fiber_action = Some(FiberAction::Yield { value });
    }

    fn set_fiber_action_transfer(&mut self, target: *mut ObjFiber, value: Value) {
        self.pending_fiber_action = Some(FiberAction::Transfer { target, value });
    }

    fn set_fiber_action_suspend(&mut self) {
        self.pending_fiber_action = Some(FiberAction::Suspend);
    }

    fn get_current_fiber(&self) -> *mut ObjFiber {
        self.fiber
    }

    fn fiber_stack_traces_enabled(&self) -> bool {
        self.config.fiber_stack_traces
    }

    fn capture_spawn_trace(&self) -> Option<Vec<StackFrame>> {
        if !self.config.fiber_stack_traces {
            return None;
        }
        Some(self.capture_current_trace())
    }

    fn get_stack_trace_string(&self, fiber: *mut ObjFiber) -> String {
        let trace = self.build_full_fiber_trace(fiber);
        if trace.is_empty() {
            "<no stack trace>".to_string()
        } else {
            trace.join("\n")
        }
    }

    fn call_method_on(&mut self, receiver: Value, method: &str, args: &[Value]) -> Option<Value> {
        // Special-case: calling a closure/function via call(...)
        // Pass only the call arguments (not the closure receiver) — closures
        // don't have a "self" block param; their block params map directly to
        // the call arguments.
        if method.starts_with("call") && receiver.is_object() {
            let ptr = receiver.as_object().unwrap();
            let header = ptr as *const ObjHeader;
            let is_closure = unsafe { (*header).obj_type == ObjType::Closure };
            if is_closure {
                let closure_ptr = ptr as *mut ObjClosure;
                return self.call_closure_sync(closure_ptr, args, None);
            }
        }

        let class = self.class_of(receiver);
        let method_sym = self.interner.intern(method);
        let method_entry = unsafe {
            match (*class).find_method(method_sym).cloned() {
                Some(m) => m,
                None => {
                    // If receiver is a class value, try static: prefix on the class itself
                    if receiver.is_object() {
                        let ptr = receiver.as_object().unwrap();
                        let header = ptr as *const ObjHeader;
                        if (*header).obj_type == ObjType::Class {
                            let recv_class = ptr as *mut ObjClass;
                            let static_sig = format!("static:{}", method);
                            let static_sym = self.interner.intern(&static_sig);
                            *(*recv_class).find_method(static_sym)?
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
            }
        };

        let mut all_args = Vec::with_capacity(1 + args.len());
        all_args.push(receiver);
        all_args.extend_from_slice(args);

        match method_entry {
            m @ (Method::Native(_) | Method::ForeignC(_)) => {
                let root_len_before = crate::codegen::runtime_fns::jit_roots_snapshot_len();
                for &arg in &all_args {
                    crate::codegen::runtime_fns::push_jit_root(arg);
                }
                let rooted_args: Vec<Value> = (0..all_args.len())
                    .map(|idx| crate::codegen::runtime_fns::jit_root_at(root_len_before + idx))
                    .collect();
                let result = Some(match m {
                    Method::Native(func) => func(self, &rooted_args),
                    Method::ForeignC(func) => {
                        crate::runtime::foreign::dispatch_foreign_c(self, func, &rooted_args)
                    }
                    _ => unreachable!(),
                });
                crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                result
            }
            Method::Closure(closure_ptr) => self.call_closure_sync(closure_ptr, &all_args, None),
            Method::Constructor(closure_ptr) => {
                let class_ptr = receiver
                    .as_object()
                    .map(|p| p as *mut ObjClass)
                    .unwrap_or(std::ptr::null_mut());
                if class_ptr.is_null() {
                    None
                } else {
                    Some(self.call_constructor_sync(class_ptr, closure_ptr, args))
                }
            }
        }
    }

    fn alloc_foreign(&mut self, data: Vec<u8>) -> *mut ObjForeign {
        let ptr = self.gc.alloc_foreign(data);
        unsafe {
            (*ptr).header.class = self.object_class;
        }
        ptr
    }

    fn new_list(&mut self) -> Value {
        self.alloc_list(Vec::new())
    }

    fn get_module_variable_names(&self, module: &str) -> Option<Vec<String>> {
        self.engine
            .modules
            .get(module)
            .map(|entry| entry.var_names.clone())
    }

    fn meta_eval(&mut self, _module: &str, source: &str) -> bool {
        // Find the calling module from the current fiber's frame
        let eval_module: String = if !self.fiber.is_null() {
            unsafe {
                (*self.fiber)
                    .mir_frames
                    .last()
                    .map(|f| (*f.module_name).clone())
                    .unwrap_or_else(|| "main".to_string())
            }
        } else {
            "main".to_string()
        };
        self.eval_source_in_module(&eval_module, source)
    }

    fn meta_compile(&mut self, source: &str) -> Option<Value> {
        self.compile_to_closure(source, false)
    }

    fn meta_compile_expression(&mut self, expr: &str) -> Option<Value> {
        self.compile_to_closure(expr, true)
    }

    fn alloc_instance(&mut self, class: *mut ObjClass) -> Value {
        let inst = self.gc.alloc_instance(class);
        unsafe {
            (*inst).header.class = class;
        }
        Value::object(inst as *mut u8)
    }

    fn lookup_class(&self, name: &str) -> Option<*mut ObjClass> {
        let cls = match name {
            "MapSequence" => self.map_sequence_class,
            "SkipSequence" => self.skip_sequence_class,
            "TakeSequence" => self.take_sequence_class,
            "WhereSequence" => self.where_sequence_class,
            "StringByteSequence" => self.string_byte_seq_class,
            "StringCodePointSequence" => self.string_code_point_seq_class,
            "MapEntry" => self.map_entry_class,
            "Object" => self.object_class,
            "Sequence" => self.sequence_class,
            "String" => self.string_class,
            "List" => self.list_class,
            "Map" => self.map_class,
            _ => return None,
        };
        if cls.is_null() {
            None
        } else {
            Some(cls)
        }
    }

    fn trigger_gc(&mut self) {
        self.gc_requested = true;
    }

    fn write_barrier(&mut self, source: Value, value: Value) {
        if let Some(ptr) = source.as_object() {
            self.gc.write_barrier(ptr as *mut ObjHeader, value);
        }
    }

    fn func_module(&self, func_id: u32) -> Option<std::rc::Rc<String>> {
        self.engine
            .func_module(crate::runtime::engine::FuncId(func_id))
            .cloned()
    }

    fn reload_module(&mut self, name: &str) -> Result<(), String> {
        VM::reload_module(self, name)
    }

    fn module_canonical_path(&self, name: &str) -> Option<String> {
        // Relative imports are canonicalised to absolute paths before
        // being registered, so the module name IS the on-disk path
        // for user modules. Built-ins register under short names
        // (`meta`, `hatch`) which don't correspond to a file.
        if std::path::Path::new(name).is_absolute() {
            Some(name.to_string())
        } else {
            None
        }
    }

    fn loaded_module_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.engine.modules.keys().cloned().collect();
        names.sort();
        names
    }

    fn register_reload_callback(&mut self, callback: Value) {
        let new_ptr = callback.as_object().map(|p| p as usize);
        for existing in &self.reload_callbacks {
            if existing.as_object().map(|p| p as usize) == new_ptr {
                return;
            }
        }
        self.reload_callbacks.push(callback);
    }

    fn register_before_reload_callback(&mut self, callback: Value) {
        let new_ptr = callback.as_object().map(|p| p as usize);
        for existing in &self.before_reload_callbacks {
            if existing.as_object().map(|p| p as usize) == new_ptr {
                return;
            }
        }
        self.before_reload_callbacks.push(callback);
    }

    fn register_file_watch(&mut self, path: String, callback: Value) {
        // De-dupe by (path, callback ptr). Same path watched with a
        // different closure registers a second entry — useful when
        // multiple subsystems care about the same file.
        let new_ptr = callback.as_object().map(|p| p as usize);
        for fw in &self.file_watches {
            if fw.path == path && fw.callback.as_object().map(|p| p as usize) == new_ptr {
                return;
            }
        }
        // Capture the current mtime as the baseline so the next
        // signal compares against it (we don't fire immediately on
        // registration).
        let last_mtime = file_mtime_secs(&path).unwrap_or(0.0);
        self.file_watches.push(FileWatch {
            path,
            callback,
            last_mtime,
        });
    }
}

// ---------------------------------------------------------------------------
// JIT re-entry helpers
// ---------------------------------------------------------------------------

impl VM {
    /// Compile and execute Wren source within an existing module's scope.
    ///
    /// The calling module's variable names are added to the resolver prelude
    /// so the eval'd code can reference them. The compiled code runs on a
    /// temporary fiber with the module's variable storage.
    fn eval_source_in_module(&mut self, module_name: &str, source: &str) -> bool {
        use crate::diagnostics::Severity;
        use crate::mir::opt::{
            constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize, licm::Licm,
            run_to_fixpoint, sra::Sra, MirPass,
        };
        use crate::parse::parser;

        // 1. Parse
        let parse_result = parser::parse(source);
        if parse_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &parse_result.errors {
                err.eprint(source);
            }
            return false;
        }

        // 2. Resolve with core classes + calling module's variables in prelude
        let mut interner = parse_result.interner;
        let core_names = [
            "Object",
            "Class",
            "Bool",
            "Num",
            "String",
            "List",
            "Map",
            "Range",
            "Null",
            "Fn",
            "Fiber",
            "System",
            "Sequence",
            "ByteArray",
            "Float32Array",
            "Float64Array",
        ];
        let mut prelude: Vec<crate::intern::SymbolId> =
            core_names.iter().map(|n| interner.intern(n)).collect();

        // Add calling module's variable names to prelude
        if let Some(entry) = self.engine.modules.get(module_name) {
            for name in &entry.var_names {
                prelude.push(interner.intern(name));
            }
        }

        let resolve_result =
            crate::sema::resolve::resolve_with_prelude(&parse_result.module, &interner, &prelude);
        if resolve_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &resolve_result.errors {
                err.eprint(source);
            }
            return false;
        }

        // 3. Lower + optimize. `Meta.eval` compiles against the
        // calling module's scope; thread field_layouts through so
        // any subclass of an already-loaded parent lines up.
        let (mut module_mir, new_layouts) = crate::mir::builder::lower_module_with_known_classes(
            &parse_result.module,
            &mut interner,
            &resolve_result,
            &self.field_layouts,
        );
        for (name, layout) in new_layouts {
            self.field_layouts.insert(name, layout);
        }
        let constfold = ConstFold;
        let dce = Dce;
        let cse = Cse::default();
        let type_spec = TypeSpecialize::with_math(&interner);
        let licm = Licm;
        let sra = Sra;
        let passes: Vec<&dyn MirPass> = vec![
            &constfold, &dce, &cse, &type_spec, &constfold, &dce, &licm, &sra, &dce,
        ];
        run_to_fixpoint(&mut module_mir.top_level, &passes, 10);

        // 4. Remap symbols
        let mut sym_map: Vec<crate::intern::SymbolId> = Vec::with_capacity(interner.len());
        for i in 0..interner.len() {
            let old_sym = crate::intern::SymbolId::from_raw(i as u32);
            let s = interner.resolve(old_sym);
            let new_sym = self.interner.intern(s);
            sym_map.push(new_sym);
        }
        module_mir
            .top_level
            .remap_symbols(|old| sym_map[old.index() as usize]);

        // 5. Register closures if any
        let eval_module_rc = std::rc::Rc::new(module_name.to_string());
        let mut closure_func_ids: Vec<u32> = Vec::new();
        for closure in module_mir.closures.drain(..) {
            let fid = self
                .engine
                .register_function_in(closure, Some(std::rc::Rc::clone(&eval_module_rc)));
            closure_func_ids.push(fid.0);
        }
        patch_closure_ids(&mut module_mir.top_level, &closure_func_ids);

        // 6. Register the eval function and execute on temp fiber with module vars
        let func_id = self
            .engine
            .register_function_in(module_mir.top_level, Some(eval_module_rc));

        // Build eval module vars in the order the resolver expects, looking up
        // values by name from the calling module (to match slot indices).
        let calling_var_names: Vec<String> = self
            .engine
            .modules
            .get(module_name)
            .map(|e| e.var_names.clone())
            .unwrap_or_default();
        let calling_vars: Vec<Value> = self
            .engine
            .modules
            .get(module_name)
            .map(|e| e.vars.clone())
            .unwrap_or_default();

        let mut eval_vars = Vec::with_capacity(resolve_result.module_vars.len());
        let mut eval_var_names = Vec::with_capacity(resolve_result.module_vars.len());
        for &parse_sym in &resolve_result.module_vars {
            let name = interner.resolve(parse_sym);
            eval_var_names.push(name.to_string());
            if let Some(idx) = calling_var_names.iter().position(|n| n == name) {
                eval_vars.push(calling_vars[idx]);
            } else if let Some(value) = self.core_class_value(name) {
                eval_vars.push(value);
            } else if let Some(value) = self.find_imported_var(name) {
                eval_vars.push(value);
            } else {
                eval_vars.push(Value::null());
            }
        }

        let eval_module_name = format!("__eval_{}__", module_name);
        self.engine.modules.insert(
            eval_module_name.clone(),
            super::engine::ModuleEntry {
                top_level: func_id,
                vars: eval_vars,
                var_names: eval_var_names,
            },
        );

        let fiber = self.gc.alloc_fiber();
        let reg_size = self
            .engine
            .get_mir(func_id)
            .map(|m| m.next_value as usize)
            .unwrap_or(0);
        unsafe {
            (*fiber).header.class = self.fiber_class;
            (*fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: crate::mir::BlockId(0),
                ip: 0,
                pc: 0,
                values: vec![Value::UNDEFINED; reg_size],
                module_name: std::rc::Rc::new(eval_module_name.clone()),
                return_dst: None,
                closure: None,
                defining_class: None,
                bc_ptr: std::ptr::null(),
            });
        }

        let prev_fiber = self.fiber;
        self.fiber = fiber;
        let result = super::vm_interp::run_fiber(self);
        self.fiber = prev_fiber;

        // Write back eval module vars to the calling module by name
        if let Some(eval_entry) = self.engine.modules.get(&eval_module_name) {
            let eval_names = eval_entry.var_names.clone();
            let eval_vals = eval_entry.vars.clone();
            if let Some(calling_entry) = self.engine.modules.get_mut(module_name) {
                for (i, name) in eval_names.iter().enumerate() {
                    if let Some(j) = calling_entry.var_names.iter().position(|n| n == name) {
                        if i < eval_vals.len() {
                            calling_entry.vars[j] = eval_vals[i];
                        }
                    }
                }
            }
        }

        // Cleanup temp module
        self.engine.modules.remove(&eval_module_name);

        result.is_ok()
    }

    /// Compile Wren source into a callable closure value.
    ///
    /// If `is_expression` is true, wraps the source as `return <expr>\n`
    /// so calling the closure evaluates the expression.
    fn compile_to_closure(&mut self, source: &str, is_expression: bool) -> Option<Value> {
        use crate::diagnostics::Severity;
        use crate::mir::opt::{
            constfold::ConstFold, cse::Cse, dce::Dce, inline::TypeSpecialize, licm::Licm,
            run_to_fixpoint, sra::Sra, MirPass,
        };
        use crate::parse::parser;

        // For expressions, wrap as a function body that returns the value
        let wrapped = if is_expression {
            format!("return {}\n", source)
        } else {
            source.to_string()
        };

        // 1. Parse
        let parse_result = parser::parse(&wrapped);
        if parse_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &parse_result.errors {
                err.eprint(&wrapped);
            }
            return None;
        }

        // 2. Resolve names with core prelude
        let mut interner = parse_result.interner;
        let core_names = [
            "Object",
            "Class",
            "Bool",
            "Num",
            "String",
            "List",
            "Map",
            "Range",
            "Null",
            "Fn",
            "Fiber",
            "System",
            "Sequence",
            "ByteArray",
            "Float32Array",
            "Float64Array",
        ];
        let prelude: Vec<crate::intern::SymbolId> =
            core_names.iter().map(|n| interner.intern(n)).collect();
        let resolve_result =
            crate::sema::resolve::resolve_with_prelude(&parse_result.module, &interner, &prelude);
        if resolve_result
            .errors
            .iter()
            .any(|d| d.severity == Severity::Error)
        {
            for err in &resolve_result.errors {
                err.eprint(&wrapped);
            }
            return None;
        }

        // 3. Lower to MIR and optimize. This is `Meta.compile` —
        // compile source against the VM's module graph so inherited
        // fields line up with parents in other modules.
        let (mut module_mir, new_layouts) = crate::mir::builder::lower_module_with_known_classes(
            &parse_result.module,
            &mut interner,
            &resolve_result,
            &self.field_layouts,
        );
        for (name, layout) in new_layouts {
            self.field_layouts.insert(name, layout);
        }
        let constfold = ConstFold;
        let dce = Dce;
        let cse = Cse::default();
        let type_spec = TypeSpecialize::with_math(&interner);
        let licm = Licm;
        let sra = Sra;
        let passes: Vec<&dyn MirPass> = vec![
            &constfold, &dce, &cse, &type_spec, &constfold, &dce, &licm, &sra, &dce,
        ];
        run_to_fixpoint(&mut module_mir.top_level, &passes, 10);

        // 4. Remap symbols to VM interner
        let mut sym_map: Vec<crate::intern::SymbolId> = Vec::with_capacity(interner.len());
        for i in 0..interner.len() {
            let old_sym = crate::intern::SymbolId::from_raw(i as u32);
            let s = interner.resolve(old_sym);
            let new_sym = self.interner.intern(s);
            sym_map.push(new_sym);
        }
        module_mir
            .top_level
            .remap_symbols(|old| sym_map[old.index() as usize]);
        for closure in &mut module_mir.closures {
            closure.remap_symbols(|old| sym_map[old.index() as usize]);
        }

        // 5. Patch + register nested closures before the top-level so
        // `MakeClosure` ops inside the compiled source resolve to the
        // engine FuncIds we actually assigned. Without this the local
        // 0-based closure slots are taken literally, which maps to
        // whatever function happens to sit at that engine FuncId
        // (often the `Meta.compile` top-level itself — hence the
        // infinite-recursion stack overflow we used to see when
        // calling the returned Fn).
        //
        // The compiled code was resolved against a core-only prelude,
        // so it expects module vars laid out in that order. Build a
        // fresh module entry populated with the core-class values and
        // bind every function we register here to it. Using a unique
        // name keeps repeat `Meta.compile` calls from stomping each
        // other's slot tables.
        let module_key = format!("<compiled#{}>", self.engine.functions.len());
        let compile_module_rc = std::rc::Rc::new(module_key.clone());

        let mut compiled_vars: Vec<Value> = Vec::with_capacity(core_names.len());
        let mut compiled_var_names: Vec<String> = Vec::with_capacity(core_names.len());
        for name in core_names.iter() {
            let v = self.core_class_value(name).unwrap_or(Value::null());
            compiled_vars.push(v);
            compiled_var_names.push((*name).to_string());
        }

        let closure_count = module_mir.closures.len();
        let base_fid = self.engine.functions.len() as u32;
        let closure_func_ids: Vec<u32> = (0..closure_count as u32).map(|i| base_fid + i).collect();
        patch_closure_ids(&mut module_mir.top_level, &closure_func_ids);
        for closure in &mut module_mir.closures {
            patch_closure_ids(closure, &closure_func_ids);
        }
        for (i, closure) in module_mir.closures.drain(..).enumerate() {
            let fid = self
                .engine
                .register_function_in(closure, Some(std::rc::Rc::clone(&compile_module_rc)));
            debug_assert_eq!(fid.0, closure_func_ids[i]);
        }

        // 6. Register and create closure for the top-level.
        let func_id = self
            .engine
            .register_function_in(module_mir.top_level, Some(compile_module_rc));

        // 6b. Register the synthetic module so `GetModuleVar` can
        // resolve the core-class slots the compiled code was bound
        // against.
        self.engine.modules.insert(
            module_key,
            super::engine::ModuleEntry {
                top_level: func_id,
                vars: compiled_vars,
                var_names: compiled_var_names,
            },
        );
        let fn_name = self.interner.intern("<compiled>");
        let fn_ptr = self.gc.alloc_fn(fn_name, 0, 0, func_id.0);
        unsafe {
            (*fn_ptr).header.class = self.fn_class;
            (*fn_ptr).trivial_getter_field = self
                .engine
                .trivial_getter_fields
                .get(func_id.0 as usize)
                .copied()
                .flatten()
                .unwrap_or(u16::MAX);
            (*fn_ptr).trivial_setter_field = self
                .engine
                .trivial_setter_fields
                .get(func_id.0 as usize)
                .copied()
                .flatten()
                .unwrap_or(u16::MAX);
        }
        let closure_ptr = self.gc.alloc_closure(fn_ptr);
        unsafe {
            (*closure_ptr).header.class = self.fn_class;
        }

        Some(Value::object(closure_ptr as *mut u8))
    }

    /// Execute a closure synchronously via a temporary fiber.
    /// Used by JIT runtime functions to re-enter the interpreter for closure bodies.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn call_closure_sync(
        &mut self,
        closure_ptr: *mut ObjClosure,
        args: &[Value],
        defining_class: Option<*mut crate::runtime::object::ObjClass>,
    ) -> Option<Value> {
        use crate::mir::{BlockId, Instruction};

        // Root the callee and argument values before alloc_fiber(), which may
        // trigger GC while these values still live only in the native caller.
        let root_len_before = crate::codegen::runtime_fns::jit_roots_snapshot_len();
        crate::codegen::runtime_fns::push_jit_root(Value::object(closure_ptr as *mut u8));
        crate::codegen::runtime_fns::push_jit_root(
            defining_class
                .map(|p| Value::object(p as *mut u8))
                .unwrap_or(Value::null()),
        );
        for &arg in args {
            crate::codegen::runtime_fns::push_jit_root(arg);
        }

        let live_closure = match crate::codegen::runtime_fns::jit_root_at(root_len_before)
            .as_object()
            .map(|p| p as *mut ObjClosure)
        {
            Some(ptr) => ptr,
            None => {
                crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                return None;
            }
        };
        let live_defining_class = crate::codegen::runtime_fns::jit_root_at(root_len_before + 1)
            .as_object()
            .map(|p| p as *mut crate::runtime::object::ObjClass);
        let current_fiber = self.fiber;
        if !current_fiber.is_null() {
            let fn_ptr = unsafe { (*live_closure).function };
            let func_id = crate::runtime::engine::FuncId(unsafe { (*fn_ptr).fn_id });

            let mir = match self.engine.get_mir(func_id) {
                Some(mir) => mir,
                None => {
                    crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                    return None;
                }
            };
            let mut values = vec![Value::UNDEFINED; mir.next_value as usize];

            let block = &mir.blocks[0];
            for (vid, inst) in &block.instructions {
                if let Instruction::BlockParam(idx) = inst {
                    let param_i = *idx as usize;
                    if param_i < args.len() {
                        let i = vid.0 as usize;
                        if i >= values.len() {
                            values.resize(i + 1, Value::UNDEFINED);
                        }
                        values[i] =
                            crate::codegen::runtime_fns::jit_root_at(root_len_before + 2 + param_i);
                    }
                } else {
                    break;
                }
            }

            // Prefer the callee's defining module (recorded at
            // register time). Fall back to the current frame's module
            // for test paths that don't go through the module registry.
            let mod_name = self
                .engine
                .func_module(func_id)
                .cloned()
                .unwrap_or_else(|| unsafe {
                    (*current_fiber)
                        .mir_frames
                        .last()
                        .map(|f| f.module_name.clone())
                        .unwrap_or_else(|| {
                            std::rc::Rc::new(crate::codegen::runtime_fns::module_name())
                        })
                });
            let stop_depth = unsafe { (*current_fiber).mir_frames.len() };
            if stop_depth >= self.config.max_call_depth {
                crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                return None;
            }

            unsafe {
                (*current_fiber).mir_frames.push(MirCallFrame {
                    func_id,
                    current_block: BlockId(0),
                    ip: 0,
                    pc: 0,
                    values,
                    module_name: mod_name,
                    return_dst: None,
                    closure: Some(live_closure),
                    defining_class: live_defining_class,
                    bc_ptr: std::ptr::null(),
                });
            }

            let result = super::vm_interp::run_fiber_until_depth(self, stop_depth);
            crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
            return result.ok();
        }

        let temp_fiber = self.acquire_sync_fiber();
        unsafe {
            (*temp_fiber).header.class = self.fiber_class;

            let fn_ptr = (*live_closure).function;
            let func_id = crate::runtime::engine::FuncId((*fn_ptr).fn_id);

            let mir = match self.engine.get_mir(func_id) {
                Some(mir) => mir,
                None => {
                    crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                    return None;
                }
            };
            let mut values = vec![Value::UNDEFINED; mir.next_value as usize];

            // Bind block params using the actual BlockParam index (not a
            // sequential counter), because unused params may be eliminated.
            let block = &mir.blocks[0];
            for (vid, inst) in &block.instructions {
                if let Instruction::BlockParam(idx) = inst {
                    let param_i = *idx as usize;
                    if param_i < args.len() {
                        let i = vid.0 as usize;
                        if i >= values.len() {
                            values.resize(i + 1, Value::UNDEFINED);
                        }
                        values[i] =
                            crate::codegen::runtime_fns::jit_root_at(root_len_before + 2 + param_i);
                    }
                } else {
                    break;
                }
            }

            // Prefer the callee's defining module. Fall back to the
            // current fiber's module for test paths that don't
            // register functions through the module registry.
            let mod_name = self
                .engine
                .func_module(func_id)
                .cloned()
                .unwrap_or_else(|| {
                    if !self.fiber.is_null() {
                        (*self.fiber)
                            .mir_frames
                            .last()
                            .map(|f| f.module_name.clone())
                            .unwrap_or_else(|| {
                                std::rc::Rc::new(crate::codegen::runtime_fns::module_name())
                            })
                    } else {
                        std::rc::Rc::new(crate::codegen::runtime_fns::module_name())
                    }
                });

            (*temp_fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                pc: 0,
                values,
                module_name: mod_name,
                return_dst: None,
                closure: Some(live_closure),
                defining_class: live_defining_class,
                bc_ptr: std::ptr::null(),
            });
        }

        let saved_fiber = self.fiber;
        // Root the saved fiber so GC can trace it during re-entrant interpreter
        // calls (it holds roots from the outer native/interpreter frame).
        // We can't use fiber.caller because that has semantic meaning (fiber resumption).
        let jit_root_idx = if !saved_fiber.is_null() {
            let idx = crate::codegen::runtime_fns::jit_roots_snapshot_len();
            crate::codegen::runtime_fns::push_jit_root(Value::object(saved_fiber as *mut u8));
            Some(idx)
        } else {
            None
        };
        self.fiber = temp_fiber;
        let result = super::vm_interp::run_fiber(self);
        let live_temp_fiber = self.fiber;
        // Restore fiber — read from JIT roots in case GC forwarded the pointer.
        if let Some(idx) = jit_root_idx {
            let forwarded = crate::codegen::runtime_fns::jit_root_at(idx);
            self.fiber = forwarded
                .as_object()
                .map(|p| p as *mut super::object::ObjFiber)
                .unwrap_or(saved_fiber);
            crate::codegen::runtime_fns::jit_roots_restore_len(idx);
        } else {
            self.fiber = saved_fiber;
        }
        crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
        self.release_sync_fiber(live_temp_fiber);

        result.ok()
    }

    /// Dispatch a constructor call from JIT code safely.
    ///
    /// When a constructor's JIT code is called, it would receive the CLASS as
    /// `this` (arg 0) instead of a newly allocated instance, causing SIGSEGV.
    /// This function allocates the instance, roots it across GC, and runs the
    /// constructor body in the interpreter.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn call_constructor_sync(
        &mut self,
        class_ptr: *mut crate::runtime::object::ObjClass,
        closure_ptr: *mut ObjClosure,
        ctor_args: &[Value], // args WITHOUT the class receiver (just the user args)
    ) -> Value {
        // Allocate the new instance. This may trigger a minor GC.
        let instance_raw = self.gc.alloc_instance(class_ptr);
        let instance_val = Value::object(instance_raw as *mut u8);

        // Root the instance across any GC that may occur inside alloc_fiber below.
        let root_len_before = crate::codegen::runtime_fns::jit_roots_snapshot_len();
        crate::codegen::runtime_fns::push_jit_root(instance_val);

        let func_id = crate::runtime::engine::FuncId(unsafe {
            let fn_ptr = (*closure_ptr).function;
            (*fn_ptr).fn_id
        });

        self.call_constructor_sync_impl(class_ptr, closure_ptr, ctor_args, func_id, root_len_before)
    }

    fn call_constructor_sync_impl(
        &mut self,
        class_ptr: *mut crate::runtime::object::ObjClass,
        closure_ptr: *mut ObjClosure,
        ctor_args: &[Value],
        func_id: crate::runtime::engine::FuncId,
        root_len_before: usize,
    ) -> Value {
        use crate::mir::{BlockId, Instruction};
        // Try JIT dispatch: if the constructor body is compiled,
        // call it directly with the new instance as `this`.
        // Don't use JIT if this constructor has active interpreter frames
        // on the fiber stack — dispatching to JIT mid-recursion while
        // interpreter frames are still executing causes incorrect results.
        let fn_idx = func_id.0 as usize;
        let has_active_interp_frames = if !self.fiber.is_null() {
            let fiber = unsafe { &*self.fiber };
            fiber.mir_frames.iter().any(|f| f.func_id == func_id)
        } else {
            false
        };
        let jit_ptr = if !has_active_interp_frames {
            self.engine
                .jit_code
                .get(fn_idx)
                .copied()
                .unwrap_or(std::ptr::null())
        } else {
            std::ptr::null()
        };
        if !jit_ptr.is_null() && ctor_args.len() <= 3 {
            // Push every ctor arg as a JIT root so a GC fired during
            // the constructor body (allocations, foreign calls, etc.)
            // updates each pointer through the shared roots Vec
            // before the body dereferences it. The instance itself
            // was already rooted at root_len_before by
            // `call_constructor_sync`. Without this, any pointer arg
            // staled to freed memory while the JIT'd ctor ran with
            // the args already loaded into registers — which is the
            // class of "Constructor JIT SIGSEGV under GC pressure"
            // bug logged in CLAUDE.md / project_cranelift_fixes.
            for arg in ctor_args.iter() {
                crate::codegen::runtime_fns::push_jit_root(*arg);
            }
            // Re-read the instance + args from roots after the
            // pushes (the Vec may have grown / moved its backing
            // buffer; the underlying Value bits are stable, but
            // reading via `jit_root_at` keeps every value in
            // lockstep with the GC's view).
            let live_instance = crate::codegen::runtime_fns::jit_root_at(root_len_before);
            let mut jit_args = [Value::null(); 4];
            jit_args[0] = live_instance;
            for i in 0..ctor_args.len() {
                jit_args[i + 1] = crate::codegen::runtime_fns::jit_root_at(root_len_before + 1 + i);
            }
            let n = ctor_args.len() + 1;
            let saved_ctx = crate::codegen::runtime_fns::read_jit_ctx();
            crate::codegen::runtime_fns::mutate_jit_ctx(|ctx| {
                ctx.current_func_id = func_id.0 as u64;
                ctx.closure = closure_ptr as *mut u8;
                ctx.defining_class = class_ptr as *mut u8;
            });
            self.engine.note_native_entry(func_id);
            // Use call_jit_fn which sets x20 (JitContext pointer).
            // The constructor body's internal calls go through wren_call_N
            // which registers the JIT frame for GC via #[naked] wrappers.
            let _ = unsafe { super::vm_interp::call_jit_fn_pub(jit_ptr, &jit_args[..n]) };
            crate::codegen::runtime_fns::set_jit_context(saved_ctx);
            // Constructor returns the instance (possibly GC-forwarded
            // — read from the root slot, not the result register).
            let result = crate::codegen::runtime_fns::jit_root_at(root_len_before);

            crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
            return result;
        }

        let mir = self.engine.get_mir(func_id);
        if mir.is_none() {
            crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
            return crate::codegen::runtime_fns::jit_root_at(root_len_before);
        }
        let mir = mir.unwrap();

        let current_fiber = self.fiber;
        if !current_fiber.is_null() {
            let live_instance = crate::codegen::runtime_fns::jit_root_at(root_len_before);
            let mut values = vec![Value::UNDEFINED; mir.next_value as usize];

            let block = &mir.blocks[0];
            for (vid, inst) in &block.instructions {
                if let Instruction::BlockParam(idx) = inst {
                    let param_i = *idx as usize;
                    let val = if param_i == 0 {
                        live_instance
                    } else if param_i - 1 < ctor_args.len() {
                        ctor_args[param_i - 1]
                    } else {
                        Value::UNDEFINED
                    };
                    let i = vid.0 as usize;
                    if i >= values.len() {
                        values.resize(i + 1, Value::UNDEFINED);
                    }
                    values[i] = val;
                } else {
                    break;
                }
            }

            let mod_name = self
                .engine
                .func_module(func_id)
                .cloned()
                .unwrap_or_else(|| unsafe {
                    (*current_fiber)
                        .mir_frames
                        .last()
                        .map(|f| f.module_name.clone())
                        .unwrap_or_else(|| {
                            std::rc::Rc::new(crate::codegen::runtime_fns::module_name())
                        })
                });
            let stop_depth = unsafe { (*current_fiber).mir_frames.len() };
            if stop_depth >= self.config.max_call_depth {
                crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
                return live_instance;
            }

            unsafe {
                (*current_fiber).mir_frames.push(MirCallFrame {
                    func_id,
                    current_block: BlockId(0),
                    ip: 0,
                    pc: 0,
                    values,
                    module_name: mod_name,
                    return_dst: None,
                    closure: Some(closure_ptr),
                    defining_class: Some(class_ptr),
                    bc_ptr: std::ptr::null(),
                });
            }

            let result = super::vm_interp::run_fiber_until_depth(self, stop_depth);
            crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);
            return match result {
                Ok(v) if !v.is_null() => v,
                _ => live_instance,
            };
        }

        // alloc_fiber may trigger GC — read back the (possibly forwarded) instance
        // from the JIT root AFTER allocation so we always have the live pointer.
        let temp_fiber = self.acquire_sync_fiber();
        let live_instance = crate::codegen::runtime_fns::jit_root_at(root_len_before);
        crate::codegen::runtime_fns::jit_roots_restore_len(root_len_before);

        unsafe {
            (*temp_fiber).header.class = self.fiber_class;

            let mut values = vec![Value::UNDEFINED; mir.next_value as usize];

            // Bind BlockParam(0) = instance (this), BlockParam(1..N) = ctor_args
            let block = &mir.blocks[0];
            for (vid, inst) in &block.instructions {
                if let Instruction::BlockParam(idx) = inst {
                    let param_i = *idx as usize;
                    let val = if param_i == 0 {
                        live_instance
                    } else if param_i - 1 < ctor_args.len() {
                        ctor_args[param_i - 1]
                    } else {
                        Value::UNDEFINED
                    };
                    let i = vid.0 as usize;
                    if i >= values.len() {
                        values.resize(i + 1, Value::UNDEFINED);
                    }
                    values[i] = val;
                } else {
                    break;
                }
            }

            let mod_name = self
                .engine
                .func_module(func_id)
                .cloned()
                .unwrap_or_else(|| {
                    if !self.fiber.is_null() {
                        (*self.fiber)
                            .mir_frames
                            .last()
                            .map(|f| f.module_name.clone())
                            .unwrap_or_else(|| {
                                std::rc::Rc::new(crate::codegen::runtime_fns::module_name())
                            })
                    } else {
                        std::rc::Rc::new(crate::codegen::runtime_fns::module_name())
                    }
                });

            (*temp_fiber).mir_frames.push(MirCallFrame {
                func_id,
                current_block: BlockId(0),
                ip: 0,
                pc: 0,
                values,
                module_name: mod_name,
                return_dst: None,
                closure: Some(closure_ptr),
                defining_class: Some(class_ptr),
                bc_ptr: std::ptr::null(),
            });
        }

        let saved_fiber = self.fiber;
        let jit_root_idx = if !saved_fiber.is_null() {
            let idx = crate::codegen::runtime_fns::jit_roots_snapshot_len();
            crate::codegen::runtime_fns::push_jit_root(Value::object(saved_fiber as *mut u8));
            Some(idx)
        } else {
            None
        };
        self.fiber = temp_fiber;
        let result = super::vm_interp::run_fiber(self);
        let live_temp_fiber = self.fiber;
        // Restore fiber — read from JIT roots in case GC forwarded the pointer.
        if let Some(idx) = jit_root_idx {
            let forwarded = crate::codegen::runtime_fns::jit_root_at(idx);
            self.fiber = forwarded
                .as_object()
                .map(|p| p as *mut super::object::ObjFiber)
                .unwrap_or(saved_fiber);
            crate::codegen::runtime_fns::jit_roots_restore_len(idx);
        } else {
            self.fiber = saved_fiber;
        }
        self.release_sync_fiber(live_temp_fiber);

        // The constructor body returns `this` (the instance). If it returns null
        // or errors, fall back to live_instance.
        match result {
            Ok(v) if !v.is_null() => v,
            _ => live_instance,
        }
    }
}

impl Drop for VM {
    fn drop(&mut self) {
        if std::env::var_os("WLIFT_TIER_STATS").is_some() {
            self.engine.dump_tier_stats(&self.interner);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Patch MakeClosure fn_id indices to actual engine FuncIds.
fn patch_closure_ids(func: &mut crate::mir::MirFunction, closure_func_ids: &[u32]) {
    for block in &mut func.blocks {
        for (_, inst) in &mut block.instructions {
            if let crate::mir::Instruction::MakeClosure { fn_id, .. } = inst {
                if let Some(&actual_id) = closure_func_ids.get(*fn_id as usize) {
                    *fn_id = actual_id;
                }
            }
        }
    }
}

#[allow(dead_code)] // Kept for future eager-compile heuristics on module entries.
fn should_eager_compile_entry(
    mir: &crate::mir::MirFunction,
    interner: &crate::intern::Interner,
) -> bool {
    let constructor_calls = mir
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .filter_map(|(_, inst)| match inst {
            crate::mir::Instruction::Call { method, .. } => Some(*method),
            _ => None,
        })
        .filter(|method| {
            let name = interner.resolve(*method);
            name == "new" || name.starts_with("new(")
        })
        .count();

    constructor_calls <= 2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// Strip ANSI escape sequences from a string for assertion matching.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                // Skip until 'm' (end of ANSI escape)
                for esc in chars.by_ref() {
                    if esc == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    fn call_primitive(vm: &mut VM, class: *mut ObjClass, sig: &str, args: &[Value]) -> Value {
        let sym = vm.interner.intern(sig);
        let method = unsafe { (*class).find_method(sym).cloned() };
        match method {
            Some(Method::Native(func)) => func(vm, args),
            Some(Method::ForeignC(func)) => {
                crate::runtime::foreign::dispatch_foreign_c(vm, func, args)
            }
            _ => panic!("Method '{}' not found", sig),
        }
    }

    fn call_static(vm: &mut VM, class: *mut ObjClass, sig: &str, args: &[Value]) -> Value {
        let full = format!("static:{}", sig);
        let sym = vm.interner.intern(&full);
        let method = unsafe { (*class).find_method(sym).cloned() };
        match method {
            Some(Method::Native(func)) => func(vm, args),
            Some(Method::ForeignC(func)) => {
                crate::runtime::foreign::dispatch_foreign_c(vm, func, args)
            }
            _ => panic!("Static method '{}' not found", sig),
        }
    }

    /// Create a VM with output capture enabled. Run source, return (result, output).
    fn run_and_capture(source: &str) -> (InterpretResult, String) {
        let mut vm = VM::new_default();
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", source);
        let output = vm.take_output();
        (result, output)
    }

    // -- VM creation --

    #[test]
    fn test_vm_creation() {
        let vm = VM::new_default();
        assert!(!vm.object_class.is_null());
        assert!(!vm.class_class.is_null());
        assert!(!vm.num_class.is_null());
        assert!(!vm.string_class.is_null());
        assert!(!vm.bool_class.is_null());
        assert!(!vm.null_class.is_null());
        assert!(!vm.list_class.is_null());
        assert!(!vm.map_class.is_null());
        assert!(!vm.range_class.is_null());
        assert!(!vm.fn_class.is_null());
        assert!(!vm.fiber_class.is_null());
        assert!(!vm.system_class.is_null());
    }

    #[test]
    fn test_sync_fiber_pool_reuses_and_resets_fibers() {
        let mut vm = VM::new_default();

        let fiber = vm.acquire_sync_fiber();
        assert!(!fiber.is_null());
        unsafe {
            (*fiber).stack.push(Value::num(42.0));
            (*fiber).state = FiberState::Running;
            (*fiber).is_try = true;
        }

        vm.release_sync_fiber(fiber);
        assert_eq!(vm.sync_fiber_pool.len(), 1);
        let pooled = vm.sync_fiber_pool[0];
        unsafe {
            assert!((*pooled).stack.is_empty());
            assert!((*pooled).mir_frames.is_empty());
            assert_eq!((*pooled).state, FiberState::New);
            assert!(!(*pooled).is_try);
        }

        vm.collect_garbage();
        assert_eq!(vm.sync_fiber_pool.len(), 1);

        let reused = vm.acquire_sync_fiber();
        assert!(!reused.is_null());
        unsafe {
            assert!((*reused).stack.is_empty());
            assert!((*reused).mir_frames.is_empty());
            assert_eq!((*reused).state, FiberState::New);
            assert!(!(*reused).is_try);
        }
    }

    #[test]
    fn test_class_hierarchy() {
        let vm = VM::new_default();
        // All classes should have Class as their class.
        unsafe {
            assert_eq!((*vm.object_class).header.class, vm.class_class);
            assert_eq!((*vm.num_class).header.class, vm.class_class);
            assert_eq!((*vm.string_class).header.class, vm.class_class);
            // Num inherits from Object; String inherits from Sequence.
            assert_eq!((*vm.num_class).superclass, vm.object_class);
            assert_eq!((*vm.string_class).superclass, vm.sequence_class);
            assert_eq!((*vm.sequence_class).superclass, vm.object_class);
        }
    }

    // -- Num primitives --

    #[test]
    fn test_num_arithmetic() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;
        let a = Value::num(10.0);
        let b = Value::num(3.0);

        let sum = call_primitive(&mut vm, cls, "+(_)", &[a, b]);
        assert_eq!(sum.as_num().unwrap(), 13.0);

        let diff = call_primitive(&mut vm, cls, "-(_)", &[a, b]);
        assert_eq!(diff.as_num().unwrap(), 7.0);

        let prod = call_primitive(&mut vm, cls, "*(_)", &[a, b]);
        assert_eq!(prod.as_num().unwrap(), 30.0);

        let quot = call_primitive(&mut vm, cls, "/(_)", &[a, b]);
        assert!((quot.as_num().unwrap() - 10.0 / 3.0).abs() < 1e-10);

        let rem = call_primitive(&mut vm, cls, "%(_)", &[a, b]);
        assert_eq!(rem.as_num().unwrap(), 1.0);
    }

    #[test]
    fn test_num_comparison() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let t = call_primitive(&mut vm, cls, "<(_)", &[Value::num(1.0), Value::num(2.0)]);
        assert!(t.as_bool().unwrap());

        let f = call_primitive(&mut vm, cls, ">(_)", &[Value::num(1.0), Value::num(2.0)]);
        assert!(!f.as_bool().unwrap());

        let eq = call_primitive(&mut vm, cls, "==(_)", &[Value::num(5.0), Value::num(5.0)]);
        assert!(eq.as_bool().unwrap());

        let neq = call_primitive(&mut vm, cls, "!=(_)", &[Value::num(5.0), Value::num(3.0)]);
        assert!(neq.as_bool().unwrap());
    }

    #[test]
    fn test_num_math() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let abs = call_primitive(&mut vm, cls, "abs", &[Value::num(-5.0)]);
        assert_eq!(abs.as_num().unwrap(), 5.0);

        let ceil = call_primitive(&mut vm, cls, "ceil", &[Value::num(2.3)]);
        assert_eq!(ceil.as_num().unwrap(), 3.0);

        let floor = call_primitive(&mut vm, cls, "floor", &[Value::num(2.7)]);
        assert_eq!(floor.as_num().unwrap(), 2.0);

        let sqrt = call_primitive(&mut vm, cls, "sqrt", &[Value::num(9.0)]);
        assert_eq!(sqrt.as_num().unwrap(), 3.0);
    }

    #[test]
    fn test_num_static() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let pi = call_static(&mut vm, cls, "pi", &[Value::null()]);
        assert!((pi.as_num().unwrap() - std::f64::consts::PI).abs() < 1e-10);

        let inf = call_static(&mut vm, cls, "infinity", &[Value::null()]);
        assert!(inf.as_num().unwrap().is_infinite());
    }

    #[test]
    fn test_num_to_string() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let s = call_primitive(&mut vm, cls, "toString", &[Value::num(42.0)]);
        assert!(s.is_object());
        let text = super::super::core::as_string(s);
        assert_eq!(text, "42");
    }

    #[test]
    fn test_num_range() {
        let mut vm = VM::new_default();
        let cls = vm.num_class;

        let range = call_primitive(&mut vm, cls, "..(_)", &[Value::num(1.0), Value::num(5.0)]);
        assert!(range.is_object());
    }

    // -- Bool/Null primitives --

    #[test]
    fn test_bool_not() {
        let mut vm = VM::new_default();
        let cls = vm.bool_class;

        let r = call_primitive(&mut vm, cls, "!", &[Value::bool(true)]);
        assert!(!r.as_bool().unwrap());

        let r2 = call_primitive(&mut vm, cls, "!", &[Value::bool(false)]);
        assert!(r2.as_bool().unwrap());
    }

    #[test]
    fn test_null_not() {
        let mut vm = VM::new_default();
        let cls = vm.null_class;

        let r = call_primitive(&mut vm, cls, "!", &[Value::null()]);
        assert!(r.as_bool().unwrap());
    }

    #[test]
    fn test_bool_to_string() {
        let mut vm = VM::new_default();
        let cls = vm.bool_class;

        let s = call_primitive(&mut vm, cls, "toString", &[Value::bool(true)]);
        assert_eq!(super::super::core::as_string(s), "true");

        let s2 = call_primitive(&mut vm, cls, "toString", &[Value::bool(false)]);
        assert_eq!(super::super::core::as_string(s2), "false");
    }

    // -- String primitives --

    #[test]
    fn test_string_contains() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;

        let hello = vm.new_string("hello world".to_string());
        let sub = vm.new_string("world".to_string());
        let r = call_primitive(&mut vm, cls, "contains(_)", &[hello, sub]);
        assert!(r.as_bool().unwrap());

        let miss = vm.new_string("xyz".to_string());
        let r2 = call_primitive(&mut vm, cls, "contains(_)", &[hello, miss]);
        assert!(!r2.as_bool().unwrap());
    }

    #[test]
    fn test_string_plus() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;

        let a = vm.new_string("hello ".to_string());
        let b = vm.new_string("world".to_string());
        let r = call_primitive(&mut vm, cls, "+(_)", &[a, b]);
        assert_eq!(super::super::core::as_string(r), "hello world");
    }

    #[test]
    fn test_string_byte_count() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;

        let s = vm.new_string("hello".to_string());
        let r = call_primitive(&mut vm, cls, "byteCount_", &[s]);
        assert_eq!(r.as_num().unwrap(), 5.0);
    }

    // -- List primitives --

    #[test]
    fn test_list_add_and_count() {
        let mut vm = VM::new_default();
        let cls = vm.list_class;

        let list = vm.new_list(vec![]);
        call_primitive(&mut vm, cls, "add(_)", &[list, Value::num(42.0)]);
        call_primitive(&mut vm, cls, "add(_)", &[list, Value::num(99.0)]);

        let count = call_primitive(&mut vm, cls, "count", &[list]);
        assert_eq!(count.as_num().unwrap(), 2.0);

        let elem = call_primitive(&mut vm, cls, "[_]", &[list, Value::num(0.0)]);
        assert_eq!(elem.as_num().unwrap(), 42.0);
    }

    #[test]
    fn test_list_remove_at() {
        let mut vm = VM::new_default();
        let cls = vm.list_class;

        let list = vm.new_list(vec![Value::num(10.0), Value::num(20.0), Value::num(30.0)]);
        let removed = call_primitive(&mut vm, cls, "removeAt(_)", &[list, Value::num(1.0)]);
        assert_eq!(removed.as_num().unwrap(), 20.0);

        let count = call_primitive(&mut vm, cls, "count", &[list]);
        assert_eq!(count.as_num().unwrap(), 2.0);
    }

    // -- Map primitives --

    #[test]
    fn test_map_set_and_get() {
        let mut vm = VM::new_default();
        let cls = vm.map_class;

        let map = vm.new_map();
        let key = vm.new_string("name".to_string());
        let val = vm.new_string("Wren".to_string());

        call_primitive(&mut vm, cls, "[_]=(_)", &[map, key, val]);
        let got = call_primitive(&mut vm, cls, "[_]", &[map, key]);
        assert_eq!(super::super::core::as_string(got), "Wren");

        let count = call_primitive(&mut vm, cls, "count", &[map]);
        assert_eq!(count.as_num().unwrap(), 1.0);
    }

    #[test]
    fn test_map_contains_key() {
        let mut vm = VM::new_default();
        let cls = vm.map_class;

        let map = vm.new_map();
        let key = vm.new_string("x".to_string());
        call_primitive(&mut vm, cls, "[_]=(_)", &[map, key, Value::num(1.0)]);

        let has = call_primitive(&mut vm, cls, "containsKey(_)", &[map, key]);
        assert!(has.as_bool().unwrap());

        let miss = vm.new_string("y".to_string());
        let no = call_primitive(&mut vm, cls, "containsKey(_)", &[map, miss]);
        assert!(!no.as_bool().unwrap());
    }

    // -- Range primitives --

    #[test]
    fn test_range_properties() {
        let mut vm = VM::new_default();
        let cls = vm.range_class;

        let range = vm.new_range(1.0, 5.0, true);
        let from = call_primitive(&mut vm, cls, "from", &[range]);
        assert_eq!(from.as_num().unwrap(), 1.0);

        let to = call_primitive(&mut vm, cls, "to", &[range]);
        assert_eq!(to.as_num().unwrap(), 5.0);

        let incl = call_primitive(&mut vm, cls, "isInclusive", &[range]);
        assert!(incl.as_bool().unwrap());
    }

    // -- Object primitives --

    #[test]
    fn test_object_equality() {
        let mut vm = VM::new_default();
        let cls = vm.object_class;

        let a = Value::num(5.0);
        let b = Value::num(5.0);
        let eq = call_primitive(&mut vm, cls, "==(_)", &[a, b]);
        assert!(eq.as_bool().unwrap());

        let neq = call_primitive(&mut vm, cls, "!=(_)", &[a, Value::num(3.0)]);
        assert!(neq.as_bool().unwrap());
    }

    #[test]
    fn test_object_not() {
        let mut vm = VM::new_default();
        let cls = vm.object_class;

        // Object.! always returns false (all objects are truthy)
        let r = call_primitive(&mut vm, cls, "!", &[Value::num(42.0)]);
        assert!(!r.as_bool().unwrap());
    }

    // -- Class primitives --

    #[test]
    fn test_class_name() {
        let mut vm = VM::new_default();
        let cls = vm.class_class;

        let num_cls_val = Value::object(vm.num_class as *mut u8);
        let name = call_primitive(&mut vm, cls, "name", &[num_cls_val]);
        assert_eq!(super::super::core::as_string(name), "Num");
    }

    // -- System primitives --

    #[test]
    fn test_system_clock() {
        let mut vm = VM::new_default();
        let cls = vm.system_class;

        let t = call_static(&mut vm, cls, "clock", &[Value::null()]);
        assert!(t.as_num().unwrap() > 0.0);
    }

    // -- class_of --

    #[test]
    fn test_class_of() {
        let vm = VM::new_default();
        assert_eq!(vm.class_of(Value::num(1.0)), vm.num_class);
        assert_eq!(vm.class_of(Value::bool(true)), vm.bool_class);
        assert_eq!(vm.class_of(Value::null()), vm.null_class);
    }

    // -- interpret (integration) --

    #[test]
    fn test_interpret_system_print() {
        let (result, output) = run_and_capture("System.print(\"hello\")");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "hello\n");
    }

    #[test]
    fn test_interpret_arithmetic() {
        let (result, output) = run_and_capture("System.print(1 + 2)");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n");
    }

    #[test]
    fn test_interpret_module_vars() {
        let (result, output) = run_and_capture("var x = 10\nSystem.print(x)");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "10\n");
    }

    #[test]
    fn test_interpret_is_type() {
        let (result, output) = run_and_capture("System.print(42 is Num)");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "true\n");
    }

    #[test]
    fn test_interpret_class_construct() {
        let (result, _) =
            run_and_capture("class Foo {\n  construct new() {}\n}\nvar f = Foo.new()");
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
    }

    #[test]
    fn test_interpret_class_with_fields() {
        let (result, output) = run_and_capture(
            r#"
class Point {
  construct new(x, y) {
    _x = x
    _y = y
  }
  x { _x }
  y { _y }
}
var p = Point.new(3, 4)
System.print(p.x)
System.print(p.y)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n4\n");
    }

    #[test]
    fn test_interpret_instance_method() {
        let (result, output) = run_and_capture(
            r#"
class Greeter {
  construct new(name) {
    _name = name
  }
  greet() { _name }
}
var g = Greeter.new("Alice")
System.print(g.greet())
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "Alice\n");
    }

    #[test]
    fn test_interpret_named_constructor() {
        let (result, output) = run_and_capture(
            r#"
class Foo {
  construct create(x) {
    _x = x
  }
  x { _x }
}
var f = Foo.create(42)
System.print(f.x)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_fn_call() {
        let (result, output) = run_and_capture(
            r#"
var fn = Fn.new {
  System.print(42)
}
fn.call()
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_fn_call_with_args() {
        let (result, output) = run_and_capture(
            r#"
var add = Fn.new {|a, b|
  System.print(a + b)
}
add.call(3, 4)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "7\n");
    }

    #[test]
    fn test_interpret_this_access() {
        let (result, output) = run_and_capture(
            r#"
class Counter {
  construct new(n) {
    _n = n
  }
  inc() { _n = _n + 1 }
  value { _n }
}
var c = Counter.new(0)
c.inc()
c.inc()
c.inc()
System.print(c.value)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n");
    }

    #[test]
    fn test_interpret_inheritance() {
        // Test method inheritance (child calls parent method)
        let (result, output) = run_and_capture(
            r#"
class Animal {
  construct new(name) {
    _name = name
  }
  name { _name }
}
class Dog is Animal {
  construct new(name) {
    super(name)
  }
  bark() { "woof" }
}
var d = Dog.new("Rex")
System.print(d.name)
System.print(d.bark())
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "Rex\nwoof\n");
    }

    #[test]
    fn test_interpret_operator_overload() {
        let (result, output) = run_and_capture(
            r#"
class Vec2 {
  construct new(x, y) {
    _x = x
    _y = y
  }
  x { _x }
  y { _y }
  +(other) { Vec2.new(_x + other.x, _y + other.y) }
  toString { "(" + _x.toString + ", " + _y.toString + ")" }
}
var a = Vec2.new(1, 2)
var b = Vec2.new(3, 4)
var c = a + b
System.print(c.x)
System.print(c.y)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "4\n6\n");
    }

    #[test]
    fn test_interpret_prefix_operator() {
        let (result, output) = run_and_capture(
            r#"
class Num2 {
  construct new(n) { _n = n }
  value { _n }
  -() { Num2.new(-_n) }
}
var a = Num2.new(5)
var b = -a
System.print(b.value)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "-5\n");
    }

    #[test]
    fn test_interpret_closure_capture() {
        let (result, output) = run_and_capture(
            r#"
var x = 10
var f = Fn.new { x }
System.print(f.call())
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "10\n");
    }

    #[test]
    fn test_interpret_closure_capture_with_args() {
        let (result, output) = run_and_capture(
            r#"
var x = 10
var f = Fn.new {|y| x + y }
System.print(f.call(5))
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "15\n");
    }

    #[test]
    fn test_interpret_closure_capture_multiple() {
        let (result, output) = run_and_capture(
            r#"
var a = 3
var b = 7
var f = Fn.new { a + b }
System.print(f.call())
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "10\n");
    }

    #[test]
    fn test_interpret_closure_capture_in_block() {
        let (result, output) = run_and_capture(
            r#"
var result = null
{
  var x = 42
  var f = Fn.new { x }
  result = f.call()
}
System.print(result)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_closure_nested_capture() {
        let (result, output) = run_and_capture(
            r#"
var x = 100
var outer = Fn.new {
  Fn.new { x }
}
var inner = outer.call()
System.print(inner.call())
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "100\n");
    }

    #[test]
    fn test_interpret_super_call() {
        let (result, output) = run_and_capture(
            r#"
class Animal {
  construct new(name) { _name = name }
  speak() { _name }
}
class Dog is Animal {
  construct new(name) { super(name) }
  speak() { super.speak() + " says woof" }
}
var d = Dog.new("Rex")
System.print(d.speak())
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "Rex says woof\n");
    }

    #[test]
    fn test_interpret_super_getter() {
        let (result, output) = run_and_capture(
            r#"
class Base {
  construct new(v) { _v = v }
  value { _v }
}
class Child is Base {
  construct new(v) { super(v) }
  value { super.value + 10 }
}
System.print(Child.new(5).value)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "15\n");
    }

    #[test]
    fn test_interpret_fiber_new_call() {
        let (result, output) = run_and_capture(
            r#"
var fiber = Fiber.new {
  System.print("inside fiber")
}
fiber.call()
System.print("after call")
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "inside fiber\nafter call\n");
    }

    #[test]
    fn test_interpret_fiber_yield() {
        let (result, output) = run_and_capture(
            r#"
var fiber = Fiber.new {
  System.print("before yield")
  Fiber.yield()
  System.print("after yield")
}
fiber.call()
System.print("yielded")
fiber.call()
System.print("done")
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "before yield\nyielded\nafter yield\ndone\n");
    }

    #[test]
    fn test_interpret_fiber_yield_value() {
        let (result, output) = run_and_capture(
            r#"
var fiber = Fiber.new {
  Fiber.yield(42)
}
var result = fiber.call()
System.print(result)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_fiber_is_done() {
        let (result, output) = run_and_capture(
            r#"
var fiber = Fiber.new {
  Fiber.yield()
}
System.print(fiber.isDone)
fiber.call()
System.print(fiber.isDone)
fiber.call()
System.print(fiber.isDone)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "false\nfalse\ntrue\n");
    }

    #[test]
    fn test_interpret_fiber_pass_value_on_resume() {
        let (result, output) = run_and_capture(
            r#"
var fiber = Fiber.new {
  var got = Fiber.yield()
  System.print(got)
}
fiber.call()
fiber.call("hello from caller")
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "hello from caller\n");
    }

    #[test]
    fn test_interpret_fiber_multiple_yields() {
        let (result, output) = run_and_capture(
            r#"
var fiber = Fiber.new {
  Fiber.yield(1)
  Fiber.yield(2)
  Fiber.yield(3)
}
System.print(fiber.call())
System.print(fiber.call())
System.print(fiber.call())
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "1\n2\n3\n");
    }

    #[test]
    fn test_interpret_while_loop() {
        let (result, output) = run_and_capture(
            r#"
var i = 0
while (i < 3) {
  System.print(i)
  i = i + 1
}
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "while loop failed: {:?}",
            result
        );
        assert_eq!(output, "0\n1\n2\n");
    }

    #[test]
    fn test_interpret_string_interpolation() {
        let (result, output) = run_and_capture(
            r#"
var name = "world"
var n = 42
System.print("hello %(name)!")
System.print("n = %(n)")
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "interpolation failed: {:?}",
            result
        );
        assert_eq!(output, "hello world!\nn = 42\n");
    }

    #[test]
    fn test_interpret_is_operator() {
        let (result, output) = run_and_capture(
            r#"
System.print(42 is Num)
System.print("hi" is String)
System.print(null is Null)
System.print(42 is String)
System.print(true is Bool)
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "is operator failed: {:?}",
            result
        );
        assert_eq!(output, "true\ntrue\ntrue\nfalse\ntrue\n");
    }

    #[test]
    fn test_interpret_setter() {
        let (result, output) = run_and_capture(
            r#"
class Point {
  construct new(x, y) {
    _x = x
    _y = y
  }
  x { _x }
  x=(v) { _x = v }
  y { _y }
}
var p = Point.new(1, 2)
System.print(p.x)
p.x = 99
System.print(p.x)
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "setter failed: {:?}",
            result
        );
        assert_eq!(output, "1\n99\n");
    }

    #[test]
    fn test_interpret_list_subscript() {
        let (result, output) = run_and_capture(
            r#"
var list = [10, 20, 30]
System.print(list[0])
System.print(list[2])
list[1] = 99
System.print(list[1])
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "list subscript failed: {:?}",
            result
        );
        assert_eq!(output, "10\n30\n99\n");
    }

    #[test]
    fn test_interpret_for_in_list() {
        let (result, output) = run_and_capture(
            r#"
var list = [10, 20, 30]
for (x in list) {
  System.print(x)
}
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "for-in list failed: {:?}",
            result
        );
        assert_eq!(output, "10\n20\n30\n");
    }

    #[test]
    fn test_interpret_for_in_range() {
        let (result, output) = run_and_capture(
            r#"
for (i in 1..4) {
  System.print(i)
}
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "for-in range failed: {:?}",
            result
        );
        assert_eq!(output, "1\n2\n3\n4\n");
    }

    #[test]
    fn test_interpret_string_plus() {
        let (result, output) = run_and_capture(
            r#"
var a = "hello"
var b = " world"
System.print(a + b)
"#,
        );
        assert!(
            matches!(result, InterpretResult::Success),
            "string concat failed: {:?}",
            result
        );
        assert_eq!(output, "hello world\n");
    }

    #[test]
    fn test_interpret_static_method_with_args() {
        let mut vm = VM::new(VMConfig::default());
        vm.output_buffer = Some(String::new());
        let result = vm.interpret(
            "main",
            r#"
class Foo {
  static double(n) { n + n }
}
System.print(Foo.double(21))
"#,
        );
        let output = vm.take_output();
        assert!(
            matches!(result, InterpretResult::Success),
            "static method with args failed: {:?}",
            result
        );
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_map_literal() {
        let (result, output) = run_and_capture(
            r#"
var m = {"a": 1, "b": 2}
System.print(m["a"])
System.print(m["b"])
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "1\n2\n");
    }

    #[test]
    fn test_interpret_map_subscript_set() {
        let (result, output) = run_and_capture(
            r#"
var m = Map.new()
m["key"] = 42
System.print(m["key"])
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    #[test]
    fn test_interpret_map_count() {
        let (result, output) = run_and_capture(
            r#"
var m = {"x": 1, "y": 2, "z": 3}
System.print(m.count)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n");
    }

    #[test]
    fn test_interpret_fn_arity() {
        let (result, output) = run_and_capture(
            r#"
var f = Fn.new {|a, b, c| a }
System.print(f.arity)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "3\n");
    }

    #[test]
    fn test_interpret_fiber_current() {
        let (result, output) = run_and_capture(
            r#"
var f = Fiber.current
System.print(f is Fiber)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "true\n");
    }

    #[test]
    fn test_interpret_fiber_suspend() {
        // Fiber.suspend() should suspend the current fiber.
        // When called from the root fiber with no caller, execution just stops.
        let (result, output) = run_and_capture(
            r#"
var fiber = Fiber.new {
  System.print("before")
  Fiber.suspend()
  System.print("after")
}
fiber.call()
System.print("main continues")
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        // After fiber.call(), the child runs "before", then suspends.
        // Control returns to main which prints "main continues".
        // "after" is never printed because the fiber is suspended and not resumed.
        assert_eq!(output, "before\nmain continues\n");
    }

    #[test]
    fn test_interpret_import_module() {
        let mut config = VMConfig::default();
        config.load_module_fn = Some(Box::new(|name: &str, _from: &str| -> Option<String> {
            if name == "math_utils" {
                Some(
                    r#"
class MathUtils {
  static greet() { "hello from import" }
  static double(n) { n + n }
}
"#
                    .to_string(),
                )
            } else {
                None
            }
        }));
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret(
            "main",
            r#"
import "math_utils" for MathUtils
System.print(MathUtils.greet())
System.print(MathUtils.double(21))
"#,
        );
        let output = vm.take_output();
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "hello from import\n42\n");
    }

    /// Regression: a method defined in module A that references a
    /// sibling class (also in A) must resolve that reference against
    /// A's module-var slots even when the method is invoked from
    /// module B. Previously each pushed frame copied the caller's
    /// `module_name`, so `GetModuleVar @0` inside the callee read
    /// the caller's slot 0 (a different variable), and importing
    /// `Outer` from a module and calling `Outer.make` that internally
    /// used `Inner.new` crashed with `Class does not implement new(_)`.
    #[test]
    fn test_cross_module_sibling_class_reference() {
        let mut config = VMConfig::default();
        config.load_module_fn = Some(Box::new(|name: &str, _from: &str| -> Option<String> {
            if name == "pair" {
                Some(
                    r#"
class Inner {
  construct new(v) { _v = v }
  value { _v }
}
class Outer {
  static make(v) { Inner.new(v) }
}
"#
                    .to_string(),
                )
            } else {
                None
            }
        }));
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret(
            "main",
            r#"
import "pair" for Outer
System.print(Outer.make(42).value)
"#,
        );
        let output = vm.take_output();
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "42\n");
    }

    /// Regression: a `Fiber.abort` that propagates back through an
    /// intermediate `body.call()` frame used to corrupt the next
    /// static-method call in the caller fiber, which returned
    /// UNDEFINED instead of its actual value.
    ///
    /// Root cause: `run_fiber_until_depth` compared its `stop_depth`
    /// against whatever fiber was currently active. When an abort
    /// switched `vm.fiber` from the try-fiber back to main mid-run,
    /// the stop_depth check fired against main's frame count — which
    /// happened to match on the first nested return after the abort
    /// — so `run_fiber_until_depth` returned early with the wrong
    /// value. Fix gates the check on `fiber == stop_fiber`.
    #[test]
    fn test_fiber_abort_through_closure_preserves_subsequent_calls() {
        let (result, output) = run_and_capture(
            r#"
class A { construct new(x) { _x = x } val { _x } }
class B { static make(x) { A.new(x) } }
class T {
  static call(body) { body.call() }
  static it(arg) { if (!(arg is Fn)) Fiber.abort("bad") }
}

var test = Fn.new { T.it("x") }
var aborted = Fiber.new { T.call(test) }.try()
System.print(aborted)
System.print(B.make(99).val)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert_eq!(output, "bad\n99\n");
    }

    #[test]
    fn test_runtime_error_has_source_location() {
        // Trigger a MethodNotFound error, verify extract_error_location returns
        // a valid SourceLoc pointing into the source code.
        // Uses interpreter mode because JIT dispatch silently returns null on
        // MethodNotFound instead of raising a RuntimeError.
        let source = "var x = 42\nx.bogus()";
        let mut config = VMConfig::default();
        config.execution_mode = crate::runtime::engine::ExecutionMode::Interpreter;
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());

        // We need to inspect the fiber before interpret() restores prev_fiber.
        // So we'll replicate the interpret pipeline up to run_fiber, then check.
        let result = vm.interpret("main", source);
        assert!(
            matches!(result, InterpretResult::RuntimeError),
            "{:?}",
            result
        );

        // Check that error_fiber was saved for post-mortem inspection
        let fiber = vm.error_fiber;
        assert!(
            !fiber.is_null(),
            "error_fiber should be saved on runtime error"
        );
        let loc = vm.extract_error_location(fiber);
        assert!(
            loc.is_some(),
            "extract_error_location should return Some for runtime error"
        );
        let loc = loc.unwrap();
        assert_eq!(loc.module.as_str(), "main");
        // Span should point somewhere in "x.bogus()" (byte offsets 11..20 in the source)
        assert!(
            loc.span.start >= 11,
            "span start {} should be >= 11",
            loc.span.start
        );
        assert!(
            loc.span.end <= 20,
            "span end {} should be <= 20",
            loc.span.end
        );
    }

    #[test]
    fn test_runtime_error_ariadne_with_snippet_and_help() {
        // Verify the full ariadne-rendered output includes source snippet,
        // error label, and contextual help note.
        use std::sync::{Arc, Mutex};

        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let errors_clone = errors.clone();

        let mut config = VMConfig::default();
        config.execution_mode = crate::runtime::engine::ExecutionMode::Interpreter;
        config.error_fn = Some(Box::new(move |_kind, _module, _line, msg| {
            errors_clone.lock().unwrap().push(msg.to_string());
        }));

        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret("main", "var x = 42\nx.bogus()");
        assert!(matches!(result, InterpretResult::RuntimeError));

        let captured = errors.lock().unwrap();
        assert!(!captured.is_empty(), "expected error to be reported");
        let rendered = &captured[0];

        // Strip ANSI color codes for easier assertion
        let plain = strip_ansi(rendered);

        // Should contain the error message
        assert!(plain.contains("bogus"), "missing method name in:\n{plain}");
        // Should contain source snippet (the source line with the error)
        assert!(
            plain.contains("x.bogus()"),
            "missing source snippet in:\n{plain}"
        );
        // Should contain the error label
        assert!(
            plain.contains("error occurred here"),
            "missing label in:\n{plain}"
        );
        // Should contain contextual help
        assert!(
            plain.contains("does not define") || plain.contains("has no getter"),
            "missing help note in:\n{plain}"
        );
    }

    #[test]
    fn test_fiber_stack_trace_disabled_by_default() {
        // When fiber_stack_traces is false (default), Fiber.stackTrace returns a message
        let (result, output) = run_and_capture(
            r#"
var f = Fiber.new { 1 }
System.print(f.stackTrace)
"#,
        );
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        assert!(
            output.contains("not enabled"),
            "expected disabled message, got: {output}"
        );
    }

    #[test]
    fn test_fiber_stack_trace_enabled() {
        let mut config = VMConfig::default();
        config.fiber_stack_traces = true;
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret(
            "main",
            r#"
var f = Fiber.new {
  Fiber.yield()
}
f.call()
System.print(f.stackTrace)
"#,
        );
        let output = vm.take_output();
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        // Should contain stack trace info, not the disabled message
        assert!(
            !output.contains("not enabled"),
            "should be enabled, got: {output}"
        );
    }

    #[test]
    fn test_fiber_cross_fiber_error_trace() {
        // When a fiber errors out, the full trace should include caller chain
        use std::sync::{Arc, Mutex};

        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let errors_clone = errors.clone();

        let mut config = VMConfig::default();
        config.fiber_stack_traces = true;
        config.error_fn = Some(Box::new(move |_kind, _module, _line, msg| {
            errors_clone.lock().unwrap().push(msg.to_string());
        }));

        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret(
            "main",
            r#"
var inner = Fiber.new {
  var x = 42
  x.bogus()
}
inner.call()
"#,
        );
        assert!(matches!(result, InterpretResult::RuntimeError));
        let captured = errors.lock().unwrap();
        assert!(!captured.is_empty(), "expected error");
        let plain = strip_ansi(&captured[0]);
        // Error should mention bogus
        assert!(plain.contains("bogus"), "should mention bogus in:\n{plain}");
    }

    #[test]
    fn test_fiber_spawn_trace_recorded() {
        // Verify that spawn_trace is populated when fiber_stack_traces is enabled
        let mut config = VMConfig::default();
        config.fiber_stack_traces = true;
        let mut vm = VM::new(config);
        vm.output_buffer = Some(String::new());
        let result = vm.interpret(
            "main",
            r#"
var f = Fiber.new { 1 }
System.print(f.stackTrace)
"#,
        );
        let output = vm.take_output();
        assert!(matches!(result, InterpretResult::Success), "{:?}", result);
        // The stack trace should mention "main" module and spawned-at info
        assert!(
            output.contains("main"),
            "should reference main module, got: {output}"
        );
    }

    // -- String new methods --

    #[test]
    fn test_string_count() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;
        let s = vm.new_string("hello".to_string());
        let count = call_primitive(&mut vm, cls, "count", &[s]);
        assert_eq!(count.as_num().unwrap(), 5.0);
    }

    #[test]
    fn test_string_is_empty() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;

        let empty = vm.new_string("".to_string());
        assert!(call_primitive(&mut vm, cls, "isEmpty", &[empty])
            .as_bool()
            .unwrap());

        let nonempty = vm.new_string("a".to_string());
        assert!(!call_primitive(&mut vm, cls, "isEmpty", &[nonempty])
            .as_bool()
            .unwrap());
    }

    #[test]
    fn test_string_split() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;
        let s = vm.new_string("a,b,c".to_string());
        let delim = vm.new_string(",".to_string());
        let result = call_primitive(&mut vm, cls, "split(_)", &[s, delim]);
        assert!(result.is_object());
        // Result is a list; check count
        let list_cls = vm.list_class;
        let count = call_primitive(&mut vm, list_cls, "count", &[result]);
        assert_eq!(count.as_num().unwrap(), 3.0);
    }

    #[test]
    fn test_string_replace() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;
        let s = vm.new_string("hello world".to_string());
        let from = vm.new_string("world".to_string());
        let to = vm.new_string("wren".to_string());
        let result = call_primitive(&mut vm, cls, "replace(_,_)", &[s, from, to]);
        assert_eq!(super::super::core::as_string(result), "hello wren");
    }

    #[test]
    fn test_string_trim() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;
        let s = vm.new_string("  hi  ".to_string());

        let trimmed = call_primitive(&mut vm, cls, "trim()", &[s]);
        assert_eq!(super::super::core::as_string(trimmed), "hi");

        let ts = call_primitive(&mut vm, cls, "trimStart()", &[s]);
        assert_eq!(super::super::core::as_string(ts), "hi  ");

        let te = call_primitive(&mut vm, cls, "trimEnd()", &[s]);
        assert_eq!(super::super::core::as_string(te), "  hi");
    }

    #[test]
    fn test_string_multiply() {
        let mut vm = VM::new_default();
        let cls = vm.string_class;
        let s = vm.new_string("ab".to_string());
        let result = call_primitive(&mut vm, cls, "*(_)", &[s, Value::num(3.0)]);
        assert_eq!(super::super::core::as_string(result), "ababab");
    }

    // -- List new methods --

    #[test]
    fn test_list_sort() {
        let mut vm = VM::new_default();
        let cls = vm.list_class;

        let list = vm.new_list(vec![Value::num(3.0), Value::num(1.0), Value::num(2.0)]);
        let sorted = call_primitive(&mut vm, cls, "sort()", &[list]);

        let first = call_primitive(&mut vm, cls, "[_]", &[sorted, Value::num(0.0)]);
        assert_eq!(first.as_num().unwrap(), 1.0);
        let second = call_primitive(&mut vm, cls, "[_]", &[sorted, Value::num(1.0)]);
        assert_eq!(second.as_num().unwrap(), 2.0);
        let third = call_primitive(&mut vm, cls, "[_]", &[sorted, Value::num(2.0)]);
        assert_eq!(third.as_num().unwrap(), 3.0);
    }

    #[test]
    fn test_list_times() {
        let mut vm = VM::new_default();
        let cls = vm.list_class;

        let list = vm.new_list(vec![Value::num(1.0), Value::num(2.0)]);
        let result = call_primitive(&mut vm, cls, "*(_)", &[list, Value::num(3.0)]);
        let count = call_primitive(&mut vm, cls, "count", &[result]);
        assert_eq!(count.as_num().unwrap(), 6.0);
    }

    // -- Map new methods --

    #[test]
    fn test_map_keys() {
        let mut vm = VM::new_default();
        let cls = vm.map_class;

        let map = vm.new_map();
        let k1 = vm.new_string("a".to_string());
        let k2 = vm.new_string("b".to_string());
        call_primitive(&mut vm, cls, "[_]=(_)", &[map, k1, Value::num(1.0)]);
        call_primitive(&mut vm, cls, "[_]=(_)", &[map, k2, Value::num(2.0)]);

        let keys = call_primitive(&mut vm, cls, "keys", &[map]);
        let list_cls = vm.list_class;
        let count = call_primitive(&mut vm, list_cls, "count", &[keys]);
        assert_eq!(count.as_num().unwrap(), 2.0);
    }

    #[test]
    fn test_map_values() {
        let mut vm = VM::new_default();
        let cls = vm.map_class;

        let map = vm.new_map();
        let k1 = vm.new_string("x".to_string());
        call_primitive(&mut vm, cls, "[_]=(_)", &[map, k1, Value::num(42.0)]);

        let vals = call_primitive(&mut vm, cls, "values", &[map]);
        let list_cls = vm.list_class;
        let count = call_primitive(&mut vm, list_cls, "count", &[vals]);
        assert_eq!(count.as_num().unwrap(), 1.0);

        let first = call_primitive(&mut vm, list_cls, "[_]", &[vals, Value::num(0.0)]);
        assert_eq!(first.as_num().unwrap(), 42.0);
    }

    // -- Object.hashCode --

    #[test]
    fn test_object_hash() {
        let mut vm = VM::new_default();
        let cls = vm.object_class;

        let h1 = call_primitive(&mut vm, cls, "hashCode", &[Value::num(42.0)]);
        let h2 = call_primitive(&mut vm, cls, "hashCode", &[Value::num(42.0)]);
        assert_eq!(h1.as_num().unwrap(), h2.as_num().unwrap());

        // Different values should (almost certainly) have different hashes
        let h3 = call_primitive(&mut vm, cls, "hashCode", &[Value::num(43.0)]);
        assert_ne!(h1.as_num().unwrap(), h3.as_num().unwrap());
    }

    // -- Object.is --

    #[test]
    fn test_object_is() {
        let mut vm = VM::new_default();
        let cls = vm.object_class;

        let num_cls = Value::object(vm.num_class as *mut u8);
        let obj_cls = Value::object(vm.object_class as *mut u8);

        // 42 is Num → true
        let r = call_primitive(&mut vm, cls, "is(_)", &[Value::num(42.0), num_cls]);
        assert!(r.as_bool().unwrap());

        // 42 is Object → true (Num inherits Object)
        let r2 = call_primitive(&mut vm, cls, "is(_)", &[Value::num(42.0), obj_cls]);
        assert!(r2.as_bool().unwrap());

        // "hi" is Num → false
        let s = vm.new_string("hi".to_string());
        let r3 = call_primitive(&mut vm, cls, "is(_)", &[s, num_cls]);
        assert!(!r3.as_bool().unwrap());
    }

    // -- Sequence hierarchy --

    #[test]
    fn test_sequence_hierarchy() {
        let vm = VM::new_default();
        unsafe {
            assert_eq!((*vm.list_class).superclass, vm.sequence_class);
            assert_eq!((*vm.map_class).superclass, vm.sequence_class);
            assert_eq!((*vm.range_class).superclass, vm.sequence_class);
            assert_eq!((*vm.string_class).superclass, vm.sequence_class);
            assert_eq!((*vm.sequence_class).superclass, vm.object_class);
        }
    }
}
