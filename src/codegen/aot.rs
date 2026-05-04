//! AOT object-file emission — Cranelift's `ObjectModule` swap-in
//! for the production JIT path.
//!
//! Drives the same parser → sema → MIR → CLIF translator the JIT
//! uses, but hands the lowered function to a `cranelift_object::
//! ObjectModule` (writes a `.o` to disk) instead of the
//! `cranelift_jit::JITModule` (mutates executable memory in
//! place). The shared translator lives in
//! `cranelift_backend::cl::lower_mir_to_module` — see Phase-2 of
//! the AOT plan: `cranelift_module::Module` is dyn-safe, so JIT
//! and AOT share the per-instruction lowering as a `&mut dyn
//! Module` parameter without cloning the codegen.
//!
//! Today's surface compiles ONE function — the entry point of
//! the parsed Wren source. Phase 3 walks the whole import graph
//! and links every reachable module's lowered MIR into a single
//! object file. Phase 4 hands the object to a system linker
//! alongside the runtime `staticlib` to land a self-contained
//! executable.
//!
//! IC pointers, code-base, and OSR-entry signals are JIT-only —
//! they patch self-call sites, on-stack replacement entries, and
//! inline-cache snapshots into mutable JIT memory. AOT outputs
//! go through a static linker, so we pass `None` for all three
//! and emit a clean object instead.
//!
//! Behind `feature = "aot"` so the production crate stays
//! untouched. Pulled in via `cargo build --features aot` or by
//! the `wlift build` driver once that lands.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use cranelift_codegen::ir::{types, AbiParam, Function, Signature, UserFuncName};
use cranelift_codegen::isa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::ast::Stmt;
use crate::codegen::cranelift_backend::cl::{lower_mir_to_module, AotLoweringConfig};
use crate::intern::Interner;
use crate::mir::builder::lower_module_with_known_classes;
use crate::mir::ModuleMir;
use crate::parse::parser::parse;
use crate::sema::resolve::resolve_with_prelude;

/// Errors the AOT pipeline can surface up to the CLI.
#[derive(Debug)]
pub enum AotError {
    /// The host triple isn't one Cranelift can target — should
    /// never fire on a supported runtime host but surfaced as a
    /// real error so the CLI can print a useful message instead
    /// of panicking.
    UnsupportedTarget(String),
    /// Cranelift couldn't construct an ISA for the resolved
    /// triple (e.g. missing CPU feature, unknown register set).
    Isa(String),
    /// CLIF lowering / object-table operation failed.
    Module(String),
    /// `std::fs::write` couldn't write the object bytes.
    Io(std::io::Error),
    /// Wren parse / sema / MIR-build error chain. Joined into a
    /// single string at the AOT boundary so the CLI's
    /// error-reporting path doesn't need to crack open each
    /// frontend's diagnostic shape.
    Frontend(String),
}

impl std::fmt::Display for AotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AotError::UnsupportedTarget(t) => write!(f, "unsupported target triple: {}", t),
            AotError::Isa(m) => write!(f, "cranelift ISA setup failed: {}", m),
            AotError::Module(m) => write!(f, "cranelift module operation failed: {}", m),
            AotError::Io(e) => write!(f, "writing object file: {}", e),
            AotError::Frontend(m) => write!(f, "wren frontend: {}", m),
        }
    }
}

impl std::error::Error for AotError {}

/// One module visited by the AOT walker.
///
/// Carries the canonical name (the absolute on-disk path for
/// relative imports — same shape `make_module_io` in the CLI driver
/// uses), the lowered MIR, and the per-parse `Interner` whose
/// `SymbolId`s appear inside `mir`. Each parse mints its own
/// interner; downstream lowering passes keep them paired so symbol
/// identity stays consistent across the module-by-module emit.
pub struct AotModule {
    // `name` is informational — kept around so emit-time errors
    // can attribute failures to a specific module path. The
    // emitted symbols use a separate, deterministic naming scheme
    // (`wlift_aot_main` / `wlift_aot_mod_<n>`) to avoid leaking
    // local filesystem paths into the produced object.
    #[allow(dead_code)]
    pub name: String,
    /// Module name the runtime should register this source under
    /// when the AOT bootstrap installs it — the `import "./foo"`
    /// path string from the FIRST importer that pulled this
    /// module in (or `"main"` for the entry). The runtime resolves
    /// imports by name match, so the bootstrap and the runtime
    /// have to agree: bundle a module under the exact spelling
    /// the source's `import` statement uses.
    pub request_name: String,
    /// Raw source bytes, for the bootstrap to embed into the
    /// produced executable. Read once during the walk so a later
    /// edit to the source file doesn't desync the embedded copy
    /// from the lowered MIR.
    pub source: Vec<u8>,
    /// Total module-var slot count (prelude + user-declared). Drives
    /// the per-module `wlift_modvars_<n>` data symbol's size — every
    /// `GetModuleVar(slot)` / `SetModuleVar(slot, _)` in this
    /// module's MIR addresses an offset inside that array, so the
    /// AOT emitter has to declare it `module_var_count * 8` bytes
    /// wide.
    pub module_var_count: usize,
    pub mir: ModuleMir,
    pub interner: Interner,
}

/// Append `.wren` to a relative-import path **without stripping
/// existing extensions**. Mirrors `with_wren_suffix` in `main.rs` —
/// `Path::with_extension` would replace `.spec` so `import "./x.spec"`
/// would silently redirect to `x.wren`.
fn with_wren_suffix(path: PathBuf) -> PathBuf {
    if path.extension().and_then(|e| e.to_str()) == Some("wren") {
        return path;
    }
    let mut s = path.into_os_string();
    s.push(".wren");
    PathBuf::from(s)
}

/// Walk the import graph rooted at `entry_path` and return one
/// `AotModule` per reachable on-disk module, in dependency-first
/// order (a module appears after every module it imports — the
/// shape Phase 4 wants when it lowers each in turn into a single
/// `ObjectModule`).
///
/// Reuses the runtime's pipeline verbatim (`parse` →
/// `resolve_with_prelude` → `lower_module_with_known_classes`) so
/// the per-module IR is bit-identical to what the JIT would build
/// on first load. Cross-module field layouts thread through via
/// the same `field_layouts` map `VM::interpret` keeps as state.
///
/// Scoped (`@hatch:foo`) and bare-builtin imports are skipped:
/// they're runtime-resolved today and have no on-disk source to
/// recurse into. Phase 4's link step decides whether to bundle
/// them, surface them as object-file imports, or assume the
/// runtime staticlib provides them.
pub fn walk_imports(entry_path: &Path) -> Result<Vec<AotModule>, AotError> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut out: Vec<AotModule> = Vec::new();
    let mut field_layouts: HashMap<String, Vec<String>> = HashMap::new();

    let entry_canonical = std::fs::canonicalize(entry_path)
        .map_err(AotError::Io)?
        .to_string_lossy()
        .into_owned();
    let entry_source_bytes = std::fs::read(entry_path).map_err(AotError::Io)?;

    walk_module(
        &entry_canonical,
        "main",
        &entry_source_bytes,
        &mut visited,
        &mut out,
        &mut field_layouts,
    )?;
    Ok(out)
}

fn walk_module(
    canonical_name: &str,
    request_name: &str,
    source_bytes: &[u8],
    visited: &mut HashSet<String>,
    out: &mut Vec<AotModule>,
    field_layouts: &mut HashMap<String, Vec<String>>,
) -> Result<(), AotError> {
    if !visited.insert(canonical_name.to_string()) {
        return Ok(());
    }

    let source = std::str::from_utf8(source_bytes)
        .map_err(|e| AotError::Frontend(format!("{} is not valid UTF-8: {}", canonical_name, e)))?;
    let parsed = parse(source);
    if !parsed.errors.is_empty() {
        return Err(AotError::Frontend(
            parsed
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let mut interner: Interner = parsed.interner;
    let prelude_syms: Vec<crate::intern::SymbolId> = crate::sema::PRELUDE_NAMES
        .iter()
        .map(|n| interner.intern(n))
        .collect();
    let resolved = resolve_with_prelude(&parsed.module, &interner, &prelude_syms);
    if !resolved.errors.is_empty() {
        return Err(AotError::Frontend(
            resolved
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    // Recurse into relative imports first — depth-first ordering
    // gives us dependency-before-importer in `out`, which matches
    // the order Phase 4 wants when feeding modules into the
    // shared `ObjectModule`.
    let module_dir = Path::new(canonical_name).parent().map(Path::to_path_buf);
    for stmt in &parsed.module {
        let Stmt::Import {
            module: spanned, ..
        } = &stmt.0
        else {
            continue;
        };
        let req = &spanned.0;
        if !(req.starts_with("./") || req.starts_with("../")) {
            continue;
        }
        let Some(base_dir) = module_dir.as_ref() else {
            continue;
        };
        let candidate = with_wren_suffix(base_dir.join(Path::new(req)));
        let imported_canonical = std::fs::canonicalize(&candidate)
            .map_err(AotError::Io)?
            .to_string_lossy()
            .into_owned();
        if visited.contains(&imported_canonical) {
            continue;
        }
        let imported_bytes = std::fs::read(&candidate).map_err(AotError::Io)?;
        walk_module(
            &imported_canonical,
            req,
            &imported_bytes,
            visited,
            out,
            field_layouts,
        )?;
    }

    let module_var_count = resolved.module_vars.len();
    let (module_mir, new_layouts) =
        lower_module_with_known_classes(&parsed.module, &mut interner, &resolved, field_layouts);
    for (k, v) in new_layouts {
        field_layouts.insert(k, v);
    }

    out.push(AotModule {
        name: canonical_name.to_string(),
        request_name: request_name.to_string(),
        source: source_bytes.to_vec(),
        module_var_count,
        mir: module_mir,
        interner,
    });
    Ok(())
}

/// Build a host-targeted `ObjectModule` with the AOT pipeline's
/// settings (PIC + speed). Shared by both single-source and
/// path-walking entry points.
fn make_object_module() -> Result<ObjectModule, AotError> {
    let triple = target_lexicon::Triple::host();

    // ISA: pessimistic settings flags — `is_pic` so the resulting
    // object can be linked into a shared library or PIE without
    // relocation issues; opt level "speed" matches the JIT path
    // (minus the JIT-only legacy register-pressure heuristics).
    let mut flag_builder = settings::builder();
    flag_builder
        .set("is_pic", "true")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| AotError::Isa(e.to_string()))?;
    let flags = settings::Flags::new(flag_builder);

    let isa_builder =
        isa::lookup(triple.clone()).map_err(|_| AotError::UnsupportedTarget(triple.to_string()))?;
    let isa = isa_builder
        .finish(flags)
        .map_err(|e| AotError::Isa(e.to_string()))?;

    // Object module: the JIT swap-in. Same `Module` trait, same
    // `define_function`-shaped API the future shared codegen
    // uses; only the backing storage differs.
    let object_builder = ObjectBuilder::new(
        isa,
        b"wlift_aot".to_vec(),
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| AotError::Module(e.to_string()))?;
    Ok(ObjectModule::new(object_builder))
}

/// Declare + lower one Wren module's top-level into the shared
/// `ObjectModule` under `fn_symbol`, with module-var reads/writes
/// rerouted through the per-module `modvars_symbol` data array.
///
/// Top-level fn: every Wren value is NaN-boxed as a `u64` —
/// signature is `(arity × i64) -> i64`. The AOT lowering emits
/// `GetModuleVar(slot)` / `SetModuleVar(slot, _)` as direct
/// `global_value` + `load`/`store` against `modvars_symbol`,
/// killing the runtime helper call entirely. The data symbol is
/// declared up front so the lowering can resolve its `DataId`
/// without reaching back into the driver.
/// Lower one MIR function under the active AOT config and define
/// it under `symbol`. Shared by top-level, every class method,
/// and every closure body — they all use the same per-module
/// `modvars` / `consts` data symbols (the lowering branches keyed
/// off `aot_cfg`), so factoring this out keeps the per-function
/// loop in `emit_aot_module` from re-declaring data per emission.
fn emit_aot_function(
    module: &mut ObjectModule,
    interner: &Interner,
    mir: &crate::mir::MirFunction,
    aot_cfg: &AotLoweringConfig,
    symbol: &str,
) -> Result<(), AotError> {
    let mut sig = Signature::new(module.target_config().default_call_conv);
    let arity = mir.arity as usize;
    for _ in 0..arity {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));

    let func_id = module
        .declare_function(symbol, Linkage::Export, &sig)
        .map_err(|e| AotError::Module(e.to_string()))?;

    let mut ctx = module.make_context();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);
    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        lower_mir_to_module(mir, interner, &mut builder, module, Some(aot_cfg))
            .map_err(AotError::Module)?;
        builder.seal_all_blocks();
        builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| AotError::Module(e.to_string()))?;
    module.clear_context(&mut ctx);
    Ok(())
}

/// Lower every reachable function in `aot_mod`'s MIR — top-level,
/// every class's methods, and every closure body — into the
/// shared `ObjectModule`. Each function gets its own exported
/// symbol; all of them share the per-module `wlift_modvars_<n>`
/// and `wlift_consts_<n>` data symbols declared here.
///
/// Symbol layout (per module):
/// - `<top_level_symbol>` — the top-level body
/// - `<top_level_symbol>__method_<class_idx>_<method_idx>` — class methods
/// - `<top_level_symbol>__closure_<closure_idx>` — closure bodies
///
/// The `__` separator keeps them grep-friendly inside `nm` output
/// and avoids colliding with whatever names the source itself uses.
fn emit_aot_module(
    module: &mut ObjectModule,
    aot_mod: &AotModule,
    fn_symbol: &str,
    modvars_symbol: &str,
    consts_symbol: &str,
) -> Result<(), AotError> {
    // Per-module .bss for module vars. `var_count == 0` is rare
    // (a module that defines no top-level names — pure imports
    // wouldn't emit any GetModuleVar / SetModuleVar) but the
    // emit must still produce a valid 0-length placeholder so
    // the `DataId` resolves cleanly.
    let var_bytes = aot_mod.module_var_count.saturating_mul(8).max(8);
    let modvars_data = module
        .declare_data(modvars_symbol, Linkage::Export, true, false)
        .map_err(|e| AotError::Module(e.to_string()))?;
    let mut data_desc = DataDescription::new();
    data_desc.define_zeroinit(var_bytes);
    module
        .define_data(modvars_data, &data_desc)
        .map_err(|e| AotError::Module(e.to_string()))?;

    // Per-module string-constant slot table. Declared up front so
    // the lowering can resolve the `DataId`; sized + defined after
    // every function in the module has been lowered.
    let consts_data = module
        .declare_data(consts_symbol, Linkage::Export, true, false)
        .map_err(|e| AotError::Module(e.to_string()))?;

    let aot_cfg = AotLoweringConfig {
        modvars_data,
        consts_data,
        const_strings: std::cell::RefCell::new(Vec::new()),
    };

    // Top-level body — the entry point Phase 7's init pass calls
    // last, after every dependency module's top-level has run.
    emit_aot_function(
        module,
        &aot_mod.interner,
        &aot_mod.mir.top_level,
        &aot_cfg,
        fn_symbol,
    )?;

    // Each class's methods. Methods share the module's `modvars`
    // and `consts` symbols, but their bodies will additionally
    // need static-field addressing (step 4) and a static class
    // metadata pointer (step 6). For now, any `wren_get_static_*`
    // or `wren_call_*` site falls through to the runtime helper
    // import — those externs persist in the produced `.o` until
    // the corresponding step lands.
    for (class_idx, class) in aot_mod.mir.classes.iter().enumerate() {
        for (method_idx, method) in class.methods.iter().enumerate() {
            let sym = format!("{}__method_{}_{}", fn_symbol, class_idx, method_idx);
            emit_aot_function(module, &aot_mod.interner, &method.mir, &aot_cfg, &sym)?;
        }
    }

    // Closure bodies — heap-allocated at runtime, so their
    // upvalue access still calls `wren_get/set_upvalue` against
    // the closure pointer threaded via TLS today. Step 4 changes
    // the ABI to take the closure as a hidden first arg; until
    // then those helpers stay as imports.
    for (closure_idx, closure_mir) in aot_mod.mir.closures.iter().enumerate() {
        let sym = format!("{}__closure_{}", fn_symbol, closure_idx);
        emit_aot_function(module, &aot_mod.interner, closure_mir, &aot_cfg, &sym)?;
    }

    // Define the consts slot array now that we know how many
    // entries the bodies actually use. Zero-fill — step 7 wires a
    // per-module init pass that walks a sibling `.rodata`
    // descriptor table to allocate `ObjString`s into these slots
    // before user code runs.
    let const_count = aot_cfg.const_strings.borrow().len();
    let const_bytes = const_count.saturating_mul(8).max(8);
    let mut consts_desc = DataDescription::new();
    consts_desc.define_zeroinit(const_bytes);
    module
        .define_data(consts_data, &consts_desc)
        .map_err(|e| AotError::Module(e.to_string()))?;

    Ok(())
}

/// Run an `AotModule` straight from a single in-memory source —
/// the parse + sema + MIR-build that `walk_module` runs per file,
/// minus the import recursion. Used by `compile_to_object` for the
/// single-module emit and by the test harness.
fn build_single_aot_module(source: &str, name: &str) -> Result<AotModule, AotError> {
    let parsed = parse(source);
    if !parsed.errors.is_empty() {
        return Err(AotError::Frontend(
            parsed
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let mut interner: Interner = parsed.interner;
    let prelude_syms: Vec<crate::intern::SymbolId> = crate::sema::PRELUDE_NAMES
        .iter()
        .map(|n| interner.intern(n))
        .collect();
    let resolved = resolve_with_prelude(&parsed.module, &interner, &prelude_syms);
    if !resolved.errors.is_empty() {
        return Err(AotError::Frontend(
            resolved
                .errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let field_layouts = std::collections::HashMap::new();
    let (module_mir, _new_layouts) =
        lower_module_with_known_classes(&parsed.module, &mut interner, &resolved, &field_layouts);

    Ok(AotModule {
        name: name.to_string(),
        request_name: name.to_string(),
        source: source.as_bytes().to_vec(),
        module_var_count: resolved.module_vars.len(),
        mir: module_mir,
        interner,
    })
}

/// Compile Wren `source` to a native object file at `output`.
///
/// Single-module entry point — no import resolution. The source is
/// parsed, MIR-built, and lowered into one `ObjectModule` whose
/// only export is `wlift_aot_main`. For multi-file programs use
/// [`compile_path_to_object`].
pub fn compile_to_object(source: &str, output: &Path) -> Result<(), AotError> {
    let aot_mod = build_single_aot_module(source, "<inline>")?;
    let mut module = make_object_module()?;
    emit_aot_module(
        &mut module,
        &aot_mod,
        "wlift_aot_main",
        "wlift_modvars_main",
        "wlift_consts_main",
    )?;
    let product = module.finish();
    let bytes = product
        .emit()
        .map_err(|e| AotError::Module(e.to_string()))?;
    std::fs::write(output, &bytes).map_err(AotError::Io)?;
    Ok(())
}

/// Compile a Wren program rooted at `entry_path` to a native
/// object file at `output`, walking the import graph and emitting
/// every reachable module's top-level into the same object.
///
/// Output symbols:
/// - The entry module's top-level is exported as `wlift_aot_main`
///   (so the runtime / a future linker shim can find the program
///   entry point unconditionally).
/// - Each imported module's top-level is exported as
///   `wlift_aot_mod_<n>` where `n` is the dependency-first index
///   in the walker output. Stable for a given source tree but not
///   stable across edits — Phase 5's runtime shim will discover
///   them by walking the symbol table at startup.
///
/// Modules are walked + lowered in dependency-first order so that
/// when Phase 5 wires up a startup shim that runs each top-level,
/// imported modules are initialised before the importer's body
/// runs — matching the runtime's `interpret` recursion order.
pub fn compile_path_to_object(entry_path: &Path, output: &Path) -> Result<(), AotError> {
    let modules = walk_imports(entry_path)?;
    compile_modules_to_object(&modules, output)
}

/// Lower a pre-walked list of modules into one native object file
/// at `output`. Same emit shape as [`compile_path_to_object`] —
/// extracted so the AOT build driver can hold onto the walker
/// output (sources, request names) for the bootstrap shim while
/// the same list drives the object emit.
///
/// Modules must already be in dependency-first order; the entry
/// is the last element. Symbol naming matches
/// [`compile_path_to_object`]: entry → `wlift_aot_main`,
/// dependencies → `wlift_aot_mod_<n>`.
pub fn compile_modules_to_object(modules: &[AotModule], output: &Path) -> Result<(), AotError> {
    if modules.is_empty() {
        return Err(AotError::Frontend("no modules to emit".into()));
    }

    let mut module = make_object_module()?;
    let last_idx = modules.len() - 1;
    for (idx, aot_mod) in modules.iter().enumerate() {
        let (fn_symbol, modvars_symbol, consts_symbol) = if idx == last_idx {
            (
                "wlift_aot_main".to_string(),
                "wlift_modvars_main".to_string(),
                "wlift_consts_main".to_string(),
            )
        } else {
            (
                format!("wlift_aot_mod_{}", idx),
                format!("wlift_modvars_{}", idx),
                format!("wlift_consts_{}", idx),
            )
        };
        emit_aot_module(
            &mut module,
            aot_mod,
            &fn_symbol,
            &modvars_symbol,
            &consts_symbol,
        )?;
    }

    let product = module.finish();
    let bytes = product
        .emit()
        .map_err(|e| AotError::Module(e.to_string()))?;
    std::fs::write(output, &bytes).map_err(AotError::Io)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase-1 deliverable from the AOT plan: emit a `.o` and
    /// confirm `nm` (or its API equivalent) sees the mangled
    /// symbol. Avoids shelling out to `nm` itself by re-parsing
    /// the bytes through the same `object` crate Cranelift uses
    /// internally — keeps the test self-contained on every host.
    #[test]
    fn emits_object_with_exported_symbol() {
        // Tempfile guarded path so the test cleans up even when
        // assertions abort.
        let tmp = tempfile::Builder::new()
            .prefix("wlift_aot_phase1_")
            .suffix(".o")
            .tempfile()
            .expect("tempfile");
        let path = tmp.path().to_path_buf();
        compile_to_object("class App { static run() { return 42 } }", &path)
            .expect("compile_to_object");

        let bytes = std::fs::read(&path).expect("read object");
        assert!(!bytes.is_empty(), "object file is empty");

        // Parse the produced object and walk its symbol table.
        // `cranelift-object` produces ELF on Linux, Mach-O on
        // macOS, and PE/COFF on Windows — `object::File::parse`
        // handles all three transparently.
        use object::{Object, ObjectSymbol};
        let obj = object::File::parse(&*bytes).expect("parse object");
        let found = obj.symbols().any(|sym| {
            sym.name()
                .map(|n| n == "wlift_aot_main" || n == "_wlift_aot_main")
                .unwrap_or(false)
        });
        assert!(
            found,
            "expected wlift_aot_main in symbol table; got {:?}",
            obj.symbols()
                .filter_map(|s| s.name().ok().map(str::to_string))
                .collect::<Vec<_>>()
        );
    }

    /// Phase-3 walker: a 2-file fixture (`entry.wren` imports
    /// `./helper`) produces two `AotModule` entries in
    /// dependency-first order, with each module's MIR carrying the
    /// expected top-level entry function.
    #[test]
    fn walk_imports_collects_reachable_modules() {
        use std::io::Write;

        let dir = tempfile::Builder::new()
            .prefix("wlift_aot_phase3_")
            .tempdir()
            .expect("tempdir");
        let entry_path = dir.path().join("entry.wren");
        let helper_path = dir.path().join("helper.wren");

        // Two trivial modules — the import is what matters; the
        // bodies just have to parse + resolve cleanly.
        std::fs::File::create(&helper_path)
            .and_then(|mut f| f.write_all(b"class Helper { static value() { return 7 } }\n"))
            .expect("write helper");
        std::fs::File::create(&entry_path)
            .and_then(|mut f| f.write_all(b"import \"./helper\"\nvar x = 1\n"))
            .expect("write entry");

        let modules = walk_imports(&entry_path).expect("walk_imports");
        assert_eq!(
            modules.len(),
            2,
            "expected entry + helper, got {} modules: {:?}",
            modules.len(),
            modules.iter().map(|m| &m.name).collect::<Vec<_>>()
        );

        // Helper canonical paths use the symlink-resolved temp dir,
        // so compare against `canonicalize`'d expectations.
        let helper_canonical = std::fs::canonicalize(&helper_path).unwrap();
        let entry_canonical = std::fs::canonicalize(&entry_path).unwrap();
        assert_eq!(
            modules[0].name,
            helper_canonical.to_string_lossy(),
            "dependency must come before importer in walk order"
        );
        assert_eq!(modules[1].name, entry_canonical.to_string_lossy());
    }

    /// A `./foo` and `../<dir>/foo` referring to the same on-disk
    /// file dedupe to a single `AotModule`. Mirrors the runtime's
    /// canonicalising resolver — without dedupe we'd emit two
    /// copies of the same function and break class identity at
    /// link time the same way two ModuleEntry copies break `is`
    /// at runtime.
    #[test]
    fn walk_imports_dedupes_aliased_paths() {
        use std::io::Write;

        let dir = tempfile::Builder::new()
            .prefix("wlift_aot_phase3_dedupe_")
            .tempdir()
            .expect("tempdir");
        // entry.wren and a.wren both import the same shared.wren
        // — entry directly via "./shared", a.wren via the same
        // "./shared". Two importers, one helper.
        let shared = dir.path().join("shared.wren");
        let a = dir.path().join("a.wren");
        let entry = dir.path().join("entry.wren");

        std::fs::File::create(&shared)
            .and_then(|mut f| f.write_all(b"class Shared {}\n"))
            .expect("write shared");
        std::fs::File::create(&a)
            .and_then(|mut f| f.write_all(b"import \"./shared\"\nvar a = 1\n"))
            .expect("write a");
        std::fs::File::create(&entry)
            .and_then(|mut f| f.write_all(b"import \"./shared\"\nimport \"./a\"\nvar e = 1\n"))
            .expect("write entry");

        let modules = walk_imports(&entry).expect("walk_imports");
        let names: Vec<&String> = modules.iter().map(|m| &m.name).collect();
        let shared_canonical = std::fs::canonicalize(&shared)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let count = names.iter().filter(|n| ***n == shared_canonical).count();
        assert_eq!(
            count,
            1,
            "shared.wren must appear exactly once across {} modules: {:?}",
            modules.len(),
            names
        );
        assert_eq!(modules.len(), 3, "expected shared + a + entry");
    }

    /// Phase-4 multi-module emit: a 2-file fixture produces one
    /// `.o` containing **both** `wlift_aot_main` (entry) and
    /// `wlift_aot_mod_0` (helper, the dependency). The shared
    /// codegen runs unchanged across modules — only the entry
    /// symbol differs.
    #[test]
    fn compile_path_emits_per_module_symbols() {
        use std::io::Write;

        let dir = tempfile::Builder::new()
            .prefix("wlift_aot_phase4_")
            .tempdir()
            .expect("tempdir");
        let entry_path = dir.path().join("entry.wren");
        let helper_path = dir.path().join("helper.wren");

        std::fs::File::create(&helper_path)
            .and_then(|mut f| f.write_all(b"class Helper { static value() { return 7 } }\n"))
            .expect("write helper");
        std::fs::File::create(&entry_path)
            .and_then(|mut f| f.write_all(b"import \"./helper\"\nvar x = 1\n"))
            .expect("write entry");

        let obj_path = dir.path().join("out.o");
        compile_path_to_object(&entry_path, &obj_path).expect("compile_path_to_object");

        let bytes = std::fs::read(&obj_path).expect("read object");
        assert!(!bytes.is_empty(), "object file is empty");

        use object::{Object, ObjectSymbol};
        let obj = object::File::parse(&*bytes).expect("parse object");
        let names: Vec<String> = obj
            .symbols()
            .filter_map(|s| s.name().ok().map(str::to_string))
            .collect();
        let has = |needle: &str| {
            names
                .iter()
                .any(|n| n == needle || n == &format!("_{}", needle))
        };
        assert!(
            has("wlift_aot_main"),
            "expected wlift_aot_main; got {:?}",
            names
        );
        assert!(
            has("wlift_aot_mod_0"),
            "expected wlift_aot_mod_0 (helper top-level); got {:?}",
            names
        );
    }
}
