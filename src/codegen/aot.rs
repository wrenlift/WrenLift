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

use std::path::Path;

use cranelift_codegen::ir::{types, AbiParam, Function, Signature, UserFuncName};
use cranelift_codegen::isa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::codegen::cranelift_backend::cl::lower_mir_to_module;
use crate::intern::Interner;
use crate::mir::builder::lower_module_with_known_classes;
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

/// Compile Wren `source` to a native object file at `output`.
///
/// Pipeline:
///
/// 1. Parse + sema + MIR-build the source through the same
///    crate-level entry points the JIT path uses.
/// 2. Pick the module's top-level function as the AOT entry
///    point (`wlift_aot_main`) — the body that runs every
///    statement at the top of a `.wren` file. Phase 3 generalises
///    to every reachable function in the module graph.
/// 3. Spin up an `ObjectModule` for the host triple, declare the
///    entry function as `Linkage::Export`, and call the shared
///    JIT/AOT translator (`cranelift_backend::cl::lower_mir_to_module`)
///    with the same `&mut dyn Module` plumbing the JIT uses.
/// 4. Finalise the module, emit the object bytes, write to
///    `output`.
pub fn compile_to_object(source: &str, output: &Path) -> Result<(), AotError> {
    // -- 1. Frontend: parse → sema → MIR --------------------------
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

    // Sema with the standard prelude name set so `System.print`
    // and friends resolve before MIR-build.
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

    // -- 2. Cranelift ISA + ObjectModule --------------------------
    //
    // Resolve the host target triple. Cross-target builds will
    // pass `--target=<triple>` explicitly later; today we stick
    // to the host so a Phase-2 test can re-parse the result
    // without installing a sysroot for another triple.
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
    let mut module = ObjectModule::new(object_builder);

    // -- 3. Declare + lower the entry function --------------------
    //
    // Top-level fn: every Wren value is NaN-boxed as a `u64` —
    // signature is `() -> i64`. The shared lowering builds the
    // CLIF body directly; runtime calls are emitted as imports
    // and the system linker resolves them against the runtime
    // staticlib at link time (Phase 4 work).
    let entry_mir = &module_mir.top_level;
    let mut sig = Signature::new(module.target_config().default_call_conv);
    let arity = entry_mir.arity as usize;
    for _ in 0..arity {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));
    let func_id = module
        .declare_function("wlift_aot_main", Linkage::Export, &sig)
        .map_err(|e| AotError::Module(e.to_string()))?;

    let mut ctx = module.make_context();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);
    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        lower_mir_to_module(entry_mir, &interner, &mut builder, &mut module)
            .map_err(AotError::Module)?;
        builder.seal_all_blocks();
        builder.finalize();
    }

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| AotError::Module(e.to_string()))?;
    module.clear_context(&mut ctx);

    // Finalise the module — emits relocations + the symbol
    // table — then write the object bytes.
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
}
