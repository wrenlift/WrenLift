//! AOT object-file emission — Cranelift's `ObjectModule` swap-in
//! for the production JIT path.
//!
//! Phase 1 of the AOT plan: prove the object pipeline lights up
//! end-to-end. Compile a single trivial Wren function to a
//! native `.o` file (no linker yet, no whole-module graph yet,
//! no runtime statically linked) and confirm the mangled symbol
//! lands on disk where `nm` can see it.
//!
//! Later phases generalise the same pipeline over the entire
//! parsed module + ship a linked executable. The Cranelift IR
//! emitted here is intentionally minimal: a single
//! `wlift_aot_main_<hash>` function returning a hardcoded `i64`,
//! which is enough to sanity-check that
//!
//! - the `cranelift-object` dep links into the workspace,
//! - the host triple resolves to a backend Cranelift can target,
//! - the resulting object file's symbol table contains the
//!   exported function.
//!
//! Once the whole-module Cranelift translator path lands (Phase 2
//! deliverable), the body fill-in here is replaced by the same
//! MIR-to-CLIF code the JIT uses; only the `Module` trait impl
//! differs.
//!
//! Behind `feature = "aot"` so the production crate stays
//! untouched. Pulled in via `cargo build --features aot` or by
//! the `wlift build` driver once that lands.

use std::path::Path;

use cranelift_codegen::ir::{types, AbiParam, Function, InstBuilder, Signature, UserFuncName};
use cranelift_codegen::isa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

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
}

impl std::fmt::Display for AotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AotError::UnsupportedTarget(t) => write!(f, "unsupported target triple: {}", t),
            AotError::Isa(m) => write!(f, "cranelift ISA setup failed: {}", m),
            AotError::Module(m) => write!(f, "cranelift module operation failed: {}", m),
            AotError::Io(e) => write!(f, "writing object file: {}", e),
        }
    }
}

impl std::error::Error for AotError {}

/// Compile a stub Wren snippet to a native object file at
/// `output`. Phase-1 stub: emits a single function named
/// `wlift_aot_main` that returns the literal `42` so the host
/// process can `nm` the result and confirm the symbol exists.
///
/// The `_source` argument is held for shape parity with the
/// future full-pipeline entry point; today's body is hardcoded.
/// Replace the body fill-in with the existing MIR-to-CLIF
/// translator once we've shared codegen across JIT + AOT
/// (Phase 2).
pub fn compile_to_object(_source: &str, output: &Path) -> Result<(), AotError> {
    // Resolve the host target triple. Cross-target builds will
    // pass `--target=<triple>` explicitly later; Phase 1 sticks
    // to the host so the test can `nm` the result without
    // installing a sysroot.
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

    // Declare a single externally-visible function returning i64.
    // Cranelift's call-conv selection matches the host ABI by
    // default — fine for Phase 1 since the test driver just
    // asserts the symbol is present, not that it's callable.
    let mut sig = Signature::new(module.target_config().default_call_conv);
    sig.returns.push(AbiParam::new(types::I64));
    let func_id = module
        .declare_function("wlift_aot_main", Linkage::Export, &sig)
        .map_err(|e| AotError::Module(e.to_string()))?;

    // Emit the trivial CLIF body. `iconst.i64 42; return r0`.
    let mut ctx = module.make_context();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);
    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let block = builder.create_block();
        builder.append_block_params_for_function_params(block);
        builder.switch_to_block(block);
        builder.seal_block(block);
        let v = builder.ins().iconst(types::I64, 42);
        builder.ins().return_(&[v]);
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
