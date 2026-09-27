//! AOT through LLVM.
//!
//! The modules the AOT walker builds, lowered by the LLVM backend into
//! one object for a target triple, with the bootstrap `main` the
//! Cranelift AOT path emits. The target reaches every level: the target
//! machine, the module's triple and data layout, each function's CPU and
//! features, and the object layout the lowering and the bootstrap read.

use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};

use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use inkwell::attributes::AttributeLoc;
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType, StructType};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, GlobalValue, IntValue,
    PointerValue,
};

use crate::codegen::aot::{
    AotBundleMeta, AotClosureManifest, AotError, AotManifest, AotModule, build_cha, module_symbols,
    plan_classes, resolve_manifest_imports,
};
use crate::codegen::llvm_backend::llvm::{AotEnv, lower_aot_function, stamp_target};
use crate::runtime::object_layout::Layout;

/// The wasm32 features a default build uses: what wasmtime and current
/// browsers run. SIMD is left to `--target-feature +simd128`.
pub const WASM32_FEATURES: &str =
    "+sign-ext,+mutable-globals,+bulk-memory,+nontrapping-fptoint,+multivalue,+reference-types";

/// A target triple with the CPU and features to compile for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlvmTarget {
    pub triple: String,
    pub cpu: String,
    pub features: String,
}

impl LlvmTarget {
    /// The machine this runs on, with its CPU and features.
    pub fn host() -> Self {
        init_targets();
        LlvmTarget {
            triple: TargetMachine::get_default_triple()
                .as_str()
                .to_string_lossy()
                .into_owned(),
            cpu: TargetMachine::get_host_cpu_name().to_string(),
            features: TargetMachine::get_host_cpu_features().to_string(),
        }
    }

    /// `triple`, with `cpu` and `features` when given. A wasm32 triple
    /// defaults to [`WASM32_FEATURES`] on the generic CPU; the host's own
    /// triple to the host CPU; any other to its generic CPU.
    pub fn new(triple: &str, cpu: Option<&str>, features: Option<&str>) -> Self {
        let host = LlvmTarget::host();
        let (dcpu, dfeat) = if triple.starts_with("wasm32") {
            ("generic".to_string(), WASM32_FEATURES.to_string())
        } else if triple == host.triple {
            (host.cpu, host.features)
        } else {
            ("generic".to_string(), String::new())
        };
        LlvmTarget {
            triple: triple.to_string(),
            cpu: cpu.map(str::to_string).unwrap_or(dcpu),
            features: features.map(str::to_string).unwrap_or(dfeat),
        }
    }

    pub fn is_wasm(&self) -> bool {
        self.triple.starts_with("wasm32")
    }

    /// The object layout of the target's pointer width.
    pub fn layout(&self) -> Layout {
        Layout::for_triple(&self.triple)
    }

    pub fn machine(&self) -> Result<TargetMachine, AotError> {
        init_targets();
        let triple = TargetTriple::create(&self.triple);
        let target = Target::from_triple(&triple)
            .map_err(|e| AotError::UnsupportedTarget(format!("{}: {e}", self.triple)))?;
        target
            .create_target_machine(
                &triple,
                &self.cpu,
                &self.features,
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| AotError::UnsupportedTarget(self.triple.clone()))
    }
}

fn init_targets() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let config = InitializationConfig::default();
        Target::initialize_native(&config).expect("native target init");
        Target::initialize_webassembly(&config);
    });
}

/// `WLIFT_LLVM_AOT_PASSES` replaces the pipeline; safe to run with.
fn aot_passes() -> String {
    std::env::var("WLIFT_LLVM_AOT_PASSES").unwrap_or_else(|_| "default<O3>".to_string())
}

/// What an object's bootstrap is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AotEntry {
    /// A program: `main` (`__main_void` and `_start` on wasm) makes a
    /// VM, runs the modules and frees it.
    #[default]
    Main,
    /// A library a host links: a static constructor registers a run
    /// function (`wlift_aot_run_programs` runs it in the host's VM), and
    /// each `#export` member is an external symbol named by caribou's
    /// link rule, the module being its `request_name`. Its arguments and
    /// result are NaN-boxed values (a Num is its `f64` bits); an
    /// instance member takes its receiver first. After a call,
    /// `wlift_aot_take_error(vm)` is nonzero if the member raised.
    Library,
}

/// Lower `modules` (dependencies first, entry last) into one object at
/// `output` for `target`, as a program.
pub fn compile_modules_to_llvm_object(
    modules: &[AotModule],
    bundle: &AotBundleMeta,
    target: &LlvmTarget,
    output: &Path,
) -> Result<Vec<AotManifest>, AotError> {
    compile_modules_to_llvm_object_as(modules, bundle, target, AotEntry::Main, output)
}

/// Lower `modules` (dependencies first, entry last) into one object at
/// `output` for `target`, with `entry` as its bootstrap.
pub fn compile_modules_to_llvm_object_as(
    modules: &[AotModule],
    bundle: &AotBundleMeta,
    target: &LlvmTarget,
    entry: AotEntry,
    output: &Path,
) -> Result<Vec<AotManifest>, AotError> {
    if modules.is_empty() {
        return Err(AotError::Frontend("no modules to emit".into()));
    }
    // A wasm program's plugins are modules the harness loads beside it
    // (see `link_wasm`); anywhere else, a native library is dlopened,
    // which this build does not do yet.
    if !target.is_wasm()
        && (!bundle.native_search_paths.is_empty() || !bundle.native_libs.is_empty())
    {
        return Err(AotError::UnsupportedTarget(format!(
            "{}: native libraries in an LLVM AOT build",
            target.triple
        )));
    }
    let machine = target.machine()?;
    let ctx = Context::create();
    let module = ctx.create_module("wlift_aot");
    module.set_triple(&machine.get_triple());
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    let layout = target.layout();
    let ptr_bytes = machine.get_target_data().get_pointer_byte_size(None);
    if ptr_bytes as i32 != layout.ptr_size {
        return Err(AotError::UnsupportedTarget(format!(
            "{}: {ptr_bytes}-byte pointers, no object layout for them",
            target.triple
        )));
    }

    let module_err = |e: String| AotError::Module(e);
    let last = modules.len() - 1;
    // A direct call checks its receiver against a class in any module,
    // so every module's variables exist before the first body.
    let modvars: Vec<GlobalValue> = modules
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            table(
                &ctx,
                &module,
                &module_symbols(idx, last).1,
                m.module_var_count,
            )
        })
        .collect();
    let cha = std::rc::Rc::new(build_cha(modules, last));
    let mut manifests = Vec::with_capacity(modules.len());
    let mut tables = Vec::with_capacity(modules.len());
    for (idx, m) in modules.iter().enumerate() {
        let (fn_symbol, modvars_symbol, consts_symbol, symbols_symbol) = module_symbols(idx, last);
        let closures_symbol = format!("{fn_symbol}__closures_data");
        let env = AotEnv {
            modvars: modvars[idx],
            consts: placeholder(&ctx, &module, &consts_symbol),
            symbols: placeholder(&ctx, &module, &symbols_symbol),
            closures: placeholder(&ctx, &module, &closures_symbol),
            const_strings: RefCell::new(Vec::new()),
            symbol_remap: RefCell::new(Vec::new()),
            defining_slot: Cell::new(None),
            wasm: target.is_wasm(),
            cha: cha.clone(),
        };
        let lower = |mir: &crate::mir::MirFunction, symbol: &str| {
            let f = lower_aot_function(
                &ctx,
                &module,
                &machine,
                mir,
                &m.interner,
                layout,
                &env,
                symbol,
            )
            .map_err(|e| module_err(format!("[{symbol}] {e}")))?;
            f.set_linkage(Linkage::Internal);
            Ok::<FunctionValue, AotError>(f)
        };

        let main_fn = lower(&m.mir.top_level, &fn_symbol)?;
        let mut classes = Vec::new();
        let mut method_fns = Vec::new();
        let mut exports = Vec::new();
        for plan in plan_classes(m, &fn_symbol) {
            if entry == AotEntry::Library
                && let Some(slot) = plan.slot
            {
                let class_name = m.interner.resolve(plan.class.name);
                for (method, body) in plan.class.methods.iter().zip(&plan.methods) {
                    if let Some(x) = export_plan(m, class_name, slot, method, body, &env)? {
                        exports.push(x);
                    }
                }
            }
            if plan.class.native_library.is_some() && !target.is_wasm() {
                return Err(AotError::UnsupportedTarget(format!(
                    "{}: foreign classes in an LLVM AOT build",
                    target.triple
                )));
            }
            env.defining_slot.set(plan.slot);
            let mut fns = Vec::with_capacity(plan.methods.len());
            for (symbol, method) in plan.methods.iter().zip(&plan.class.methods) {
                let body = lower(&method.mir, symbol)?;
                fns.push(entry_for(
                    &ctx,
                    &module,
                    &machine,
                    layout,
                    target,
                    body,
                    method.mir.arity,
                )?);
            }
            if let Some(manifest) = plan.manifest {
                classes.push(manifest);
                method_fns.push(fns);
            }
        }
        env.defining_slot.set(None);
        let mut closures = Vec::with_capacity(m.mir.closures.len());
        let mut closure_fns = Vec::with_capacity(m.mir.closures.len());
        for (k, closure) in m.mir.closures.iter().enumerate() {
            let symbol = format!("{fn_symbol}__closure_{k}");
            let body = lower(closure, &symbol)?;
            closure_fns.push(entry_for(
                &ctx,
                &module,
                &machine,
                layout,
                target,
                body,
                closure.arity,
            )?);
            closures.push(AotClosureManifest {
                fn_symbol: symbol,
                arity: closure.arity,
                debug_name: m.interner.resolve(closure.name).to_string(),
            });
        }

        let const_texts: Vec<String> = env.const_strings.take().into_iter().map(|e| e.1).collect();
        let symbol_names: Vec<String> = env.symbol_remap.take().into_iter().map(|e| e.1).collect();
        let consts = finish_table(&ctx, &module, env.consts, &consts_symbol, const_texts.len());
        let symbols = finish_table(
            &ctx,
            &module,
            env.symbols,
            &symbols_symbol,
            symbol_names.len(),
        );
        let closures_table = finish_table(
            &ctx,
            &module,
            env.closures,
            &closures_symbol,
            closures.len(),
        );
        tables.push(ModuleTables {
            main_fn,
            modvars: env.modvars,
            consts,
            symbols,
            closures: closures_table,
            closure_fns,
            method_fns,
            exports,
        });
        manifests.push(AotManifest {
            fn_symbol,
            modvars_symbol,
            modvars_count: m.module_var_count,
            consts_symbol,
            const_texts,
            symbols_symbol,
            symbol_names,
            module_name: m.request_name.clone(),
            module_aliases: m.aliases.clone(),
            classes,
            imports: Vec::new(),
            closures,
            closures_symbol,
            runtime_imports: Vec::new(),
        });
    }
    resolve_manifest_imports(modules, &mut manifests);

    Bootstrap {
        ctx: &ctx,
        module: &module,
        machine: &machine,
        layout,
        b: ctx.create_builder(),
        strings: 0,
    }
    .emit(&manifests, &tables, target.is_wasm(), entry)
    .map_err(module_err)?;

    module
        .verify()
        .map_err(|e| module_err(format!("llvm verifier: {}", e.to_string())))?;
    module
        .run_passes(&aot_passes(), &machine, PassBuilderOptions::create())
        .map_err(|e| module_err(format!("llvm passes: {e}")))?;
    if std::env::var_os("WLIFT_LLVM_IR").is_some() {
        eprintln!("{}", module.print_to_string().to_string());
    }
    machine
        .write_to_file(&module, FileType::Object, output)
        .map_err(|e| module_err(e.to_string()))?;
    Ok(manifests)
}

/// What the runtime is handed for a closure or method body: on wasm,
/// which checks every indirect call's signature, a wrapper
/// `entry(args, n)` that passes the first `arity` of the `n` values and
/// null for any missing, as a native callee reads only the registers it
/// declares; elsewhere the body itself.
fn entry_for<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    machine: &TargetMachine,
    layout: Layout,
    target: &LlvmTarget,
    body: FunctionValue<'ctx>,
    arity: u8,
) -> Result<FunctionValue<'ctx>, AotError> {
    if !target.is_wasm() {
        return Ok(body);
    }
    let e = |e: inkwell::builder::BuilderError| AotError::Module(e.to_string());
    let i64t = ctx.i64_type();
    let word = ctx.custom_width_int_type(layout.ptr_size as u32 * 8);
    let ptr = ctx.ptr_type(AddressSpace::default());
    let name = format!("{}__entry", body.get_name().to_string_lossy());
    let f = module.add_function(&name, i64t.fn_type(&[ptr.into(), word.into()], false), None);
    f.set_linkage(Linkage::Internal);
    stamp_target(ctx, machine, f);
    let b = ctx.create_builder();
    b.position_at_end(ctx.append_basic_block(f, "entry"));
    let args = f.get_nth_param(0).unwrap().into_pointer_value();
    let n = f.get_nth_param(1).unwrap().into_int_value();
    let null = b.build_alloca(i64t, "null").map_err(e)?;
    b.build_store(
        null,
        i64t.const_int(crate::runtime::value::Value::null().to_bits(), false),
    )
    .map_err(e)?;
    let mut vals: Vec<BasicMetadataValueEnum> = Vec::with_capacity(arity as usize);
    for i in 0..arity as u64 {
        let at = unsafe { b.build_in_bounds_gep(i64t, args, &[i64t.const_int(i, false)], "a") }
            .map_err(e)?;
        let have = b
            .build_int_compare(
                inkwell::IntPredicate::ULT,
                word.const_int(i, false),
                n,
                "have",
            )
            .map_err(e)?;
        let p = b
            .build_select(have, at, null, "ap")
            .map_err(e)?
            .into_pointer_value();
        vals.push(b.build_load(i64t, p, "v").map_err(e)?.into());
    }
    let call = b.build_call(body, &vals, "r").map_err(e)?;
    call.set_tail_call(true);
    let r = call.try_as_basic_value().basic().unwrap();
    b.build_return(Some(&r)).map_err(e)?;
    Ok(f)
}

/// The prelinked wasm runtime object: `WLIFT_WASM_RUNTIME`, else
/// `wasm32-wasip1/wlift_runtime.o` beside the running binary or under
/// `target/{release,debug}` from the working directory.
pub fn locate_wasm_runtime() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Some(p) = std::env::var_os("WLIFT_WASM_RUNTIME") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let beside = std::env::current_exe().ok().and_then(|exe| {
        exe.parent()
            .map(|d| d.join("wasm32-wasip1/wlift_runtime.o"))
    });
    beside
        .into_iter()
        .chain(
            ["release", "debug"]
                .map(|p| PathBuf::from(format!("target/{p}/wasm32-wasip1/wlift_runtime.o"))),
        )
        .find(|p| p.is_file())
}

/// Link a wasm32 program object against the prelinked runtime object
/// (releases ship it beside `wlift`) into one WASI command module at
/// `output`, in process.
///
/// Side modules already beside `output` (see [`place_wasm_libraries`])
/// are native libraries its host loads at start-up, as Ash's does: the
/// program exports its memory, table, `malloc` and exactly the runtime
/// functions and data they import.
pub fn link_wasm(program: &Path, runtime: &Path, output: &Path) -> Result<(), AotError> {
    let read = |path: &Path| -> Result<ash_wasm_link::Object, AotError> {
        let bytes = std::fs::read(path).map_err(AotError::Io)?;
        let name = path.display().to_string();
        ash_wasm_link::read(&name, &bytes).map_err(|e| AotError::Module(format!("{name}: {e:#}")))
    };
    let objects = vec![read(program)?, read(runtime)?];
    let mut options = ash_wasm_link::LinkOptions::default();
    for side in side_modules_beside(output) {
        options.hdll_imports.extend(side.functions);
        options.hdll_data.extend(side.data);
    }
    if !options.hdll_imports.is_empty() {
        // A side module's data is placed with the program's allocator.
        options.hdll_imports.push("malloc".to_string());
    }
    options.hdll_imports.sort();
    options.hdll_imports.dedup();
    options.hdll_data.sort();
    options.hdll_data.dedup();
    let module = ash_wasm_link::link(objects, &options)
        .map_err(|e| AotError::Module(link_error(&format!("{e:#}"))))?;
    std::fs::write(output, module).map_err(AotError::Io)
}

/// The linker's error in wlift's terms. Imports nothing can supply mean
/// the runtime object is out of step with this wlift or lacks a libc
/// function, and the linker's own advice there names another project's
/// build script.
fn link_error(msg: &str) -> String {
    match msg
        .split_once("no host can supply: ")
        .and_then(|(_, rest)| rest.split_once(".\n"))
    {
        Some((names, _)) => format!(
            "the program needs {names}, which the runtime object does not define: the \
             object is older than this wlift, or the name is a libc function it leaves \
             out. Reinstall wlift, or set WLIFT_WASM_RUNTIME to a runtime object built \
             from the same sources"
        ),
        None => format!("link: {msg}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{link_error, link_symbol};

    #[test]
    fn a_link_symbol_spells_every_part_with_its_length() {
        assert_eq!(
            link_symbol("wren", "bench/tally", "Tally", 'm', "add", 1),
            "caribou_4wren_13bench_2ftally_5Tally_m3add_1"
        );
        assert_eq!(
            link_symbol("haxe", "game.Player", "Player", 't', "spawnAt", 2),
            "caribou_4haxe_13game_2ePlayer_6Player_t7spawnAt_2"
        );
        assert_eq!(
            link_symbol("math", "Math", "Math", 't', "hypot", 2),
            "caribou_4math_4Math_4Math_t5hypot_2"
        );
        assert_eq!(
            link_symbol("wren", "m", "C", 's', "hp", 1),
            "caribou_4wren_1m_1C_s2hp_1"
        );
        assert_eq!(
            link_symbol("wren", "m", "C", 'm', "+", 1),
            "caribou_4wren_1m_1C_m3_2b_1"
        );
        assert_eq!(
            link_symbol("wren", "m", "C", 'm', "a_b", 0),
            "caribou_4wren_1m_1C_m5a_5fb_0"
        );
    }

    #[test]
    fn an_out_of_step_runtime_is_reported_in_wlifts_terms() {
        let msg = "the linked module would import 3 symbol(s) no host can supply: \
                   env.pthread_create, env.pthread_join, env.pthread_detach.\nThese come from \
                   the runtime object, so it is older than the compiler that emitted the calls. \
                   Rebuild it:\n    scripts/build_wasm_runtime.py";
        let out = link_error(msg);
        assert!(out.contains("env.pthread_create, env.pthread_join, env.pthread_detach"));
        assert!(out.contains("Reinstall wlift"));
        assert!(!out.contains("scripts/"));
        assert_eq!(link_error("bad object"), "link: bad object");
    }
}

/// The side modules in `output`'s directory, other than `output`.
fn side_modules_beside(output: &Path) -> Vec<ash_wasm_link::SideModule> {
    let dir = match output.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name() != output.file_name())
        .filter(|p| p.extension().is_some_and(|e| e == "wasm"))
        .collect();
    paths.sort();
    paths
        .iter()
        .filter_map(|p| std::fs::read(p).ok())
        .filter_map(|bytes| ash_wasm_link::read_side_module(&bytes).ok().flatten())
        .collect()
}

/// Put each of `libraries` (name, bytes) beside `output` as
/// `<name>.wasm`, where the host looks for it. A side module is copied;
/// a position-independent archive is linked into one first, exporting
/// the `exports` the program's foreign classes name for it.
pub fn place_wasm_libraries(
    libraries: &[(String, Vec<u8>)],
    exports: &std::collections::HashMap<String, Vec<String>>,
    output: &Path,
) -> Result<Vec<PathBuf>, AotError> {
    use crate::side_module;
    let dir = match output.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let mut placed = Vec::with_capacity(libraries.len());
    for (lib, bytes) in libraries {
        let path = dir.join(format!("{lib}.wasm"));
        if side_module::is_archive(bytes) {
            let work = tempfile::Builder::new()
                .prefix("wlift_side_")
                .tempdir()
                .map_err(AotError::Io)?;
            let archive = work.path().join(format!("lib{lib}.a"));
            std::fs::write(&archive, bytes).map_err(AotError::Io)?;
            let wanted = exports.get(lib).map_or(&[][..], Vec::as_slice);
            side_module::link(&archive, wanted, &path).map_err(|e| {
                AotError::Module(format!(
                    "linking native library {lib} into a side module: {e}"
                ))
            })?;
        } else if side_module::is_side_module(bytes) {
            std::fs::write(&path, bytes).map_err(AotError::Io)?;
        } else {
            return Err(AotError::Module(format!(
                "native library {lib}: its wasm build is neither a side module nor an \
                 archive of one"
            )));
        }
        placed.push(path);
    }
    Ok(placed)
}

/// What the bootstrap reaches of one lowered module.
struct ModuleTables<'ctx> {
    main_fn: FunctionValue<'ctx>,
    modvars: GlobalValue<'ctx>,
    consts: GlobalValue<'ctx>,
    symbols: GlobalValue<'ctx>,
    closures: GlobalValue<'ctx>,
    closure_fns: Vec<FunctionValue<'ctx>>,
    /// Per installed class, its method bodies in manifest order.
    method_fns: Vec<Vec<FunctionValue<'ctx>>>,
    /// In a library, the `#export` members to give symbols.
    exports: Vec<ExportPlan>,
}

/// The symbol a member links under across languages: `caribou`, then the
/// language, the module as its language spells it, the class, the kind's
/// letter (`m` method, `g` getter, `s` setter, `t` static, `c`
/// constructor) with the member's name, and the arity, each after a `_`,
/// and each name as its length and its text. A byte that is not a C
/// identifier's is `_` and two hex digits, `_` included. The rule is
/// caribou's (`caribou_mangle`); its test vectors are below.
fn link_symbol(
    lang: &str,
    module: &str,
    class: &str,
    kind: char,
    name: &str,
    arity: usize,
) -> String {
    fn segment(out: &mut String, text: &str) {
        let mut escaped = String::with_capacity(text.len());
        for b in text.bytes() {
            if b.is_ascii_alphanumeric() {
                escaped.push(b as char);
            } else {
                escaped.push_str(&format!("_{b:02x}"));
            }
        }
        out.push_str(&escaped.len().to_string());
        out.push_str(&escaped);
    }
    let mut out = String::from("caribou");
    for part in [lang, module, class] {
        out.push('_');
        segment(&mut out, part);
    }
    out.push('_');
    out.push(kind);
    segment(&mut out, name);
    out.push('_');
    out.push_str(&arity.to_string());
    out
}

/// An `#export` member of a library: the symbol it links under and how
/// to call it.
struct ExportPlan {
    symbol: String,
    /// The class's slot in its module's variables.
    class_slot: u32,
    /// The method's signature in the module's symbol table.
    sig_slot: u32,
    /// Its Wren parameters, the receiver aside.
    arity: usize,
    /// An instance member takes its receiver; a static or constructor
    /// is called on the class.
    has_receiver: bool,
    /// The compiled body, when the export can call it directly: not one
    /// that needs its defining class installed (super, static fields).
    body: Option<String>,
    /// A constructor's body is its initializer, run on a fresh instance.
    constructor: bool,
}

/// The export plan for `method` of the class `class_name` in slot
/// `slot`, when it carries `#export`.
fn export_plan(
    m: &AotModule,
    class_name: &str,
    slot: u32,
    method: &crate::mir::MethodMir,
    body: &str,
    env: &AotEnv,
) -> Result<Option<ExportPlan>, AotError> {
    let export = crate::sema::export::Export::from_entries(&method.attributes)
        .map_err(|e| AotError::Frontend(format!("{class_name}.{}: {e}", method.signature)))?;
    let Some(export) = export else {
        return Ok(None);
    };
    let arity = export.params.len();
    if arity > 8 {
        return Err(AotError::UnsupportedTarget(format!(
            "{class_name}.{}: an #export member takes at most 8 parameters",
            method.signature
        )));
    }
    let kind = if method.is_constructor {
        'c'
    } else if method.is_static {
        't'
    } else if export.is_setter {
        's'
    } else if !export.has_params {
        'g'
    } else {
        'm'
    };
    let symbol = link_symbol(
        "wren",
        &m.request_name,
        class_name,
        kind,
        &export.name,
        arity,
    );
    // The signature goes in the symbol table beside the ones the bodies
    // use, under an id no interned symbol has.
    let mut remap = env.symbol_remap.borrow_mut();
    let sig_slot = match remap.iter().position(|(_, text)| *text == method.signature) {
        Some(i) => i,
        None => {
            let id = u32::MAX - remap.len() as u32;
            remap.push((id, method.signature.clone()));
            remap.len() - 1
        }
    };
    Ok(Some(ExportPlan {
        symbol,
        class_slot: slot,
        sig_slot: sig_slot as u32,
        arity,
        has_receiver: !(method.is_static || method.is_constructor),
        body: (!crate::codegen::aot::method_uses_defining_class(&method.mir))
            .then(|| body.to_string()),
        constructor: method.is_constructor,
    }))
}

/// A zeroed `[i64; n]` (at least one slot) under `name`.
fn table<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    name: &str,
    n: usize,
) -> GlobalValue<'ctx> {
    let ty = ctx.i64_type().array_type(n.max(1) as u32);
    let g = module.add_global(ty, None, name);
    g.set_linkage(Linkage::Internal);
    g.set_initializer(&ty.const_zero());
    g.set_alignment(8);
    g
}

/// A stand-in for a table sized once the module is lowered.
fn placeholder<'ctx>(ctx: &'ctx Context, module: &Module<'ctx>, name: &str) -> GlobalValue<'ctx> {
    module.add_global(
        ctx.i64_type().array_type(0),
        None,
        &format!("{name}.pending"),
    )
}

/// Replace `pending` with the real `n`-slot table.
fn finish_table<'ctx>(
    ctx: &'ctx Context,
    module: &Module<'ctx>,
    pending: GlobalValue<'ctx>,
    name: &str,
    n: usize,
) -> GlobalValue<'ctx> {
    let g = table(ctx, module, name, n);
    pending
        .as_pointer_value()
        .replace_all_uses_with(g.as_pointer_value());
    unsafe { pending.delete() };
    g
}

/// The bootstrap: `main` makes the VM, fills every module's tables,
/// installs its closures and classes, and runs each module's body.
struct Bootstrap<'ctx, 'a> {
    ctx: &'ctx Context,
    module: &'a Module<'ctx>,
    machine: &'a TargetMachine,
    layout: Layout,
    b: inkwell::builder::Builder<'ctx>,
    strings: usize,
}

/// A bootstrap import's parameter: a pointer, a `usize`, or a narrow
/// integer the callee takes zero-extended.
#[derive(Clone, Copy)]
enum P {
    Ptr,
    Word,
    I8,
    I16,
    I32,
    I64,
}

impl<'ctx> Bootstrap<'ctx, '_> {
    fn word(&self) -> IntType<'ctx> {
        self.ctx
            .custom_width_int_type(self.layout.ptr_size as u32 * 8)
    }

    fn ptr(&self) -> inkwell::types::PointerType<'ctx> {
        self.ctx.ptr_type(AddressSpace::default())
    }

    fn ty(&self, p: P) -> BasicTypeEnum<'ctx> {
        match p {
            P::Ptr => self.ptr().into(),
            P::Word => self.word().into(),
            P::I8 => self.ctx.i8_type().into(),
            P::I16 => self.ctx.i16_type().into(),
            P::I32 => self.ctx.i32_type().into(),
            P::I64 => self.ctx.i64_type().into(),
        }
    }

    /// An import with the runtime's exact signature: wasm checks it.
    fn import(&self, name: &str, params: &[P], ret: Option<P>) -> FunctionValue<'ctx> {
        if let Some(f) = self.module.get_function(name) {
            return f;
        }
        let ps: Vec<BasicMetadataTypeEnum> = params.iter().map(|p| self.ty(*p).into()).collect();
        let fty = match ret {
            Some(r) => self.ty(r).fn_type(&ps, false),
            None => self.ctx.void_type().fn_type(&ps, false),
        };
        let f = self.module.add_function(name, fty, None);
        let zext = inkwell::attributes::Attribute::get_named_enum_kind_id("zeroext");
        for (i, p) in params.iter().enumerate() {
            if matches!(p, P::I8 | P::I16) {
                f.add_attribute(
                    AttributeLoc::Param(i as u32),
                    self.ctx.create_enum_attribute(zext, 0),
                );
            }
        }
        f
    }

    fn call(
        &self,
        f: FunctionValue<'ctx>,
        args: &[BasicValueEnum<'ctx>],
    ) -> Result<Option<BasicValueEnum<'ctx>>, String> {
        let a: Vec<BasicMetadataValueEnum> = args.iter().map(|v| (*v).into()).collect();
        let call = self.b.build_call(f, &a, "").map_err(|e| e.to_string())?;
        // The declaration's zero-extension holds at the call too.
        let zext = inkwell::attributes::Attribute::get_named_enum_kind_id("zeroext");
        for (i, p) in f.get_param_iter().enumerate() {
            if let BasicValueEnum::IntValue(v) = p
                && v.get_type().get_bit_width() < 32
            {
                call.add_attribute(
                    AttributeLoc::Param(i as u32),
                    self.ctx.create_enum_attribute(zext, 0),
                );
            }
        }
        Ok(call.try_as_basic_value().basic())
    }

    fn wordc(&self, v: u64) -> BasicValueEnum<'ctx> {
        self.word().const_int(v, false).into()
    }

    /// A private NUL-terminated copy of `text`; its address and length.
    fn string(&mut self, text: &str) -> (BasicValueEnum<'ctx>, BasicValueEnum<'ctx>) {
        self.strings += 1;
        let bytes = self.ctx.const_string(text.as_bytes(), true);
        let g = self.module.add_global(
            bytes.get_type(),
            None,
            &format!("wlift_str_{}", self.strings),
        );
        g.set_linkage(Linkage::Private);
        g.set_constant(true);
        g.set_unnamed_addr(true);
        g.set_initializer(&bytes);
        (g.as_pointer_value().into(), self.wordc(text.len() as u64))
    }

    fn slot_ptr(&self, g: GlobalValue<'ctx>, slot: u64) -> Result<PointerValue<'ctx>, String> {
        let i64t = self.ctx.i64_type();
        unsafe {
            self.b.build_in_bounds_gep(
                i64t,
                g.as_pointer_value(),
                &[i64t.const_int(slot, false)],
                "slot",
            )
        }
        .map_err(|e| e.to_string())
    }

    /// `WliftAotMethodDesc` as an LLVM struct, checked against the
    /// layout table for the target.
    fn method_desc_type(&self) -> Result<StructType<'ctx>, String> {
        let i8t = self.ctx.i8_type();
        let ty = self.ctx.struct_type(
            &[
                self.ptr().into(),
                self.word().into(),
                self.ptr().into(),
                i8t.into(),
                i8t.into(),
            ],
            false,
        );
        let data = self.machine.get_target_data();
        let at = |i: u32| data.offset_of_element(&ty, i).unwrap_or(u64::MAX) as i32;
        let l = &self.layout;
        let got = (at(1), at(2), at(3), at(4), data.get_abi_size(&ty) as i32);
        let want = (
            l.method_desc_sig_len,
            l.method_desc_fn_ptr,
            l.method_desc_arity,
            l.method_desc_flags,
            l.method_desc_size,
        );
        if got != want {
            return Err(format!(
                "WliftAotMethodDesc is {got:?} on this target, the layout table says {want:?}"
            ));
        }
        Ok(ty)
    }

    fn emit(
        mut self,
        manifests: &[AotManifest],
        tables: &[ModuleTables<'ctx>],
        wasm: bool,
        entry_kind: AotEntry,
    ) -> Result<(), String> {
        let library = entry_kind == AotEntry::Library;
        use P::*;
        let i32t = self.ctx.i32_type();
        let new_vm = self.import("wlift_aot_new_vm", &[], Some(Ptr));
        let free_vm = self.import("wrenFreeVM", &[Ptr], None);
        let init_prelude = self.import("wlift_aot_init_prelude", &[Ptr, Ptr, Word], Some(I32));
        let root_region = self.import(
            "wlift_aot_register_root_region",
            &[Ptr, Ptr, Word],
            Some(I32),
        );
        let alloc_const = self.import("wlift_aot_alloc_const_string", &[Ptr, Ptr, Word], Some(I64));
        let intern = self.import("wlift_aot_intern_symbol", &[Ptr, Ptr, Word], Some(I64));
        let register_closure = self.import(
            "wlift_aot_register_closure",
            &[Ptr, I8, Ptr, Ptr, Word],
            Some(I64),
        );
        let runtime_import = self.import(
            "wlift_aot_resolve_runtime_import",
            &[Ptr, Ptr, Word, Ptr, Word, Ptr, Word],
            Some(I32),
        );
        let install_class = self.import(
            "wlift_aot_install_class",
            &[Ptr, Ptr, Word, Ptr, Word, Word, I16, Ptr, Word],
            Some(I32),
        );
        let enter = self.import("wlift_aot_enter", &[Ptr, Ptr, Word, Ptr, Word, Ptr], None);
        let exit = self.import("wlift_aot_exit", &[Ptr], None);
        let invoke = self.import("wlift_aot_invoke_module_body", &[Ptr], Some(I64));
        let take_error = self.import("wlift_aot_take_error", &[Ptr], Some(I32));
        let bind_foreign = self.import(
            "wlift_aot_bind_foreign_class",
            &[Ptr, Ptr, Word, Ptr, Word, Ptr, Word],
            Some(I32),
        );
        let desc_ty = self.method_desc_type()?;
        // `WliftAotForeignMethodDesc`: repr(C), so the target's natural
        // struct layout.
        let foreign_desc_ty = self.ctx.struct_type(
            &[
                self.ptr().into(),
                self.word().into(),
                self.ptr().into(),
                self.word().into(),
                self.ctx.i8_type().into(),
            ],
            false,
        );

        // wasi-libc's startup calls `__main_void` when main takes no
        // arguments. A library's run takes its host's VM instead.
        let (name, main_ty) = if library {
            (
                "wlift_program_run",
                i32t.fn_type(&[self.ptr().into()], false),
            )
        } else if wasm {
            ("__main_void", i32t.fn_type(&[], false))
        } else {
            (
                "main",
                i32t.fn_type(&[i32t.into(), self.ptr().into()], false),
            )
        };
        let main = self
            .module
            .add_function(name, main_ty, library.then_some(Linkage::Internal));
        stamp_target(self.ctx, self.machine, main);
        let entry = self.ctx.append_basic_block(main, "entry");
        let body = self.ctx.append_basic_block(main, "body");
        let fail = self.ctx.append_basic_block(main, "fail");
        let e = |e: inkwell::builder::BuilderError| e.to_string();

        self.b.position_at_end(entry);
        let saved = self
            .b
            .build_array_alloca(
                self.ctx.i64_type(),
                self.ctx.i64_type().const_int(16, false),
                "saved",
            )
            .map_err(e)?;
        if wasm {
            // Compiled frames all run below this one: the collector scans
            // the shadow stack up to the end of the context buffer. A
            // library's run only raises it, below its host's frames.
            let stack_top = self.import(
                if library {
                    "wlift_aot_raise_stack_top"
                } else {
                    "wlift_aot_stack_top"
                },
                &[Ptr],
                None,
            );
            let end = unsafe {
                self.b.build_in_bounds_gep(
                    self.ctx.i64_type(),
                    saved,
                    &[self.ctx.i64_type().const_int(16, false)],
                    "top",
                )
            }
            .map_err(e)?;
            self.call(stack_top, &[end.into()])?;
        }
        let vm = if library {
            let vm = main
                .get_nth_param(0)
                .ok_or("run takes the VM")?
                .into_pointer_value();
            self.b
                .build_store(self.program_vm().as_pointer_value(), vm)
                .map_err(e)?;
            vm
        } else {
            self.call(new_vm, &[])?
                .ok_or("new_vm")?
                .into_pointer_value()
        };
        let null = self.b.build_is_null(vm, "novm").map_err(e)?;
        self.b
            .build_conditional_branch(null, fail, body)
            .map_err(e)?;

        self.b.position_at_end(body);
        let vmv: BasicValueEnum = vm.into();
        // Only a program with foreign classes reaches the side modules its
        // host loaded, and only it imports Ash's dlopen and dlsym.
        if wasm
            && manifests
                .iter()
                .any(|m| m.classes.iter().any(|c| c.foreign_library.is_some()))
        {
            let set = self.import("wlift_aot_set_native_loader", &[Ptr, Ptr], None);
            let open = self.forward("wlift_native_open", "ash_host_dlopen", &[Ptr, Word])?;
            let sym = self.forward(
                "wlift_native_sym",
                "ash_host_dlsym",
                &[Ptr, Word, Ptr, Word],
            )?;
            let fp = |f: FunctionValue<'ctx>| f.as_global_value().as_pointer_value().into();
            self.call(set, &[fp(open), fp(sym)])?;
        }
        for (m, t) in manifests.iter().zip(tables) {
            let modvars: BasicValueEnum = t.modvars.as_pointer_value().into();
            let consts: BasicValueEnum = t.consts.as_pointer_value().into();
            let count = self.wordc(m.modvars_count as u64);
            self.call(init_prelude, &[vmv, modvars, count])?;
            self.call(root_region, &[vmv, modvars, count])?;
            let nconsts = self.wordc(m.const_texts.len() as u64);
            self.call(root_region, &[vmv, consts, nconsts])?;
            for (k, text) in m.const_texts.iter().enumerate() {
                let (p, len) = self.string(text);
                let v = self
                    .call(alloc_const, &[vmv, p, len])?
                    .ok_or("alloc_const")?;
                let slot = self.slot_ptr(t.consts, k as u64)?;
                self.b.build_store(slot, v).map_err(e)?;
            }
            for (k, name) in m.symbol_names.iter().enumerate() {
                let (p, len) = self.string(name);
                let v = self.call(intern, &[vmv, p, len])?.ok_or("intern")?;
                let slot = self.slot_ptr(t.symbols, k as u64)?;
                self.b.build_store(slot, v).map_err(e)?;
            }
            for (k, (c, f)) in m.closures.iter().zip(&t.closure_fns).enumerate() {
                let (p, len) = self.string(&c.debug_name);
                let arity = self.ctx.i8_type().const_int(c.arity as u64, false).into();
                let fp = f.as_global_value().as_pointer_value().into();
                let v = self
                    .call(register_closure, &[vmv, arity, fp, p, len])?
                    .ok_or("register_closure")?;
                let slot = self.slot_ptr(t.closures, k as u64)?;
                self.b.build_store(slot, v).map_err(e)?;
            }
            for ri in &m.runtime_imports {
                let (mp, ml) = self.string(&ri.module_name);
                let (vp, vl) = self.string(&ri.var_name);
                let slot = self.wordc(ri.target_slot as u64);
                self.call(runtime_import, &[vmv, modvars, slot, mp, ml, vp, vl])?;
            }
            for imp in &m.imports {
                let src = manifests
                    .iter()
                    .position(|s| s.modvars_symbol == imp.source_modvars_symbol)
                    .ok_or_else(|| format!("no module owns {}", imp.source_modvars_symbol))?;
                let from = self.slot_ptr(tables[src].modvars, imp.source_slot as u64)?;
                let v = self
                    .b
                    .build_load(self.ctx.i64_type(), from, "imp")
                    .map_err(e)?;
                let to = self.slot_ptr(t.modvars, imp.target_slot as u64)?;
                self.b.build_store(to, v).map_err(e)?;
            }
            for (class, fns) in m.classes.iter().zip(&t.method_fns) {
                let mut descs = Vec::with_capacity(class.methods.len());
                for (method, f) in class.methods.iter().zip(fns) {
                    let (sig, sig_len) = self.string(&method.signature);
                    let flags = (method.is_static as u64) | ((method.is_constructor as u64) << 1);
                    descs.push(
                        desc_ty.const_named_struct(&[
                            sig.as_basic_value_enum(),
                            sig_len,
                            f.as_global_value().as_pointer_value().into(),
                            self.ctx
                                .i8_type()
                                .const_int(method.arity as u64, false)
                                .into(),
                            self.ctx.i8_type().const_int(flags, false).into(),
                        ]),
                    );
                }
                let arr = desc_ty.const_array(&descs);
                let g = self.module.add_global(
                    arr.get_type(),
                    None,
                    &format!("{}__class_{}", m.fn_symbol, class.slot),
                );
                g.set_linkage(Linkage::Private);
                g.set_constant(true);
                g.set_initializer(&arr);
                let (name, name_len) = self.string(&class.name);
                let parent = class.parent_slot.map(u64::from).unwrap_or(u64::MAX);
                self.call(
                    install_class,
                    &[
                        vmv,
                        modvars,
                        self.wordc(class.slot as u64),
                        name,
                        name_len,
                        self.word().const_int(parent, false).into(),
                        self.ctx
                            .i16_type()
                            .const_int(class.num_fields as u64, false)
                            .into(),
                        g.as_pointer_value().into(),
                        self.wordc(class.methods.len() as u64),
                    ],
                )?;
                if let Some(lib) = &class.foreign_library {
                    let mut descs = Vec::with_capacity(class.foreign_methods.len());
                    for fm in &class.foreign_methods {
                        let (sig, sig_len) = self.string(&fm.signature);
                        let (sym, sym_len) = match &fm.symbol {
                            Some(s) => self.string(s),
                            None => (self.ptr().const_null().into(), self.wordc(0)),
                        };
                        descs.push(
                            foreign_desc_ty.const_named_struct(&[
                                sig,
                                sig_len,
                                sym,
                                sym_len,
                                self.ctx
                                    .i8_type()
                                    .const_int(fm.is_static as u64, false)
                                    .into(),
                            ]),
                        );
                    }
                    let arr = foreign_desc_ty.const_array(&descs);
                    let g = self.module.add_global(
                        arr.get_type(),
                        None,
                        &format!("{}__foreign_{}", m.fn_symbol, class.slot),
                    );
                    g.set_linkage(Linkage::Private);
                    g.set_constant(true);
                    g.set_initializer(&arr);
                    let (lp, ll) = self.string(lib);
                    let rc = self
                        .call(
                            bind_foreign,
                            &[
                                vmv,
                                modvars,
                                self.wordc(class.slot as u64),
                                lp,
                                ll,
                                g.as_pointer_value().into(),
                                self.wordc(class.foreign_methods.len() as u64),
                            ],
                        )?
                        .ok_or("bind_foreign_class")?
                        .into_int_value();
                    // A plugin the harness did not load ends the program
                    // here, with the loader's error already reported.
                    let failed = self
                        .b
                        .build_int_compare(
                            inkwell::IntPredicate::NE,
                            rc,
                            i32t.const_zero(),
                            "unbound",
                        )
                        .map_err(e)?;
                    let bail = self.ctx.append_basic_block(main, "unbound");
                    let next = self.ctx.append_basic_block(main, "bound");
                    self.b
                        .build_conditional_branch(failed, bail, next)
                        .map_err(e)?;
                    self.b.position_at_end(bail);
                    self.b.build_return(Some(&rc)).map_err(e)?;
                    self.b.position_at_end(next);
                }
            }
            let (mname, mlen) = self.string(&m.module_name);
            self.call(enter, &[vmv, modvars, count, mname, mlen, saved.into()])?;
            let fp = t.main_fn.as_global_value().as_pointer_value().into();
            self.call(invoke, &[fp])?;
            self.call(exit, &[saved.into()])?;
            // An error the body left uncaught ends the program with it.
            let rc = self
                .call(take_error, &[vmv])?
                .ok_or("take_error")?
                .into_int_value();
            let raised = self
                .b
                .build_int_compare(inkwell::IntPredicate::NE, rc, i32t.const_zero(), "raised")
                .map_err(e)?;
            let bail = self.ctx.append_basic_block(main, "raised");
            let next = self.ctx.append_basic_block(main, "next");
            self.b
                .build_conditional_branch(raised, bail, next)
                .map_err(e)?;
            self.b.position_at_end(bail);
            self.b.build_return(Some(&rc)).map_err(e)?;
            self.b.position_at_end(next);
        }
        if !library {
            self.call(free_vm, &[vmv])?;
        }
        self.b.build_return(Some(&i32t.const_zero())).map_err(e)?;

        self.b.position_at_end(fail);
        self.b
            .build_return(Some(&i32t.const_int(70, false)))
            .map_err(e)?;
        if library {
            self.emit_registration(main)?;
            for (m, t) in manifests.iter().zip(tables) {
                for x in &t.exports {
                    self.emit_export(m, t, x, wasm)?;
                }
            }
        } else if wasm {
            self.emit_start(main)?;
        }
        Ok(())
    }

    /// The VM a library program runs in, set by its run function and read
    /// by its exported members.
    fn program_vm(&self) -> GlobalValue<'ctx> {
        if let Some(g) = self.module.get_global("wlift_program_vm") {
            return g;
        }
        let g = self.module.add_global(self.ptr(), None, "wlift_program_vm");
        g.set_linkage(Linkage::Internal);
        g.set_initializer(&self.ptr().const_null());
        g
    }

    /// A static constructor handing `run` to the runtime, so a host
    /// linking this library runs it without naming it.
    fn emit_registration(&self, run: FunctionValue<'ctx>) -> Result<(), String> {
        let e = |e: inkwell::builder::BuilderError| e.to_string();
        let register = self.import("wlift_aot_register_program", &[P::Ptr], None);
        let f = self.module.add_function(
            "wlift_program_register",
            self.ctx.void_type().fn_type(&[], false),
            Some(Linkage::Internal),
        );
        stamp_target(self.ctx, self.machine, f);
        self.b
            .position_at_end(self.ctx.append_basic_block(f, "entry"));
        self.call(register, &[run.as_global_value().as_pointer_value().into()])?;
        self.b.build_return(None).map_err(e)?;
        let i32t = self.ctx.i32_type();
        let entry_ty = self
            .ctx
            .struct_type(&[i32t.into(), self.ptr().into(), self.ptr().into()], false);
        let ctors = entry_ty.const_array(&[entry_ty.const_named_struct(&[
            i32t.const_int(65535, false).into(),
            f.as_global_value().as_pointer_value().into(),
            self.ptr().const_null().into(),
        ])]);
        let g = self
            .module
            .add_global(ctors.get_type(), None, "llvm.global_ctors");
        g.set_linkage(Linkage::Appending);
        g.set_initializer(&ctors);
        Ok(())
    }

    /// The external symbol for an `#export` member: it enters the
    /// program's context and makes the call Wren code would, so a
    /// constructor, a static and an inherited method behave as they do
    /// there.
    fn emit_export(
        &mut self,
        m: &AotManifest,
        t: &ModuleTables<'ctx>,
        x: &ExportPlan,
        wasm: bool,
    ) -> Result<(), String> {
        use P::*;
        let e = |e: inkwell::builder::BuilderError| e.to_string();
        let i64t = self.ctx.i64_type();
        let n_params = x.arity + x.has_receiver as usize;
        let params: Vec<BasicMetadataTypeEnum> = (0..n_params).map(|_| i64t.into()).collect();
        let f = self
            .module
            .add_function(&x.symbol, i64t.fn_type(&params, false), None);
        stamp_target(self.ctx, self.machine, f);
        let entry = self.ctx.append_basic_block(f, "entry");
        let call = self.ctx.append_basic_block(f, "call");
        let unrun = self.ctx.append_basic_block(f, "unrun");
        self.b.position_at_end(entry);
        let vm = self
            .b
            .build_load(self.ptr(), self.program_vm().as_pointer_value(), "vm")
            .map_err(e)?
            .into_pointer_value();
        let none = self.b.build_is_null(vm, "unrun").map_err(e)?;
        self.b
            .build_conditional_branch(none, unrun, call)
            .map_err(e)?;

        self.b.position_at_end(unrun);
        let null = i64t.const_int(crate::runtime::value::Value::null().to_bits(), false);
        self.b.build_return(Some(&null)).map_err(e)?;

        self.b.position_at_end(call);
        let saved = self
            .b
            .build_array_alloca(i64t, i64t.const_int(16, false), "saved")
            .map_err(e)?;
        if wasm {
            let raise = self.import("wlift_aot_raise_stack_top", &[Ptr], None);
            let end = unsafe {
                self.b
                    .build_in_bounds_gep(i64t, saved, &[i64t.const_int(16, false)], "top")
            }
            .map_err(e)?;
            self.call(raise, &[end.into()])?;
        }
        let enter = self.import("wlift_aot_enter", &[Ptr, Ptr, Word, Ptr, Word, Ptr], None);
        let exit = self.import("wlift_aot_exit", &[Ptr], None);
        let (mname, mlen) = self.string(&m.module_name);
        self.call(
            enter,
            &[
                vm.into(),
                t.modvars.as_pointer_value().into(),
                self.wordc(m.modvars_count as u64),
                mname,
                mlen,
                saved.into(),
            ],
        )?;
        let receiver: IntValue<'ctx> = if x.has_receiver {
            f.get_nth_param(0).ok_or("receiver")?.into_int_value()
        } else {
            let slot = self.slot_ptr(t.modvars, x.class_slot as u64)?;
            self.b
                .build_load(i64t, slot, "class")
                .map_err(e)?
                .into_int_value()
        };
        let user_args: Vec<BasicValueEnum> =
            f.get_param_iter().skip(x.has_receiver as usize).collect();
        let dispatch = |this: &mut Self| -> Result<IntValue<'ctx>, String> {
            let sym = this.slot_ptr(t.symbols, x.sig_slot as u64)?;
            let sig = this
                .b
                .build_load(i64t, sym, "sig")
                .map_err(|e| e.to_string())?;
            let mut args: Vec<BasicValueEnum> = vec![receiver.into(), sig];
            args.extend(user_args.iter().copied());
            let call_fn = this.import(
                &format!("wren_call_{}", x.arity),
                &vec![I64; x.arity + 2],
                Some(I64),
            );
            Ok(this
                .call(call_fn, &args)?
                .ok_or("wren_call")?
                .into_int_value())
        };
        let body = x
            .body
            .as_deref()
            .and_then(|s| self.module.get_function(s))
            .filter(|b| b.count_params() as usize == x.arity + 1);
        let result = match body {
            None => dispatch(self)?,
            Some(body) => {
                let direct = self.ctx.append_basic_block(f, "direct");
                let join = self.ctx.append_basic_block(f, "join");
                let mut incoming: Vec<(IntValue<'ctx>, inkwell::basic_block::BasicBlock<'ctx>)> =
                    Vec::new();
                if x.has_receiver {
                    // Straight to the body only for an instance of this very
                    // class; a subclass may override the method.
                    use crate::codegen::cranelift_backend::cl::{PTR_MASK, TAG_OBJ};
                    let check = self.ctx.append_basic_block(f, "check");
                    let other = self.ctx.append_basic_block(f, "dispatch");
                    let tag = i64t.const_int(TAG_OBJ, false);
                    let high = self.b.build_and(receiver, tag, "high").map_err(e)?;
                    let is_obj = self
                        .b
                        .build_int_compare(inkwell::IntPredicate::EQ, high, tag, "isobj")
                        .map_err(e)?;
                    self.b
                        .build_conditional_branch(is_obj, check, other)
                        .map_err(e)?;
                    self.b.position_at_end(check);
                    let mask = i64t.const_int(PTR_MASK, false);
                    let addr = self.b.build_and(receiver, mask, "addr").map_err(e)?;
                    let header = self
                        .b
                        .build_int_add(
                            addr,
                            i64t.const_int(self.layout.header_class as u64, false),
                            "hdr",
                        )
                        .map_err(e)?;
                    let header = self
                        .b
                        .build_int_to_ptr(header, self.ptr(), "hdrp")
                        .map_err(e)?;
                    let class_word = self
                        .b
                        .build_load(self.word(), header, "cls")
                        .map_err(e)?
                        .into_int_value();
                    let class_ptr = self
                        .b
                        .build_int_z_extend_or_bit_cast(class_word, i64t, "clsw")
                        .map_err(e)?;
                    let slot = self.slot_ptr(t.modvars, x.class_slot as u64)?;
                    let class = self
                        .b
                        .build_load(i64t, slot, "class")
                        .map_err(e)?
                        .into_int_value();
                    let expected = self.b.build_and(class, mask, "want").map_err(e)?;
                    let same = self
                        .b
                        .build_int_compare(inkwell::IntPredicate::EQ, class_ptr, expected, "same")
                        .map_err(e)?;
                    self.b
                        .build_conditional_branch(same, direct, other)
                        .map_err(e)?;
                    self.b.position_at_end(other);
                    let v = dispatch(self)?;
                    incoming.push((v, self.b.get_insert_block().unwrap()));
                    self.b.build_unconditional_branch(join).map_err(e)?;
                } else {
                    self.b.build_unconditional_branch(direct).map_err(e)?;
                }
                self.b.position_at_end(direct);
                // A constructor allocates the instance its initializer
                // runs on; the receiver is then the class.
                let this = if x.constructor {
                    let alloc = self.import("wren_alloc_instance", &[I64], Some(I64));
                    self.call(alloc, &[receiver.into()])?
                        .ok_or("wren_alloc_instance")?
                        .into_int_value()
                } else {
                    receiver
                };
                let mut args: Vec<BasicMetadataValueEnum> = vec![this.into()];
                args.extend(user_args.iter().map(|a| BasicMetadataValueEnum::from(*a)));
                let returned = self
                    .b
                    .build_call(body, &args, "direct")
                    .map_err(e)?
                    .try_as_basic_value()
                    .basic()
                    .ok_or("a method body returns a value")?
                    .into_int_value();
                let v = if x.constructor { this } else { returned };
                incoming.push((v, self.b.get_insert_block().unwrap()));
                self.b.build_unconditional_branch(join).map_err(e)?;
                self.b.position_at_end(join);
                let phi = self.b.build_phi(i64t, "result").map_err(e)?;
                for (v, bb) in &incoming {
                    phi.add_incoming(&[(v, *bb)]);
                }
                phi.as_basic_value().into_int_value()
            }
        };
        self.call(exit, &[saved.into()])?;
        self.b.build_return(Some(&result)).map_err(e)?;
        Ok(())
    }

    /// An internal function `name` forwarding its arguments to the host
    /// import `import`, both taking `params` and answering an i32.
    fn forward(
        &self,
        name: &str,
        import: &str,
        params: &[P],
    ) -> Result<FunctionValue<'ctx>, String> {
        let tys: Vec<BasicMetadataTypeEnum> = params.iter().map(|p| self.ty(*p).into()).collect();
        let f = self.module.add_function(
            name,
            self.ctx.i32_type().fn_type(&tys, false),
            Some(Linkage::Internal),
        );
        stamp_target(self.ctx, self.machine, f);
        let resume = self.b.get_insert_block();
        self.b
            .position_at_end(self.ctx.append_basic_block(f, "entry"));
        let host = self.import(import, params, Some(P::I32));
        let args: Vec<BasicValueEnum> = f.get_param_iter().collect();
        let v = self.call(host, &args)?.ok_or(import.to_string())?;
        self.b.build_return(Some(&v)).map_err(|e| e.to_string())?;
        if let Some(block) = resume {
            self.b.position_at_end(block);
        }
        Ok(f)
    }

    /// A WASI command's `_start`: the bootstrap, then `exit` with its code
    /// when it failed. Weak, so a libc `crt1` linked in takes its place.
    /// The linker runs the constructors from the module's start function.
    fn emit_start(&self, main: FunctionValue<'ctx>) -> Result<(), String> {
        let e = |e: inkwell::builder::BuilderError| e.to_string();
        let i32t = self.ctx.i32_type();
        let exit = self.import("exit", &[P::I32], None);
        let start =
            self.module
                .add_function("_start", self.ctx.void_type().fn_type(&[], false), None);
        start.set_linkage(Linkage::WeakAny);
        stamp_target(self.ctx, self.machine, start);
        let entry = self.ctx.append_basic_block(start, "entry");
        let fail = self.ctx.append_basic_block(start, "fail");
        let done = self.ctx.append_basic_block(start, "done");
        self.b.position_at_end(entry);
        let code = self.call(main, &[])?.ok_or("main")?.into_int_value();
        let failed = self
            .b
            .build_int_compare(inkwell::IntPredicate::NE, code, i32t.const_zero(), "failed")
            .map_err(e)?;
        self.b
            .build_conditional_branch(failed, fail, done)
            .map_err(e)?;
        self.b.position_at_end(fail);
        self.call(exit, &[code.into()])?;
        self.b.build_unreachable().map_err(e)?;
        self.b.position_at_end(done);
        self.b.build_return(None).map_err(e)?;
        Ok(())
    }
}
