//! AOT through LLVM.
//!
//! The modules the AOT walker builds, lowered by the LLVM backend into
//! one object for a target triple, with the bootstrap `main` the
//! Cranelift AOT path emits. The target reaches every level: the target
//! machine, the module's triple and data layout, each function's CPU and
//! features, and the object layout the lowering and the bootstrap read.

use std::cell::{Cell, RefCell};
use std::path::Path;

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
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, GlobalValue, PointerValue,
};

use crate::codegen::aot::{
    AotBundleMeta, AotClosureManifest, AotError, AotManifest, AotModule, module_symbols,
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

/// Lower `modules` (dependencies first, entry last) into one object at
/// `output` for `target`.
pub fn compile_modules_to_llvm_object(
    modules: &[AotModule],
    bundle: &AotBundleMeta,
    target: &LlvmTarget,
    output: &Path,
) -> Result<Vec<AotManifest>, AotError> {
    if modules.is_empty() {
        return Err(AotError::Frontend("no modules to emit".into()));
    }
    if !bundle.native_search_paths.is_empty() || !bundle.native_libs.is_empty() {
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
    let mut manifests = Vec::with_capacity(modules.len());
    let mut tables = Vec::with_capacity(modules.len());
    for (idx, m) in modules.iter().enumerate() {
        let (fn_symbol, modvars_symbol, consts_symbol, symbols_symbol) = module_symbols(idx, last);
        let closures_symbol = format!("{fn_symbol}__closures_data");
        let env = AotEnv {
            modvars: table(&ctx, &module, &modvars_symbol, m.module_var_count),
            consts: placeholder(&ctx, &module, &consts_symbol),
            symbols: placeholder(&ctx, &module, &symbols_symbol),
            closures: placeholder(&ctx, &module, &closures_symbol),
            const_strings: RefCell::new(Vec::new()),
            symbol_remap: RefCell::new(Vec::new()),
            defining_slot: Cell::new(None),
            wasm: target.is_wasm(),
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
        for plan in plan_classes(m, &fn_symbol) {
            if plan.class.native_library.is_some() {
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
    .emit(&manifests, &tables, target.is_wasm())
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
    ) -> Result<(), String> {
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
        let desc_ty = self.method_desc_type()?;

        // wasi-libc's startup calls `__main_void` when main takes no
        // arguments.
        let (name, main_ty) = if wasm {
            ("__main_void", i32t.fn_type(&[], false))
        } else {
            (
                "main",
                i32t.fn_type(&[i32t.into(), self.ptr().into()], false),
            )
        };
        let main = self.module.add_function(name, main_ty, None);
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
        let vm = self
            .call(new_vm, &[])?
            .ok_or("new_vm")?
            .into_pointer_value();
        let null = self.b.build_is_null(vm, "novm").map_err(e)?;
        self.b
            .build_conditional_branch(null, fail, body)
            .map_err(e)?;

        self.b.position_at_end(body);
        let vmv: BasicValueEnum = vm.into();
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
        self.call(free_vm, &[vmv])?;
        self.b.build_return(Some(&i32t.const_zero())).map_err(e)?;

        self.b.position_at_end(fail);
        self.b
            .build_return(Some(&i32t.const_int(70, false)))
            .map_err(e)?;
        Ok(())
    }
}
