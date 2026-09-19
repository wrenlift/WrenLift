//! LLVM top tier.
//!
//! Lowers the same MIR the optimised Cranelift tier takes to LLVM IR
//! and compiles it with MCJIT. The generated code has the same ABI as
//! the Cranelift tier: every parameter and the result are NaN-boxed
//! `i64`s, runtime helpers are the `wren_*` functions reached through
//! their addresses, and each loop header that qualifies gets an OSR
//! entry taking a pointer to its live-ins.
//!
//! Block parameters live in `alloca`s so the lowering can store to
//! them on every edge without building phis; mem2reg in the O2
//! pipeline turns them back into SSA.
#[cfg(feature = "llvm")]
pub mod llvm {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use inkwell::attributes::AttributeLoc;
    use inkwell::basic_block::BasicBlock;
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::execution_engine::ExecutionEngine;
    use inkwell::intrinsics::Intrinsic;
    use inkwell::module::Module;
    use inkwell::passes::PassBuilderOptions;
    use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
    use inkwell::types::{BasicMetadataTypeEnum, BasicTypeEnum, FunctionType};
    use inkwell::values::{
        BasicMetadataValueEnum, BasicValue, BasicValueEnum, FloatValue, FunctionValue, IntValue,
        MetadataValue, PointerValue,
    };
    use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};

    use crate::codegen::NativeOsrEntry;
    use crate::codegen::cranelift_backend::cl::{
        OsrEntryLayout, PTR_MASK, QNAN, TAG_FALSE, TAG_NULL, TAG_OBJ, TAG_TRUE,
        collect_osr_targets, const_f64_of, direct_calls_enabled, env_jit_callsite_ic,
        infer_osr_value_types, is_positive_power_of_two, jit_func_id, jit_modvar_in_range,
        jit_modvars_cell, osr_entry_layout, should_compile_osr_entries,
    };
    use crate::intern::Interner;
    use crate::mir::{
        BlockId, DeoptReg, Instruction, MirFunction, MirType, Terminator, ValueId,
        osr_reachable_blocks, osr_rematerializable_defs,
    };
    use crate::runtime::object_layout::*;

    /// Compiled output of the LLVM tier. The engine owns the code; the
    /// context outlives it (fields drop in order).
    pub struct LlvmCompiledCode {
        pub fn_ptr: *const u8,
        pub osr_entries: Vec<NativeOsrEntry>,
        _engine: ExecutionEngine<'static>,
        _context: Box<Context>,
    }

    // SAFETY: the engine's memory is self-contained; nothing else
    // touches the context after the compile.
    unsafe impl Send for LlvmCompiledCode {}
    unsafe impl Sync for LlvmCompiledCode {}

    type InlineBodies = Arc<HashMap<u32, Arc<MirFunction>>>;

    /// An OSR entry awaiting its address: target block, parameter
    /// count, symbol, and the live-in register descriptions.
    type OsrDef = (
        BlockId,
        u16,
        String,
        Vec<u32>,
        Vec<bool>,
        Vec<Option<u16>>,
        Vec<bool>,
    );

    /// `WLIFT_LLVM_IR=1` prints every module after optimisation.
    fn env_llvm_ir() -> bool {
        std::env::var_os("WLIFT_LLVM_IR").is_some()
    }

    /// `WLIFT_LLVM_PASSES` overrides the middle-end pipeline; `off` skips it.
    fn pass_spec() -> String {
        std::env::var("WLIFT_LLVM_PASSES").unwrap_or_else(|_| "default<O2>".to_string())
    }

    /// `WLIFT_LLVM_CODEGEN=0|1|2|3` sets MCJIT's code generation level.
    fn codegen_level() -> OptimizationLevel {
        match std::env::var("WLIFT_LLVM_CODEGEN").as_deref() {
            Ok("0") => OptimizationLevel::None,
            Ok("1") => OptimizationLevel::Less,
            Ok("3") => OptimizationLevel::Aggressive,
            _ => OptimizationLevel::Default,
        }
    }

    fn init_llvm() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            Target::initialize_native(&InitializationConfig::default())
                .expect("native target init");
            ExecutionEngine::link_in_mc_jit();
        });
    }

    fn host_target_machine() -> Result<TargetMachine, String> {
        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
        let cpu = TargetMachine::get_host_cpu_name().to_string();
        let features = TargetMachine::get_host_cpu_features().to_string();
        target
            .create_target_machine(
                &triple,
                &cpu,
                &features,
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::JITDefault,
            )
            .ok_or_else(|| "no target machine for host".to_string())
    }

    /// Compile a MIR function with LLVM. Same inputs as the Cranelift
    /// tier so the two lower identical MIR.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_mir(
        mir: &MirFunction,
        interner: &Interner,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
        inline_bodies: Option<InlineBodies>,
        cha_by_method: crate::runtime::engine::SharedCha,
    ) -> Result<LlvmCompiledCode, String> {
        init_llvm();
        let t0 = std::time::Instant::now();
        let context = Box::new(Context::create());
        // SAFETY: the box lives in the returned struct and drops after
        // the engine, so nothing built from this reference outlives it.
        let ctx: &'static Context = unsafe { &*(Box::as_ref(&context) as *const Context) };
        let func_name = interner.resolve(mir.name);
        let safe_name = format!(
            "wlift_{}",
            func_name.replace(['(', ')', ',', ' ', '='], "_")
        );
        let module = ctx.create_module(&safe_name);
        let machine = host_target_machine()?;
        module.set_triple(&machine.get_triple());
        module.set_data_layout(&machine.get_target_data().get_data_layout());

        let i64t = ctx.i64_type();
        let ptr_ty = ctx.ptr_type(AddressSpace::default());
        let params: Vec<BasicMetadataTypeEnum> =
            (0..mir.arity as usize).map(|_| i64t.into()).collect();
        let main_ty = i64t.fn_type(&params, false);
        let main_fn = module.add_function(&safe_name, main_ty, None);
        stamp_function(ctx, &machine, main_fn);

        // Every OSR header that qualifies; the body is lowered once with
        // an entry switch over the main entry and these.
        let mut layouts: Vec<OsrEntryLayout> = Vec::new();
        if should_compile_osr_entries(mir, interner) {
            for target in collect_osr_targets(mir) {
                if let Some(layout) = osr_entry_layout(mir, target) {
                    layouts.push(layout);
                }
            }
        }

        let shared = Shared {
            ctx,
            module: &module,
            machine: &machine,
            mir,
            callsite_ic_ptrs,
            callsite_ic_live_ptrs,
            jit_code_base,
            inline_bodies: inline_bodies.as_ref(),
            cha_by_method: cha_by_method.as_deref(),
            main_fn,
            iterate_sym: interner.lookup("iterate(_)"),
            iter_value_sym: interner.lookup("iteratorValue(_)"),
            add_sym: interner.lookup("add(_)"),
            globals: std::cell::RefCell::new(Vec::new()),
        };

        let mut osr_defs: Vec<OsrDef> = Vec::new();
        if layouts.is_empty() {
            Lower::new(&shared, main_fn, &[]).run()?;
        } else {
            // `body(which, args)`: which 0 is the main entry with the
            // parameters in `args`, which k the k-th header with its
            // live-ins there. Kept out of line so the trampolines stay
            // one call each instead of copies of the body.
            let body_ty = i64t.fn_type(&[i64t.into(), ptr_ty.into()], false);
            let body_fn = module.add_function(&format!("{}_body", safe_name), body_ty, None);
            stamp_function(ctx, &machine, body_fn);
            body_fn.add_attribute(
                AttributeLoc::Function,
                ctx.create_enum_attribute(
                    inkwell::attributes::Attribute::get_named_enum_kind_id("noinline"),
                    0,
                ),
            );
            Lower::new(&shared, body_fn, &layouts).run()?;

            let b = ctx.create_builder();
            let entry = ctx.append_basic_block(main_fn, "entry");
            b.position_at_end(entry);
            let n = (mir.arity as usize).max(1);
            let buf = b
                .build_alloca(i64t.array_type(n as u32), "args")
                .map_err(|e| e.to_string())?;
            for i in 0..mir.arity as usize {
                let slot = unsafe {
                    b.build_in_bounds_gep(i64t, buf, &[i64t.const_int(i as u64, false)], "slot")
                }
                .map_err(|e| e.to_string())?;
                b.build_store(slot, main_fn.get_nth_param(i as u32).unwrap())
                    .map_err(|e| e.to_string())?;
            }
            let r = b
                .build_call(body_fn, &[i64t.const_zero().into(), buf.into()], "r")
                .map_err(|e| e.to_string())?
                .try_as_basic_value()
                .basic()
                .unwrap();
            b.build_return(Some(&r)).map_err(|e| e.to_string())?;

            let i64_params: HashSet<ValueId> = mir
                .blocks
                .iter()
                .flat_map(|b| b.params.iter())
                .filter(|(_, t)| *t == MirType::I64)
                .map(|(p, _)| *p)
                .collect();
            for (k, layout) in layouts.iter().enumerate() {
                let target = layout.target_block;
                let name = format!("{}_osr_bb{}", safe_name, target.0);
                let ty = i64t.fn_type(&[ptr_ty.into()], false);
                let f = module.add_function(&name, ty, None);
                stamp_function(ctx, &machine, f);
                let entry = ctx.append_basic_block(f, "entry");
                b.position_at_end(entry);
                let which = i64t.const_int((k + 1) as u64, false);
                let r = b
                    .build_call(
                        body_fn,
                        &[which.into(), f.get_nth_param(0).unwrap().into()],
                        "r",
                    )
                    .map_err(|e| e.to_string())?
                    .try_as_basic_value()
                    .basic()
                    .unwrap();
                b.build_return(Some(&r)).map_err(|e| e.to_string())?;
                let live: Vec<ValueId> = layout
                    .external_args
                    .iter()
                    .copied()
                    .chain(mir.blocks[target.0 as usize].params.iter().map(|(p, _)| *p))
                    .collect();
                osr_defs.push((
                    target,
                    layout.param_count,
                    name,
                    live.iter()
                        .map(|v| {
                            mir.scalar_param_sources
                                .get(v)
                                .map(|(o, _)| o.0)
                                .unwrap_or(v.0)
                        })
                        .collect(),
                    live.iter()
                        .map(|v| mir.speculated_num_params.contains(v))
                        .collect(),
                    live.iter()
                        .map(|v| mir.scalar_param_sources.get(v).map(|(_, f)| *f))
                        .collect(),
                    live.iter().map(|v| i64_params.contains(v)).collect(),
                ));
            }
        }

        let t_build = t0.elapsed();
        if let Err(e) = module.verify() {
            if std::env::var_os("WLIFT_JIT_DEBUG").is_some() || env_llvm_ir() {
                eprintln!("{}", module.print_to_string().to_string());
            }
            return Err(format!("llvm verifier: {}", e.to_string()));
        }
        let spec = pass_spec();
        if spec != "off" {
            module
                .run_passes(&spec, &machine, PassBuilderOptions::create())
                .map_err(|e| format!("run_passes({spec}): {}", e))?;
        }
        let t_opt = t0.elapsed();
        if env_llvm_ir() {
            eprintln!("=== LLVM IR for {} ===", safe_name);
            eprintln!("{}", module.print_to_string().to_string());
            eprintln!("=== end ===");
        }
        // `WLIFT_LLVM_ASM=1` prints the host assembly of every module.
        if std::env::var_os("WLIFT_LLVM_ASM").is_some() {
            let buf = machine
                .write_to_memory_buffer(&module, inkwell::targets::FileType::Assembly)
                .map_err(|e| e.to_string())?;
            eprintln!("=== LLVM ASM for {} ===", safe_name);
            eprintln!("{}", String::from_utf8_lossy(buf.as_slice()));
            eprintln!("=== end ===");
        }

        let engine = module
            .create_jit_execution_engine(codegen_level())
            .map_err(|e| e.to_string())?;
        for (global, addr) in shared.globals.borrow().iter() {
            engine.add_global_mapping(global, *addr as usize);
        }
        let t_engine = t0.elapsed();
        let fn_ptr = engine
            .get_function_address(&safe_name)
            .map_err(|e| e.to_string())? as *const u8;
        if std::env::var_os("WLIFT_TIER_TRACE").is_some() {
            eprintln!(
                "tier-trace: llvm {} build={:?} opt={:?} engine={:?} codegen={:?}",
                safe_name,
                t_build,
                t_opt - t_build,
                t_engine - t_opt,
                t0.elapsed() - t_engine
            );
        }
        let mut osr_entries = Vec::with_capacity(osr_defs.len());
        for (target_block, param_count, name, regs, num, field, int) in osr_defs {
            let Ok(addr) = engine.get_function_address(&name) else {
                continue;
            };
            osr_entries.push(NativeOsrEntry {
                target_block,
                param_count,
                ptr: addr as *const u8,
                live_in_regs: regs,
                live_in_num: num,
                live_in_field: field,
                live_in_int: int,
            });
        }
        Ok(LlvmCompiledCode {
            fn_ptr,
            osr_entries,
            _engine: engine,
            _context: context,
        })
    }

    /// Frame pointers on every body (the conservative scanner and the
    /// call stubs read them) and the host CPU, which MCJIT's default
    /// machine would not select on its own.
    fn stamp_function(ctx: &Context, machine: &TargetMachine, f: FunctionValue) {
        f.add_attribute(
            AttributeLoc::Function,
            ctx.create_string_attribute("frame-pointer", "all"),
        );
        f.add_attribute(
            AttributeLoc::Function,
            ctx.create_string_attribute("target-cpu", &machine.get_cpu().to_string()),
        );
        f.add_attribute(
            AttributeLoc::Function,
            ctx.create_string_attribute(
                "target-features",
                &machine.get_feature_string().to_string_lossy(),
            ),
        );
    }

    /// Inputs shared by the main body and every OSR entry of one compile.
    struct Shared<'ctx, 'a> {
        ctx: &'ctx Context,
        module: &'a Module<'ctx>,
        #[allow(dead_code)]
        machine: &'a TargetMachine,
        mir: &'a MirFunction,
        callsite_ic_ptrs: Option<&'a [crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&'a [usize]>,
        jit_code_base: Option<*const *const u8>,
        inline_bodies: Option<&'a InlineBodies>,
        cha_by_method: Option<&'a crate::runtime::engine::ChaMap>,
        main_fn: FunctionValue<'ctx>,
        /// `iterate(_)`, `iteratorValue(_)` and `add(_)`, lowered inline
        /// for lists.
        iterate_sym: Option<crate::intern::SymbolId>,
        iter_value_sym: Option<crate::intern::SymbolId>,
        add_sym: Option<crate::intern::SymbolId>,
        /// Externals the module reads through a global of known size,
        /// so LLVM may hoist their loads: `(global, address)`, mapped
        /// into the execution engine before the code is finalised.
        globals: std::cell::RefCell<Vec<(inkwell::values::GlobalValue<'ctx>, u64)>>,
    }

    /// Lowering state for one LLVM function (the body or an OSR entry).
    struct Lower<'ctx, 'a> {
        sh: &'a Shared<'ctx, 'a>,
        b: Builder<'ctx>,
        f: FunctionValue<'ctx>,
        /// OSR headers this body can be entered at; empty means the
        /// function's own entry only.
        entries: &'a [OsrEntryLayout],
        blocks: Vec<BasicBlock<'ctx>>,
        /// Alloca per block parameter and per live-in the OSR region
        /// redefines.
        slots: HashMap<ValueId, (PointerValue<'ctx>, BasicTypeEnum<'ctx>)>,
        osr_vars: HashSet<ValueId>,
        vals: HashMap<ValueId, BasicValueEnum<'ctx>>,
        /// TBAA access tag per instance field index: field arrays are
        /// separate allocations, so slot `i` of any instance never
        /// overlaps slot `j != i` of another and loads of one field
        /// survive stores to another.
        field_tbaa: HashMap<u16, MetadataValue<'ctx>>,
        field_tbaa_root: Option<MetadataValue<'ctx>>,
        miss_exit: Option<(u32, Vec<DeoptReg>)>,
        /// Boxed values known to be Nums; a store of one needs no
        /// field-kind note.
        num_values: HashSet<IntValue<'ctx>>,
        /// Instances this body allocated, by class: their fields lie
        /// right after the header and their kind notes have a fixed
        /// address.
        fresh: HashMap<IntValue<'ctx>, usize>,
        /// Receivers whose class a dominating `ClassIs` guard proved,
        /// per block.
        class_facts: HashMap<usize, Vec<(ValueId, usize)>>,
        /// The value each copy was made from, followed to the original:
        /// a class fact about a copy holds for every copy.
        move_roots: HashMap<ValueId, ValueId>,
        /// Boxed values that are an i64 converted to f64: an index
        /// read as the integer skips the integrality test.
        int_sources: HashMap<ValueId, ValueId>,
        cur_block: usize,
        /// The receiver of a body spliced behind its class check.
        inline_class: Option<(IntValue<'ctx>, usize)>,

        /// A guarded getter whose class keeps its field as Nums: the
        /// guard that follows checks the class's field-kind byte at
        /// this address instead of the value.
        /// The value, the class's field-kind bytes with their count,
        /// and the field: the guard that consumes it reads the byte.
        field_invariant: Option<(ValueId, *mut u8, usize, u16)>,
        /// The instruction being lowered, at inline depth zero.
        cur_vid: ValueId,
        raw_bools: HashSet<ValueId>,
        value_types: Vec<MirType>,

        receiver: Option<IntValue<'ctx>>,
        /// Main-entry parameters when the body has an entry switch.
        param_regs: Vec<IntValue<'ctx>>,
        /// Set while lowering an inlined callee body: its own value map,
        /// receiver, and no inline caches.
        inline_depth: u32,
        tmp: u32,
    }

    macro_rules! bail {
        ($($t:tt)*) => { return Err(format!($($t)*)) };
    }

    impl<'ctx, 'a> Lower<'ctx, 'a> {
        fn new(
            sh: &'a Shared<'ctx, 'a>,
            f: FunctionValue<'ctx>,
            entries: &'a [OsrEntryLayout],
        ) -> Self {
            let mir = sh.mir;
            Self {
                sh,
                b: sh.ctx.create_builder(),
                f,
                entries,
                blocks: Vec::new(),
                slots: HashMap::new(),
                osr_vars: HashSet::new(),
                vals: HashMap::new(),
                field_tbaa: HashMap::new(),
                field_tbaa_root: None,
                miss_exit: None,
                num_values: HashSet::new(),
                fresh: HashMap::new(),
                class_facts: HashMap::new(),
                move_roots: HashMap::new(),
                int_sources: HashMap::new(),
                cur_block: 0,
                inline_class: None,
                field_invariant: None,
                cur_vid: ValueId(u32::MAX),
                raw_bools: HashSet::new(),
                value_types: infer_osr_value_types(mir),

                receiver: None,
                param_regs: Vec::new(),
                inline_depth: 0,
                tmp: 0,
            }
        }

        // ── Types and constants ────────────────────────────────────────

        fn i64t(&self) -> inkwell::types::IntType<'ctx> {
            self.sh.ctx.i64_type()
        }
        fn f64t(&self) -> inkwell::types::FloatType<'ctx> {
            self.sh.ctx.f64_type()
        }
        fn i1t(&self) -> inkwell::types::IntType<'ctx> {
            self.sh.ctx.bool_type()
        }
        fn ptrt(&self) -> inkwell::types::PointerType<'ctx> {
            self.sh.ctx.ptr_type(AddressSpace::default())
        }
        fn c64(&self, v: u64) -> IntValue<'ctx> {
            self.i64t().const_int(v, false)
        }
        fn cf64(&self, v: f64) -> FloatValue<'ctx> {
            self.f64t().const_float(v)
        }
        fn name(&mut self, base: &str) -> String {
            self.tmp += 1;
            format!("{base}{}", self.tmp)
        }

        fn helper_type(&self, n: usize) -> FunctionType<'ctx> {
            let params: Vec<BasicMetadataTypeEnum> = (0..n).map(|_| self.i64t().into()).collect();
            self.i64t().fn_type(&params, false)
        }

        /// Call a `wren_*` runtime helper by name; all-`i64` ABI.
        fn call_helper(
            &mut self,
            name: &str,
            args: &[IntValue<'ctx>],
        ) -> Result<IntValue<'ctx>, String> {
            let Some(addr) = crate::codegen::runtime_fns::resolve(name) else {
                bail!("unknown runtime helper {name}");
            };
            let ty = self.helper_type(args.len());
            self.call_addr(
                ty,
                addr,
                &args.iter().map(|a| (*a).into()).collect::<Vec<_>>(),
                name,
            )
            .map(|v| v.into_int_value())
        }

        fn call_addr(
            &mut self,
            ty: FunctionType<'ctx>,
            addr: usize,
            args: &[BasicMetadataValueEnum<'ctx>],
            name: &str,
        ) -> Result<BasicValueEnum<'ctx>, String> {
            let ptr = self
                .b
                .build_int_to_ptr(self.c64(addr as u64), self.ptrt(), "fp")
                .map_err(|e| e.to_string())?;
            let call = self
                .b
                .build_indirect_call(ty, ptr, args, name)
                .map_err(|e| e.to_string())?;
            call.try_as_basic_value()
                .basic()
                .ok_or_else(|| "helper returned void".to_string())
        }

        /// A libm function declared by name, so LLVM knows it is pure
        /// and MCJIT binds it from the process.
        fn libm_decl(&mut self, name: &str, arity: usize) -> FunctionValue<'ctx> {
            if let Some(f) = self.sh.module.get_function(name) {
                return f;
            }
            let params: Vec<BasicMetadataTypeEnum> =
                (0..arity).map(|_| self.f64t().into()).collect();
            let f = self
                .sh
                .module
                .add_function(name, self.f64t().fn_type(&params, false), None);
            let ctx = self.sh.ctx;
            f.add_attribute(
                AttributeLoc::Function,
                ctx.create_enum_attribute(
                    inkwell::attributes::Attribute::get_named_enum_kind_id("nounwind"),
                    0,
                ),
            );
            f.add_attribute(
                AttributeLoc::Function,
                ctx.create_string_attribute("memory", "none"),
            );
            f
        }

        fn libm1(&mut self, name: &str, x: FloatValue<'ctx>) -> Result<FloatValue<'ctx>, String> {
            let f = self.libm_decl(name, 1);
            let call = self
                .b
                .build_call(f, &[x.into()], name)
                .map_err(|e| e.to_string())?;
            Ok(call
                .try_as_basic_value()
                .basic()
                .unwrap()
                .into_float_value())
        }

        fn libm2(
            &mut self,
            name: &str,
            x: FloatValue<'ctx>,
            y: FloatValue<'ctx>,
        ) -> Result<FloatValue<'ctx>, String> {
            let f = self.libm_decl(name, 2);
            let call = self
                .b
                .build_call(f, &[x.into(), y.into()], name)
                .map_err(|e| e.to_string())?;
            Ok(call
                .try_as_basic_value()
                .basic()
                .unwrap()
                .into_float_value())
        }

        fn intrinsic1(
            &mut self,
            name: &str,
            x: FloatValue<'ctx>,
        ) -> Result<FloatValue<'ctx>, String> {
            let intr = Intrinsic::find(name).ok_or_else(|| format!("no intrinsic {name}"))?;
            let decl = intr
                .get_declaration(self.sh.module, &[self.f64t().into()])
                .ok_or_else(|| format!("no declaration for {name}"))?;
            let call = self
                .b
                .build_call(decl, &[x.into()], "intr")
                .map_err(|e| e.to_string())?;
            Ok(call
                .try_as_basic_value()
                .basic()
                .unwrap()
                .into_float_value())
        }

        fn intrinsic2(
            &mut self,
            name: &str,
            x: FloatValue<'ctx>,
            y: FloatValue<'ctx>,
        ) -> Result<FloatValue<'ctx>, String> {
            let intr = Intrinsic::find(name).ok_or_else(|| format!("no intrinsic {name}"))?;
            let decl = intr
                .get_declaration(self.sh.module, &[self.f64t().into()])
                .ok_or_else(|| format!("no declaration for {name}"))?;
            let call = self
                .b
                .build_call(decl, &[x.into(), y.into()], "intr")
                .map_err(|e| e.to_string())?;
            Ok(call
                .try_as_basic_value()
                .basic()
                .unwrap()
                .into_float_value())
        }

        // ── Memory ─────────────────────────────────────────────────────

        fn addr(&mut self, base: IntValue<'ctx>, off: i64) -> Result<PointerValue<'ctx>, String> {
            let a = if off == 0 {
                base
            } else {
                self.b
                    .build_int_add(base, self.c64(off as u64), "addr")
                    .map_err(|e| e.to_string())?
            };
            self.b
                .build_int_to_ptr(a, self.ptrt(), "p")
                .map_err(|e| e.to_string())
        }

        fn field_tag(&mut self, idx: u16) -> MetadataValue<'ctx> {
            let ctx = self.sh.ctx;
            let root = *self.field_tbaa_root.get_or_insert_with(|| {
                ctx.metadata_node(&[ctx.metadata_string("wren fields").into()])
            });
            *self.field_tbaa.entry(idx).or_insert_with(|| {
                let ty = ctx.metadata_node(&[
                    ctx.metadata_string(&format!("field {idx}")).into(),
                    root.into(),
                    ctx.i64_type().const_zero().into(),
                ]);
                ctx.metadata_node(&[ty.into(), ty.into(), ctx.i64_type().const_zero().into()])
            })
        }

        /// Load instance field `idx` from a fields array.
        fn field_load(
            &mut self,
            fields: IntValue<'ctx>,
            idx: u16,
        ) -> Result<IntValue<'ctx>, String> {
            let p = self.addr(fields, idx as i64 * VALUE_SIZE as i64)?;
            let ld = self
                .b
                .build_load(self.i64t(), p, "fld")
                .map_err(|e| e.to_string())?;
            let tag = self.field_tag(idx);
            let kind = self.sh.ctx.get_kind_id("tbaa");
            ld.as_instruction_value()
                .ok_or("load is not an instruction")?
                .set_metadata(tag, kind)
                .map_err(|e| e.to_string())?;
            Ok(ld.into_int_value())
        }

        /// Or the kind of `v` into the class's byte for field `idx` of
        /// the instance at `obj`, as `ObjInstance::note_field_kind`
        /// does; a class without the bytes is skipped.
        fn note_field_kind(
            &mut self,
            obj: IntValue<'ctx>,
            idx: u16,
            v: IntValue<'ctx>,
        ) -> Result<(), String> {
            use crate::runtime::object::{FIELD_NUM, FIELD_OTHER};
            let class = self.load64_stable(obj, HEADER_CLASS as i64)?;
            let kinds = self.load64_stable(class, CLASS_FIELD_KINDS as i64)?;
            let has = self.icmp(IntPredicate::NE, kinds, self.c64(0))?;
            let note = self.new_block("fkn");
            let done = self.new_block("fkd");
            self.cbr(has, note, done)?;
            self.b.position_at_end(note);
            let p = self.addr(kinds, idx as i64)?;
            let seen = self
                .b
                .build_load(self.sh.ctx.i8_type(), p, "seen")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let is_box = self.is_nan_boxed(v)?;
            let bit = self
                .b
                .build_select(
                    is_box,
                    self.sh.ctx.i8_type().const_int(FIELD_OTHER as u64, false),
                    self.sh.ctx.i8_type().const_int(FIELD_NUM as u64, false),
                    "kind",
                )
                .map_err(|e| e.to_string())?
                .into_int_value();
            self.store_kind_bit(p, seen, bit)?;
            self.br(done)?;
            self.b.position_at_end(done);
            Ok(())
        }

        /// The class a guard proved for `recv` in the current block.
        fn known_class(&self, recv: ValueId) -> Option<usize> {
            let recv = self.root(recv);
            self.class_facts
                .get(&self.cur_block)?
                .iter()
                .find(|(v, _)| *v == recv)
                .map(|(_, c)| *c)
        }

        fn root(&self, v: ValueId) -> ValueId {
            self.move_roots.get(&v).copied().unwrap_or(v)
        }

        /// Store `seen | bit` at `p` when that changes the byte.
        fn store_kind_bit(
            &mut self,
            p: PointerValue<'ctx>,
            seen: IntValue<'ctx>,
            bit: IntValue<'ctx>,
        ) -> Result<(), String> {
            let new = self
                .b
                .build_or(seen, bit, "seen")
                .map_err(|e| e.to_string())?;
            let changed = self.icmp(IntPredicate::NE, new, seen)?;
            let store = self.new_block("fks");
            let done = self.new_block("fkd");
            self.cbr(changed, store, done)?;
            self.b.position_at_end(store);
            let st = self.b.build_store(p, new).map_err(|e| e.to_string())?;
            let tag = self.kinds_tag();
            st.set_metadata(tag, self.sh.ctx.get_kind_id("tbaa"))
                .map_err(|e| e.to_string())?;
            self.br(done)?;
            self.b.position_at_end(done);
            Ok(())
        }

        /// The fields array of the instance `obj`: right after the
        /// header.
        fn instance_fields(&mut self, obj: IntValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            self.b
                .build_int_add(obj, self.c64(INSTANCE_SIZE as u64), "fields")
                .map_err(|e| e.to_string())
        }

        /// `note_field_kind` for an instance of a class known at compile
        /// time: the byte has a fixed address, and one that already
        /// records another kind, or the kind being stored, never changes
        /// again.
        fn note_field_kind_static(
            &mut self,
            class: usize,
            idx: u16,
            v: IntValue<'ctx>,
        ) -> Result<(), String> {
            use crate::runtime::object::{FIELD_NUM, FIELD_OTHER};
            let class = class as *const crate::runtime::object::ObjClass;
            let kinds = unsafe { (*class).field_kinds_ptr };
            let len = unsafe { (*class).field_kinds.len() };
            if kinds.is_null() || idx as usize >= len {
                return Ok(());
            }
            let seen = unsafe { std::ptr::read_volatile(kinds.add(idx as usize)) };
            let known_num = self.num_values.contains(&v);
            if seen & FIELD_OTHER != 0 || (known_num && seen & FIELD_NUM != 0) {
                return Ok(());
            }
            let p = self.addr(self.c64(kinds as u64), idx as i64)?;
            let cur = self
                .b
                .build_load(self.sh.ctx.i8_type(), p, "seen")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let bit = if known_num {
                self.sh.ctx.i8_type().const_int(FIELD_NUM as u64, false)
            } else {
                let is_box = self.is_nan_boxed(v)?;
                self.b
                    .build_select(
                        is_box,
                        self.sh.ctx.i8_type().const_int(FIELD_OTHER as u64, false),
                        self.sh.ctx.i8_type().const_int(FIELD_NUM as u64, false),
                        "kind",
                    )
                    .map_err(|e| e.to_string())?
                    .into_int_value()
            };
            self.store_kind_bit(p, cur, bit)
        }

        /// When `class` has only ever held Nums in field `idx` and every
        /// instance starts with one, let the guard on `dst` check the
        /// class's byte rather than the value.
        fn note_field_invariant(&mut self, class: usize, idx: u16, dst: ValueId) {
            if class == 0 {
                return;
            }
            let class = class as *const crate::runtime::object::ObjClass;
            let kinds = unsafe { (*class).field_kinds_ptr };
            let len = unsafe { (*class).field_kinds.len() };
            if kinds.is_null() || idx as usize >= len {
                return;
            }
            // The main thread may be or'ing this byte while the compile
            // reads it; the compiled check reads it again at run time.
            let seen = unsafe { std::ptr::read_volatile(kinds.add(idx as usize)) };
            if seen != crate::runtime::object::FIELD_NUM {
                return;
            }
            self.field_invariant = Some((dst, kinds, len, idx));
        }

        /// The class's field-kind bytes as a global of the array's size
        /// rather than a bare address, which lets LLVM hoist the checks
        /// out of loops; the engine maps it to the array. Made only when
        /// a guard reads it, so the module never declares one unused.
        fn kinds_global(
            &mut self,
            kinds: *mut u8,
            len: usize,
        ) -> inkwell::values::GlobalValue<'ctx> {
            let name = format!("wren_field_kinds_{:x}", kinds as usize);
            match self.sh.module.get_global(&name) {
                Some(g) => g,
                None => {
                    let ty = self.sh.ctx.i8_type().array_type(len as u32);
                    let g = self.sh.module.add_global(ty, None, &name);
                    g.set_linkage(inkwell::module::Linkage::External);
                    g.set_alignment(8);
                    self.sh.globals.borrow_mut().push((g, kinds as u64));
                    g
                }
            }
        }

        /// The TBAA tag of field-kind bytes: disjoint from field data,
        /// so a check survives stores to fields but not stores of a kind.
        fn kinds_tag(&mut self) -> MetadataValue<'ctx> {
            let ctx = self.sh.ctx;
            let root = *self.field_tbaa_root.get_or_insert_with(|| {
                ctx.metadata_node(&[ctx.metadata_string("wren fields").into()])
            });
            *self.field_tbaa.entry(u16::MAX).or_insert_with(|| {
                let ty = ctx.metadata_node(&[
                    ctx.metadata_string("field kinds").into(),
                    root.into(),
                    ctx.i64_type().const_zero().into(),
                ]);
                ctx.metadata_node(&[ty.into(), ty.into(), ctx.i64_type().const_zero().into()])
            })
        }

        /// Store instance field `idx` of a fields array.
        fn field_store(
            &mut self,
            fields: IntValue<'ctx>,
            idx: u16,
            v: IntValue<'ctx>,
        ) -> Result<(), String> {
            let p = self.addr(fields, idx as i64 * VALUE_SIZE as i64)?;
            let st = self.b.build_store(p, v).map_err(|e| e.to_string())?;
            let tag = self.field_tag(idx);
            let kind = self.sh.ctx.get_kind_id("tbaa");
            st.set_metadata(tag, kind).map_err(|e| e.to_string())?;
            Ok(())
        }

        fn load64(&mut self, base: IntValue<'ctx>, off: i64) -> Result<IntValue<'ctx>, String> {
            let p = self.addr(base, off)?;
            self.b
                .build_load(self.i64t(), p, "ld")
                .map(|v| v.into_int_value())
                .map_err(|e| e.to_string())
        }

        /// A load of an object word that never changes while the object
        /// is alive (its class, its fields pointer): LLVM may keep it
        /// across stores it cannot disambiguate.
        fn load64_stable(
            &mut self,
            base: IntValue<'ctx>,
            off: i64,
        ) -> Result<IntValue<'ctx>, String> {
            let p = self.addr(base, off)?;
            let ld = self
                .b
                .build_load(self.i64t(), p, "ld")
                .map_err(|e| e.to_string())?;
            let ctx = self.sh.ctx;
            let kind = ctx.get_kind_id("invariant.load");
            ld.as_instruction_value()
                .ok_or("load is not an instruction")?
                .set_metadata(ctx.metadata_node(&[]), kind)
                .map_err(|e| e.to_string())?;
            Ok(ld.into_int_value())
        }

        fn load8(&mut self, base: IntValue<'ctx>, off: i64) -> Result<IntValue<'ctx>, String> {
            let p = self.addr(base, off)?;
            let v = self
                .b
                .build_load(self.sh.ctx.i8_type(), p, "ld8")
                .map(|v| v.into_int_value())
                .map_err(|e| e.to_string())?;
            self.b
                .build_int_z_extend(v, self.i64t(), "zx")
                .map_err(|e| e.to_string())
        }

        fn store64(
            &mut self,
            base: IntValue<'ctx>,
            off: i64,
            v: IntValue<'ctx>,
        ) -> Result<(), String> {
            let p = self.addr(base, off)?;
            self.b.build_store(p, v).map_err(|e| e.to_string())?;
            Ok(())
        }

        /// Store zero to the stack slot `p` at function entry.
        fn zero_at_entry(&mut self, p: PointerValue<'ctx>) -> Result<(), String> {
            let entry = self.f.get_first_basic_block().unwrap();
            let cur = self.b.get_insert_block().unwrap();
            match entry.get_terminator() {
                Some(term) => self.b.position_before(&term),
                None => self.b.position_at_end(entry),
            }
            self.b
                .build_store(p, self.c64(0))
                .map_err(|e| e.to_string())?;
            self.b.position_at_end(cur);
            Ok(())
        }

        /// Stack buffer of `n` i64 slots, as an integer address.
        fn stack_buf(&mut self, n: usize) -> Result<(PointerValue<'ctx>, IntValue<'ctx>), String> {
            let entry = self.f.get_first_basic_block().unwrap();
            let cur = self.b.get_insert_block().unwrap();
            match entry.get_first_instruction() {
                Some(first) => self.b.position_before(&first),
                None => self.b.position_at_end(entry),
            }
            let ty = self.i64t().array_type(n.max(1) as u32);
            let p = self.b.build_alloca(ty, "buf").map_err(|e| e.to_string())?;
            self.b.position_at_end(cur);
            let addr = self
                .b
                .build_ptr_to_int(p, self.i64t(), "bufaddr")
                .map_err(|e| e.to_string())?;
            Ok((p, addr))
        }

        // ── Value conversions ──────────────────────────────────────────

        fn get(&self, v: &ValueId) -> Result<BasicValueEnum<'ctx>, String> {
            self.vals
                .get(v)
                .copied()
                .ok_or_else(|| format!("undefined value {:?}", v))
        }
        fn geti(&self, v: &ValueId) -> Result<IntValue<'ctx>, String> {
            match self.get(v)? {
                BasicValueEnum::IntValue(i) => Ok(i),
                BasicValueEnum::FloatValue(f) => self
                    .b
                    .build_bit_cast(f, self.i64t(), "bits")
                    .map(|v| v.into_int_value())
                    .map_err(|e| e.to_string()),
                other => bail!("expected int for {:?}, got {:?}", v, other),
            }
        }
        fn getf(&self, v: &ValueId) -> Result<FloatValue<'ctx>, String> {
            match self.get(v)? {
                BasicValueEnum::FloatValue(f) => Ok(f),
                BasicValueEnum::IntValue(i) if i.get_type().get_bit_width() == 64 => self
                    .b
                    .build_bit_cast(i, self.f64t(), "f")
                    .map(|v| v.into_float_value())
                    .map_err(|e| e.to_string()),
                other => bail!("expected f64 for {:?}, got {:?}", v, other),
            }
        }

        /// A boxed Wren value for `v`, boxing raw booleans and floats.
        fn boxed(&mut self, v: &ValueId) -> Result<IntValue<'ctx>, String> {
            match self.get(v)? {
                BasicValueEnum::IntValue(i) if i.get_type().get_bit_width() == 1 => {
                    self.box_bool(i)
                }
                BasicValueEnum::IntValue(i) => {
                    if self.value_types.get(v.0 as usize) == Some(&MirType::I64)
                        && !self.raw_bools.contains(v)
                    {
                        let f = self
                            .b
                            .build_signed_int_to_float(i, self.f64t(), "i2f")
                            .map_err(|e| e.to_string())?;
                        let b = self.bits(f)?;
                        self.num_values.insert(b);
                        Ok(b)
                    } else {
                        Ok(i)
                    }
                }
                BasicValueEnum::FloatValue(f) => {
                    let b = self.bits(f)?;
                    self.num_values.insert(b);
                    Ok(b)
                }
                other => bail!("cannot box {:?}", other),
            }
        }

        fn bits(&self, f: FloatValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            self.b
                .build_bit_cast(f, self.i64t(), "bits")
                .map(|v| v.into_int_value())
                .map_err(|e| e.to_string())
        }
        fn f64_of(&self, i: IntValue<'ctx>) -> Result<FloatValue<'ctx>, String> {
            self.b
                .build_bit_cast(i, self.f64t(), "f")
                .map(|v| v.into_float_value())
                .map_err(|e| e.to_string())
        }
        fn box_bool(&self, c: IntValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            self.b
                .build_select(c, self.c64(TAG_TRUE), self.c64(TAG_FALSE), "boolbox")
                .map(|v| v.into_int_value())
                .map_err(|e| e.to_string())
        }

        fn icmp(
            &self,
            p: IntPredicate,
            a: IntValue<'ctx>,
            b: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            self.b
                .build_int_compare(p, a, b, "icmp")
                .map_err(|e| e.to_string())
        }
        fn fcmp(
            &self,
            p: FloatPredicate,
            a: FloatValue<'ctx>,
            b: FloatValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            self.b
                .build_float_compare(p, a, b, "fcmp")
                .map_err(|e| e.to_string())
        }
        fn and(&self, a: IntValue<'ctx>, b: IntValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            self.b.build_and(a, b, "and").map_err(|e| e.to_string())
        }

        fn is_nan_boxed(&self, v: IntValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            let m = self.and(v, self.c64(QNAN))?;
            self.icmp(IntPredicate::EQ, m, self.c64(QNAN))
        }

        /// At a loop header: load a word from the safepoint page,
        /// unreadable while a collector waits, so the load faults and
        /// the fault handler parks the thread. Volatile, so it stays.
        fn safepoint_poll(&mut self) -> Result<(), String> {
            let page = crate::codegen::jit_safepoint_page();
            if page == 0 {
                return Ok(());
            }
            let p = self
                .b
                .build_int_to_ptr(self.c64(page as u64), self.ptrt(), "spp")
                .map_err(|e| e.to_string())?;
            let load = self
                .b
                .build_load(self.sh.ctx.i32_type(), p, "sp")
                .map_err(|e| e.to_string())?;
            load.as_instruction_value()
                .ok_or_else(|| "safepoint load".to_string())?
                .set_volatile(true)
                .map_err(|e| e.to_string())?;
            Ok(())
        }

        fn new_block(&mut self, base: &str) -> BasicBlock<'ctx> {
            let n = self.name(base);
            self.sh.ctx.append_basic_block(self.f, &n)
        }
        fn br(&self, bb: BasicBlock<'ctx>) -> Result<(), String> {
            self.b
                .build_unconditional_branch(bb)
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
        fn cbr(
            &self,
            c: IntValue<'ctx>,
            t: BasicBlock<'ctx>,
            f: BasicBlock<'ctx>,
        ) -> Result<(), String> {
            self.b
                .build_conditional_branch(c, t, f)
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
        fn phi(
            &mut self,
            ty: BasicTypeEnum<'ctx>,
            incoming: &[(BasicValueEnum<'ctx>, BasicBlock<'ctx>)],
        ) -> Result<BasicValueEnum<'ctx>, String> {
            let phi = self.b.build_phi(ty, "phi").map_err(|e| e.to_string())?;
            for (v, bb) in incoming {
                phi.add_incoming(&[(v as &dyn BasicValue, *bb)]);
            }
            Ok(phi.as_basic_value())
        }

        /// Guarded receiver-class load: `(obj_ptr, class)` on the returned
        /// block, or a branch to `not_object` for a non-object value.
        fn class_load_guarded(
            &mut self,
            recv: IntValue<'ctx>,
            not_object: BasicBlock<'ctx>,
        ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), String> {
            let high = self.and(recv, self.c64(TAG_OBJ))?;
            let is_obj = self.icmp(IntPredicate::EQ, high, self.c64(TAG_OBJ))?;
            let obj = self.new_block("obj");
            self.cbr(is_obj, obj, not_object)?;
            self.b.position_at_end(obj);
            let ptr = self.and(recv, self.c64(PTR_MASK))?;
            let class = self.load64_stable(ptr, HEADER_CLASS as i64)?;
            Ok((ptr, class))
        }

        /// `(is_object, ptr, class)` of `r` without a branch: a
        /// non-object reads the null object's zero class instead, so a
        /// class check is a pure function of `r` and repeated checks on
        /// one receiver fold into one.
        fn class_of(
            &mut self,
            r: IntValue<'ctx>,
        ) -> Result<(IntValue<'ctx>, IntValue<'ctx>, IntValue<'ctx>), String> {
            let high = self.and(r, self.c64(TAG_OBJ))?;
            let is_obj = self.icmp(IntPredicate::EQ, high, self.c64(TAG_OBJ))?;
            let masked = self.and(r, self.c64(PTR_MASK))?;
            let null_obj = self.c64(crate::codegen::runtime_fns::JIT_NULL_OBJECT.as_ptr() as u64);
            let ptr = self
                .b
                .build_select(is_obj, masked, null_obj, "optr")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let class = self.load64_stable(ptr, HEADER_CLASS as i64)?;
            Ok((is_obj, ptr, class))
        }

        /// `(hit, fields)` for a receiver expected to be an instance of
        /// `class`; `fields` reads the null object when it is not.
        fn instance_check(
            &mut self,
            r: IntValue<'ctx>,
            class: u64,
        ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), String> {
            let (is_obj, ptr, recv_class) = self.class_of(r)?;
            let same = self.icmp(IntPredicate::EQ, recv_class, self.c64(class))?;
            let hit = self
                .b
                .build_and(is_obj, same, "hit")
                .map_err(|e| e.to_string())?;
            let null_obj = self.c64(crate::codegen::runtime_fns::JIT_NULL_OBJECT.as_ptr() as u64);
            let safe = self
                .b
                .build_select(hit, ptr, null_obj, "iptr")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let fields = self
                .b
                .build_int_add(safe, self.c64(INSTANCE_SIZE as u64), "fields")
                .map_err(|e| e.to_string())?;
            Ok((hit, fields))
        }

        // ── Driver ─────────────────────────────────────────────────────

        fn run(mut self) -> Result<(), String> {
            let mir = self.sh.mir;
            let ctx = self.sh.ctx;
            let prologue = ctx.append_basic_block(self.f, "prologue");
            for i in 0..mir.blocks.len() {
                self.blocks
                    .push(ctx.append_basic_block(self.f, &format!("bb{i}")));
            }
            self.b.position_at_end(prologue);

            let slot_type = |vid: ValueId, this: &Self| -> BasicTypeEnum<'ctx> {
                match this.value_types.get(vid.0 as usize) {
                    Some(MirType::F64) => this.f64t().into(),
                    _ => this.i64t().into(),
                }
            };
            // Parameter slots.
            for block in &mir.blocks {
                for (p, _) in &block.params {
                    let lt = slot_type(*p, &self);
                    let slot = self
                        .b
                        .build_alloca(lt, &format!("v{}", p.0))
                        .map_err(|e| e.to_string())?;
                    self.slots.insert(*p, (slot, lt));
                }
            }
            // Values an OSR entry defines that the body also defines
            // (live-ins and the constants an entry rematerialises) get a
            // slot too; every block reads them from it.
            for layout in self.entries {
                let mut vars: Vec<ValueId> = layout.external_args.clone();
                vars.extend(osr_rematerializable_defs(mir, layout.target_block).into_keys());
                for vid in vars {
                    // A block parameter already has a slot, but a block
                    // the entry switch reaches directly is no longer
                    // dominated by the parameter's block, so every block
                    // reads it from the slot.
                    if !self.slots.contains_key(&vid) {
                        let lt = slot_type(vid, &self);
                        let slot = self
                            .b
                            .build_alloca(lt, &format!("o{}", vid.0))
                            .map_err(|e| e.to_string())?;
                        self.slots.insert(vid, (slot, lt));
                    }
                    self.osr_vars.insert(vid);
                }
            }

            if self.entries.is_empty() {
                let entry = &mir.blocks[0];
                let params: Vec<IntValue> = (0..mir.arity as usize)
                    .map(|i| self.f.get_nth_param(i as u32).unwrap().into_int_value())
                    .collect();
                self.receiver = params.first().copied();
                for &(vid, ref inst) in &entry.instructions {
                    if let Instruction::BlockParam(idx) = inst
                        && let Some(p) = params.get(*idx as usize)
                    {
                        self.vals.insert(vid, (*p).into());
                    }
                }
                self.br(self.blocks[0])?;
            } else {
                // Entry switch: 0 is the main entry, k the k-th header.
                let which = self.f.get_nth_param(0).unwrap().into_int_value();
                let args_ptr = self.f.get_nth_param(1).unwrap().into_pointer_value();
                let args_addr = self
                    .b
                    .build_ptr_to_int(args_ptr, self.i64t(), "args")
                    .map_err(|e| e.to_string())?;
                let main_entry = self.new_block("main");
                let mut cases: Vec<(IntValue<'ctx>, BasicBlock<'ctx>)> = Vec::new();
                let mut entry_blocks = Vec::new();
                for k in 0..self.entries.len() {
                    let bb = self.new_block("osr");
                    cases.push((self.c64((k + 1) as u64), bb));
                    entry_blocks.push(bb);
                }
                self.b
                    .build_switch(which, main_entry, &cases)
                    .map_err(|e| e.to_string())?;

                // Main entry: parameters arrive through `args`. They are
                // read into slots so the loop headers see one definition
                // whichever entry was taken.
                self.b.position_at_end(main_entry);
                let entry = &mir.blocks[0];
                let mut params: Vec<IntValue<'ctx>> = Vec::new();
                for i in 0..mir.arity as usize {
                    params.push(self.load64(args_addr, (i as i64) * VALUE_SIZE as i64)?);
                }
                self.receiver = params.first().copied();
                for &(vid, ref inst) in &entry.instructions {
                    if let Instruction::BlockParam(idx) = inst
                        && let Some(p) = params.get(*idx as usize)
                        && let Some((slot, _)) = self.slots.get(&vid).copied()
                    {
                        self.b.build_store(slot, *p).map_err(|e| e.to_string())?;
                    }
                }
                self.param_regs = params;
                self.br(self.blocks[0])?;

                for (k, layout) in self.entries.iter().enumerate() {
                    self.b.position_at_end(entry_blocks[k]);
                    let mut slot = 0i64;
                    for vid in &layout.external_args {
                        let raw = self.load64(args_addr, slot * VALUE_SIZE as i64)?;
                        slot += 1;
                        let v = self.osr_incoming(*vid, raw)?;
                        let (p, _) = self.slots[vid];
                        self.b.build_store(p, v).map_err(|e| e.to_string())?;
                    }
                    let target = &mir.blocks[layout.target_block.0 as usize];
                    for (p, _) in &target.params {
                        let raw = self.load64(args_addr, slot * VALUE_SIZE as i64)?;
                        slot += 1;
                        let v = self.osr_incoming(*p, raw)?;
                        let (slotp, _) = self.slots[p];
                        self.b.build_store(slotp, v).map_err(|e| e.to_string())?;
                    }
                    for (vid, inst) in osr_rematerializable_defs(mir, layout.target_block) {
                        let v: BasicValueEnum = match inst {
                            Instruction::ConstNum(n) => self.c64(n.to_bits()).into(),
                            Instruction::ConstBool(b) => {
                                self.c64(if b { TAG_TRUE } else { TAG_FALSE }).into()
                            }
                            Instruction::ConstNull => self.c64(TAG_NULL).into(),
                            Instruction::ConstF64(n) => self.cf64(n).into(),
                            Instruction::ConstI64(n) => self.c64(n as u64).into(),
                            _ => bail!("non-rematerializable OSR external value"),
                        };
                        let (p, _) = self.slots[&vid];
                        self.b.build_store(p, v).map_err(|e| e.to_string())?;
                    }
                    self.br(self.blocks[layout.target_block.0 as usize])?;
                }
            }

            let rpo = crate::codegen::cranelift_backend::cl::compute_rpo(mir);
            let reachable: HashSet<usize> = osr_reachable_blocks(mir, BlockId(0));
            self.move_roots = move_roots(mir);
            self.int_sources = int_sources(mir);
            self.class_facts = class_facts(mir, &self.move_roots);
            let loop_headers: HashSet<usize> = mir
                .blocks
                .iter()
                .enumerate()
                .filter(|(i, b)| b.predecessors.iter().any(|p| p.0 as usize >= *i))
                .map(|(i, _)| i)
                .collect();
            for &bi in &rpo {
                let bb = self.blocks[bi];
                self.b.position_at_end(bb);
                if !reachable.contains(&bi) {
                    self.b.build_unreachable().map_err(|e| e.to_string())?;
                    continue;
                }
                self.cur_block = bi;
                if loop_headers.contains(&bi) {
                    self.safepoint_poll()?;
                }
                self.lower_block(bi)?;
            }
            for bb in self.blocks.iter() {
                if bb.get_terminator().is_none() {
                    self.b.position_at_end(*bb);
                    self.b.build_unreachable().map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }

        /// A live-in read from the OSR argument array, in the type the
        /// body carries it: boxed, raw f64, or integer.
        fn osr_incoming(
            &mut self,
            vid: ValueId,
            raw: IntValue<'ctx>,
        ) -> Result<BasicValueEnum<'ctx>, String> {
            Ok(match self.value_types.get(vid.0 as usize) {
                Some(MirType::F64) => self.f64_of(raw)?.into(),
                Some(MirType::I64) => {
                    let f = self.f64_of(raw)?;
                    self.b
                        .build_float_to_signed_int(f, self.i64t(), "f2i")
                        .map_err(|e| e.to_string())?
                        .into()
                }
                _ => raw.into(),
            })
        }

        fn lower_block(&mut self, bi: usize) -> Result<(), String> {
            let mir = self.sh.mir;
            let block = &mir.blocks[bi];
            for (p, _) in &block.params {
                let (slot, ty) = *self
                    .slots
                    .get(p)
                    .ok_or_else(|| format!("block parameter {:?} has no slot", p))?;
                let v = self
                    .b
                    .build_load(ty, slot, &format!("v{}", p.0))
                    .map_err(|e| e.to_string())?;
                self.vals.insert(*p, v);
            }
            let osr_vars: Vec<ValueId> = self.osr_vars.iter().copied().collect();
            for vid in osr_vars {
                let (slot, ty) = *self
                    .slots
                    .get(&vid)
                    .ok_or_else(|| format!("OSR live-in {:?} has no slot", vid))?;
                let v = self
                    .b
                    .build_load(ty, slot, &format!("o{}", vid.0))
                    .map_err(|e| e.to_string())?;
                self.vals.insert(vid, v);
            }
            for (i, &(vid, ref inst)) in block.instructions.iter().enumerate() {
                // A call whose result is guarded next may leave the
                // function on a class miss instead of calling.
                self.miss_exit = match block.instructions.get(i + 1) {
                    Some((
                        _,
                        Instruction::GuardNumAt {
                            value,
                            live,
                            call_pc,
                            call_live,
                            ..
                        },
                    )) if *value == vid
                        && matches!(
                            inst,
                            Instruction::Call { .. } | Instruction::CallKnownFunc { .. }
                        ) =>
                    {
                        let regs: Vec<DeoptReg> = live
                            .iter()
                            .filter(|r| r.reg != vid.0)
                            .chain(call_live.iter())
                            .cloned()
                            .collect();
                        Some((*call_pc, regs))
                    }
                    Some((_, Instruction::SlowPathExit { pc, live })) => Some((*pc, live.clone())),
                    _ => None,
                };
                self.cur_vid = vid;
                let v = self.lower_instruction(vid, inst)?;
                self.miss_exit = None;
                if !matches!(
                    block.instructions.get(i + 1),
                    Some((_, Instruction::GuardNumAt { value, .. })) if *value == vid
                ) && !matches!(
                    block.instructions.get(i + 1),
                    Some((_, Instruction::Move(s))) if *s == vid
                ) {
                    self.field_invariant = None;
                }
                if let Some(v) = v {
                    self.vals.insert(vid, v);
                    if is_raw_bool(inst) {
                        self.raw_bools.insert(vid);
                    }
                    if self.osr_vars.contains(&vid) {
                        let (slot, _) = self.slots[&vid];
                        self.b.build_store(slot, v).map_err(|e| e.to_string())?;
                    }
                }
            }
            self.lower_terminator(&block.terminator)
        }

        /// Store the edge's arguments into the target's parameter slots.
        fn pass_args(&mut self, target: BlockId, args: &[ValueId]) -> Result<(), String> {
            let params = &self.sh.mir.blocks[target.0 as usize].params;
            let mut stores: Vec<(PointerValue<'ctx>, BasicValueEnum<'ctx>)> = Vec::new();
            for (i, a) in args.iter().enumerate() {
                let Some((p, ty)) = params.get(i) else {
                    continue;
                };
                let (slot, _) = self.slots[p];
                let v: BasicValueEnum = match ty {
                    MirType::F64 => self.getf(a)?.into(),
                    MirType::I64 => match self.get(a)? {
                        BasicValueEnum::IntValue(i) if i.get_type().get_bit_width() == 64 => {
                            i.into()
                        }
                        BasicValueEnum::FloatValue(f) => self
                            .b
                            .build_float_to_signed_int(f, self.i64t(), "f2i")
                            .map_err(|e| e.to_string())?
                            .into(),
                        other => bail!("bad i64 edge arg {:?}", other),
                    },
                    _ => self.boxed(a)?.into(),
                };
                stores.push((slot, v));
            }
            for (slot, v) in stores {
                self.b.build_store(slot, v).map_err(|e| e.to_string())?;
            }
            Ok(())
        }

        fn truthy(&mut self, c: &ValueId) -> Result<IntValue<'ctx>, String> {
            match self.get(c)? {
                BasicValueEnum::IntValue(i) if i.get_type().get_bit_width() == 1 => Ok(i),
                _ => {
                    let v = self.boxed(c)?;
                    let nf = self.icmp(IntPredicate::NE, v, self.c64(TAG_FALSE))?;
                    let nn = self.icmp(IntPredicate::NE, v, self.c64(TAG_NULL))?;
                    self.and(nf, nn)
                }
            }
        }

        fn lower_terminator(&mut self, term: &Terminator) -> Result<(), String> {
            match term {
                Terminator::Return(v) => {
                    let r = self.boxed(v)?;
                    self.b.build_return(Some(&r)).map_err(|e| e.to_string())?;
                }
                Terminator::ReturnNull => {
                    let r = self.c64(TAG_NULL);
                    self.b.build_return(Some(&r)).map_err(|e| e.to_string())?;
                }
                Terminator::Branch { target, args } => {
                    self.pass_args(*target, args)?;
                    self.br(self.blocks[target.0 as usize])?;
                }
                Terminator::CondBranch {
                    condition,
                    true_target,
                    true_args,
                    false_target,
                    false_args,
                } => {
                    let c = self.truthy(condition)?;
                    if true_args.is_empty() && false_args.is_empty() {
                        self.cbr(
                            c,
                            self.blocks[true_target.0 as usize],
                            self.blocks[false_target.0 as usize],
                        )?;
                    } else {
                        // Each edge stores its own arguments on a
                        // trampoline block.
                        let tb = self.new_block("t");
                        let fb = self.new_block("f");
                        self.cbr(c, tb, fb)?;
                        self.b.position_at_end(tb);
                        self.pass_args(*true_target, true_args)?;
                        self.br(self.blocks[true_target.0 as usize])?;
                        self.b.position_at_end(fb);
                        self.pass_args(*false_target, false_args)?;
                        self.br(self.blocks[false_target.0 as usize])?;
                    }
                }
                Terminator::Unreachable => {
                    self.b.build_unreachable().map_err(|e| e.to_string())?;
                }
            }
            Ok(())
        }

        // ── Instructions ───────────────────────────────────────────────

        fn lower_instruction(
            &mut self,
            vid: ValueId,
            inst: &Instruction,
        ) -> Result<Option<BasicValueEnum<'ctx>>, String> {
            use Instruction as I;
            let v: BasicValueEnum<'ctx> = match inst {
                I::ConstNum(n) => {
                    let c = self.c64(n.to_bits());
                    self.num_values.insert(c);
                    c.into()
                }
                I::ConstBool(b) => self.c64(if *b { TAG_TRUE } else { TAG_FALSE }).into(),
                I::ConstNull => self.c64(TAG_NULL).into(),
                I::ConstF64(n) => self.cf64(*n).into(),
                I::ConstI64(n) => self.c64(*n as u64).into(),
                I::BlockParam(idx) => {
                    if self.entries.is_empty() || self.inline_depth > 0 {
                        return Ok(None);
                    }
                    match self.param_regs.get(*idx as usize) {
                        Some(p) => (*p).into(),
                        None => return Ok(None),
                    }
                }
                I::Move(s) => {
                    // A copy carries a pending field invariant of its
                    // source.
                    if let Some((guarded, kinds, len, idx)) = self.field_invariant
                        && guarded == *s
                    {
                        self.field_invariant = Some((vid, kinds, len, idx));
                    }
                    self.get(s)?
                }

                I::Add(a, b) => self.boxed_binop(a, b, BinOp::Add, "wren_num_add")?.into(),
                I::Sub(a, b) => self.boxed_binop(a, b, BinOp::Sub, "wren_num_sub")?.into(),
                I::Mul(a, b) => self.boxed_binop(a, b, BinOp::Mul, "wren_num_mul")?.into(),
                I::Div(a, b) => self.boxed_binop(a, b, BinOp::Div, "wren_num_div")?.into(),
                I::Mod(a, b) => self.boxed_binop(a, b, BinOp::Rem, "wren_num_mod")?.into(),
                I::CmpLt(a, b) => self
                    .boxed_binop(a, b, BinOp::Cmp(FloatPredicate::OLT), "wren_cmp_lt")?
                    .into(),
                I::CmpGt(a, b) => self
                    .boxed_binop(a, b, BinOp::Cmp(FloatPredicate::OGT), "wren_cmp_gt")?
                    .into(),
                I::CmpLe(a, b) => self
                    .boxed_binop(a, b, BinOp::Cmp(FloatPredicate::OLE), "wren_cmp_le")?
                    .into(),
                I::CmpGe(a, b) => self
                    .boxed_binop(a, b, BinOp::Cmp(FloatPredicate::OGE), "wren_cmp_ge")?
                    .into(),
                I::CmpEq(a, b) => self
                    .boxed_binop(a, b, BinOp::Cmp(FloatPredicate::OEQ), "wren_cmp_eq")?
                    .into(),
                I::CmpNe(a, b) => self
                    .boxed_binop(a, b, BinOp::Cmp(FloatPredicate::UNE), "wren_cmp_ne")?
                    .into(),
                I::Neg(a) => {
                    let la = self.boxed(a)?;
                    let is_box = self.is_nan_boxed(la)?;
                    let fast = self.new_block("negf");
                    let slow = self.new_block("negs");
                    let merge = self.new_block("negm");
                    self.cbr(is_box, slow, fast)?;
                    self.b.position_at_end(fast);
                    let fa = self.f64_of(la)?;
                    let n = self
                        .b
                        .build_float_neg(fa, "fneg")
                        .map_err(|e| e.to_string())?;
                    let nb = self.bits(n)?;
                    self.br(merge)?;
                    self.b.position_at_end(slow);
                    let s = self.call_helper("wren_num_neg", &[la])?;
                    let slow_end = self.b.get_insert_block().unwrap();
                    self.br(merge)?;
                    self.b.position_at_end(merge);
                    self.phi(
                        self.i64t().into(),
                        &[(nb.into(), fast), (s.into(), slow_end)],
                    )?
                }
                I::Not(a) => {
                    let v = self.boxed(a)?;
                    let f = self.icmp(IntPredicate::EQ, v, self.c64(TAG_FALSE))?;
                    let n = self.icmp(IntPredicate::EQ, v, self.c64(TAG_NULL))?;
                    let falsy = self.b.build_or(f, n, "falsy").map_err(|e| e.to_string())?;
                    self.box_bool(falsy)?.into()
                }

                I::GetField(recv, idx) => {
                    let r = self.boxed(recv)?;
                    let obj = self.and(r, self.c64(PTR_MASK))?;
                    let fields = self.instance_fields(obj)?;
                    if let Some(class) = self.known_class(*recv) {
                        self.note_field_invariant(class, *idx, vid);
                    }
                    self.field_load(fields, *idx)?.into()
                }
                I::SetField(recv, idx, val) => {
                    let r = self.boxed(recv)?;
                    let v = self.boxed(val)?;
                    let obj = self.and(r, self.c64(PTR_MASK))?;
                    let fields = self.instance_fields(obj)?;
                    self.field_store(fields, *idx, v)?;
                    let known = self
                        .fresh
                        .get(&r)
                        .copied()
                        .or_else(|| match self.inline_class {
                            Some((recv, class)) if recv == r => Some(class),
                            _ => None,
                        })
                        .or_else(|| self.known_class(*recv));
                    if let Some(class) = known {
                        self.note_field_kind_static(class, *idx, v)?;
                    } else if !self.num_values.contains(&v) {
                        self.note_field_kind(obj, *idx, v)?;
                    }
                    v.into()
                }
                I::GetModuleVar(idx) => {
                    let cell = jit_modvars_cell();
                    if jit_modvar_in_range(*idx) {
                        let base = self.load64(self.c64(cell as u64), 0)?;
                        self.load64(base, (*idx as i64) * 8)?.into()
                    } else if cell != 0 {
                        let cellv = self.c64(cell as u64);
                        let base = self.load64(cellv, 0)?;
                        let len = self.load64(cellv, 8)?;
                        let in_range = self.icmp(IntPredicate::ULT, self.c64(*idx as u64), len)?;
                        let hit = self.new_block("mvh");
                        let miss = self.new_block("mvm");
                        let merge = self.new_block("mvj");
                        self.cbr(in_range, hit, miss)?;
                        self.b.position_at_end(hit);
                        let v = self.load64(base, (*idx as i64) * 8)?;
                        self.br(merge)?;
                        self.b.position_at_end(miss);
                        let null = self.c64(TAG_NULL);
                        self.br(merge)?;
                        self.b.position_at_end(merge);
                        self.phi(self.i64t().into(), &[(v.into(), hit), (null.into(), miss)])?
                    } else {
                        self.call_helper("wren_get_module_var", &[self.c64(*idx as u64)])?
                            .into()
                    }
                }
                I::SetModuleVar(idx, val) => {
                    let v = self.boxed(val)?;
                    let cell = jit_modvars_cell();
                    if jit_modvar_in_range(*idx) {
                        let base = self.load64(self.c64(cell as u64), 0)?;
                        self.store64(base, (*idx as i64) * 8, v)?;
                    } else if cell != 0 {
                        let cellv = self.c64(cell as u64);
                        let base = self.load64(cellv, 0)?;
                        let len = self.load64(cellv, 8)?;
                        let in_range = self.icmp(IntPredicate::ULT, self.c64(*idx as u64), len)?;
                        let hit = self.new_block("svh");
                        let miss = self.new_block("svm");
                        let merge = self.new_block("svj");
                        self.cbr(in_range, hit, miss)?;
                        self.b.position_at_end(hit);
                        self.store64(base, (*idx as i64) * 8, v)?;
                        self.br(merge)?;
                        self.b.position_at_end(miss);
                        self.call_helper("wren_set_module_var", &[self.c64(*idx as u64), v])?;
                        self.br(merge)?;
                        self.b.position_at_end(merge);
                    } else {
                        self.call_helper("wren_set_module_var", &[self.c64(*idx as u64), v])?;
                    }
                    v.into()
                }

                I::Call {
                    receiver,
                    method,
                    args,
                    ..
                } => self.lower_call(receiver, *method, args)?.into(),
                I::CallKnownFunc {
                    func_id,
                    method,
                    expected_class,
                    inline_getter_field,
                    direct,
                    receiver,
                    args,
                } => self
                    .lower_known_call(
                        *func_id,
                        *method,
                        *expected_class,
                        *inline_getter_field,
                        *direct,
                        receiver,
                        args,
                    )?
                    .into(),
                I::SuperCall { method, args } => {
                    if args.len() > 4 {
                        bail!("SuperCall with arity {} not supported by JIT", args.len());
                    }
                    let name = [
                        "wren_super_call_0",
                        "wren_super_call_1",
                        "wren_super_call_2",
                        "wren_super_call_3",
                        "wren_super_call_4",
                    ][args.len()];
                    let mut call_args = vec![self.c64(method.index() as u64)];
                    for a in args {
                        call_args.push(self.boxed(a)?);
                    }
                    self.call_helper(name, &call_args)?.into()
                }
                I::CallStaticSelf { args } => {
                    if self.inline_depth > 0 {
                        bail!("CallStaticSelf inside an inlined body");
                    }
                    let mut call_args: Vec<BasicMetadataValueEnum> = Vec::new();
                    if let Some(r) = self.receiver {
                        call_args.push(r.into());
                    }
                    for a in args {
                        call_args.push(self.boxed(a)?.into());
                    }
                    let call = self
                        .b
                        .build_call(self.sh.main_fn, &call_args, "self")
                        .map_err(|e| e.to_string())?;
                    call.try_as_basic_value().basic().unwrap()
                }

                I::MakeList(elems) => {
                    let mut a = Vec::with_capacity(elems.len());
                    for e in elems {
                        a.push(self.boxed(e)?);
                    }
                    self.alloc_list(&a)?.into()
                }
                I::MakeMap(pairs) => {
                    let map = self.call_helper("wren_make_map", &[])?;
                    for (k, v) in pairs {
                        let k = self.boxed(k)?;
                        let v = self.boxed(v)?;
                        self.call_helper("wren_map_set", &[map, k, v])?;
                    }
                    map.into()
                }
                I::MakeRange(from, to, inclusive) => {
                    let f = self.boxed(from)?;
                    let t = self.boxed(to)?;
                    let i = self.c64(*inclusive as u64);
                    self.call_helper("wren_make_range", &[f, t, i])?.into()
                }
                I::StringConcat(parts) => {
                    if parts.is_empty() {
                        self.c64(TAG_NULL).into()
                    } else {
                        let mut acc = self.boxed(&parts[0])?;
                        for p in &parts[1..] {
                            let v = self.boxed(p)?;
                            acc = self.call_helper("wren_string_concat", &[acc, v])?;
                        }
                        acc.into()
                    }
                }
                I::ToString(a) => {
                    let v = self.boxed(a)?;
                    self.call_helper("wren_to_string", &[v])?.into()
                }
                I::GetUpvalue(idx) => self
                    .call_helper("wren_get_upvalue", &[self.c64(*idx as u64)])?
                    .into(),
                I::SetUpvalue(idx, val) => {
                    let v = self.boxed(val)?;
                    self.call_helper("wren_set_upvalue", &[self.c64(*idx as u64), v])?
                        .into()
                }
                I::GetStaticField(sym) => self
                    .call_helper("wren_get_static_field", &[self.c64(sym.index() as u64)])?
                    .into(),
                I::SetStaticField(sym, val) => {
                    let v = self.boxed(val)?;
                    self.call_helper("wren_set_static_field", &[self.c64(sym.index() as u64), v])?
                        .into()
                }
                I::MakeClosure { fn_id, upvalues } => {
                    let n = upvalues.len();
                    let fid = self.c64(*fn_id as u64);
                    if n <= 8 {
                        let name = format!("wren_make_closure_{n}");
                        let mut a = vec![fid];
                        for uv in upvalues {
                            a.push(self.boxed(uv)?);
                        }
                        self.call_helper(&name, &a)?.into()
                    } else {
                        let (_, buf) = self.stack_buf(n)?;
                        for (i, uv) in upvalues.iter().enumerate() {
                            let v = self.boxed(uv)?;
                            self.store64(buf, (i * 8) as i64, v)?;
                        }
                        self.call_helper("wren_make_closure_n", &[fid, self.c64(n as u64), buf])?
                            .into()
                    }
                }
                I::SubscriptGet { receiver, args } if args.len() == 1 => {
                    let r = self.boxed(receiver)?;
                    let idx = self.boxed(&args[0])?;
                    let int = self.int_source(&args[0])?;
                    self.typed_array_get(r, idx, int)?.into()
                }
                I::SubscriptGet { receiver, args } => {
                    let mut a = vec![self.boxed(receiver)?];
                    for x in args {
                        a.push(self.boxed(x)?);
                    }
                    self.call_helper("wren_subscript_get", &a)?.into()
                }
                I::SubscriptSet {
                    receiver,
                    args,
                    value,
                } if args.len() == 1 => {
                    let r = self.boxed(receiver)?;
                    let idx = self.boxed(&args[0])?;
                    let v = self.boxed(value)?;
                    let int = self.int_source(&args[0])?;
                    self.typed_array_set(r, idx, int, v)?.into()
                }
                I::SubscriptSet {
                    receiver,
                    args,
                    value,
                } => {
                    let mut a = vec![self.boxed(receiver)?];
                    for x in args {
                        a.push(self.boxed(x)?);
                    }
                    a.push(self.boxed(value)?);
                    self.call_helper("wren_subscript_set", &a)?.into()
                }
                I::BitAnd(a, b) => self.helper2("wren_bit_and", a, b)?.into(),
                I::BitOr(a, b) => self.helper2("wren_bit_or", a, b)?.into(),
                I::BitXor(a, b) => self.helper2("wren_bit_xor", a, b)?.into(),
                I::Shl(a, b) => self.helper2("wren_bit_shl", a, b)?.into(),
                I::Shr(a, b) => self.helper2("wren_bit_shr", a, b)?.into(),
                I::BitNot(a) => {
                    let v = self.boxed(a)?;
                    self.call_helper("wren_bit_not", &[v])?.into()
                }
                I::IsType(a, class_sym) => {
                    let v = self.boxed(a)?;
                    self.call_helper("wren_is_type", &[v, self.c64(class_sym.index() as u64)])?
                        .into()
                }
                I::ConstString(idx) => self
                    .call_helper("wren_const_string", &[self.c64(*idx as u64)])?
                    .into(),

                I::ClassIs(a, class_ptr) => {
                    let v = self.boxed(a)?;
                    let cur = self.b.get_insert_block().unwrap();
                    let merge = self.new_block("cim");
                    let (_, class) = self.class_load_guarded(v, merge)?;
                    let hit = self.icmp(IntPredicate::EQ, class, self.c64(*class_ptr as u64))?;
                    let obj_end = self.b.get_insert_block().unwrap();
                    self.br(merge)?;
                    self.b.position_at_end(merge);
                    let no = self.i1t().const_zero();
                    self.phi(
                        self.i1t().into(),
                        &[(no.into(), cur), (hit.into(), obj_end)],
                    )?
                }
                I::ObjectIs(a, obj_ptr) => {
                    let v = self.boxed(a)?;
                    let expected = self.c64(TAG_OBJ | (*obj_ptr as u64 & PTR_MASK));
                    self.icmp(IntPredicate::EQ, v, expected)?.into()
                }
                I::GuardClassAt {
                    value,
                    class,
                    pc,
                    live,
                } => {
                    let v = self.boxed(value)?;
                    // Branchless: the class load runs on every path, so
                    // LLVM hoists it out of a loop the value is invariant
                    // in and the exit with it. A non-object reads the
                    // null object's zero class and misses.
                    let (_, _, recv_class) = self.class_of(v)?;
                    let hit = self.icmp(IntPredicate::EQ, recv_class, self.c64(*class as u64))?;
                    let fails = self.b.build_not(hit, "fails").map_err(|e| e.to_string())?;
                    self.guard_deopt_at(fails, *pc, live)?;
                    // From here on in this block the value has the class.
                    let root = self.root(*value);
                    self.class_facts
                        .entry(self.cur_block)
                        .or_default()
                        .push((root, *class));
                    v.into()
                }
                I::NewInstance { class, assigned } => {
                    let class_val = self.c64(TAG_OBJ | (*class as u64 & PTR_MASK));
                    let nf = unsafe {
                        (*(*class as *const crate::runtime::object::ObjClass)).num_fields
                    };
                    let inst = self.alloc_instance(class_val, Some((nf, *assigned)))?;
                    // The slow path allocates the same layout: fields
                    // follow the header whenever a bump region exists.
                    if crate::codegen::jit_bump_region() != 0 {
                        self.fresh.insert(inst, *class);
                    }
                    inst.into()
                }
                I::ClosureFnIs(a, fn_ptr) => {
                    let v = self.boxed(a)?;
                    let high = self.and(v, self.c64(TAG_OBJ))?;
                    let is_obj = self.icmp(IntPredicate::EQ, high, self.c64(TAG_OBJ))?;
                    let cur = self.b.get_insert_block().unwrap();
                    let obj = self.new_block("cfo");
                    let clo = self.new_block("cfc");
                    let merge = self.new_block("cfm");
                    self.cbr(is_obj, obj, merge)?;
                    self.b.position_at_end(obj);
                    let ptr = self.and(v, self.c64(PTR_MASK))?;
                    let ty = self.load8(ptr, HEADER_OBJ_TYPE as i64)?;
                    let is_clo = self.icmp(
                        IntPredicate::EQ,
                        ty,
                        self.c64(crate::runtime::object::ObjType::Closure as u64),
                    )?;
                    self.cbr(is_clo, clo, merge)?;
                    self.b.position_at_end(clo);
                    let function = self.load64(ptr, CLOSURE_FUNCTION as i64)?;
                    let hit = self.icmp(IntPredicate::EQ, function, self.c64(*fn_ptr as u64))?;
                    self.br(merge)?;
                    self.b.position_at_end(merge);
                    let no = self.i1t().const_zero();
                    self.phi(
                        self.i1t().into(),
                        &[(no.into(), cur), (no.into(), obj), (hit.into(), clo)],
                    )?
                }

                I::AddI64(a, b) => self
                    .b
                    .build_int_add(self.geti(a)?, self.geti(b)?, "add")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::SubI64(a, b) => self
                    .b
                    .build_int_sub(self.geti(a)?, self.geti(b)?, "sub")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::MulI64(a, b) => self
                    .b
                    .build_int_mul(self.geti(a)?, self.geti(b)?, "mul")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::RemI64(a, b) => self
                    .b
                    .build_int_signed_rem(self.geti(a)?, self.geti(b)?, "rem")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::BandI64(a, b) => self.and(self.geti(a)?, self.geti(b)?)?.into(),
                I::NegI64(a) => self
                    .b
                    .build_int_neg(self.geti(a)?, "neg")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::CmpLtI64(a, b) => self
                    .icmp(IntPredicate::SLT, self.geti(a)?, self.geti(b)?)?
                    .into(),
                I::CmpGtI64(a, b) => self
                    .icmp(IntPredicate::SGT, self.geti(a)?, self.geti(b)?)?
                    .into(),
                I::CmpLeI64(a, b) => self
                    .icmp(IntPredicate::SLE, self.geti(a)?, self.geti(b)?)?
                    .into(),
                I::CmpGeI64(a, b) => self
                    .icmp(IntPredicate::SGE, self.geti(a)?, self.geti(b)?)?
                    .into(),
                I::I64ToF64(a) => self
                    .b
                    .build_signed_int_to_float(self.geti(a)?, self.f64t(), "i2f")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::F64ToI64(a) => {
                    let f = self.getf(a)?;
                    self.b
                        .build_float_to_signed_int(f, self.i64t(), "f2i")
                        .map_err(|e| e.to_string())?
                        .into()
                }
                I::ListCount(recv) => {
                    let r = self.boxed(recv)?;
                    let obj = self.and(r, self.c64(PTR_MASK))?;
                    let p = self.addr(obj, LIST_COUNT as i64)?;
                    let count32 = self
                        .b
                        .build_load(self.sh.ctx.i32_type(), p, "count")
                        .map_err(|e| e.to_string())?
                        .into_int_value();
                    let f = self
                        .b
                        .build_unsigned_int_to_float(count32, self.f64t(), "countf")
                        .map_err(|e| e.to_string())?;
                    self.bits(f)?.into()
                }
                I::IsNum(a) => {
                    let v = self.boxed(a)?;
                    let boxed = self.is_nan_boxed(v)?;
                    self.b
                        .build_not(boxed, "isnum")
                        .map_err(|e| e.to_string())?
                        .into()
                }

                I::AddF64(a, b) => self
                    .b
                    .build_float_add(self.getf(a)?, self.getf(b)?, "fadd")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::SubF64(a, b) => self
                    .b
                    .build_float_sub(self.getf(a)?, self.getf(b)?, "fsub")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::MulF64(a, b) => self
                    .b
                    .build_float_mul(self.getf(a)?, self.getf(b)?, "fmul")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::DivF64(a, b) => self
                    .b
                    .build_float_div(self.getf(a)?, self.getf(b)?, "fdiv")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::ModF64(a, b) => {
                    let av = self.getf(a)?;
                    let bv = self.getf(b)?;
                    if let Some(c) =
                        const_f64_of(self.sh.mir, *b).filter(|c| is_positive_power_of_two(*c))
                    {
                        let q = self
                            .b
                            .build_float_mul(av, self.cf64(1.0 / c), "q")
                            .map_err(|e| e.to_string())?;
                        let q = self.intrinsic1("llvm.trunc.f64", q)?;
                        let m = self
                            .b
                            .build_float_mul(q, bv, "m")
                            .map_err(|e| e.to_string())?;
                        let r = self
                            .b
                            .build_float_sub(av, m, "r")
                            .map_err(|e| e.to_string())?;
                        self.intrinsic2("llvm.copysign.f64", r, av)?.into()
                    } else {
                        self.f64_rem(av, bv)?.into()
                    }
                }
                I::NegF64(a) => self
                    .b
                    .build_float_neg(self.getf(a)?, "fneg")
                    .map_err(|e| e.to_string())?
                    .into(),
                I::CmpLtF64(a, b) => self
                    .fcmp(FloatPredicate::OLT, self.getf(a)?, self.getf(b)?)?
                    .into(),
                I::CmpGtF64(a, b) => self
                    .fcmp(FloatPredicate::OGT, self.getf(a)?, self.getf(b)?)?
                    .into(),
                I::CmpLeF64(a, b) => self
                    .fcmp(FloatPredicate::OLE, self.getf(a)?, self.getf(b)?)?
                    .into(),
                I::CmpGeF64(a, b) => self
                    .fcmp(FloatPredicate::OGE, self.getf(a)?, self.getf(b)?)?
                    .into(),
                I::Unbox(a) => self.getf(a)?.into(),
                I::Box(a) => self.boxed(a)?.into(),
                I::GuardNum(s) => {
                    let v = self.boxed(s)?;
                    let fails = self.is_nan_boxed(v)?;
                    self.guard_deopt(fails)?;
                    self.num_values.insert(v);
                    v.into()
                }
                // In a block that ends unreachable, the exit is the
                // block: the guard that led here has already failed.
                I::SlowPathExit { pc, live } => {
                    let ends = matches!(
                        self.sh.mir.blocks[self.cur_block].terminator,
                        Terminator::Unreachable
                    );
                    if ends && self.inline_depth == 0 {
                        self.deopt_exit(*pc, live)?;
                        let dead = self.new_block("after_exit");
                        self.b.position_at_end(dead);
                    }
                    return Ok(None);
                }
                // A loop compiled cold: its generic calls fill their
                // caches as it runs, and the 256th iteration asks for
                // the function to be compiled again from them. The
                // body finishes its call on this code; there is no
                // transfer out of it.
                I::ColdLoopExit { .. } => {
                    if self.inline_depth == 0 {
                        let (slot, counter) = self.stack_buf(1)?;
                        self.zero_at_entry(slot)?;
                        let c = self.load64(counter, 0)?;
                        let c1 = self
                            .b
                            .build_int_add(c, self.c64(1), "cold")
                            .map_err(|e| e.to_string())?;
                        self.store64(counter, 0, c1)?;
                        let hot = self.icmp(
                            IntPredicate::EQ,
                            c1,
                            self.c64(crate::codegen::COLD_LOOP_EXIT_AFTER as u64),
                        )?;
                        let ask = self.new_block("cold_hot");
                        let cont = self.new_block("cold_cont");
                        self.cbr(hot, ask, cont)?;
                        self.b.position_at_end(ask);
                        let fid = self.c64(jit_func_id() as u64);
                        self.call_helper("wren_cold_loop_hot", &[fid])?;
                        self.b
                            .build_unconditional_branch(cont)
                            .map_err(|e| e.to_string())?;
                        self.b.position_at_end(cont);
                    }
                    return Ok(None);
                }
                I::GuardNumAt {
                    value, pc, live, ..
                } => {
                    let v = self.boxed(value)?;
                    let fails = match self.field_invariant.take() {
                        Some((guarded, kinds, len, idx)) if guarded == *value => {
                            let global = self.kinds_global(kinds, len);
                            let p = unsafe {
                                self.b
                                    .build_in_bounds_gep(
                                        self.sh.ctx.i8_type(),
                                        global.as_pointer_value(),
                                        &[self.sh.ctx.i64_type().const_int(idx as u64, false)],
                                        "fkp",
                                    )
                                    .map_err(|e| e.to_string())?
                            };
                            let kind = self
                                .b
                                .build_load(self.sh.ctx.i8_type(), p, "fk")
                                .map_err(|e| e.to_string())?;
                            let tag = self.kinds_tag();
                            kind.as_instruction_value()
                                .ok_or("load is not an instruction")?
                                .set_metadata(tag, self.sh.ctx.get_kind_id("tbaa"))
                                .map_err(|e| e.to_string())?;
                            self.b
                                .build_int_compare(
                                    IntPredicate::NE,
                                    kind.into_int_value(),
                                    self.sh
                                        .ctx
                                        .i8_type()
                                        .const_int(crate::runtime::object::FIELD_NUM as u64, false),
                                    "fkfail",
                                )
                                .map_err(|e| e.to_string())?
                        }
                        _ => self.is_nan_boxed(v)?,
                    };
                    self.guard_deopt_at(fails, *pc, live)?;
                    self.num_values.insert(v);
                    v.into()
                }
                I::GuardBool(s) => {
                    let v = self.boxed(s)?;
                    let t = self.icmp(IntPredicate::EQ, v, self.c64(TAG_TRUE))?;
                    let f = self.icmp(IntPredicate::EQ, v, self.c64(TAG_FALSE))?;
                    let is_bool = self.b.build_or(t, f, "isbool").map_err(|e| e.to_string())?;
                    let fails = self
                        .b
                        .build_not(is_bool, "fails")
                        .map_err(|e| e.to_string())?;
                    self.guard_deopt(fails)?;
                    v.into()
                }
                I::GuardClass(s, _) | I::GuardProtocol(s, _) => self.get(s)?,
                I::MathUnaryF64(op, a) => {
                    use crate::mir::MathUnaryOp::*;
                    let x = self.getf(a)?;
                    let r = match op {
                        Floor => self.intrinsic1("llvm.floor.f64", x)?,
                        Ceil => self.intrinsic1("llvm.ceil.f64", x)?,
                        Sqrt => self.intrinsic1("llvm.sqrt.f64", x)?,
                        Abs => self.intrinsic1("llvm.fabs.f64", x)?,
                        Trunc => self.intrinsic1("llvm.trunc.f64", x)?,
                        Round => self.intrinsic1("llvm.rint.f64", x)?,
                        Fract => {
                            let fl = self.intrinsic1("llvm.floor.f64", x)?;
                            self.b
                                .build_float_sub(x, fl, "fract")
                                .map_err(|e| e.to_string())?
                        }
                        Sign => {
                            let zero = self.cf64(0.0);
                            let pos = self.fcmp(FloatPredicate::OGT, x, zero)?;
                            let neg = self.fcmp(FloatPredicate::OLT, x, zero)?;
                            let pz = self
                                .b
                                .build_select(pos, self.cf64(1.0), zero, "pz")
                                .map_err(|e| e.to_string())?
                                .into_float_value();
                            self.b
                                .build_select(neg, self.cf64(-1.0), pz, "sign")
                                .map_err(|e| e.to_string())?
                                .into_float_value()
                        }
                        Sin => self.intrinsic1("llvm.sin.f64", x)?,
                        Cos => self.intrinsic1("llvm.cos.f64", x)?,
                        Tan => self.libm1("tan", x)?,
                        Asin => self.libm1("asin", x)?,
                        Acos => self.libm1("acos", x)?,
                        Atan => self.libm1("atan", x)?,
                        Log => self.intrinsic1("llvm.log.f64", x)?,
                        Log2 => self.intrinsic1("llvm.log2.f64", x)?,
                        Exp => self.intrinsic1("llvm.exp.f64", x)?,
                        Cbrt => self.libm1("cbrt", x)?,
                    };
                    r.into()
                }
                I::MathBinaryF64(op, a, b) => {
                    use crate::mir::MathBinaryOp::*;
                    let x = self.getf(a)?;
                    let y = self.getf(b)?;
                    let r = match op {
                        Min => self.intrinsic2("llvm.minimum.f64", x, y)?,
                        Max => self.intrinsic2("llvm.maximum.f64", x, y)?,
                        Pow => self.intrinsic2("llvm.pow.f64", x, y)?,
                        Atan2 => self.libm2("atan2", x, y)?,
                    };
                    r.into()
                }
            };
            Ok(Some(v))
        }

        /// Branch to `slow` unless `r` is a typed array and `idx` a Num
        /// within its count; on the returned block, `(obj_ptr, index,
        /// data, kind)` are ready.
        fn typed_array_probe(
            &mut self,
            r: IntValue<'ctx>,
            idx: IntValue<'ctx>,
            slow: BasicBlock<'ctx>,
        ) -> Result<
            (
                IntValue<'ctx>,
                IntValue<'ctx>,
                IntValue<'ctx>,
                IntValue<'ctx>,
            ),
            String,
        > {
            let high = self
                .b
                .build_right_shift(r, self.c64(48), false, "tag")
                .map_err(|e| e.to_string())?;
            let is_obj = self.icmp(IntPredicate::EQ, high, self.c64(0xFFFC))?;
            let obj_bb = self.new_block("tao");
            self.cbr(is_obj, obj_bb, slow)?;
            self.b.position_at_end(obj_bb);
            let obj = self.and(r, self.c64(PTR_MASK))?;
            let ty = self.load8(obj, HEADER_OBJ_TYPE as i64)?;
            let is_ta = self.icmp(IntPredicate::EQ, ty, self.c64(OBJ_TYPE_TYPED_ARRAY as u64))?;
            let ta_bb = self.new_block("ta");
            self.cbr(is_ta, ta_bb, slow)?;
            self.b.position_at_end(ta_bb);
            let idx_i = self.int_index(idx, obj, TYPED_ARRAY_COUNT as i64, slow)?;
            let data = self.load64(obj, TYPED_ARRAY_DATA as i64)?;
            let kind = self.load8(obj, TYPED_ARRAY_KIND as i64)?;
            Ok((obj, idx_i, data, kind))
        }

        /// `idx` as an i64 when it is an integral Num below the u32
        /// count at `obj + count_off`; otherwise branch to `slow`.
        /// An i64 index checked against the count at `count_off`.
        fn bounded_index(
            &mut self,
            idx_i: IntValue<'ctx>,
            obj: IntValue<'ctx>,
            count_off: i64,
            slow: BasicBlock<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            let count_p = self.addr(obj, count_off)?;
            let count32 = self
                .b
                .build_load(self.sh.ctx.i32_type(), count_p, "count")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let count = self
                .b
                .build_int_z_extend(count32, self.i64t(), "count64")
                .map_err(|e| e.to_string())?;
            let in_range = self.icmp(IntPredicate::ULT, idx_i, count)?;
            let ok_bb = self.new_block("ixr");
            self.cbr(in_range, ok_bb, slow)?;
            self.b.position_at_end(ok_bb);
            Ok(idx_i)
        }

        fn int_index(
            &mut self,
            idx: IntValue<'ctx>,
            obj: IntValue<'ctx>,
            count_off: i64,
            slow: BasicBlock<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            let is_box = self.is_nan_boxed(idx)?;
            let num_bb = self.new_block("ixn");
            self.cbr(is_box, slow, num_bb)?;
            self.b.position_at_end(num_bb);
            let idx_f = self.f64_of(idx)?;
            let idx_i = self
                .b
                .build_float_to_signed_int(idx_f, self.i64t(), "idx")
                .map_err(|e| e.to_string())?;
            let back = self
                .b
                .build_signed_int_to_float(idx_i, self.f64t(), "idxf")
                .map_err(|e| e.to_string())?;
            let integral = self
                .b
                .build_float_compare(FloatPredicate::OEQ, back, idx_f, "int")
                .map_err(|e| e.to_string())?;
            let int_bb = self.new_block("ixi");
            self.cbr(integral, int_bb, slow)?;
            self.b.position_at_end(int_bb);
            let count_p = self.addr(obj, count_off)?;
            let count32 = self
                .b
                .build_load(self.sh.ctx.i32_type(), count_p, "count")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let count = self
                .b
                .build_int_z_extend(count32, self.i64t(), "count64")
                .map_err(|e| e.to_string())?;
            let in_range = self.icmp(IntPredicate::ULT, idx_i, count)?;
            let ok_bb = self.new_block("ixr");
            self.cbr(in_range, ok_bb, slow)?;
            self.b.position_at_end(ok_bb);
            Ok(idx_i)
        }

        /// The address of `r[idx]` when `r` is a List and `idx` an
        /// integral Num within its count; otherwise branch to `other`.
        fn list_element(
            &mut self,
            r: IntValue<'ctx>,
            idx: IntValue<'ctx>,
            int: Option<IntValue<'ctx>>,
            other: BasicBlock<'ctx>,
        ) -> Result<PointerValue<'ctx>, String> {
            let high = self
                .b
                .build_right_shift(r, self.c64(48), false, "tag")
                .map_err(|e| e.to_string())?;
            let is_obj = self.icmp(IntPredicate::EQ, high, self.c64(0xFFFC))?;
            let obj_bb = self.new_block("leo");
            self.cbr(is_obj, obj_bb, other)?;
            self.b.position_at_end(obj_bb);
            let obj = self.and(r, self.c64(PTR_MASK))?;
            let ty = self.load8(obj, HEADER_OBJ_TYPE as i64)?;
            let is_list = self.icmp(
                IntPredicate::EQ,
                ty,
                self.c64(crate::runtime::object::ObjType::List as u64),
            )?;
            let list_bb = self.new_block("lel");
            self.cbr(is_list, list_bb, other)?;
            self.b.position_at_end(list_bb);
            let i = match int {
                Some(i) => self.bounded_index(i, obj, LIST_COUNT as i64, other)?,
                None => self.int_index(idx, obj, LIST_COUNT as i64, other)?,
            };
            let elements = self.load64(obj, LIST_ELEMENTS as i64)?;
            self.element_addr(elements, i, VALUE_SIZE as u64)
        }

        fn element_addr(
            &mut self,
            data: IntValue<'ctx>,
            idx: IntValue<'ctx>,
            size: u64,
        ) -> Result<PointerValue<'ctx>, String> {
            let off = self
                .b
                .build_int_mul(idx, self.c64(size), "off")
                .map_err(|e| e.to_string())?;
            let a = self
                .b
                .build_int_add(data, off, "ea")
                .map_err(|e| e.to_string())?;
            self.b
                .build_int_to_ptr(a, self.ptrt(), "ep")
                .map_err(|e| e.to_string())
        }

        /// `r[idx]` with List elements and the typed-array element kinds
        /// inline and everything else through the helper.
        /// The i64 a boxed index was converted from, when it was.
        fn int_source(&self, idx: &ValueId) -> Result<Option<IntValue<'ctx>>, String> {
            match self.int_sources.get(idx) {
                Some(i) => Ok(Some(self.geti(i)?)),
                None => Ok(None),
            }
        }

        fn typed_array_get(
            &mut self,
            r: IntValue<'ctx>,
            idx: IntValue<'ctx>,
            int: Option<IntValue<'ctx>>,
        ) -> Result<IntValue<'ctx>, String> {
            let exit = self.miss_exit_block()?;
            let slow = exit.unwrap_or_else(|| self.new_block("sgs"));
            let merge = self.new_block("sgm");
            let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> = Vec::new();
            let other = self.new_block("sgo");
            let p = self.list_element(r, idx, int, other)?;
            let v = self
                .b
                .build_load(self.i64t(), p, "elem")
                .map_err(|e| e.to_string())?
                .into_int_value();
            incoming.push((v.into(), self.b.get_insert_block().unwrap()));
            self.br(merge)?;
            self.b.position_at_end(other);
            let (_, i, data, kind) = self.typed_array_probe(r, idx, slow)?;
            let ctx = self.sh.ctx;
            let kinds: [(u8, u64); 4] = [
                (TA_KIND_F64, 8),
                (TA_KIND_F32, 4),
                (TA_KIND_I32, 4),
                (TA_KIND_U8, 1),
            ];
            for (k, size) in kinds {
                let hit = self.icmp(IntPredicate::EQ, kind, self.c64(k as u64))?;
                let yes = self.new_block("sgk");
                let next = self.new_block("sgn");
                self.cbr(hit, yes, next)?;
                self.b.position_at_end(yes);
                let p = self.element_addr(data, i, size)?;
                let f: FloatValue<'ctx> = match k {
                    TA_KIND_F64 => self
                        .b
                        .build_load(self.f64t(), p, "e")
                        .map_err(|e| e.to_string())?
                        .into_float_value(),
                    TA_KIND_F32 => {
                        let v = self
                            .b
                            .build_load(ctx.f32_type(), p, "e")
                            .map_err(|e| e.to_string())?
                            .into_float_value();
                        self.b
                            .build_float_ext(v, self.f64t(), "ext")
                            .map_err(|e| e.to_string())?
                    }
                    TA_KIND_I32 => {
                        let v = self
                            .b
                            .build_load(ctx.i32_type(), p, "e")
                            .map_err(|e| e.to_string())?
                            .into_int_value();
                        self.b
                            .build_signed_int_to_float(v, self.f64t(), "i2f")
                            .map_err(|e| e.to_string())?
                    }
                    _ => {
                        let v = self
                            .b
                            .build_load(ctx.i8_type(), p, "e")
                            .map_err(|e| e.to_string())?
                            .into_int_value();
                        self.b
                            .build_unsigned_int_to_float(v, self.f64t(), "u2f")
                            .map_err(|e| e.to_string())?
                    }
                };
                let bits = self.bits(f)?;
                incoming.push((bits.into(), self.b.get_insert_block().unwrap()));
                self.br(merge)?;
                self.b.position_at_end(next);
            }
            self.br(slow)?;
            if exit.is_none() {
                self.b.position_at_end(slow);
                let sv = self.call_helper("wren_subscript_get", &[r, idx])?;
                incoming.push((sv.into(), self.b.get_insert_block().unwrap()));
                self.br(merge)?;
            }
            self.b.position_at_end(merge);
            Ok(self.phi(self.i64t().into(), &incoming)?.into_int_value())
        }

        /// `r[idx] = v` with List and f32/f64 typed-array stores inline
        /// and everything else through the helper.
        fn typed_array_set(
            &mut self,
            r: IntValue<'ctx>,
            idx: IntValue<'ctx>,
            int: Option<IntValue<'ctx>>,
            v: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            let exit = self.miss_exit_block()?;
            let slow = exit.unwrap_or_else(|| self.new_block("sss"));
            let merge = self.new_block("ssm");
            let other = self.new_block("sso");
            let p = self.list_element(r, idx, int, other)?;
            self.b.build_store(p, v).map_err(|e| e.to_string())?;
            self.br(merge)?;
            self.b.position_at_end(other);
            let (_, i, data, kind) = self.typed_array_probe(r, idx, slow)?;
            let is_box = self.is_nan_boxed(v)?;
            let num_bb = self.new_block("ssn");
            self.cbr(is_box, slow, num_bb)?;
            self.b.position_at_end(num_bb);
            let vf = self.f64_of(v)?;
            let is_f64 = self.icmp(IntPredicate::EQ, kind, self.c64(TA_KIND_F64 as u64))?;
            let f64_bb = self.new_block("ss64");
            let chk32 = self.new_block("ssc");
            self.cbr(is_f64, f64_bb, chk32)?;
            self.b.position_at_end(f64_bb);
            let p = self.element_addr(data, i, 8)?;
            self.b.build_store(p, vf).map_err(|e| e.to_string())?;
            self.br(merge)?;
            self.b.position_at_end(chk32);
            let is_f32 = self.icmp(IntPredicate::EQ, kind, self.c64(TA_KIND_F32 as u64))?;
            let f32_bb = self.new_block("ss32");
            self.cbr(is_f32, f32_bb, slow)?;
            self.b.position_at_end(f32_bb);
            let p = self.element_addr(data, i, 4)?;
            let v32 = self
                .b
                .build_float_trunc(vf, self.sh.ctx.f32_type(), "f32")
                .map_err(|e| e.to_string())?;
            self.b.build_store(p, v32).map_err(|e| e.to_string())?;
            self.br(merge)?;
            if exit.is_none() {
                self.b.position_at_end(slow);
                self.call_helper("wren_subscript_set", &[r, idx, v])?;
                self.br(merge)?;
            }
            self.b.position_at_end(merge);
            Ok(v)
        }

        /// Branch on `fails` to a cold block that re-executes the call in
        /// the interpreter with the entry parameters and returns its
        /// result; lowering continues on the other edge.
        fn guard_deopt(&mut self, fails: IntValue<'ctx>) -> Result<(), String> {
            if self.inline_depth > 0 {
                bail!("speculative guard inside an inlined body");
            }
            let params: Vec<IntValue<'ctx>> = if self.entries.is_empty() {
                (0..self.sh.mir.arity as usize)
                    .map(|i| self.f.get_nth_param(i as u32).unwrap().into_int_value())
                    .collect()
            } else {
                self.param_regs.clone()
            };
            let deopt = self.new_block("deopt");
            let cont = self.new_block("cont");
            self.cbr(fails, deopt, cont)?;
            self.b.position_at_end(deopt);
            let (_, buf) = self.stack_buf(params.len())?;
            for (i, p) in params.iter().enumerate() {
                self.store64(buf, (i * 8) as i64, *p)?;
            }
            let fid = self.c64(jit_func_id() as u64);
            let n = self.c64(params.len() as u64);
            let result = self.call_helper("wren_deopt_n", &[fid, n, buf])?;
            self.b
                .build_return(Some(&result))
                .map_err(|e| e.to_string())?;
            self.b.position_at_end(cont);
            Ok(())
        }

        /// A mid-body guard: when `fails`, store `live` in the word
        /// layout `wren_deopt_at` reads and return whatever it computes
        /// by resuming the interpreter at `pc`.
        fn guard_deopt_at(
            &mut self,
            fails: IntValue<'ctx>,
            pc: u32,
            live: &[DeoptReg],
        ) -> Result<(), String> {
            if self.inline_depth > 0 {
                bail!("mid-body guard inside an inlined body");
            }
            let deopt = self.new_block("deopt_at");
            let cont = self.new_block("cont");
            self.cbr(fails, deopt, cont)?;
            self.b.position_at_end(deopt);
            self.deopt_exit(pc, live)?;
            self.b.position_at_end(cont);
            Ok(())
        }

        /// Leave the function from the current block: hand `live` to
        /// `wren_deopt_at` for offset `pc` and return its result.
        fn deopt_exit(&mut self, pc: u32, live: &[DeoptReg]) -> Result<(), String> {
            let words = live
                .iter()
                .map(crate::codegen::runtime_fns::deopt_words)
                .sum::<usize>();
            let (_, buf) = self.stack_buf(words)?;
            let mut at = 0i64;
            for r in live {
                self.store64(
                    buf,
                    at * 8,
                    self.c64(crate::codegen::runtime_fns::deopt_tag(r)),
                )?;
                at += 1;
                for c in crate::codegen::runtime_fns::deopt_consts(r) {
                    self.store64(buf, at * 8, self.c64(c))?;
                    at += 1;
                }
                for v in r.source.operands() {
                    let v = self.boxed(&v)?;
                    self.store64(buf, at * 8, v)?;
                    at += 1;
                }
            }
            let fid = self.c64(jit_func_id() as u64);
            let pcv = self.c64(pc as u64);
            let n = self.c64(words as u64);
            let result = self.call_helper("wren_deopt_at", &[fid, pcv, n, buf])?;
            self.b
                .build_return(Some(&result))
                .map_err(|e| e.to_string())?;
            Ok(())
        }

        /// A fresh instance of the class object `class_val`: bumped out
        /// of the Immix small region when the compile knows it, its
        /// fields null, the start byte written; the helper when the
        /// region is out of room, the object exceeds a line, or there
        /// is no region.
        /// `size` bytes (16-aligned, at most a line) of plain object
        /// from the bump region at `bump`, its start byte written; the
        /// builder ends in the block holding the address, and `slow`
        /// is taken when the region has no room.
        fn bump_alloc(
            &mut self,
            bump: usize,
            size: IntValue<'ctx>,
            slow: BasicBlock<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            use crate::runtime::gc_immix_heap::{
                BUMP_CODES, BUMP_CUR, BUMP_LIMIT, BUMP_PLAIN_FLAG,
            };
            let bump_v = self.c64(bump as u64);
            let cur = self.load64(bump_v, BUMP_CUR as i64)?;
            let limit = self.load64(bump_v, BUMP_LIMIT as i64)?;
            // Never straddle a line: start at the next line if the
            // object would.
            let off = self.and(cur, self.c64(127))?;
            let end_in_line = self
                .b
                .build_int_add(off, size, "eil")
                .map_err(|e| e.to_string())?;
            let straddles = self.icmp(IntPredicate::UGT, end_in_line, self.c64(128))?;
            let aligned = self.and(
                self.b
                    .build_int_add(cur, self.c64(127), "c127")
                    .map_err(|e| e.to_string())?,
                self.c64(!127u64),
            )?;
            let p = self
                .b
                .build_select(straddles, aligned, cur, "p")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let np = self
                .b
                .build_int_add(p, size, "np")
                .map_err(|e| e.to_string())?;
            let room = self.icmp(IntPredicate::ULE, np, limit)?;
            let fast = self.new_block("alf");
            self.cbr(room, fast, slow)?;
            self.b.position_at_end(fast);
            self.store64(bump_v, BUMP_CUR as i64, np)?;
            let codes = self.load64(bump_v, BUMP_CODES as i64)?;
            let q = self
                .b
                .build_right_shift(p, self.c64(4), false, "q")
                .map_err(|e| e.to_string())?;
            let code_p = self.addr(
                self.b
                    .build_int_add(codes, q, "cp")
                    .map_err(|e| e.to_string())?,
                0,
            )?;
            let code = self
                .b
                .build_int_truncate(
                    self.b
                        .build_or(
                            self.b
                                .build_right_shift(size, self.c64(4), false, "sq")
                                .map_err(|e| e.to_string())?,
                            self.c64(BUMP_PLAIN_FLAG as u64),
                            "code",
                        )
                        .map_err(|e| e.to_string())?,
                    self.sh.ctx.i8_type(),
                    "code8",
                )
                .map_err(|e| e.to_string())?;
            self.b
                .build_store(code_p, code)
                .map_err(|e| e.to_string())?;
            Ok(p)
        }

        /// A list of `elems` in one plain allocation, its elements
        /// after the header; a literal too long for a line, or no bump
        /// region, takes the helper.
        fn alloc_list(&mut self, elems: &[IntValue<'ctx>]) -> Result<IntValue<'ctx>, String> {
            use crate::runtime::object::{FLAG_HEAP_BUFFER, ObjType};
            let bump = crate::codegen::jit_bump_region();
            let list_class = crate::codegen::jit_list_class();
            let n = elems.len();
            // An empty literal starts with the room the runtime gives it.
            let cap = if n == 0 { 8 } else { n };
            let size = (LIST_SIZE as u64 + VALUE_SIZE as u64 * cap as u64 + 15) & !15;
            if bump == 0 || list_class == 0 || size > 128 || n > 4 {
                return self.make_list_helper(elems);
            }
            let slow = self.new_block("lls");
            let merge = self.new_block("llm");
            let p = self.bump_alloc(bump, self.c64(size), slow)?;
            // Header: list type with the heap-buffer flag, no next, the
            // list class; count and capacity share a word; the elements
            // follow.
            let type_word = ObjType::List as u64 | ((FLAG_HEAP_BUFFER as u64) << 24);
            self.store64(p, 0, self.c64(type_word))?;
            self.store64(p, HEADER_NEXT as i64, self.c64(0))?;
            self.store64(p, HEADER_CLASS as i64, self.c64(list_class as u64))?;
            self.store64(
                p,
                LIST_COUNT as i64,
                self.c64(n as u64 | ((cap as u64) << 32)),
            )?;
            let elements = self
                .b
                .build_int_add(p, self.c64(LIST_SIZE as u64), "elements")
                .map_err(|e| e.to_string())?;
            self.store64(p, LIST_ELEMENTS as i64, elements)?;
            for (i, v) in elems.iter().enumerate() {
                self.store64(elements, i as i64 * VALUE_SIZE as i64, *v)?;
            }
            let boxed = self
                .b
                .build_or(p, self.c64(TAG_OBJ), "list")
                .map_err(|e| e.to_string())?;
            let fast_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(slow);
            let sv = self.make_list_helper(elems)?;
            let slow_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(merge);
            Ok(self
                .phi(
                    self.i64t().into(),
                    &[(boxed.into(), fast_end), (sv.into(), slow_end)],
                )?
                .into_int_value())
        }

        fn make_list_helper(&mut self, elems: &[IntValue<'ctx>]) -> Result<IntValue<'ctx>, String> {
            if elems.len() <= 4 {
                let name = [
                    "wren_make_list",
                    "wren_make_list_1",
                    "wren_make_list_2",
                    "wren_make_list_3",
                    "wren_make_list_4",
                ][elems.len()];
                return self.call_helper(name, elems);
            }
            let list = self.call_helper("wren_make_list", &[])?;
            for v in elems {
                self.call_helper("wren_list_add", &[list, *v])?;
            }
            Ok(list)
        }

        /// A new instance of `class_val`. With `known`, the field count
        /// is a compile-time constant and the fields in the `assigned`
        /// mask are left for the stores that follow.
        fn alloc_instance(
            &mut self,
            class_val: IntValue<'ctx>,
            known: Option<(u16, u64)>,
        ) -> Result<IntValue<'ctx>, String> {
            let bump = crate::codegen::jit_bump_region();
            let known_size = known
                .map(|(nf, _)| (INSTANCE_SIZE as u64 + VALUE_SIZE as u64 * nf as u64 + 15) & !15);
            if bump == 0 || known_size.is_some_and(|s| s > 128) {
                return self.call_helper("wren_alloc_instance", &[class_val]);
            }
            let slow = self.new_block("als");
            let merge = self.new_block("alm");
            let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> = Vec::new();
            let class = self.and(class_val, self.c64(PTR_MASK))?;
            let (nf, size) = match known {
                Some((nf, _)) => (self.c64(nf as u64), self.c64(known_size.unwrap())),
                None => {
                    let nf_p = self.addr(class, CLASS_NUM_FIELDS as i64)?;
                    let nf16 = self
                        .b
                        .build_load(self.sh.ctx.i16_type(), nf_p, "nf16")
                        .map_err(|e| e.to_string())?
                        .into_int_value();
                    let nf = self
                        .b
                        .build_int_z_extend(nf16, self.i64t(), "nf")
                        .map_err(|e| e.to_string())?;
                    // size = (40 + 8 * nf + 15) & !15
                    let raw = self
                        .b
                        .build_int_add(
                            self.b
                                .build_int_mul(nf, self.c64(VALUE_SIZE as u64), "fb")
                                .map_err(|e| e.to_string())?,
                            self.c64(INSTANCE_SIZE as u64 + 15),
                            "raw",
                        )
                        .map_err(|e| e.to_string())?;
                    let size = self.and(raw, self.c64(!15u64))?;
                    let fits = self.icmp(IntPredicate::ULE, size, self.c64(128))?;
                    let sized = self.new_block("alz");
                    self.cbr(fits, sized, slow)?;
                    self.b.position_at_end(sized);
                    (nf, size)
                }
            };
            let p = self.bump_alloc(bump, size, slow)?;
            // Header: type byte, clear mark/generation/flags, no next,
            // the class, the field count, no owned fields, the fields
            // right after the header.
            self.store64(p, 0, self.c64(OBJ_TYPE_INSTANCE as u64))?;
            self.store64(p, HEADER_NEXT as i64, self.c64(0))?;
            self.store64(p, HEADER_CLASS as i64, class)?;
            self.store64(p, INSTANCE_NUM_FIELDS as i64, nf)?;
            let fields = self
                .b
                .build_int_add(p, self.c64(INSTANCE_SIZE as u64), "fields")
                .map_err(|e| e.to_string())?;
            let has_fields = self.icmp(IntPredicate::NE, nf, self.c64(0))?;
            let fields_or_null = self
                .b
                .build_select(has_fields, fields, self.c64(0), "fp")
                .map_err(|e| e.to_string())?
                .into_int_value();
            self.store64(p, INSTANCE_FIELDS as i64, fields_or_null)?;
            if let Some((count, assigned)) = known {
                // Null the fields nothing stores before the object can
                // be seen.
                for i in 0..count {
                    if i < 64 && assigned & (1 << i) != 0 {
                        continue;
                    }
                    self.field_store(fields, i, self.c64(TAG_NULL))?;
                }
                let boxed = self
                    .b
                    .build_or(p, self.c64(TAG_OBJ), "inst")
                    .map_err(|e| e.to_string())?;
                incoming.push((boxed.into(), self.b.get_insert_block().unwrap()));
                self.br(merge)?;
                self.b.position_at_end(slow);
                let sv = self.call_helper("wren_alloc_instance", &[class_val])?;
                incoming.push((sv.into(), self.b.get_insert_block().unwrap()));
                self.br(merge)?;
                self.b.position_at_end(merge);
                return Ok(self.phi(self.i64t().into(), &incoming)?.into_int_value());
            }
            // Null every field.
            let loop_bb = self.new_block("alnull");
            let done = self.new_block("aldone");
            let entry_bb = self.b.get_insert_block().unwrap();
            self.cbr(has_fields, loop_bb, done)?;
            self.b.position_at_end(loop_bb);
            let i = self
                .b
                .build_phi(self.i64t(), "i")
                .map_err(|e| e.to_string())?;
            let slot = self.addr(
                self.b
                    .build_int_add(
                        fields,
                        self.b
                            .build_int_mul(i.as_basic_value().into_int_value(), self.c64(8), "io")
                            .map_err(|e| e.to_string())?,
                        "sa",
                    )
                    .map_err(|e| e.to_string())?,
                0,
            )?;
            self.b
                .build_store(slot, self.c64(TAG_NULL))
                .map_err(|e| e.to_string())?;
            let next = self
                .b
                .build_int_add(i.as_basic_value().into_int_value(), self.c64(1), "i1")
                .map_err(|e| e.to_string())?;
            i.add_incoming(&[(&self.c64(0), entry_bb), (&next, loop_bb)]);
            let more = self.icmp(IntPredicate::ULT, next, nf)?;
            self.cbr(more, loop_bb, done)?;
            self.b.position_at_end(done);
            let boxed = self
                .b
                .build_or(p, self.c64(TAG_OBJ), "inst")
                .map_err(|e| e.to_string())?;
            incoming.push((boxed.into(), done));
            self.br(merge)?;
            self.b.position_at_end(slow);
            let sv = self.call_helper("wren_alloc_instance", &[class_val])?;
            incoming.push((sv.into(), self.b.get_insert_block().unwrap()));
            self.br(merge)?;
            self.b.position_at_end(merge);
            Ok(self.phi(self.i64t().into(), &incoming)?.into_int_value())
        }

        /// `(slot, value)` of the direct-call depth counter.
        fn direct_depth(&mut self) -> Result<(PointerValue<'ctx>, IntValue<'ctx>), String> {
            let addr = &crate::codegen::runtime_fns::JIT_DIRECT_DEPTH
                as *const std::sync::atomic::AtomicU32 as u64;
            let p = self
                .b
                .build_int_to_ptr(self.c64(addr), self.ptrt(), "depthp")
                .map_err(|e| e.to_string())?;
            let d = self
                .b
                .build_load(self.sh.ctx.i32_type(), p, "depth")
                .map_err(|e| e.to_string())?
                .into_int_value();
            Ok((p, d))
        }

        /// The class-miss exit a guarded call may take instead of its
        /// slow path: `(offset of the call, registers live before it)`,
        /// set while the call just before a `GuardNumAt` is lowered.
        fn take_miss_exit(&mut self) -> Option<(u32, Vec<DeoptReg>)> {
            if self.inline_depth > 0 {
                return None;
            }
            self.miss_exit.take()
        }

        /// A `slow` block that leaves the function through the pending
        /// miss exit, or `None` when the call must keep its slow path.
        fn miss_exit_block(&mut self) -> Result<Option<BasicBlock<'ctx>>, String> {
            let Some((pc, live)) = self.take_miss_exit() else {
                return Ok(None);
            };
            let cur = self.b.get_insert_block().unwrap();
            let bb = self.new_block("miss");
            self.b.position_at_end(bb);
            self.deopt_exit(pc, &live)?;
            self.b.position_at_end(cur);
            Ok(Some(bb))
        }

        fn helper2(
            &mut self,
            name: &str,
            a: &ValueId,
            b: &ValueId,
        ) -> Result<IntValue<'ctx>, String> {
            let x = self.boxed(a)?;
            let y = self.boxed(b)?;
            self.call_helper(name, &[x, y])
        }

        /// Integer fast path for `a % b`, `fmod` otherwise.
        fn f64_rem(
            &mut self,
            av: FloatValue<'ctx>,
            bv: FloatValue<'ctx>,
        ) -> Result<FloatValue<'ctx>, String> {
            let fast = self.new_block("remf");
            let slow = self.new_block("rems");
            let merge = self.new_block("remm");
            let ai = self.intrinsic_fptosi_sat(av)?;
            let bi = self.intrinsic_fptosi_sat(bv)?;
            let a_back = self
                .b
                .build_signed_int_to_float(ai, self.f64t(), "ab")
                .map_err(|e| e.to_string())?;
            let b_back = self
                .b
                .build_signed_int_to_float(bi, self.f64t(), "bb")
                .map_err(|e| e.to_string())?;
            let a_int = self.fcmp(FloatPredicate::OEQ, a_back, av)?;
            let b_int = self.fcmp(FloatPredicate::OEQ, b_back, bv)?;
            let limit = self.cf64(9007199254740992.0);
            let a_abs = self.intrinsic1("llvm.fabs.f64", av)?;
            let b_abs = self.intrinsic1("llvm.fabs.f64", bv)?;
            let a_small = self.fcmp(FloatPredicate::OLT, a_abs, limit)?;
            let b_small = self.fcmp(FloatPredicate::OLT, b_abs, limit)?;
            let b_nz = self.icmp(IntPredicate::NE, bi, self.c64(0))?;
            let ok = self.and(a_int, b_int)?;
            let ok = self.and(ok, a_small)?;
            let ok = self.and(ok, b_small)?;
            let ok = self.and(ok, b_nz)?;
            self.cbr(ok, fast, slow)?;
            self.b.position_at_end(fast);
            let r = self
                .b
                .build_int_signed_rem(ai, bi, "irem")
                .map_err(|e| e.to_string())?;
            let rf = self
                .b
                .build_signed_int_to_float(r, self.f64t(), "rf")
                .map_err(|e| e.to_string())?;
            let rf = self.intrinsic2("llvm.copysign.f64", rf, av)?;
            self.br(merge)?;
            self.b.position_at_end(slow);
            let sr = self.libm2("fmod", av, bv)?;
            self.br(merge)?;
            self.b.position_at_end(merge);
            Ok(self
                .phi(self.f64t().into(), &[(rf.into(), fast), (sr.into(), slow)])?
                .into_float_value())
        }

        fn intrinsic_fptosi_sat(&mut self, x: FloatValue<'ctx>) -> Result<IntValue<'ctx>, String> {
            let intr = Intrinsic::find("llvm.fptosi.sat").ok_or("no fptosi.sat")?;
            let decl = intr
                .get_declaration(self.sh.module, &[self.i64t().into(), self.f64t().into()])
                .ok_or("no fptosi.sat declaration")?;
            let call = self
                .b
                .build_call(decl, &[x.into()], "sat")
                .map_err(|e| e.to_string())?;
            Ok(call.try_as_basic_value().basic().unwrap().into_int_value())
        }

        /// Boxed arithmetic or comparison with the inline f64 fast path.
        fn boxed_binop(
            &mut self,
            a: &ValueId,
            b: &ValueId,
            op: BinOp,
            slow_fn: &str,
        ) -> Result<IntValue<'ctx>, String> {
            let la = self.boxed(a)?;
            let lb = self.boxed(b)?;
            let check_b = self.new_block("chk");
            let fast = self.new_block("fast");
            let slow = self.new_block("slow");
            let merge = self.new_block("merge");
            let a_nan = self.is_nan_boxed(la)?;
            self.cbr(a_nan, slow, check_b)?;
            self.b.position_at_end(check_b);
            let b_nan = self.is_nan_boxed(lb)?;
            self.cbr(b_nan, slow, fast)?;
            self.b.position_at_end(fast);
            let fa = self.f64_of(la)?;
            let fb = self.f64_of(lb)?;
            let fast_v = match op {
                BinOp::Add => self.bits(
                    self.b
                        .build_float_add(fa, fb, "fadd")
                        .map_err(|e| e.to_string())?,
                )?,
                BinOp::Sub => self.bits(
                    self.b
                        .build_float_sub(fa, fb, "fsub")
                        .map_err(|e| e.to_string())?,
                )?,
                BinOp::Mul => self.bits(
                    self.b
                        .build_float_mul(fa, fb, "fmul")
                        .map_err(|e| e.to_string())?,
                )?,
                BinOp::Div => self.bits(
                    self.b
                        .build_float_div(fa, fb, "fdiv")
                        .map_err(|e| e.to_string())?,
                )?,
                BinOp::Rem => {
                    let r = self.f64_rem(fa, fb)?;
                    self.bits(r)?
                }
                BinOp::Cmp(p) => {
                    let c = self.fcmp(p, fa, fb)?;
                    self.box_bool(c)?
                }
            };
            let fast_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(slow);
            let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> =
                vec![(fast_v.into(), fast_end)];
            // Equality on anything but a Num is identity unless the left
            // operand's class says otherwise, which a helper decides.
            let identity = match op {
                BinOp::Cmp(FloatPredicate::OEQ) => Some(true),
                BinOp::Cmp(FloatPredicate::UNE) => Some(false),
                _ => None,
            };
            if let Some(eq) = identity {
                let same = self.icmp(IntPredicate::EQ, la, lb)?;
                let (is_obj, _, class) = self.class_of(la)?;
                // A non-object reads the null object's flags: none.
                let null_obj =
                    self.c64(crate::codegen::runtime_fns::JIT_NULL_OBJECT.as_ptr() as u64);
                let src = self
                    .b
                    .build_select(is_obj, class, null_obj, "flagsrc")
                    .map_err(|e| e.to_string())?
                    .into_int_value();
                let flags = self.load8(src, CLASS_FLAGS as i64)?;
                let has_eq = self.icmp(
                    IntPredicate::NE,
                    self.and(flags, self.c64(CLASS_FLAG_EQ as u64))?,
                    self.c64(0),
                )?;
                let custom = self
                    .b
                    .build_and(is_obj, has_eq, "customeq")
                    .map_err(|e| e.to_string())?;
                let by_identity = self.new_block("ideq");
                let call = self.new_block("eqcall");
                self.cbr(custom, call, by_identity)?;
                self.b.position_at_end(by_identity);
                let r = if eq {
                    same
                } else {
                    self.b.build_not(same, "ne").map_err(|e| e.to_string())?
                };
                let v = self.box_bool(r)?;
                incoming.push((v.into(), self.b.get_insert_block().unwrap()));
                self.br(merge)?;
                self.b.position_at_end(call);
            }
            let slow_v = self.call_helper(slow_fn, &[la, lb])?;
            let slow_end = self.b.get_insert_block().unwrap();
            incoming.push((slow_v.into(), slow_end));
            self.br(merge)?;
            self.b.position_at_end(merge);
            Ok(self.phi(self.i64t().into(), &incoming)?.into_int_value())
        }

        // ── Calls ──────────────────────────────────────────────────────

        fn wren_call(
            &mut self,
            receiver: IntValue<'ctx>,
            method_val: IntValue<'ctx>,
            args: &[IntValue<'ctx>],
        ) -> Result<IntValue<'ctx>, String> {
            if args.len() > 8 {
                let (_, buf) = self.stack_buf(args.len())?;
                for (i, a) in args.iter().enumerate() {
                    self.store64(buf, (i * 8) as i64, *a)?;
                }
                let count = self.c64(args.len() as u64);
                return self.call_helper("wren_call_dynamic", &[receiver, method_val, count, buf]);
            }
            let name = format!("wren_call_{}", args.len());
            let mut a = vec![receiver, method_val];
            a.extend_from_slice(args);
            self.call_helper(&name, &a)
        }

        fn method_bits(
            &self,
            method: crate::intern::SymbolId,
            ic_idx: Option<usize>,
        ) -> IntValue<'ctx> {
            let mut bits = method.index() as u64;
            if let Some(i) = ic_idx.filter(|_| env_jit_callsite_ic()) {
                bits |= ((i as u64) + 1) << 32;
            }
            self.c64(bits)
        }

        /// Call `func_id`'s compiled code through its `jit_code` slot,
        /// counting the direct-call depth around it. Branches to `slow`
        /// when the slot is empty or the depth is out; returns the
        /// result and the block it arrives from, or `None` when there
        /// is no slot table (the builder is then unmoved).
        fn slot_call(
            &mut self,
            func_id: u32,
            r: IntValue<'ctx>,
            args: &[IntValue<'ctx>],
            slow: BasicBlock<'ctx>,
        ) -> Result<Option<(IntValue<'ctx>, BasicBlock<'ctx>)>, String> {
            let Some(base) = self.sh.jit_code_base else {
                return Ok(None);
            };
            let call_bb = self.new_block("slc");
            // The body calling itself is the code in its own slot; it
            // needs no slot to tell it so.
            let is_self = self.inline_depth == 0 && func_id == jit_func_id();
            let jit_ptr = if is_self {
                None
            } else {
                let depth_bb = self.new_block("sld");
                let slot_addr = unsafe { base.add(func_id as usize) } as u64;
                let jit_ptr = self.load64(self.c64(slot_addr), 0)?;
                let has = self.icmp(IntPredicate::NE, jit_ptr, self.c64(0))?;
                self.cbr(has, depth_bb, slow)?;
                self.b.position_at_end(depth_bb);
                Some(jit_ptr)
            };
            let (depth_p, depth) = self.direct_depth()?;
            let room = self.icmp(
                IntPredicate::ULT,
                depth,
                self.sh
                    .ctx
                    .i32_type()
                    .const_int(crate::codegen::runtime_fns::MAX_JIT_DEPTH as u64, false),
            )?;
            self.cbr(room, call_bb, slow)?;
            self.b.position_at_end(call_bb);
            let deeper = self
                .b
                .build_int_add(depth, self.sh.ctx.i32_type().const_int(1, false), "d1")
                .map_err(|e| e.to_string())?;
            self.b
                .build_store(depth_p, deeper)
                .map_err(|e| e.to_string())?;
            let mut a: Vec<BasicMetadataValueEnum> = vec![r.into()];
            a.extend(args.iter().map(|v| BasicMetadataValueEnum::from(*v)));
            let call = match jit_ptr {
                Some(jit_ptr) => {
                    let ty = self.helper_type(1 + args.len());
                    let ptr = self
                        .b
                        .build_int_to_ptr(jit_ptr, self.ptrt(), "jp")
                        .map_err(|e| e.to_string())?;
                    self.b
                        .build_indirect_call(ty, ptr, &a, "direct")
                        .map_err(|e| e.to_string())?
                }
                None => self
                    .b
                    .build_call(self.sh.main_fn, &a, "self")
                    .map_err(|e| e.to_string())?,
            };
            let fv = call.try_as_basic_value().basic().unwrap().into_int_value();
            self.b
                .build_store(depth_p, depth)
                .map_err(|e| e.to_string())?;
            Ok(Some((fv, call_bb)))
        }

        fn known_call_nocheck(
            &mut self,
            func_id: u32,
            method: crate::intern::SymbolId,
            r: IntValue<'ctx>,
            args: &[IntValue<'ctx>],
        ) -> Result<IntValue<'ctx>, String> {
            let packed = (func_id as u64) | ((method.index() as u64) << 32);
            let name = format!("wren_known_call_{}_nocheck", args.len());
            let mut a = vec![self.c64(packed), r];
            a.extend_from_slice(args);
            self.call_helper(&name, &a)
        }

        /// Inline a single-block callee body; `None` when the body cannot
        /// be inlined here.
        fn inline_body(
            &mut self,
            callee: &Arc<MirFunction>,
            r: IntValue<'ctx>,
            class: usize,
            args: &[IntValue<'ctx>],
        ) -> Result<Option<IntValue<'ctx>>, String> {
            let block = &callee.blocks[0];
            let mut callee_args = vec![r];
            callee_args.extend_from_slice(args);
            let saved_class = self.inline_class.replace((r, class));
            let saved_vals = std::mem::take(&mut self.vals);
            let saved_bools = std::mem::take(&mut self.raw_bools);
            let saved_types =
                std::mem::replace(&mut self.value_types, infer_osr_value_types(callee));
            let saved_recv = self.receiver.replace(r);
            self.inline_depth += 1;
            let mut result: Result<Option<IntValue<'ctx>>, String> = Ok(None);
            let mut failed = false;
            for (vid, inst) in &block.instructions {
                match inst {
                    Instruction::BlockParam(idx) => match callee_args.get(*idx as usize) {
                        Some(v) => {
                            self.vals.insert(*vid, (*v).into());
                        }
                        None => {
                            failed = true;
                            break;
                        }
                    },
                    _ => match self.lower_instruction(*vid, inst) {
                        Ok(Some(v)) => {
                            self.vals.insert(*vid, v);
                            if is_raw_bool(inst) {
                                self.raw_bools.insert(*vid);
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    },
                }
            }
            if result.is_ok() && !failed {
                result = match &block.terminator {
                    Terminator::Return(v) => self.boxed(v).map(Some),
                    Terminator::ReturnNull => Ok(Some(self.c64(TAG_NULL))),
                    _ => Ok(None),
                };
            }
            self.inline_depth -= 1;
            self.inline_class = saved_class;
            self.receiver = saved_recv;
            self.value_types = saved_types;
            self.raw_bools = saved_bools;
            self.vals = saved_vals;
            result
        }

        fn lower_call(
            &mut self,
            receiver: &ValueId,
            method: crate::intern::SymbolId,
            args: &[ValueId],
        ) -> Result<IntValue<'ctx>, String> {
            let r = self.boxed(receiver)?;
            let mut arg_vals = Vec::with_capacity(args.len());
            for a in args {
                arg_vals.push(self.boxed(a)?);
            }
            if args.len() > 8 {
                let m = self.c64(method.index() as u64);
                return self.wren_call(r, m, &arg_vals);
            }
            if args.len() == 1 && Some(method) == self.sh.iterate_sym {
                let ic_idx = self.take_ic_idx();
                return self.list_iterate(r, arg_vals[0], method, ic_idx);
            }
            if args.len() == 1 && Some(method) == self.sh.iter_value_sym {
                let ic_idx = self.take_ic_idx();
                return self.list_iterator_value(r, arg_vals[0], method, ic_idx);
            }
            if args.len() == 1 && Some(method) == self.sh.add_sym {
                let ic_idx = self.take_ic_idx();
                return self.list_add(r, arg_vals[0], method, ic_idx);
            }
            let ic_idx = self.take_ic_idx();

            // Class-hierarchy devirtualisation: one guarded direct call
            // per known implementation.
            if let Some(cha) = self.sh.cha_by_method
                && args.len() <= 4
            {
                let impls: Vec<crate::runtime::engine::ChaImpl> =
                    cha.get(&method).cloned().unwrap_or_default();
                if !impls.is_empty() {
                    let merge = self.new_block("cham");
                    let slow = self.new_block("chas");
                    let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> = Vec::new();
                    let (is_obj, _, recv_class) = self.class_of(r)?;
                    for imp in &impls {
                        let (class_ptr, fid) = (&imp.class, &imp.fid);
                        let next = self.new_block("chan");
                        let fast = self.new_block("chaf");
                        let same =
                            self.icmp(IntPredicate::EQ, recv_class, self.c64(*class_ptr as u64))?;
                        let hit = self
                            .b
                            .build_and(is_obj, same, "hit")
                            .map_err(|e| e.to_string())?;
                        self.cbr(hit, fast, next)?;
                        self.b.position_at_end(fast);
                        let body = self.sh.inline_bodies.and_then(|b| b.get(fid)).cloned();
                        let mut done = false;
                        if let Some(callee) = body
                            && let Some(v) = self.inline_body(&callee, r, *class_ptr, &arg_vals)?
                        {
                            incoming.push((v.into(), self.b.get_insert_block().unwrap()));
                            self.br(merge)?;
                            done = true;
                        }
                        if !done && imp.direct && args.len() <= 4 && self.sh.jit_code_base.is_some()
                        {
                            // Straight through the slot when the callee
                            // is compiled; the helper otherwise.
                            let helper = self.new_block("chah");
                            let (v, end) = self
                                .slot_call(*fid, r, &arg_vals, helper)?
                                .expect("a slot table");
                            incoming.push((v.into(), end));
                            self.br(merge)?;
                            self.b.position_at_end(helper);
                        }
                        if !done {
                            if args.len() <= 3 {
                                let v = self.known_call_nocheck(*fid, method, r, &arg_vals)?;
                                incoming.push((v.into(), self.b.get_insert_block().unwrap()));
                                self.br(merge)?;
                            } else {
                                self.br(next)?;
                            }
                        }
                        self.b.position_at_end(next);
                    }
                    self.br(slow)?;
                    self.b.position_at_end(slow);
                    let m = self.c64(method.index() as u64);
                    let sv = self.wren_call(r, m, &arg_vals)?;
                    incoming.push((sv.into(), self.b.get_insert_block().unwrap()));
                    self.br(merge)?;
                    self.b.position_at_end(merge);
                    return Ok(self.phi(self.i64t().into(), &incoming)?.into_int_value());
                }
            }

            // Monomorphic getter from the inline cache.
            let ic = ic_idx.and_then(|i| self.sh.callsite_ic_ptrs.and_then(|ics| ics.get(i)));
            let _ = self.sh.callsite_ic_live_ptrs;
            if let Some(ic) = ic {
                // A constructor on a resolved class: allocate and run the
                // initialiser directly when the receiver is that class.
                if ic.kind == 3 && ic.class != 0 && ic.func_id != 0 && args.len() <= 3 {
                    let fast = self.new_block("ctf");
                    let slow = self.new_block("cts");
                    let merge = self.new_block("ctm");
                    let hit =
                        self.icmp(IntPredicate::EQ, r, self.c64(TAG_OBJ | ic.class as u64))?;
                    self.cbr(hit, fast, slow)?;
                    self.b.position_at_end(fast);
                    let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> = Vec::new();
                    let helper = self.new_block("cth");
                    // The initialiser is called straight through its slot
                    // when it is compiled; the helper otherwise.
                    match (direct_calls_enabled(), self.sh.jit_code_base) {
                        (true, Some(base)) => {
                            let slot_addr = unsafe { base.add(ic.func_id as usize) } as u64;
                            let jit_ptr = self.load64(self.c64(slot_addr), 0)?;
                            let has = self.icmp(IntPredicate::NE, jit_ptr, self.c64(0))?;
                            let depth_bb = self.new_block("ctd");
                            let call_bb = self.new_block("ctc");
                            self.cbr(has, depth_bb, helper)?;
                            self.b.position_at_end(depth_bb);
                            let (depth_p, depth) = self.direct_depth()?;
                            let room = self.icmp(
                                IntPredicate::ULT,
                                depth,
                                self.sh.ctx.i32_type().const_int(
                                    crate::codegen::runtime_fns::MAX_JIT_DEPTH as u64,
                                    false,
                                ),
                            )?;
                            self.cbr(room, call_bb, helper)?;
                            self.b.position_at_end(call_bb);
                            let inst = self.alloc_instance(r, None)?;
                            let deeper = self
                                .b
                                .build_int_add(
                                    depth,
                                    self.sh.ctx.i32_type().const_int(1, false),
                                    "d1",
                                )
                                .map_err(|e| e.to_string())?;
                            self.b
                                .build_store(depth_p, deeper)
                                .map_err(|e| e.to_string())?;
                            let ty = self.helper_type(1 + args.len());
                            let ptr = self
                                .b
                                .build_int_to_ptr(jit_ptr, self.ptrt(), "cp")
                                .map_err(|e| e.to_string())?;
                            let mut a: Vec<BasicMetadataValueEnum> = vec![inst.into()];
                            a.extend(arg_vals.iter().map(|v| BasicMetadataValueEnum::from(*v)));
                            self.b
                                .build_indirect_call(ty, ptr, &a, "init")
                                .map_err(|e| e.to_string())?;
                            self.b
                                .build_store(depth_p, depth)
                                .map_err(|e| e.to_string())?;
                            incoming.push((inst.into(), self.b.get_insert_block().unwrap()));
                            self.br(merge)?;
                        }
                        _ => self.br(helper)?,
                    }
                    self.b.position_at_end(helper);
                    let packed = self.c64(ic.func_id | ((method.index() as u64) << 32));
                    let name = [
                        "wren_construct_0",
                        "wren_construct_1",
                        "wren_construct_2",
                        "wren_construct_3",
                    ][args.len()];
                    let mut call_args = vec![packed, r];
                    call_args.extend(arg_vals.iter().copied());
                    let fv = self.call_helper(name, &call_args)?;
                    incoming.push((fv.into(), self.b.get_insert_block().unwrap()));
                    self.br(merge)?;
                    self.b.position_at_end(slow);
                    let m = self.method_bits(method, ic_idx);
                    let sv = self.wren_call(r, m, &arg_vals)?;
                    incoming.push((sv.into(), self.b.get_insert_block().unwrap()));
                    self.br(merge)?;
                    self.b.position_at_end(merge);
                    return Ok(self.phi(self.i64t().into(), &incoming)?.into_int_value());
                }
                if ic.kind == 5 && ic.class != 0 {
                    let fast = self.new_block("icf");
                    let (hit, fields) = self.instance_check(r, ic.class as u64)?;
                    if let Some(miss) = self.miss_exit_block()? {
                        self.cbr(hit, fast, miss)?;
                        self.b.position_at_end(fast);
                        let dst = self.cur_vid;
                        self.note_field_invariant(ic.class, ic.func_id as u16, dst);
                        return self.field_load(fields, ic.func_id as u16);
                    }
                    let slow = self.new_block("ics");
                    let merge = self.new_block("icm");
                    self.cbr(hit, fast, slow)?;
                    self.b.position_at_end(fast);
                    let fv = self.field_load(fields, ic.func_id as u16)?;
                    self.br(merge)?;
                    self.b.position_at_end(slow);
                    let m = self.method_bits(method, ic_idx);
                    let sv = self.wren_call(r, m, &arg_vals)?;
                    let slow_end = self.b.get_insert_block().unwrap();
                    self.br(merge)?;
                    self.b.position_at_end(merge);
                    return Ok(self
                        .phi(
                            self.i64t().into(),
                            &[(fv.into(), fast), (sv.into(), slow_end)],
                        )?
                        .into_int_value());
                }
            }
            let m = self.method_bits(method, ic_idx);
            self.wren_call(r, m, &arg_vals)
        }

        /// The inline-cache entry of the call being lowered: only a
        /// call of this body's own MIR has one.
        fn take_ic_idx(&mut self) -> Option<usize> {
            if self.inline_depth == 0 {
                self.sh.mir.ic_sites.get(&self.cur_vid).map(|i| *i as usize)
            } else {
                None
            }
        }

        /// Branch to `slow` unless `r` is a List; on the returned block
        /// `(obj_ptr, count)` are ready.
        fn list_probe(
            &mut self,
            r: IntValue<'ctx>,
            slow: BasicBlock<'ctx>,
        ) -> Result<(IntValue<'ctx>, IntValue<'ctx>), String> {
            let high = self.and(r, self.c64(TAG_OBJ))?;
            let is_obj = self.icmp(IntPredicate::EQ, high, self.c64(TAG_OBJ))?;
            let obj_bb = self.new_block("lo");
            self.cbr(is_obj, obj_bb, slow)?;
            self.b.position_at_end(obj_bb);
            let obj = self.and(r, self.c64(PTR_MASK))?;
            let ty = self.load8(obj, HEADER_OBJ_TYPE as i64)?;
            let is_list = self.icmp(
                IntPredicate::EQ,
                ty,
                self.c64(crate::runtime::object::ObjType::List as u64),
            )?;
            let list_bb = self.new_block("ll");
            self.cbr(is_list, list_bb, slow)?;
            self.b.position_at_end(list_bb);
            let count_p = self.addr(obj, LIST_COUNT as i64)?;
            let count32 = self
                .b
                .build_load(self.sh.ctx.i32_type(), count_p, "count")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let count = self
                .b
                .build_int_z_extend(count32, self.i64t(), "count64")
                .map_err(|e| e.to_string())?;
            Ok((obj, count))
        }

        /// `list.iterate(i)`: null starts at 0, a Num steps by one, and
        /// the end of the list is `false`; anything else takes the call.
        fn list_iterate(
            &mut self,
            r: IntValue<'ctx>,
            iter: IntValue<'ctx>,
            method: crate::intern::SymbolId,
            ic_idx: Option<usize>,
        ) -> Result<IntValue<'ctx>, String> {
            let exit = self.miss_exit_block()?;
            let slow = exit.unwrap_or_else(|| self.new_block("its"));
            let merge = self.new_block("itm");
            let (_, count) = self.list_probe(r, slow)?;
            let countf = self
                .b
                .build_unsigned_int_to_float(count, self.f64t(), "countf")
                .map_err(|e| e.to_string())?;
            let is_null = self.icmp(IntPredicate::EQ, iter, self.c64(TAG_NULL))?;
            let is_box = self.is_nan_boxed(iter)?;
            let num_bb = self.new_block("itn");
            let null_bb = self.new_block("it0");
            // A non-null box that is not a Num is not an iterator we
            // know; the call answers.
            let step_bb = self.new_block("itp");
            let test_bb = self.b.get_insert_block().unwrap();
            self.cbr(is_null, null_bb, num_bb)?;
            self.b.position_at_end(num_bb);
            self.cbr(is_box, slow, step_bb)?;
            self.b.position_at_end(step_bb);
            let f = self.f64_of(iter)?;
            let next = self
                .b
                .build_float_add(f, self.cf64(1.0), "next")
                .map_err(|e| e.to_string())?;
            self.br(null_bb)?;
            self.b.position_at_end(null_bb);
            let cand = self
                .phi(
                    self.f64t().into(),
                    &[(self.cf64(0.0).into(), test_bb), (next.into(), step_bb)],
                )?
                .into_float_value();
            let in_range = self.fcmp(FloatPredicate::OLT, cand, countf)?;
            let bits = self.bits(cand)?;
            let fast = self
                .b
                .build_select(in_range, bits, self.c64(TAG_FALSE), "iter")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let fast_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            if exit.is_some() {
                self.b.position_at_end(merge);
                return Ok(fast);
            }
            self.b.position_at_end(slow);
            let m = self.method_bits(method, ic_idx);
            let sv = self.wren_call(r, m, &[iter])?;
            let slow_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(merge);
            Ok(self
                .phi(
                    self.i64t().into(),
                    &[(fast.into(), fast_end), (sv.into(), slow_end)],
                )?
                .into_int_value())
        }

        /// `list.add(v)` with room in the buffer: store and count; a
        /// full list or another receiver takes the call. Returns the
        /// list.
        fn list_add(
            &mut self,
            r: IntValue<'ctx>,
            v: IntValue<'ctx>,
            method: crate::intern::SymbolId,
            ic_idx: Option<usize>,
        ) -> Result<IntValue<'ctx>, String> {
            let slow = self.new_block("las");
            let merge = self.new_block("lam");
            let (obj, count) = self.list_probe(r, slow)?;
            let cap_p = self.addr(obj, LIST_CAPACITY as i64)?;
            let cap32 = self
                .b
                .build_load(self.sh.ctx.i32_type(), cap_p, "cap")
                .map_err(|e| e.to_string())?
                .into_int_value();
            let cap = self
                .b
                .build_int_z_extend(cap32, self.i64t(), "cap64")
                .map_err(|e| e.to_string())?;
            let room = self.icmp(IntPredicate::ULT, count, cap)?;
            let store_bb = self.new_block("lat");
            let grow_bb = self.new_block("lag");
            self.cbr(room, store_bb, grow_bb)?;
            // A full list grows through the helper, not the call.
            self.b.position_at_end(grow_bb);
            self.call_helper("wren_list_add", &[r, v])?;
            let grow_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(store_bb);
            let elements = self.load64(obj, LIST_ELEMENTS as i64)?;
            let p = self.element_addr(elements, count, 8)?;
            self.b.build_store(p, v).map_err(|e| e.to_string())?;
            let next = self
                .b
                .build_int_add(count, self.c64(1), "count1")
                .map_err(|e| e.to_string())?;
            let next32 = self
                .b
                .build_int_truncate(next, self.sh.ctx.i32_type(), "count32")
                .map_err(|e| e.to_string())?;
            let count_p = self.addr(obj, LIST_COUNT as i64)?;
            self.b
                .build_store(count_p, next32)
                .map_err(|e| e.to_string())?;
            let fast_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(slow);
            let m = self.method_bits(method, ic_idx);
            let sv = self.wren_call(r, m, &[v])?;
            let slow_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(merge);
            Ok(self
                .phi(
                    self.i64t().into(),
                    &[
                        (r.into(), fast_end),
                        (r.into(), grow_end),
                        (sv.into(), slow_end),
                    ],
                )?
                .into_int_value())
        }

        /// `list.iteratorValue(i)`: the element at a Num index within
        /// the count; anything else takes the call.
        fn list_iterator_value(
            &mut self,
            r: IntValue<'ctx>,
            iter: IntValue<'ctx>,
            method: crate::intern::SymbolId,
            ic_idx: Option<usize>,
        ) -> Result<IntValue<'ctx>, String> {
            let exit = self.miss_exit_block()?;
            let slow = exit.unwrap_or_else(|| self.new_block("ivs"));
            let merge = self.new_block("ivm");
            let (obj, count) = self.list_probe(r, slow)?;
            let is_box = self.is_nan_boxed(iter)?;
            let num_bb = self.new_block("ivn");
            self.cbr(is_box, slow, num_bb)?;
            self.b.position_at_end(num_bb);
            let f = self.f64_of(iter)?;
            let idx = self
                .b
                .build_float_to_signed_int(f, self.i64t(), "idx")
                .map_err(|e| e.to_string())?;
            let in_range = self.icmp(IntPredicate::ULT, idx, count)?;
            let load_bb = self.new_block("ivl");
            self.cbr(in_range, load_bb, slow)?;
            self.b.position_at_end(load_bb);
            let elements = self.load64(obj, LIST_ELEMENTS as i64)?;
            let p = self.element_addr(elements, idx, 8)?;
            let v = self
                .b
                .build_load(self.i64t(), p, "elem")
                .map_err(|e| e.to_string())?
                .into_int_value();
            self.br(merge)?;
            if exit.is_some() {
                self.b.position_at_end(merge);
                return Ok(v);
            }
            self.b.position_at_end(slow);
            let m = self.method_bits(method, ic_idx);
            let sv = self.wren_call(r, m, &[iter])?;
            let slow_end = self.b.get_insert_block().unwrap();
            self.br(merge)?;
            self.b.position_at_end(merge);
            Ok(self
                .phi(
                    self.i64t().into(),
                    &[(v.into(), load_bb), (sv.into(), slow_end)],
                )?
                .into_int_value())
        }

        #[allow(clippy::too_many_arguments)]
        fn lower_known_call(
            &mut self,
            func_id: u32,
            method: crate::intern::SymbolId,
            expected_class: usize,
            inline_getter_field: Option<u16>,
            direct: bool,
            receiver: &ValueId,
            args: &[ValueId],
        ) -> Result<IntValue<'ctx>, String> {
            let r = self.boxed(receiver)?;
            let mut arg_vals = Vec::with_capacity(args.len());
            for a in args {
                arg_vals.push(self.boxed(a)?);
            }
            let m = self.c64(method.index() as u64);

            if let Some(bodies) = self.sh.inline_bodies
                && expected_class != 0
                && args.len() <= 4
                && let Some(callee) = bodies.get(&func_id).cloned()
            {
                let fast = self.new_block("kif");
                let (hit, _) = self.instance_check(r, expected_class as u64)?;
                if let Some(miss) = self.miss_exit_block()? {
                    // A class miss leaves the function, so the
                    // inlined body needs no merge.
                    self.cbr(hit, fast, miss)?;
                    self.b.position_at_end(fast);
                    return match self.inline_body(&callee, r, expected_class, &arg_vals)? {
                        Some(v) => {
                            if let Some(field) = inline_getter_field {
                                let dst = self.cur_vid;
                                self.note_field_invariant(expected_class, field, dst);
                            }
                            Ok(v)
                        }
                        None => {
                            let slow = self.new_block("kis");
                            self.br(slow)?;
                            self.b.position_at_end(slow);
                            self.wren_call(r, m, &arg_vals)
                        }
                    };
                }
                let slow = self.new_block("kis");
                let merge = self.new_block("kim");
                self.cbr(hit, fast, slow)?;
                self.b.position_at_end(fast);
                let mut incoming: Vec<(BasicValueEnum<'ctx>, BasicBlock<'ctx>)> = Vec::new();
                match self.inline_body(&callee, r, expected_class, &arg_vals)? {
                    Some(v) => {
                        incoming.push((v.into(), self.b.get_insert_block().unwrap()));
                        self.br(merge)?;
                    }
                    None => self.br(slow)?,
                }
                self.b.position_at_end(slow);
                let sv = self.wren_call(r, m, &arg_vals)?;
                incoming.push((sv.into(), self.b.get_insert_block().unwrap()));
                self.br(merge)?;
                self.b.position_at_end(merge);
                return Ok(self.phi(self.i64t().into(), &incoming)?.into_int_value());
            }

            if direct_calls_enabled()
                && inline_getter_field.is_none()
                && direct
                && expected_class != 0
                && args.len() <= 4
                && self.sh.jit_code_base.is_some()
            {
                let fast = self.new_block("plf");
                let slow = self.new_block("pls");
                let merge = self.new_block("plm");
                let (hit, _) = self.instance_check(r, expected_class as u64)?;
                self.cbr(hit, fast, slow)?;
                self.b.position_at_end(fast);
                let (fv, call_bb) = self
                    .slot_call(func_id, r, &arg_vals, slow)?
                    .expect("a slot table");
                self.br(merge)?;
                self.b.position_at_end(slow);
                let sv = self.wren_call(r, m, &arg_vals)?;
                let slow_end = self.b.get_insert_block().unwrap();
                self.br(merge)?;
                self.b.position_at_end(merge);
                return Ok(self
                    .phi(
                        self.i64t().into(),
                        &[(fv.into(), call_bb), (sv.into(), slow_end)],
                    )?
                    .into_int_value());
            }

            if let Some(field) = inline_getter_field {
                let fast = self.new_block("gf");
                let (hit, fields) = self.instance_check(r, expected_class as u64)?;
                if let Some(miss) = self.miss_exit_block()? {
                    self.cbr(hit, fast, miss)?;
                    self.b.position_at_end(fast);
                    let dst = self.cur_vid;
                    self.note_field_invariant(expected_class, field, dst);
                    return self.field_load(fields, field);
                }
                let slow = self.new_block("gs");
                let merge = self.new_block("gm");
                self.cbr(hit, fast, slow)?;
                self.b.position_at_end(fast);
                let fv = self.field_load(fields, field)?;
                self.br(merge)?;
                self.b.position_at_end(slow);
                let sv = self.wren_call(r, m, &arg_vals)?;
                let slow_end = self.b.get_insert_block().unwrap();
                self.br(merge)?;
                self.b.position_at_end(merge);
                return Ok(self
                    .phi(
                        self.i64t().into(),
                        &[(fv.into(), fast), (sv.into(), slow_end)],
                    )?
                    .into_int_value());
            }

            if expected_class != 0 && args.len() <= 3 {
                let fast = self.new_block("kf");
                let slow = self.new_block("ks");
                let merge = self.new_block("km");
                let (hit, _) = self.instance_check(r, expected_class as u64)?;
                self.cbr(hit, fast, slow)?;
                self.b.position_at_end(fast);
                let fv = self.known_call_nocheck(func_id, method, r, &arg_vals)?;
                let fast_end = self.b.get_insert_block().unwrap();
                self.br(merge)?;
                self.b.position_at_end(slow);
                let sv = self.wren_call(r, m, &arg_vals)?;
                let slow_end = self.b.get_insert_block().unwrap();
                self.br(merge)?;
                self.b.position_at_end(merge);
                return Ok(self
                    .phi(
                        self.i64t().into(),
                        &[(fv.into(), fast_end), (sv.into(), slow_end)],
                    )?
                    .into_int_value());
            }

            if args.len() <= 3 {
                let packed = (func_id as u64) | ((method.index() as u64) << 32);
                let name = format!("wren_known_call_{}", args.len());
                let mut a = vec![self.c64(packed), r];
                a.extend_from_slice(&arg_vals);
                return self.call_helper(&name, &a);
            }
            self.wren_call(r, m, &arg_vals)
        }
    }

    enum BinOp {
        Add,
        Sub,
        Mul,
        Div,
        Rem,
        Cmp(FloatPredicate),
    }

    /// For each block, the receivers a dominating `ClassIs` guard
    /// proved: the guard's true edge is the only way into a block that
    /// dominates it.
    /// Every `Box(I64ToF64(i))` value, mapped to `i`.
    fn int_sources(mir: &MirFunction) -> HashMap<ValueId, ValueId> {
        let conv: HashMap<ValueId, ValueId> = mir
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|(v, inst)| match inst {
                Instruction::I64ToF64(i) => Some((*v, *i)),
                _ => None,
            })
            .collect();
        mir.blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|(v, inst)| match inst {
                Instruction::Box(f) => conv.get(f).map(|i| (*v, *i)),
                _ => None,
            })
            .collect()
    }

    /// Every copy's original value, through chains of copies.
    fn move_roots(mir: &MirFunction) -> HashMap<ValueId, ValueId> {
        let mut roots: HashMap<ValueId, ValueId> = HashMap::new();
        for b in &mir.blocks {
            for (v, inst) in &b.instructions {
                if let Instruction::Move(s) = inst {
                    let root = roots.get(s).copied().unwrap_or(*s);
                    roots.insert(*v, root);
                }
            }
        }
        roots
    }

    fn class_facts(
        mir: &MirFunction,
        roots: &HashMap<ValueId, ValueId>,
    ) -> HashMap<usize, Vec<(ValueId, usize)>> {
        use crate::mir::opt::licm::{compute_dominators, compute_rpo};
        let n = mir.blocks.len();
        let mut preds = vec![0usize; n];
        for b in &mir.blocks {
            for s in b.terminator.successors() {
                if let Some(c) = preds.get_mut(s.0 as usize) {
                    *c += 1;
                }
            }
        }
        let class_of: HashMap<ValueId, (ValueId, usize)> = mir
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|(v, inst)| match inst {
                Instruction::ClassIs(r, c) => Some((*v, (*r, *c))),
                _ => None,
            })
            .collect();
        let guards: Vec<(usize, ValueId, usize)> = mir
            .blocks
            .iter()
            .filter_map(|b| match &b.terminator {
                Terminator::CondBranch {
                    condition,
                    true_target,
                    ..
                } if preds.get(true_target.0 as usize) == Some(&1) => class_of
                    .get(condition)
                    .map(|&(r, c)| (true_target.0 as usize, r, c)),
                _ => None,
            })
            .collect();
        // An in-place guard holds for the rest of its block, which the
        // lowering adds as it passes it, and for every block the guard's
        // block dominates.
        let in_place: Vec<(usize, ValueId, usize)> = mir
            .blocks
            .iter()
            .flat_map(|b| {
                b.instructions
                    .iter()
                    .filter_map(move |(_, inst)| match inst {
                        Instruction::GuardClassAt { value, class, .. } => Some((
                            b.id.0 as usize,
                            roots.get(value).copied().unwrap_or(*value),
                            *class,
                        )),
                        _ => None,
                    })
            })
            .collect();
        let mut facts: HashMap<usize, Vec<(ValueId, usize)>> = HashMap::new();
        if guards.is_empty() && in_place.is_empty() {
            return facts;
        }
        let rpo = compute_rpo(mir);
        let idom = compute_dominators(mir, &rpo);
        for bi in 0..n {
            let mut d = bi;
            loop {
                for &(t, r, c) in &guards {
                    if t == d {
                        facts.entry(bi).or_default().push((r, c));
                    }
                }
                if d != bi {
                    for &(t, r, c) in &in_place {
                        if t == d {
                            facts.entry(bi).or_default().push((r, c));
                        }
                    }
                }
                let up = idom.get(d).copied().unwrap_or(usize::MAX);
                if up == usize::MAX || up == d {
                    break;
                }
                d = up;
            }
        }
        facts
    }

    fn is_raw_bool(inst: &Instruction) -> bool {
        matches!(
            inst,
            Instruction::CmpLtF64(..)
                | Instruction::CmpGtF64(..)
                | Instruction::CmpLeF64(..)
                | Instruction::CmpGeF64(..)
                | Instruction::ClassIs(..)
                | Instruction::ObjectIs(..)
                | Instruction::ClosureFnIs(..)
                | Instruction::CmpLtI64(..)
                | Instruction::CmpGtI64(..)
                | Instruction::CmpLeI64(..)
                | Instruction::CmpGeI64(..)
                | Instruction::IsNum(..)
        )
    }
}
