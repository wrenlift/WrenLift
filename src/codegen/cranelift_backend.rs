/// Cranelift-based JIT backend.
///
/// Translates MIR directly to Cranelift IR, bypassing the custom MachInst layer.
/// This provides correct register allocation and instruction encoding for x86_64
/// without the SCRATCH_GP / spill-slot conflicts of the hand-written emitter.
#[cfg(feature = "cranelift")]
pub mod cl {
    use crate::intern::Interner;
    use crate::mir::{
        osr_external_live_values, osr_reachable_blocks, osr_rematerializable_defs, BlockId,
        Instruction, MirFunction, MirType, Terminator, ValueId,
    };
    use crate::runtime::object_layout::*;
    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{AbiParam, BlockArg, Function, InstBuilder, MemFlags, Value};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_codegen::Context;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    use std::collections::{HashMap, HashSet};

    const QNAN: u64 = 0x7FFC_0000_0000_0000;
    const TAG_NULL: u64 = QNAN; // 0x7FFC_0000_0000_0000 — no extra bits
    const TAG_FALSE: u64 = QNAN | 1;
    const TAG_TRUE: u64 = QNAN | 2;
    // Note: QNAN | 3 = TAG_UNDEFINED (not null!)
    const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    /// Compiled output from the Cranelift backend.
    pub struct CraneliftCompiledCode {
        /// The JIT module (keeps executable memory alive).
        _module: JITModule,
        /// Callable function pointer.
        pub fn_ptr: *const u8,
        /// Optional compiled loop/header OSR entry points.
        pub osr_entries: Vec<crate::codegen::NativeOsrEntry>,
        /// Size of the generated code.
        pub code_size: usize,
    }

    // SAFETY: The JITModule's memory is self-contained and the fn_ptr
    // points into it. Safe to send across threads for installation.
    unsafe impl Send for CraneliftCompiledCode {}
    unsafe impl Sync for CraneliftCompiledCode {}

    /// Compile a MIR function to native code using Cranelift.
    pub fn compile_mir(
        mir: &MirFunction,
        interner: &Interner,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
    ) -> Result<CraneliftCompiledCode, String> {
        // 1. Create Cranelift ISA for the host
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;

        // Disable frame pointers for smaller/faster code
        flag_builder
            .set("preserve_frame_pointers", "false")
            .map_err(|e| format!("Failed to set preserve_frame_pointers: {}", e))?;

        // Disable probestack — macOS aarch64 inline probestack can cause
        // false SIGSEGV (interpreted as stack overflow by the Rust runtime).
        flag_builder
            .set("enable_probestack", "false")
            .map_err(|e| format!("Failed to set enable_probestack: {}", e))?;

        flag_builder
            .set("enable_verifier", "true")
            .map_err(|e| e.to_string())?;
        let isa = cranelift_native::builder()
            .map_err(|e| e.to_string())?
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        // 2. Create JIT module with runtime symbol resolution
        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register all runtime function symbols
        for (name, addr) in runtime_symbols() {
            jit_builder.symbol(name, addr as *const u8);
        }

        let mut module = JITModule::new(jit_builder);

        // 3. Build the function signature: all args are i64 (NaN-boxed values)
        // Use mir.arity (total params INCLUDING receiver) to match the caller's ABI.
        // BlockParam instructions may be fewer (dead receiver eliminated by DCE),
        // but the function must still accept all args the caller passes.
        let param_count = mir.arity as usize;

        // Check if this function is num-specialized (all params guarded as Num).
        // If so, create an inner f64→f64 version for direct recursive calls
        // to avoid the box/unbox roundtrip per recursion (~370ns → ~5ns).
        let has_num_guards = mir.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|(_, inst)| matches!(inst, Instruction::GuardNum(_)))
        });
        let has_self_calls = mir.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|(_, inst)| matches!(inst, Instruction::CallStaticSelf { .. }))
        });
        let use_f64_inner = has_num_guards && has_self_calls && param_count > 0;

        let mut sig = module.make_signature();
        for _ in 0..param_count {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));

        // 4. Declare and define the function
        let func_name = interner.resolve(mir.name);
        let safe_name = format!(
            "wlift_{}",
            func_name.replace(['(', ')', ',', ' ', '='], "_")
        );
        let func_id = module
            .declare_function(&safe_name, Linkage::Local, &sig)
            .map_err(|e| e.to_string())?;

        // Count actually-used params (BlockParam instructions in bb0) —
        // this may be fewer than arity (e.g., unused receiver after DCE).
        let used_param_count = mir.blocks[0]
            .instructions
            .iter()
            .filter(|(_, inst)| matches!(inst, Instruction::BlockParam(_)))
            .count();

        // If num-specialized, declare an inner f64→f64 function for recursion.
        // The outer i64→i64 wrapper does guard+unbox, calls inner, then boxes result.
        // The inner function only takes the USED params (typically just n, not receiver).
        let inner_func_id = if use_f64_inner {
            let inner_name = format!("{}_f64", safe_name);
            let mut inner_sig = module.make_signature();
            for _ in 0..used_param_count {
                inner_sig.params.push(AbiParam::new(types::F64));
            }
            inner_sig.returns.push(AbiParam::new(types::F64));
            let inner_id = module
                .declare_function(&inner_name, Linkage::Local, &inner_sig)
                .map_err(|e| e.to_string())?;
            Some((inner_id, inner_sig))
        } else {
            None
        };

        // 5. Lower MIR to Cranelift IR
        if std::env::var_os("WLIFT_CL_MIR").is_some() {
            eprintln!("=== CL MIR input for {} ===", safe_name);
            eprintln!("{}", mir.pretty_print(interner));
            eprintln!("=== end ===");
        }

        if let Some((inner_id, ref inner_sig)) = inner_func_id {
            // ── Build the INNER f64→f64 function (the hot recursive path) ──
            let mut inner_func = Function::with_name_signature(
                cranelift_codegen::ir::UserFuncName::user(0, inner_id.as_u32()),
                inner_sig.clone(),
            );
            {
                let mut fb_ctx = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut inner_func, &mut fb_ctx);
                lower_mir_impl(
                    mir,
                    interner,
                    &mut builder,
                    &mut module,
                    callsite_ic_ptrs,
                    None, // f64 inner functions don't use IC
                    None, // no jit_code_base for inner
                    Some(inner_id),
                    None,
                )?;
                builder.seal_all_blocks();
                builder.finalize();
            }
            if std::env::var_os("WLIFT_CL_IR").is_some() {
                eprintln!("=== Cranelift IR (inner f64) for {} ===", safe_name);
                eprintln!("{}", inner_func.display());
                eprintln!("=== end ===");
            }
            // Verify inner function before defining
            if let Err(errors) = cranelift_codegen::verify_function(&inner_func, module.isa()) {
                return Err(format!(
                    "Verifier errors in inner {}: {}",
                    safe_name, errors
                ));
            }
            let mut inner_ctx = Context::for_function(inner_func);
            module
                .define_function(inner_id, &mut inner_ctx)
                .map_err(|e| e.to_string())?;

            // ── Build the OUTER i64→i64 wrapper ──
            // unbox params → call inner → box result
            let mut func = Function::with_name_signature(
                cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
                sig,
            );
            {
                let mut fb_ctx = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);
                let entry = builder.create_block();
                builder.switch_to_block(entry);
                // Add i64 params
                for _ in 0..param_count {
                    builder.append_block_param(entry, types::I64);
                }
                let entry_params = builder.block_params(entry).to_vec();
                // Collect the BlockParam indices used by the MIR, then
                // unbox only those params to pass to the inner f64 function.
                let used_indices: Vec<usize> = mir.blocks[0]
                    .instructions
                    .iter()
                    .filter_map(|(_, inst)| {
                        if let Instruction::BlockParam(idx) = inst {
                            Some(*idx as usize)
                        } else {
                            None
                        }
                    })
                    .collect();
                let f64_args: Vec<Value> = used_indices
                    .iter()
                    .map(|&idx| {
                        builder
                            .ins()
                            .bitcast(types::F64, MemFlags::new(), entry_params[idx])
                    })
                    .collect();
                // Call inner
                let inner_ref = module.declare_func_in_func(inner_id, builder.func);
                let call = builder.ins().call(inner_ref, &f64_args);
                let f64_result = builder.inst_results(call)[0];
                // Box result back to i64
                let i64_result = builder
                    .ins()
                    .bitcast(types::I64, MemFlags::new(), f64_result);
                builder.ins().return_(&[i64_result]);
                builder.seal_all_blocks();
                builder.finalize();
            }
            if std::env::var_os("WLIFT_CL_IR").is_some() {
                eprintln!("=== Cranelift IR (wrapper) for {} ===", safe_name);
                eprintln!("{}", func.display());
                eprintln!("=== end ===");
            }
            let mut ctx = Context::for_function(func);
            module
                .define_function(func_id, &mut ctx)
                .map_err(|e| e.to_string())?;
            module.finalize_definitions().map_err(|e| e.to_string())?;
            let fn_ptr = module.get_finalized_function(func_id);
            let code_size = ctx.compiled_code().unwrap().code_info().total_size as usize;
            return Ok(CraneliftCompiledCode {
                _module: module,
                fn_ptr,
                osr_entries: Vec::new(),
                code_size,
            });
        }

        // Standard path (no f64 specialization)
        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig,
        );
        {
            let mut fb_ctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);

            lower_mir_to_cranelift(
                mir,
                interner,
                &mut builder,
                &mut module,
                callsite_ic_ptrs,
                callsite_ic_live_ptrs,
                jit_code_base,
            )?;

            builder.seal_all_blocks();
            builder.finalize();
        }

        // Dump Cranelift IR if requested
        if std::env::var_os("WLIFT_CL_IR").is_some() {
            eprintln!("=== Cranelift IR for {} ===", safe_name);
            eprintln!("{}", func.display());
            eprintln!("=== end ===");
        }

        // 6. Compile
        if std::env::var_os("WLIFT_CL_VERIFY").is_some() {
            if let Err(errs) = cranelift_codegen::verify_function(&func, module.isa()) {
                eprintln!(
                    "cl-verify: {} (FuncId u0:{}) failed:\n{}\nIR:\n{}",
                    safe_name,
                    func_id.as_u32(),
                    errs,
                    func.display()
                );
            }
        }
        let mut ctx = Context::for_function(func);
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        let osr_defs = if should_compile_osr_entries(mir, interner) {
            compile_osr_entries(
                mir,
                interner,
                &mut module,
                &safe_name,
                callsite_ic_ptrs,
                callsite_ic_live_ptrs,
                jit_code_base,
            )
        } else {
            Vec::new()
        };
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let fn_ptr = module.get_finalized_function(func_id);
        let code_size = ctx.compiled_code().unwrap().code_info().total_size as usize;
        let osr_entries = osr_defs
            .into_iter()
            .map(|def| crate::codegen::NativeOsrEntry {
                target_block: def.target_block,
                param_count: def.param_count,
                ptr: module.get_finalized_function(def.func_id),
            })
            .collect();

        Ok(CraneliftCompiledCode {
            _module: module,
            fn_ptr,
            osr_entries,
            code_size,
        })
    }

    struct PendingOsrDefinition {
        target_block: BlockId,
        param_count: u16,
        func_id: cranelift_module::FuncId,
    }

    #[derive(Clone)]
    struct OsrEntryLayout {
        target_block: BlockId,
        external_args: Vec<ValueId>,
        param_count: u16,
    }

    fn should_compile_osr_entries(mir: &MirFunction, interner: &Interner) -> bool {
        // Runtime OSR transfer covers top-level/module frames and now
        // method/closure frames reached from the interpreter. The per-block
        // `osr_entry_layout` analysis still rejects loops whose live-in layout
        // or reachable region is unsupported.
        if interner.resolve(mir.name) == "<module>" {
            return mir.arity == 0;
        }
        // Only compile OSR entries if this function has at least one backward
        // branch. Saves code bloat on straight-line methods.
        mir.blocks.iter().any(has_backward_successor)
    }

    fn compile_osr_entries(
        mir: &MirFunction,
        interner: &Interner,
        module: &mut JITModule,
        safe_name: &str,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
    ) -> Vec<PendingOsrDefinition> {
        let mut defs = Vec::new();
        for target_block in collect_osr_targets(mir) {
            let Some(layout) = osr_entry_layout(mir, target_block) else {
                if std::env::var_os("WLIFT_OSR_TRACE").is_some() {
                    eprintln!(
                        "osr-trace: skip {} bb{} unsupported live-in layout",
                        safe_name, target_block.0
                    );
                }
                continue;
            };

            let mut sig = module.make_signature();
            for _ in 0..layout.param_count {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));

            let osr_name = format!("{}_osr_bb{}", safe_name, target_block.0);
            let Ok(func_id) = module.declare_function(&osr_name, Linkage::Local, &sig) else {
                continue;
            };
            let mut func = Function::with_name_signature(
                cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
                sig,
            );
            let mut fb_ctx = FunctionBuilderContext::new();
            let lower_result = {
                let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);
                let result = lower_mir_impl(
                    mir,
                    interner,
                    &mut builder,
                    module,
                    callsite_ic_ptrs,
                    callsite_ic_live_ptrs,
                    jit_code_base,
                    None,
                    Some(layout.clone()),
                );
                if result.is_ok() {
                    builder.seal_all_blocks();
                    builder.finalize();
                }
                result
            };
            if lower_result.is_err() {
                if std::env::var_os("WLIFT_OSR_TRACE").is_some() {
                    eprintln!(
                        "osr-trace: skip {} bb{} lowering failed: {:?}",
                        safe_name,
                        target_block.0,
                        lower_result.err()
                    );
                }
                continue;
            }
            if let Err(errors) = cranelift_codegen::verify_function(&func, module.isa()) {
                if std::env::var_os("WLIFT_OSR_TRACE").is_some() {
                    eprintln!(
                        "osr-trace: skip {} bb{} verifier failed: {}",
                        safe_name, target_block.0, errors
                    );
                }
                continue;
            }
            let mut ctx = Context::for_function(func);
            if let Err(error) = module.define_function(func_id, &mut ctx) {
                if std::env::var_os("WLIFT_OSR_TRACE").is_some() {
                    eprintln!(
                        "osr-trace: skip {} bb{} define failed: {}",
                        safe_name, target_block.0, error
                    );
                }
                continue;
            }
            defs.push(PendingOsrDefinition {
                target_block,
                param_count: layout.param_count,
                func_id,
            });
        }
        defs
    }

    fn collect_osr_targets(mir: &MirFunction) -> Vec<BlockId> {
        let mut seen = HashSet::new();
        let mut targets = Vec::new();
        for block in &mir.blocks {
            for target in block.terminator.successors() {
                if target.0 <= block.id.0 && seen.insert(target) {
                    targets.push(target);
                }
            }
        }
        targets
    }

    fn has_backward_successor(block: &crate::mir::BasicBlock) -> bool {
        block
            .terminator
            .successors()
            .into_iter()
            .any(|target| target.0 <= block.id.0)
    }

    fn osr_entry_layout(mir: &MirFunction, target: BlockId) -> Option<OsrEntryLayout> {
        let target_idx = target.0 as usize;
        let target_block = mir.blocks.get(target_idx)?;
        if target_block
            .params
            .iter()
            .any(|(_, ty)| !matches!(ty, MirType::Value))
        {
            return None;
        }

        let value_types = infer_osr_value_types(mir);
        let external_args = osr_external_live_values(mir, target);
        if external_args.iter().any(|vid| {
            !matches!(
                value_types.get(vid.0 as usize).copied(),
                Some(MirType::Value)
            )
        }) {
            return None;
        }

        let param_count = external_args.len() + target_block.params.len();
        if param_count > 4 {
            return None;
        }

        // Use `mir::osr_reachable_blocks` — the same helper
        // `osr_external_live_values` uses — so the `defs` set this
        // function builds stays in lockstep with the `external_args`
        // it consumes. A local DFS that diverges in even one edge
        // case (e.g. missing bounds guard) lets a validity check
        // pass while lowering subsequently panics.
        let reachable = osr_reachable_blocks(mir, target);
        let mut defs = HashSet::new();
        for &idx in &reachable {
            let block = &mir.blocks[idx];
            for &(param, _) in &block.params {
                defs.insert(param);
            }
            for &(dst, _) in &block.instructions {
                defs.insert(dst);
            }
        }
        let rematerializable = osr_rematerializable_defs(mir, target);
        let external_arg_set: HashSet<ValueId> = external_args.iter().copied().collect();

        for &idx in &reachable {
            let block = &mir.blocks[idx];
            for (_, inst) in &block.instructions {
                if matches!(inst, Instruction::CallStaticSelf { .. }) {
                    return None;
                }
                for op in inst.operands() {
                    if !defs.contains(&op)
                        && !rematerializable.contains_key(&op)
                        && !external_arg_set.contains(&op)
                    {
                        return None;
                    }
                }
            }
            for op in block.terminator.operands() {
                if !defs.contains(&op)
                    && !rematerializable.contains_key(&op)
                    && !external_arg_set.contains(&op)
                {
                    return None;
                }
            }
        }

        Some(OsrEntryLayout {
            target_block: target,
            external_args,
            param_count: param_count as u16,
        })
    }

    fn infer_osr_value_types(mir: &MirFunction) -> Vec<MirType> {
        let mut value_types = vec![MirType::Void; mir.next_value as usize];
        for block in &mir.blocks {
            for &(value, ty) in &block.params {
                value_types[value.0 as usize] = ty;
            }
        }
        for block in &mir.blocks {
            for &(dst, ref inst) in &block.instructions {
                let ty = match inst {
                    Instruction::ConstNum(_)
                    | Instruction::ConstBool(_)
                    | Instruction::ConstNull
                    | Instruction::ConstString(_)
                    | Instruction::Add(..)
                    | Instruction::Sub(..)
                    | Instruction::Mul(..)
                    | Instruction::Div(..)
                    | Instruction::Mod(..)
                    | Instruction::Neg(..)
                    | Instruction::Box(_)
                    | Instruction::GetField(..)
                    | Instruction::GetStaticField(_)
                    | Instruction::GetModuleVar(_)
                    | Instruction::Call { .. }
                    | Instruction::CallKnownFunc { .. }
                    | Instruction::CallStaticSelf { .. }
                    | Instruction::SuperCall { .. }
                    | Instruction::MakeClosure { .. }
                    | Instruction::GetUpvalue(_)
                    | Instruction::MakeList(_)
                    | Instruction::MakeMap(_)
                    | Instruction::MakeRange(..)
                    | Instruction::StringConcat(_)
                    | Instruction::ToString(_)
                    | Instruction::SubscriptGet { .. }
                    | Instruction::BitAnd(..)
                    | Instruction::BitOr(..)
                    | Instruction::BitXor(..)
                    | Instruction::BitNot(_)
                    | Instruction::Shl(..)
                    | Instruction::Shr(..) => MirType::Value,
                    Instruction::ConstF64(_)
                    | Instruction::MathUnaryF64(..)
                    | Instruction::MathBinaryF64(..)
                    | Instruction::AddF64(..)
                    | Instruction::SubF64(..)
                    | Instruction::MulF64(..)
                    | Instruction::DivF64(..)
                    | Instruction::ModF64(..)
                    | Instruction::NegF64(_)
                    | Instruction::Unbox(_) => MirType::F64,
                    Instruction::ConstI64(_) => MirType::I64,
                    Instruction::CmpLt(..)
                    | Instruction::CmpGt(..)
                    | Instruction::CmpLe(..)
                    | Instruction::CmpGe(..)
                    | Instruction::CmpEq(..)
                    | Instruction::CmpNe(..)
                    | Instruction::CmpLtF64(..)
                    | Instruction::CmpGtF64(..)
                    | Instruction::CmpLeF64(..)
                    | Instruction::CmpGeF64(..)
                    | Instruction::Not(_)
                    | Instruction::IsType(..) => MirType::Bool,
                    Instruction::GuardNum(src)
                    | Instruction::GuardBool(src)
                    | Instruction::Move(src)
                    | Instruction::SetField(_, _, src)
                    | Instruction::SetStaticField(_, src)
                    | Instruction::SetModuleVar(_, src)
                    | Instruction::SetUpvalue(_, src) => value_types[src.0 as usize],
                    Instruction::GuardClass(src, _) | Instruction::GuardProtocol(src, _) => {
                        value_types[src.0 as usize]
                    }
                    Instruction::SubscriptSet { value, .. } => value_types[value.0 as usize],
                    Instruction::BlockParam(idx) => block
                        .params
                        .get(*idx as usize)
                        .map(|(_, ty)| *ty)
                        .unwrap_or(MirType::Value),
                };
                value_types[dst.0 as usize] = ty;
            }
        }
        value_types
    }

    fn emit_osr_external_constants(
        mir: &MirFunction,
        target: BlockId,
        builder: &mut FunctionBuilder,
        val_map: &mut HashMap<ValueId, Value>,
    ) -> Result<(), String> {
        for (vid, inst) in osr_rematerializable_defs(mir, target) {
            let value = match inst {
                Instruction::ConstNum(n) => builder.ins().iconst(types::I64, n.to_bits() as i64),
                Instruction::ConstBool(b) => {
                    let bits = if b { TAG_TRUE } else { TAG_FALSE } as i64;
                    builder.ins().iconst(types::I64, bits)
                }
                Instruction::ConstNull => builder.ins().iconst(types::I64, TAG_NULL as i64),
                Instruction::ConstF64(n) => builder.ins().f64const(n),
                Instruction::ConstI64(n) => builder.ins().iconst(types::I64, n),
                _ => return Err("non-rematerializable OSR external value".to_string()),
            };
            val_map.insert(vid, value);
        }
        Ok(())
    }

    /// Collect all runtime function name→address pairs for Cranelift symbol resolution.
    fn runtime_symbols() -> Vec<(&'static str, usize)> {
        let mut syms = Vec::new();
        // Iterate through all known runtime function names
        let names = [
            "wren_call_0",
            "wren_call_1",
            "wren_call_2",
            "wren_call_3",
            "wren_call_4",
            "wren_load_jit_ptr",
            "wren_known_call_0",
            "wren_known_call_1",
            "wren_known_call_2",
            "wren_known_call_3",
            "wren_known_call_0_nocheck",
            "wren_known_call_1_nocheck",
            "wren_known_call_2_nocheck",
            "wren_known_call_3_nocheck",
            "wren_ic_call_0",
            "wren_ic_call_1",
            "wren_ic_call_2",
            "wren_ic_call_3",
            "wren_super_call_0",
            "wren_super_call_1",
            "wren_super_call_2",
            "wren_super_call_3",
            "wren_super_call_4",
            "wren_make_list",
            "wren_make_list_1",
            "wren_make_list_2",
            "wren_make_list_3",
            "wren_make_list_4",
            "wren_list_add",
            "wren_make_map",
            "wren_map_set",
            "wren_make_range",
            "wren_make_closure_0",
            "wren_make_closure_1",
            "wren_make_closure_2",
            "wren_make_closure_3",
            "wren_make_closure_4",
            "wren_get_module_var",
            "wren_const_string",
            "wren_set_module_var",
            "wren_get_upvalue",
            "wren_set_upvalue",
            "wren_get_static_field",
            "wren_set_static_field",
            "wren_num_add",
            "wren_num_sub",
            "wren_num_mul",
            "wren_num_div",
            "wren_num_mod",
            "wren_num_neg",
            "wren_cmp_lt",
            "wren_cmp_gt",
            "wren_cmp_le",
            "wren_cmp_ge",
            "wren_cmp_eq",
            "wren_cmp_ne",
            "wren_not",
            "wren_is_truthy",
            "wren_write_barrier",
            "wren_string_concat",
            "wren_to_string",
            "wren_is_type",
            "wren_subscript_get",
            "wren_subscript_set",
            "wren_bit_and",
            "wren_bit_or",
            "wren_bit_xor",
            "wren_bit_not",
            "wren_bit_shl",
            "wren_bit_shr",
        ];

        for name in &names {
            if let Some(addr) = crate::codegen::runtime_fns::resolve(name) {
                syms.push((*name, addr));
            }
        }
        syms
    }

    /// Declare a runtime function in the Cranelift module and return its FuncRef.
    fn declare_runtime_fn(
        module: &mut JITModule,
        builder: &mut FunctionBuilder,
        name: &str,
        param_count: usize,
    ) -> Result<cranelift_codegen::ir::FuncRef, String> {
        let mut sig = module.make_signature();
        for _ in 0..param_count {
            sig.params.push(AbiParam::new(types::I64));
        }
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| e.to_string())?;
        let func_ref = module.declare_func_in_func(func_id, builder.func);
        Ok(func_ref)
    }

    /// Lower a MIR function into Cranelift IR using the FunctionBuilder.
    fn lower_mir_to_cranelift(
        mir: &MirFunction,
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
    ) -> Result<(), String> {
        lower_mir_impl(
            mir,
            interner,
            builder,
            module,
            callsite_ic_ptrs,
            callsite_ic_live_ptrs,
            jit_code_base,
            None,
            None,
        )
    }

    /// Inner lowering with optional f64 specialization.
    /// When `f64_self_id` is Some, this function is the f64→f64 inner version:
    /// - BlockParam types are f64 (not i64)
    /// - Unbox/Box of params/returns are no-ops
    /// - CallStaticSelf calls the inner function directly with f64 args
    #[allow(clippy::too_many_arguments)] // Lowering context is inherently wide (IC, OSR, f64 inner, ...).
    fn lower_mir_impl(
        mir: &MirFunction,
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
        f64_self_id: Option<cranelift_module::FuncId>,
        osr_entry: Option<OsrEntryLayout>,
    ) -> Result<(), String> {
        // Map MIR blocks to Cranelift blocks
        let mut block_map: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
        for (i, _) in mir.blocks.iter().enumerate() {
            let cl_block = builder.create_block();
            block_map.insert(BlockId(i as u32), cl_block);
        }

        // Map MIR values to Cranelift values.
        let mut val_map: HashMap<ValueId, Value> = HashMap::new();

        if let Some(ref layout) = osr_entry {
            let osr_entry = builder.create_block();
            builder.switch_to_block(osr_entry);
            for vid in &layout.external_args {
                let param = builder.append_block_param(osr_entry, types::I64);
                val_map.insert(*vid, param);
            }
            let target_block = &mir.blocks[layout.target_block.0 as usize];
            for (_, ty) in &target_block.params {
                let cl_type = match ty {
                    MirType::F64 => types::F64,
                    _ => types::I64,
                };
                builder.append_block_param(osr_entry, cl_type);
            }
            emit_osr_external_constants(mir, layout.target_block, builder, &mut val_map)?;
            let args: Vec<BlockArg> = builder
                .block_params(osr_entry)
                .iter()
                .skip(layout.external_args.len())
                .copied()
                .map(BlockArg::Value)
                .collect();
            builder.ins().jump(block_map[&layout.target_block], &args);
        }

        // Receiver (entry_params[0]) saved for CallStaticSelf
        let mut receiver_val: Option<Value> = None;

        // Track which MIR values are raw Cranelift booleans (i8) rather than
        // NaN-boxed TAG_TRUE/TAG_FALSE. Used to skip the expensive truthiness
        // check in CondBranch when the condition is a direct fcmp/icmp result.
        let mut raw_bools: std::collections::HashSet<ValueId> = std::collections::HashSet::new();

        // Cache for declared runtime functions
        let mut runtime_cache: HashMap<String, cranelift_codegen::ir::FuncRef> = HashMap::new();

        // Pre-compute per-block call site base index.
        // The IC table is indexed by sequential block order (bb0, bb1, ...),
        // but we process blocks in RPO order. Without this map, call_site_idx
        // would assign wrong IC entries to call sites in reordered blocks.
        let mut block_call_site_base: Vec<usize> = Vec::with_capacity(mir.blocks.len());
        {
            let mut running = 0usize;
            for blk in &mir.blocks {
                block_call_site_base.push(running);
                for (_, inst) in &blk.instructions {
                    if matches!(
                        inst,
                        Instruction::Call { .. } | Instruction::SuperCall { .. }
                    ) {
                        running += 1;
                    }
                }
            }
        }
        // Helper to get or declare a runtime function
        let mut get_runtime_fn = |module: &mut JITModule,
                                  builder: &mut FunctionBuilder,
                                  name: &str,
                                  param_count: usize|
         -> Result<cranelift_codegen::ir::FuncRef, String> {
            if let Some(&func_ref) = runtime_cache.get(name) {
                return Ok(func_ref);
            }
            let func_ref = declare_runtime_fn(module, builder, name, param_count)?;
            runtime_cache.insert(name.to_string(), func_ref);
            Ok(func_ref)
        };

        // Process blocks in reverse post-order (dominance order).
        // The MIR block array may have preheader blocks (bb4) listed after
        // loop bodies (bb2), but Cranelift requires values to be defined
        // before use. RPO guarantees dominators come first.
        let rpo = match osr_entry.as_ref() {
            Some(layout) => compute_rpo_from(mir, layout.target_block),
            None => compute_rpo(mir),
        };
        for &block_idx in &rpo {
            let block = &mir.blocks[block_idx];
            let bid = BlockId(block_idx as u32);
            let cl_block = block_map[&bid];
            builder.switch_to_block(cl_block);

            // Reset call_site_idx to the pre-computed base for this block.
            // This ensures IC entries are read from the correct sequential
            // position even though blocks are processed in RPO order.
            let mut call_site_idx = block_call_site_base[block_idx];

            // Add block parameters (from loop back-edges / CondBranch args)
            for (vid, ty) in &block.params {
                let cl_type = match ty {
                    MirType::F64 => types::F64,
                    _ => types::I64,
                };
                let param = builder.append_block_param(cl_block, cl_type);
                val_map.insert(*vid, param);
            }

            // For the entry block (first in RPO = bb0), map BlockParam
            // instructions to Cranelift's function parameters.
            // Cranelift adds signature params to the first switched-to block.
            // For the entry block, add function params as block params
            // THEN map BlockParam instructions to those params.
            if osr_entry.is_none() && block_idx == 0 {
                if f64_self_id.is_some() {
                    // f64 inner function: params are only the USED ones
                    // (sequential f64 params, no receiver).
                    let bp_count = block
                        .instructions
                        .iter()
                        .filter(|(_, inst)| matches!(inst, Instruction::BlockParam(_)))
                        .count();
                    for _ in 0..bp_count {
                        builder.append_block_param(cl_block, types::F64);
                    }
                    let entry_params = builder.block_params(cl_block).to_vec();
                    let mut param_idx = 0usize;
                    for &(vid, ref inst) in &block.instructions {
                        if matches!(inst, Instruction::BlockParam(_)) {
                            if param_idx < entry_params.len() {
                                val_map.insert(vid, entry_params[param_idx]);
                            }
                            param_idx += 1;
                        }
                    }
                } else {
                    // i64 path: add mir.arity params to match the caller ABI
                    // (includes receiver even if dead).
                    let arity = mir.arity as usize;
                    for _ in 0..arity {
                        builder.append_block_param(cl_block, types::I64);
                    }
                    let entry_params = builder.block_params(cl_block).to_vec();
                    if !entry_params.is_empty() {
                        receiver_val = Some(entry_params[0]);
                    }
                    // Map BlockParam(idx) → entry_params[idx]
                    for &(vid, ref inst) in &block.instructions {
                        if let Instruction::BlockParam(idx) = inst {
                            let idx = *idx as usize;
                            if idx < entry_params.len() {
                                val_map.insert(vid, entry_params[idx]);
                            }
                        }
                    }
                }
            }

            // Lower each instruction
            for &(vid, ref inst) in &block.instructions {
                // Track raw booleans from f64 comparisons
                let is_raw_bool = matches!(
                    inst,
                    Instruction::CmpLtF64(..)
                        | Instruction::CmpGtF64(..)
                        | Instruction::CmpLeF64(..)
                        | Instruction::CmpGeF64(..)
                );
                let result = lower_instruction(
                    inst,
                    mir,
                    interner,
                    builder,
                    module,
                    &val_map,
                    &mut get_runtime_fn,
                    callsite_ic_ptrs,
                    callsite_ic_live_ptrs,
                    jit_code_base,
                    &mut call_site_idx,
                    f64_self_id,
                    receiver_val,
                )?;
                if let Some(val) = result {
                    val_map.insert(vid, val);
                    if is_raw_bool {
                        raw_bools.insert(vid);
                    }
                }
            }

            // Lower terminator
            lower_terminator(&block.terminator, builder, &val_map, &block_map, &raw_bools)?;
        }

        Ok(())
    }

    /// Describes what the fast-path of an inline boxed binary operation does.
    enum InlineBinOp {
        /// f64 arithmetic: "fadd", "fsub", "fmul", "fdiv"
        Arith(&'static str),
        /// f64 comparison producing TAG_TRUE / TAG_FALSE
        Cmp(FloatCC),
    }

    /// Emit an inline NaN-box check with fast path for two boxed operands.
    ///
    /// Fast path: both operands are numbers → bitcast to f64, do the operation,
    ///            bitcast back (arith) or produce TAG_TRUE/TAG_FALSE (cmp).
    /// Slow path: call the runtime function.
    #[allow(clippy::type_complexity)] // Runtime-fn resolver closure: one-shot type used only here.
    fn emit_inline_boxed_binop(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        get_runtime_fn: &mut dyn FnMut(
            &mut JITModule,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        la: Value,
        lb: Value,
        op: InlineBinOp,
        slow_fn: &str,
    ) -> Result<Option<Value>, String> {
        let qnan = builder.ins().iconst(types::I64, QNAN as i64);

        let check_b_block = builder.create_block();
        let fast_block = builder.create_block();
        let slow_block = builder.create_block();
        let merge_block = builder.create_block();
        builder.append_block_param(merge_block, types::I64);

        // Check a: (a & QNAN) == QNAN means NOT a number → slow path
        let a_masked = builder.ins().band(la, qnan);
        let a_is_nan = builder.ins().icmp(IntCC::Equal, a_masked, qnan);
        builder
            .ins()
            .brif(a_is_nan, slow_block, &[], check_b_block, &[]);

        // Check b: (b & QNAN) == QNAN means NOT a number → slow path
        builder.switch_to_block(check_b_block);
        let b_masked = builder.ins().band(lb, qnan);
        let b_is_nan = builder.ins().icmp(IntCC::Equal, b_masked, qnan);
        builder
            .ins()
            .brif(b_is_nan, slow_block, &[], fast_block, &[]);

        // Fast path: bitcast to f64, do the operation, bitcast result back
        builder.switch_to_block(fast_block);
        let fa = builder.ins().bitcast(types::F64, MemFlags::new(), la);
        let fb = builder.ins().bitcast(types::F64, MemFlags::new(), lb);
        let iresult = match op {
            InlineBinOp::Arith(name) => {
                let fresult = match name {
                    "fadd" => builder.ins().fadd(fa, fb),
                    "fsub" => builder.ins().fsub(fa, fb),
                    "fmul" => builder.ins().fmul(fa, fb),
                    "fdiv" => builder.ins().fdiv(fa, fb),
                    _ => unreachable!(),
                };
                builder.ins().bitcast(types::I64, MemFlags::new(), fresult)
            }
            InlineBinOp::Cmp(cc) => {
                let cmp = builder.ins().fcmp(cc, fa, fb);
                let true_val = builder.ins().iconst(types::I64, TAG_TRUE as i64);
                let false_val = builder.ins().iconst(types::I64, TAG_FALSE as i64);
                builder.ins().select(cmp, true_val, false_val)
            }
        };
        builder.ins().jump(merge_block, &[BlockArg::Value(iresult)]);

        // Slow path: call runtime function
        builder.switch_to_block(slow_block);
        let f = get_runtime_fn(module, builder, slow_fn, 2)?;
        let call = builder.ins().call(f, &[la, lb]);
        let slow_result = builder.inst_results(call)[0];
        builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(slow_result)]);

        // Merge block: result from whichever path was taken
        builder.switch_to_block(merge_block);
        Ok(Some(builder.block_params(merge_block)[0]))
    }

    /// Lower a single MIR instruction to Cranelift IR.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)] // Instruction lowering threads builder/module/val-map/IC/JIT-code-base — wide by design.
    fn lower_instruction(
        inst: &Instruction,
        _mir: &MirFunction,
        _interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        val_map: &HashMap<ValueId, Value>,
        get_runtime_fn: &mut dyn FnMut(
            &mut JITModule,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
        call_site_idx: &mut usize,
        f64_self_id: Option<cranelift_module::FuncId>,
        receiver_val: Option<Value>,
    ) -> Result<Option<Value>, String> {
        // Investigation mode — convert undefined-value to a graceful
        // Err so the broker thread survives, letting other functions
        // keep JITing. Exposes a latent miscompile we're bisecting.
        let dummy: Value = builder.ins().iconst(types::I64, 0);
        let err_sink: std::cell::Cell<Option<ValueId>> = std::cell::Cell::new(None);
        let get = |vid: &ValueId| -> Value {
            match val_map.get(vid) {
                Some(v) => *v,
                None => {
                    if err_sink.get().is_none() {
                        err_sink.set(Some(*vid));
                    }
                    dummy
                }
            }
        };

        let result = match inst {
            // === Constants ===
            Instruction::ConstNum(n) => {
                let bits = n.to_bits() as i64;
                Ok(Some(builder.ins().iconst(types::I64, bits)))
            }
            Instruction::ConstBool(b) => {
                let bits = if *b { TAG_TRUE } else { TAG_FALSE } as i64;
                Ok(Some(builder.ins().iconst(types::I64, bits)))
            }
            Instruction::ConstNull => Ok(Some(builder.ins().iconst(types::I64, TAG_NULL as i64))),
            Instruction::ConstF64(n) => Ok(Some(builder.ins().f64const(*n))),
            Instruction::ConstI64(n) => Ok(Some(builder.ins().iconst(types::I64, *n))),

            Instruction::BlockParam(_) => {
                // Already handled when creating block params
                Ok(None)
            }

            Instruction::Move(src) => Ok(Some(get(src))),

            // === Boxed arithmetic → inline fast path + runtime slow path ===
            Instruction::Add(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Arith("fadd"),
                    "wren_num_add",
                )
            }
            Instruction::Sub(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Arith("fsub"),
                    "wren_num_sub",
                )
            }
            Instruction::Mul(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Arith("fmul"),
                    "wren_num_mul",
                )
            }
            Instruction::Div(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Arith("fdiv"),
                    "wren_num_div",
                )
            }
            Instruction::Mod(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_num_mod", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::Neg(a) => {
                let f = get_runtime_fn(module, builder, "wren_num_neg", 1)?;
                let result = builder.ins().call(f, &[get(a)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Boxed comparisons → inline fast path + runtime slow path ===
            Instruction::CmpLt(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Cmp(FloatCC::LessThan),
                    "wren_cmp_lt",
                )
            }
            Instruction::CmpGt(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Cmp(FloatCC::GreaterThan),
                    "wren_cmp_gt",
                )
            }
            Instruction::CmpLe(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Cmp(FloatCC::LessThanOrEqual),
                    "wren_cmp_le",
                )
            }
            Instruction::CmpGe(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Cmp(FloatCC::GreaterThanOrEqual),
                    "wren_cmp_ge",
                )
            }
            Instruction::CmpEq(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_cmp_eq", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::CmpNe(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_cmp_ne", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Logical ===
            Instruction::Not(a) => {
                // Inline: is_falsy(v) → TAG_TRUE, else → TAG_FALSE
                // falsy = (v == TAG_FALSE || v == TAG_NULL)
                let val = get(a);
                let tag_false = builder.ins().iconst(types::I64, TAG_FALSE as i64);
                let tag_null = builder.ins().iconst(types::I64, TAG_NULL as i64);
                let tag_true = builder.ins().iconst(types::I64, TAG_TRUE as i64);
                let is_false = builder.ins().icmp(IntCC::Equal, val, tag_false);
                let is_null = builder.ins().icmp(IntCC::Equal, val, tag_null);
                let is_falsy = builder.ins().bor(is_false, is_null);
                Ok(Some(builder.ins().select(is_falsy, tag_true, tag_false)))
            }

            // === Field access (inline GEP) ===
            Instruction::GetField(recv, idx) => {
                let recv_val = get(recv);
                // Extract obj pointer: recv & PTR_MASK
                let mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                let obj_ptr = builder.ins().band(recv_val, mask);
                // Load fields pointer: obj_ptr + INSTANCE_FIELDS
                let fields_ptr =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj_ptr, INSTANCE_FIELDS);
                // Load field value: fields_ptr + idx * VALUE_SIZE
                let offset = (*idx as i32) * VALUE_SIZE;
                let field_val =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), fields_ptr, offset);
                Ok(Some(field_val))
            }
            Instruction::SetField(recv, idx, val) => {
                let recv_val = get(recv);
                let store_val = get(val);
                // Extract obj pointer
                let mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                let obj_ptr = builder.ins().band(recv_val, mask);
                // Load fields pointer
                let fields_ptr =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj_ptr, INSTANCE_FIELDS);
                // Store field value
                let offset = (*idx as i32) * VALUE_SIZE;
                builder
                    .ins()
                    .store(MemFlags::trusted(), store_val, fields_ptr, offset);
                // Write barrier
                let wb = get_runtime_fn(module, builder, "wren_write_barrier", 2)?;
                let _result = builder.ins().call(wb, &[recv_val, store_val]);
                // SetField result is the stored value
                Ok(Some(store_val))
            }

            // === Module variables ===
            Instruction::GetModuleVar(idx) => {
                let f = get_runtime_fn(module, builder, "wren_get_module_var", 1)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SetModuleVar(idx, val) => {
                let f = get_runtime_fn(module, builder, "wren_set_module_var", 2)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val, get(val)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Method calls — inline IC fast path + wren_call_N slow path ===
            Instruction::Call {
                receiver,
                method,
                args,
                pure_call: _,
            } => {
                // wren_call_N helpers only exist up to arity 4. Calls with
                // more than 4 args silently truncated on the JIT path,
                // which corrupts callee parameters (e.g. Render_.new with
                // 7 args saw its final three become undefined). Bail out
                // so the function falls back to the interpreter until we
                // grow wren_call_5..wren_call_8 wrappers.
                if args.len() > 4 {
                    return Err(format!(
                        "Call with arity {} not supported by JIT (need wren_call_{})",
                        args.len(),
                        args.len()
                    ));
                }
                let r = get(receiver);
                let ic_idx = *call_site_idx;
                *call_site_idx += 1;

                // Try inline IC: emit class-check + fast path.
                // Kind=5 (getter): inline field load (class baked as constant).
                // Kind=1: currently only used for IC index encoding in slow path.
                let ic = callsite_ic_ptrs.and_then(|ics| ics.get(ic_idx));
                let _live_ptr = callsite_ic_live_ptrs.and_then(|ptrs| ptrs.get(ic_idx).copied());

                if let Some(ic) = ic {
                    // Only emit IC fast path for kind=5 (getter inline).
                    // Kind=1 uses the slow path with IC index encoding so
                    // dispatch_call_rooted can use cached method lookups.
                    if ic.kind == 5 && ic.class != 0 {
                        let fast_block = builder.create_block();
                        let slow_block = builder.create_block();
                        let merge_block = builder.create_block();
                        builder.append_block_param(merge_block, types::I64);

                        // Extract receiver class: recv & PTR_MASK → load class at offset 16
                        let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                        let obj_ptr = builder.ins().band(r, ptr_mask);
                        let recv_class = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            obj_ptr,
                            HEADER_CLASS,
                        );

                        // Kind=5 getter: class is baked as constant.
                        let cached_class = builder.ins().iconst(types::I64, ic.class as i64);
                        let class_match =
                            builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
                        builder
                            .ins()
                            .brif(class_match, fast_block, &[], slow_block, &[]);

                        // Fast path: inline field load (kind=5 only)
                        builder.switch_to_block(fast_block);
                        let field_idx = ic.func_id as i32;
                        let fields_ptr = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            obj_ptr,
                            INSTANCE_FIELDS,
                        );
                        let offset = field_idx * VALUE_SIZE;
                        let fast_result =
                            builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), fields_ptr, offset);
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(fast_result)]);

                        // Slow path: full dispatch via wren_call_N
                        builder.switch_to_block(slow_block);
                        let mut method_bits = method.index() as u64;
                        if std::env::var_os("WLIFT_ENABLE_JIT_CALLSITE_IC").is_some() {
                            method_bits |= ((ic_idx as u64) + 1) << 32;
                        }
                        let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                        let call_name = match args.len() {
                            0 => "wren_call_0",
                            1 => "wren_call_1",
                            2 => "wren_call_2",
                            3 => "wren_call_3",
                            _ => "wren_call_4",
                        };
                        let arg_count = 2 + args.len().min(4);
                        let f = get_runtime_fn(module, builder, call_name, arg_count)?;
                        let mut slow_args = vec![r, method_val];
                        for a in args.iter().take(4) {
                            slow_args.push(get(a));
                        }
                        let slow_call = builder.ins().call(f, &slow_args);
                        let slow_result = builder.inst_results(slow_call)[0];
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(slow_result)]);

                        // Merge
                        builder.switch_to_block(merge_block);
                        return Ok(Some(builder.block_params(merge_block)[0]));
                    }
                }

                // No IC or unsupported IC kind: full dispatch
                let mut method_bits = method.index() as u64;
                if std::env::var_os("WLIFT_ENABLE_JIT_CALLSITE_IC").is_some() {
                    method_bits |= ((ic_idx as u64) + 1) << 32;
                }
                let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                let call_name = match args.len() {
                    0 => "wren_call_0",
                    1 => "wren_call_1",
                    2 => "wren_call_2",
                    3 => "wren_call_3",
                    _ => "wren_call_4",
                };
                let arg_count = 2 + args.len().min(4);
                let f = get_runtime_fn(module, builder, call_name, arg_count)?;
                let mut call_args = vec![r, method_val];
                for a in args.iter().take(4) {
                    call_args.push(get(a));
                }
                let result = builder.ins().call(f, &call_args);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Direct known-function call (devirtualized) ===
            Instruction::CallKnownFunc {
                func_id,
                method,
                expected_class,
                inline_getter_field,
                pure_leaf,
                receiver,
                args,
            } => {
                // Same wren_call_N arity limit as Instruction::Call — the
                // slow-path fallback inside this branch also truncates.
                if args.len() > 4 {
                    return Err(format!(
                        "CallKnownFunc with arity {} not supported by JIT",
                        args.len()
                    ));
                }
                let r = get(receiver);

                // === Pure-leaf direct call (ZERO FFI) ===
                // Callee has no internal method calls, so no context
                // setup is needed. Emit: class check + load callee ptr
                // + call_indirect. The JIT code slot address is stable
                // because engine.jit_code doesn't reallocate post-load.
                // Skip if we have a getter-inline hint — that path is cheaper.
                if inline_getter_field.is_none()
                    && *pure_leaf
                    && *expected_class != 0
                    && args.len() <= 4
                {
                    if let Some(jit_base_ptr) = jit_code_base {
                        let fast_block = builder.create_block();
                        let slow_block = builder.create_block();
                        let merge_block = builder.create_block();
                        builder.append_block_param(merge_block, types::I64);

                        // Class check
                        let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                        let obj_ptr = builder.ins().band(r, ptr_mask);
                        let recv_class = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            obj_ptr,
                            HEADER_CLASS,
                        );
                        let cached_class = builder.ins().iconst(types::I64, *expected_class as i64);
                        let class_match =
                            builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
                        builder
                            .ins()
                            .brif(class_match, fast_block, &[], slow_block, &[]);

                        // Fast path: load the callee's JIT slot and call_indirect.
                        // slot_addr = jit_code_base + func_id * 8
                        builder.switch_to_block(fast_block);
                        let slot_addr = unsafe { jit_base_ptr.add(*func_id as usize) as i64 };
                        let slot_addr_val = builder.ins().iconst(types::I64, slot_addr);
                        let jit_ptr =
                            builder
                                .ins()
                                .load(types::I64, MemFlags::new(), slot_addr_val, 0);
                        // Guard: if slot is null (callee not yet compiled),
                        // fall to slow path.
                        let zero = builder.ins().iconst(types::I64, 0);
                        let has_jit = builder.ins().icmp(IntCC::NotEqual, jit_ptr, zero);
                        let pure_call_block = builder.create_block();
                        builder
                            .ins()
                            .brif(has_jit, pure_call_block, &[], slow_block, &[]);

                        builder.switch_to_block(pure_call_block);
                        // Direct call signature: (recv, args...) -> i64
                        let mut sig = module.make_signature();
                        sig.params.push(AbiParam::new(types::I64)); // recv
                        for _ in args.iter() {
                            sig.params.push(AbiParam::new(types::I64));
                        }
                        sig.returns.push(AbiParam::new(types::I64));
                        let sig_ref = builder.import_signature(sig);
                        let mut call_args = vec![r];
                        for a in args {
                            call_args.push(get(a));
                        }
                        let call = builder.ins().call_indirect(sig_ref, jit_ptr, &call_args);
                        let fast_result = builder.inst_results(call)[0];
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(fast_result)]);

                        // Slow path: wren_call_N full dispatch.
                        builder.switch_to_block(slow_block);
                        let method_bits = method.index() as u64;
                        let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                        let slow_name = match args.len() {
                            0 => "wren_call_0",
                            1 => "wren_call_1",
                            2 => "wren_call_2",
                            3 => "wren_call_3",
                            _ => "wren_call_4",
                        };
                        let slow_arg_count = 2 + args.len().min(4);
                        let slow_f = get_runtime_fn(module, builder, slow_name, slow_arg_count)?;
                        let mut slow_args = vec![r, method_val];
                        for a in args.iter().take(4) {
                            slow_args.push(get(a));
                        }
                        let slow_call = builder.ins().call(slow_f, &slow_args);
                        let slow_result = builder.inst_results(slow_call)[0];
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(slow_result)]);

                        builder.switch_to_block(merge_block);
                        return Ok(Some(builder.block_params(merge_block)[0]));
                    }
                }

                // === Trivial-getter inline path ===
                // If the callee is a trivial getter (one-instruction
                // GetField), inline the field load directly. Class check
                // guards against polymorphic misuse. Zero FFI — pure load.
                if let Some(field_idx) = inline_getter_field {
                    let fast_block = builder.create_block();
                    let slow_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);

                    let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                    let obj_ptr = builder.ins().band(r, ptr_mask);
                    let recv_class =
                        builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), obj_ptr, HEADER_CLASS);
                    let cached_class = builder.ins().iconst(types::I64, *expected_class as i64);
                    let class_match = builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
                    builder
                        .ins()
                        .brif(class_match, fast_block, &[], slow_block, &[]);

                    // Fast path: load fields_ptr then indexed field.
                    builder.switch_to_block(fast_block);
                    let fields_ptr = builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        obj_ptr,
                        INSTANCE_FIELDS,
                    );
                    let offset = (*field_idx as i32) * VALUE_SIZE;
                    let field_val =
                        builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), fields_ptr, offset);
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(field_val)]);

                    // Slow path: class mismatch → wren_call_N.
                    builder.switch_to_block(slow_block);
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let slow_name = match args.len() {
                        0 => "wren_call_0",
                        1 => "wren_call_1",
                        2 => "wren_call_2",
                        3 => "wren_call_3",
                        _ => "wren_call_4",
                    };
                    let slow_arg_count = 2 + args.len().min(4);
                    let slow_f = get_runtime_fn(module, builder, slow_name, slow_arg_count)?;
                    let mut slow_args = vec![r, method_val];
                    for a in args.iter().take(4) {
                        slow_args.push(get(a));
                    }
                    let slow_call = builder.ins().call(slow_f, &slow_args);
                    let slow_result = builder.inst_results(slow_call)[0];
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(slow_result)]);

                    builder.switch_to_block(merge_block);
                    return Ok(Some(builder.block_params(merge_block)[0]));
                }

                // Only inline-dispatch when we have a cached class pointer.
                // Otherwise fall back to the slow helper.
                if *expected_class != 0 && args.len() <= 4 {
                    let fast_block = builder.create_block();
                    let slow_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);

                    // Check receiver is an object (MSB of NaN-box set means obj).
                    // For simplicity we just extract obj_ptr and load class —
                    // if recv is not an object this reads garbage but the class
                    // comparison below will fail safely.
                    let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                    let obj_ptr = builder.ins().band(r, ptr_mask);
                    let recv_class =
                        builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), obj_ptr, HEADER_CLASS);
                    let cached_class = builder.ins().iconst(types::I64, *expected_class as i64);
                    let class_match = builder.ins().icmp(IntCC::Equal, recv_class, cached_class);

                    // Fast path: class matches — load jit_ptr and call direct.
                    // We still have to go through wren_known_call_N because it
                    // handles context setup and depth tracking. But at least we
                    // skipped the class check in Rust (saves ~15ns).
                    builder
                        .ins()
                        .brif(class_match, fast_block, &[], slow_block, &[]);

                    // Fast path: class matched — use _nocheck variant which
                    // skips the Rust-side class verification (we already did
                    // it inline). Still goes through Rust to set up context
                    // + depth tracking, but ~15ns faster than the checked
                    // version.
                    builder.switch_to_block(fast_block);
                    let packed = (*func_id as u64) | ((method.index() as u64) << 32);
                    let fid_val = builder.ins().iconst(types::I64, packed as i64);
                    let fast_name = match args.len() {
                        0 => "wren_known_call_0_nocheck",
                        1 => "wren_known_call_1_nocheck",
                        2 => "wren_known_call_2_nocheck",
                        _ => "wren_known_call_3_nocheck",
                    };
                    let fast_arg_count = 2 + args.len().min(3);
                    let fast_f = get_runtime_fn(module, builder, fast_name, fast_arg_count)?;
                    let mut fast_args = vec![fid_val, r];
                    for a in args.iter().take(3) {
                        fast_args.push(get(a));
                    }
                    let fast_call = builder.ins().call(fast_f, &fast_args);
                    let fast_result = builder.inst_results(fast_call)[0];
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(fast_result)]);

                    // Slow path: class mismatch → wren_call_N full dispatch.
                    builder.switch_to_block(slow_block);
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let slow_name = match args.len() {
                        0 => "wren_call_0",
                        1 => "wren_call_1",
                        2 => "wren_call_2",
                        3 => "wren_call_3",
                        _ => "wren_call_4",
                    };
                    let slow_arg_count = 2 + args.len().min(4);
                    let slow_f = get_runtime_fn(module, builder, slow_name, slow_arg_count)?;
                    let mut slow_args = vec![r, method_val];
                    for a in args.iter().take(4) {
                        slow_args.push(get(a));
                    }
                    let slow_call = builder.ins().call(slow_f, &slow_args);
                    let slow_result = builder.inst_results(slow_call)[0];
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(slow_result)]);

                    builder.switch_to_block(merge_block);
                    return Ok(Some(builder.block_params(merge_block)[0]));
                }

                // No cached class — just call the helper.
                let packed = (*func_id as u64) | ((method.index() as u64) << 32);
                let fid_val = builder.ins().iconst(types::I64, packed as i64);
                let call_name = match args.len() {
                    0 => "wren_known_call_0",
                    1 => "wren_known_call_1",
                    2 => "wren_known_call_2",
                    _ => "wren_known_call_3",
                };
                let arg_count = 2 + args.len().min(3);
                let f = get_runtime_fn(module, builder, call_name, arg_count)?;
                let mut call_args = vec![fid_val, r];
                for a in args.iter().take(3) {
                    call_args.push(get(a));
                }
                let result = builder.ins().call(f, &call_args);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Super calls ===
            Instruction::SuperCall { method, args } => {
                if args.len() > 4 {
                    return Err(format!(
                        "SuperCall with arity {} not supported by JIT",
                        args.len()
                    ));
                }
                let method_val = builder.ins().iconst(types::I64, method.index() as i64);
                let call_name = match args.len() {
                    0 => "wren_super_call_0",
                    1 => "wren_super_call_1",
                    2 => "wren_super_call_2",
                    3 => "wren_super_call_3",
                    _ => "wren_super_call_4",
                };
                let arg_count = 1 + args.len().min(4);
                let f = get_runtime_fn(module, builder, call_name, arg_count)?;

                let mut call_args = vec![method_val];
                for a in args.iter().take(4) {
                    call_args.push(get(a));
                }
                let result = builder.ins().call(f, &call_args);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Collections ===
            Instruction::MakeList(elems) => {
                if elems.len() <= 4 {
                    let name = match elems.len() {
                        0 => "wren_make_list",
                        1 => "wren_make_list_1",
                        2 => "wren_make_list_2",
                        3 => "wren_make_list_3",
                        _ => "wren_make_list_4",
                    };
                    let f = get_runtime_fn(module, builder, name, elems.len())?;
                    let args: Vec<Value> = elems.iter().map(&get).collect();
                    let result = builder.ins().call(f, &args);
                    Ok(Some(builder.inst_results(result)[0]))
                } else {
                    // >4 elements: create empty + add each
                    let f_make = get_runtime_fn(module, builder, "wren_make_list", 0)?;
                    let make_result = builder.ins().call(f_make, &[]);
                    let list = builder.inst_results(make_result)[0];

                    let f_add = get_runtime_fn(module, builder, "wren_list_add", 2)?;
                    for e in elems {
                        builder.ins().call(f_add, &[list, get(e)]);
                    }
                    Ok(Some(list))
                }
            }

            Instruction::MakeMap(pairs) => {
                let f_make = get_runtime_fn(module, builder, "wren_make_map", 0)?;
                let make_result = builder.ins().call(f_make, &[]);
                let map = builder.inst_results(make_result)[0];

                let f_set = get_runtime_fn(module, builder, "wren_map_set", 3)?;
                for (k, v) in pairs {
                    builder.ins().call(f_set, &[map, get(k), get(v)]);
                }
                Ok(Some(map))
            }

            Instruction::MakeRange(from, to, inclusive) => {
                let f = get_runtime_fn(module, builder, "wren_make_range", 3)?;
                let incl = builder
                    .ins()
                    .iconst(types::I64, if *inclusive { 1i64 } else { 0 });
                let result = builder.ins().call(f, &[get(from), get(to), incl]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === String operations ===
            Instruction::StringConcat(parts) => {
                let f = get_runtime_fn(module, builder, "wren_string_concat", 2)?;
                if parts.is_empty() {
                    let empty = builder.ins().iconst(types::I64, TAG_NULL as i64);
                    return Ok(Some(empty));
                }
                let mut result = get(&parts[0]);
                for p in &parts[1..] {
                    let call = builder.ins().call(f, &[result, get(p)]);
                    result = builder.inst_results(call)[0];
                }
                Ok(Some(result))
            }
            Instruction::ToString(a) => {
                let f = get_runtime_fn(module, builder, "wren_to_string", 1)?;
                let result = builder.ins().call(f, &[get(a)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Upvalues ===
            Instruction::GetUpvalue(idx) => {
                let f = get_runtime_fn(module, builder, "wren_get_upvalue", 1)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SetUpvalue(idx, val) => {
                let f = get_runtime_fn(module, builder, "wren_set_upvalue", 2)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val, get(val)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Static fields ===
            Instruction::GetStaticField(sym) => {
                let f = get_runtime_fn(module, builder, "wren_get_static_field", 1)?;
                let idx_val = builder.ins().iconst(types::I64, sym.index() as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SetStaticField(sym, val) => {
                let f = get_runtime_fn(module, builder, "wren_set_static_field", 2)?;
                let idx_val = builder.ins().iconst(types::I64, sym.index() as i64);
                let result = builder.ins().call(f, &[idx_val, get(val)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Closures ===
            Instruction::MakeClosure { fn_id, upvalues } => {
                let name = match upvalues.len() {
                    0 => "wren_make_closure_0",
                    1 => "wren_make_closure_1",
                    2 => "wren_make_closure_2",
                    3 => "wren_make_closure_3",
                    _ => "wren_make_closure_4",
                };
                let f = get_runtime_fn(module, builder, name, 1 + upvalues.len().min(4))?;
                let fn_id_val = builder.ins().iconst(types::I64, *fn_id as i64);
                let mut args = vec![fn_id_val];
                for uv in upvalues.iter().take(4) {
                    args.push(get(uv));
                }
                let result = builder.ins().call(f, &args);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Subscript operations ===
            //
            // Single-index subscripts get an inline fast path for
            // `ObjTypedArray` receivers: check the header's obj_type
            // byte against the TypedArray tag, bounds-check the
            // index, then dispatch on the kind byte to a direct
            // f32/f64/u8 load. The fast path costs two byte loads +
            // one compare on the way out of the receiver guard.
            //
            // The guard intentionally lives at EVERY single-index
            // subscript site — no compile-time receiver-class info
            // is required, so typed arrays passed in as params,
            // stored in fields, or returned from factories all hit
            // this path. The slow path is the pre-existing
            // `wren_subscript_get` runtime function, which already
            // handles List / Map / String / TypedArray correctly.
            Instruction::SubscriptGet { receiver, args } if args.len() == 1 => {
                let r = get(receiver);
                let idx = get(&args[0]);

                let after_is_obj = builder.create_block();
                let fast_block = builder.create_block();
                let in_bounds_block = builder.create_block();
                let check_f32_block = builder.create_block();
                let get_u8_block = builder.create_block();
                let get_f32_block = builder.create_block();
                let get_f64_block = builder.create_block();
                let slow_block = builder.create_block();
                let merge_block = builder.create_block();
                builder.append_block_param(merge_block, types::I64);

                // 1. Receiver must be an object-kind NaN-boxed
                //    value. Object values have their top 16 bits
                //    equal to 0xFFFC (QNAN | sign bit).
                let shr48 = builder.ins().ushr_imm(r, 48);
                let obj_tag = builder.ins().iconst(types::I64, 0xFFFC);
                let is_obj = builder.ins().icmp(IntCC::Equal, shr48, obj_tag);
                builder
                    .ins()
                    .brif(is_obj, after_is_obj, &[], slow_block, &[]);

                // 2. Unbox pointer, load obj_type byte, branch on
                //    TypedArray tag.
                builder.switch_to_block(after_is_obj);
                let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                let obj_ptr = builder.ins().band(r, ptr_mask);
                let obj_type_byte =
                    builder
                        .ins()
                        .uload8(types::I64, MemFlags::trusted(), obj_ptr, HEADER_OBJ_TYPE);
                let ta_tag = builder
                    .ins()
                    .iconst(types::I64, OBJ_TYPE_TYPED_ARRAY as i64);
                let is_ta = builder.ins().icmp(IntCC::Equal, obj_type_byte, ta_tag);
                builder.ins().brif(is_ta, fast_block, &[], slow_block, &[]);

                // 3. Fast path: convert NaN-boxed Num index to i64,
                //    bounds-check against element count. Negative
                //    indices (Wren convention: `-1` → last) fall to
                //    the slow path for simplicity — Wren's integer
                //    API returns the same values, just more slowly.
                builder.switch_to_block(fast_block);
                let idx_f = builder.ins().bitcast(types::F64, MemFlags::new(), idx);
                let idx_i = builder.ins().fcvt_to_sint(types::I64, idx_f);
                // `uload32` already zero-extends the 32-bit load into
                // i64 — no separate uextend required.
                let count = builder
                    .ins()
                    .uload32(MemFlags::trusted(), obj_ptr, TYPED_ARRAY_COUNT);
                let zero = builder.ins().iconst(types::I64, 0);
                let in_range_low = builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, idx_i, zero);
                let in_range_high = builder.ins().icmp(IntCC::SignedLessThan, idx_i, count);
                let in_range = builder.ins().band(in_range_low, in_range_high);
                builder
                    .ins()
                    .brif(in_range, in_bounds_block, &[], slow_block, &[]);

                // 4. In-bounds: load kind byte + data pointer,
                //    dispatch to the element-typed load.
                builder.switch_to_block(in_bounds_block);
                let data =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj_ptr, TYPED_ARRAY_DATA);
                let kind = builder.ins().uload8(
                    types::I64,
                    MemFlags::trusted(),
                    obj_ptr,
                    TYPED_ARRAY_KIND,
                );
                let k_u8_const = builder.ins().iconst(types::I64, TA_KIND_U8 as i64);
                let is_u8 = builder.ins().icmp(IntCC::Equal, kind, k_u8_const);
                builder
                    .ins()
                    .brif(is_u8, get_u8_block, &[], check_f32_block, &[]);
                builder.switch_to_block(check_f32_block);
                let k_f32_const = builder.ins().iconst(types::I64, TA_KIND_F32 as i64);
                let is_f32 = builder.ins().icmp(IntCC::Equal, kind, k_f32_const);
                builder
                    .ins()
                    .brif(is_f32, get_f32_block, &[], get_f64_block, &[]);

                // 5a. U8: byte load → f64 (unsigned convert) →
                //     NaN-box bits.
                builder.switch_to_block(get_u8_block);
                let u8_addr = builder.ins().iadd(data, idx_i);
                let byte_val = builder
                    .ins()
                    .uload8(types::I64, MemFlags::trusted(), u8_addr, 0);
                let byte_f64 = builder.ins().fcvt_from_uint(types::F64, byte_val);
                let byte_bits = builder.ins().bitcast(types::I64, MemFlags::new(), byte_f64);
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(byte_bits)]);

                // 5b. F32: 4-byte float load → f64 promote → box.
                builder.switch_to_block(get_f32_block);
                let four = builder.ins().iconst(types::I64, 4);
                let f32_offset = builder.ins().imul(idx_i, four);
                let f32_addr = builder.ins().iadd(data, f32_offset);
                let f32_val = builder
                    .ins()
                    .load(types::F32, MemFlags::trusted(), f32_addr, 0);
                let f32_as_f64 = builder.ins().fpromote(types::F64, f32_val);
                let f32_bits = builder
                    .ins()
                    .bitcast(types::I64, MemFlags::new(), f32_as_f64);
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(f32_bits)]);

                // 5c. F64: direct 8-byte float load → box.
                builder.switch_to_block(get_f64_block);
                let eight = builder.ins().iconst(types::I64, 8);
                let f64_offset = builder.ins().imul(idx_i, eight);
                let f64_addr = builder.ins().iadd(data, f64_offset);
                let f64_val = builder
                    .ins()
                    .load(types::F64, MemFlags::trusted(), f64_addr, 0);
                let f64_bits = builder.ins().bitcast(types::I64, MemFlags::new(), f64_val);
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(f64_bits)]);

                // 6. Slow path: existing runtime dispatch.
                builder.switch_to_block(slow_block);
                let slow_fn = get_runtime_fn(module, builder, "wren_subscript_get", 2)?;
                let slow_call = builder.ins().call(slow_fn, &[r, idx]);
                let slow_result = builder.inst_results(slow_call)[0];
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(slow_result)]);

                builder.switch_to_block(merge_block);
                Ok(Some(builder.block_params(merge_block)[0]))
            }
            Instruction::SubscriptGet { receiver, args } => {
                // Multi-index subscript: fall back to runtime call.
                let f = get_runtime_fn(module, builder, "wren_subscript_get", 1 + args.len())?;
                let mut call_args = vec![get(receiver)];
                for a in args {
                    call_args.push(get(a));
                }
                let result = builder.ins().call(f, &call_args);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SubscriptSet {
                receiver,
                args,
                value,
            } if args.len() == 1 => {
                // Mirror of the SubscriptGet inline fast path. Only
                // F32 and F64 writes are inlined — ByteArray writes
                // require 0..=255 integer validation which is
                // cheaper to leave in the slow path (also less
                // hot for the graphics / audio / physics use
                // cases that drive this whole optimization).
                let r = get(receiver);
                let idx = get(&args[0]);
                let val = get(value);

                let after_is_obj = builder.create_block();
                let fast_block = builder.create_block();
                let in_bounds_block = builder.create_block();
                let check_f32_block = builder.create_block();
                let set_f32_block = builder.create_block();
                let set_f64_block = builder.create_block();
                let slow_block = builder.create_block();
                let merge_block = builder.create_block();
                builder.append_block_param(merge_block, types::I64);

                // 1. Receiver must be an object (NaN-boxed pointer).
                let shr48 = builder.ins().ushr_imm(r, 48);
                let obj_tag = builder.ins().iconst(types::I64, 0xFFFC);
                let is_obj = builder.ins().icmp(IntCC::Equal, shr48, obj_tag);
                builder
                    .ins()
                    .brif(is_obj, after_is_obj, &[], slow_block, &[]);

                // 2. Obj_type must be TypedArray.
                builder.switch_to_block(after_is_obj);
                let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                let obj_ptr = builder.ins().band(r, ptr_mask);
                let obj_type_byte =
                    builder
                        .ins()
                        .uload8(types::I64, MemFlags::trusted(), obj_ptr, HEADER_OBJ_TYPE);
                let ta_tag = builder
                    .ins()
                    .iconst(types::I64, OBJ_TYPE_TYPED_ARRAY as i64);
                let is_ta = builder.ins().icmp(IntCC::Equal, obj_type_byte, ta_tag);
                builder.ins().brif(is_ta, fast_block, &[], slow_block, &[]);

                // 3. Index must be a Num in [0, count). Negative
                //    indices → slow path (preserves Wren semantics
                //    via the runtime helper).
                builder.switch_to_block(fast_block);
                let idx_f = builder.ins().bitcast(types::F64, MemFlags::new(), idx);
                let idx_i = builder.ins().fcvt_to_sint(types::I64, idx_f);
                // `uload32` already zero-extends the 32-bit load into
                // i64 — no separate uextend required.
                let count = builder
                    .ins()
                    .uload32(MemFlags::trusted(), obj_ptr, TYPED_ARRAY_COUNT);
                let zero = builder.ins().iconst(types::I64, 0);
                let in_range_low = builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThanOrEqual, idx_i, zero);
                let in_range_high = builder.ins().icmp(IntCC::SignedLessThan, idx_i, count);
                let in_range = builder.ins().band(in_range_low, in_range_high);
                builder
                    .ins()
                    .brif(in_range, in_bounds_block, &[], slow_block, &[]);

                // 4. Value must be a Num. `(value & QNAN) == QNAN`
                //    means a singleton or object — go slow. Real
                //    f64 NaN values ALSO fail this test (they'd be
                //    stored correctly, but the simpler rule keeps
                //    the fast path predictable).
                builder.switch_to_block(in_bounds_block);
                let qnan_const = builder.ins().iconst(types::I64, QNAN as i64);
                let val_masked = builder.ins().band(val, qnan_const);
                let val_is_non_num = builder.ins().icmp(IntCC::Equal, val_masked, qnan_const);
                let after_val_check = builder.create_block();
                builder
                    .ins()
                    .brif(val_is_non_num, slow_block, &[], after_val_check, &[]);

                // 5. Load kind, dispatch to the typed store. U8
                //    falls through to the slow path (range check).
                builder.switch_to_block(after_val_check);
                let data =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj_ptr, TYPED_ARRAY_DATA);
                let kind = builder.ins().uload8(
                    types::I64,
                    MemFlags::trusted(),
                    obj_ptr,
                    TYPED_ARRAY_KIND,
                );
                let k_f32_const = builder.ins().iconst(types::I64, TA_KIND_F32 as i64);
                let is_f32 = builder.ins().icmp(IntCC::Equal, kind, k_f32_const);
                builder
                    .ins()
                    .brif(is_f32, set_f32_block, &[], check_f32_block, &[]);
                builder.switch_to_block(check_f32_block);
                let k_f64_const = builder.ins().iconst(types::I64, TA_KIND_F64 as i64);
                let is_f64 = builder.ins().icmp(IntCC::Equal, kind, k_f64_const);
                builder
                    .ins()
                    .brif(is_f64, set_f64_block, &[], slow_block, &[]);

                // 5a. F32: demote f64 → f32 and store 4 bytes.
                builder.switch_to_block(set_f32_block);
                let val_f64 = builder.ins().bitcast(types::F64, MemFlags::new(), val);
                let val_f32 = builder.ins().fdemote(types::F32, val_f64);
                let four = builder.ins().iconst(types::I64, 4);
                let f32_offset = builder.ins().imul(idx_i, four);
                let f32_addr = builder.ins().iadd(data, f32_offset);
                builder
                    .ins()
                    .store(MemFlags::trusted(), val_f32, f32_addr, 0);
                builder.ins().jump(merge_block, &[BlockArg::Value(val)]);

                // 5b. F64: store 8 bytes directly.
                builder.switch_to_block(set_f64_block);
                let val_f64b = builder.ins().bitcast(types::F64, MemFlags::new(), val);
                let eight = builder.ins().iconst(types::I64, 8);
                let f64_offset = builder.ins().imul(idx_i, eight);
                let f64_addr = builder.ins().iadd(data, f64_offset);
                builder
                    .ins()
                    .store(MemFlags::trusted(), val_f64b, f64_addr, 0);
                builder.ins().jump(merge_block, &[BlockArg::Value(val)]);

                // 6. Slow path: runtime handles byte writes +
                //    validation + anything non-TypedArray.
                builder.switch_to_block(slow_block);
                let slow_fn = get_runtime_fn(module, builder, "wren_subscript_set", 3)?;
                let slow_call = builder.ins().call(slow_fn, &[r, idx, val]);
                let slow_result = builder.inst_results(slow_call)[0];
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(slow_result)]);

                builder.switch_to_block(merge_block);
                Ok(Some(builder.block_params(merge_block)[0]))
            }
            Instruction::SubscriptSet {
                receiver,
                args,
                value,
            } => {
                let f = get_runtime_fn(module, builder, "wren_subscript_set", 2 + args.len())?;
                let mut call_args = vec![get(receiver)];
                for a in args {
                    call_args.push(get(a));
                }
                call_args.push(get(value));
                let result = builder.ins().call(f, &call_args);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Bitwise ===
            Instruction::BitAnd(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_bit_and", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::BitOr(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_bit_or", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::BitXor(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_bit_xor", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::BitNot(a) => {
                let f = get_runtime_fn(module, builder, "wren_bit_not", 1)?;
                let result = builder.ins().call(f, &[get(a)]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::Shl(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_bit_shl", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::Shr(a, b) => {
                let f = get_runtime_fn(module, builder, "wren_bit_shr", 2)?;
                let result = builder.ins().call(f, &[get(a), get(b)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Type checks ===
            Instruction::IsType(a, class_sym) => {
                let f = get_runtime_fn(module, builder, "wren_is_type", 2)?;
                let class_val = builder.ins().iconst(types::I64, class_sym.index() as i64);
                let result = builder.ins().call(f, &[get(a), class_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Unboxed f64 arithmetic (used by optimized tier) ===
            Instruction::AddF64(a, b) => Ok(Some(builder.ins().fadd(get(a), get(b)))),
            Instruction::SubF64(a, b) => Ok(Some(builder.ins().fsub(get(a), get(b)))),
            Instruction::MulF64(a, b) => Ok(Some(builder.ins().fmul(get(a), get(b)))),
            Instruction::DivF64(a, b) => Ok(Some(builder.ins().fdiv(get(a), get(b)))),
            Instruction::ModF64(a, b) => {
                // f64 modulo: a - floor(a/b) * b
                let div = builder.ins().fdiv(get(a), get(b));
                let floored = builder.ins().floor(div);
                let mul = builder.ins().fmul(floored, get(b));
                Ok(Some(builder.ins().fsub(get(a), mul)))
            }
            Instruction::NegF64(a) => Ok(Some(builder.ins().fneg(get(a)))),

            // === Unboxed f64 comparisons → raw Cranelift booleans ===
            // These produce raw i8 booleans (not NaN-boxed). The CondBranch
            // handler detects raw_bools and uses brif directly without the
            // expensive NaN-box truthiness check.
            Instruction::CmpLtF64(a, b) => {
                Ok(Some(builder.ins().fcmp(FloatCC::LessThan, get(a), get(b))))
            }
            Instruction::CmpGtF64(a, b) => Ok(Some(builder.ins().fcmp(
                FloatCC::GreaterThan,
                get(a),
                get(b),
            ))),
            Instruction::CmpLeF64(a, b) => Ok(Some(builder.ins().fcmp(
                FloatCC::LessThanOrEqual,
                get(a),
                get(b),
            ))),
            Instruction::CmpGeF64(a, b) => Ok(Some(builder.ins().fcmp(
                FloatCC::GreaterThanOrEqual,
                get(a),
                get(b),
            ))),

            // === Box/Unbox ===
            Instruction::Unbox(a) => {
                if f64_self_id.is_some() {
                    // In f64 inner function: values are already f64, no-op
                    Ok(Some(get(a)))
                } else {
                    // i64 (NaN-boxed) → f64 bitcast
                    Ok(Some(builder.ins().bitcast(
                        types::F64,
                        MemFlags::new(),
                        get(a),
                    )))
                }
            }
            Instruction::Box(a) => {
                if f64_self_id.is_some() {
                    // In f64 inner function: keep as f64, no boxing
                    Ok(Some(get(a)))
                } else {
                    // f64 → i64 (NaN-boxed) bitcast
                    Ok(Some(builder.ins().bitcast(
                        types::I64,
                        MemFlags::new(),
                        get(a),
                    )))
                }
            }

            // === Guards ===
            Instruction::GuardNum(src) => {
                // In f64 mode: no guard needed, values are already f64
                // In i64 mode: pass through (guards are for optimization hints)
                Ok(Some(get(src)))
            }
            Instruction::GuardBool(src) => Ok(Some(get(src))),
            Instruction::GuardClass(src, _class_id) => Ok(Some(get(src))),
            Instruction::GuardProtocol(src, _proto) => Ok(Some(get(src))),

            // === Math intrinsics ===
            Instruction::MathUnaryF64(op, a) => {
                use crate::mir::MathUnaryOp::*;
                let val = get(a);
                let result = match op {
                    // Cranelift native instructions
                    Floor => builder.ins().floor(val),
                    Ceil => builder.ins().ceil(val),
                    Sqrt => builder.ins().sqrt(val),
                    Abs => builder.ins().fabs(val),
                    Trunc => builder.ins().trunc(val),
                    Round => builder.ins().nearest(val),
                    // Compute from primitives
                    Fract => {
                        let floored = builder.ins().floor(val);
                        builder.ins().fsub(val, floored)
                    }
                    Sign => {
                        let zero = builder.ins().f64const(0.0);
                        let one = builder.ins().f64const(1.0);
                        let neg_one = builder.ins().f64const(-1.0);
                        let is_pos = builder.ins().fcmp(FloatCC::GreaterThan, val, zero);
                        let is_neg = builder.ins().fcmp(FloatCC::LessThan, val, zero);
                        let pos_or_zero = builder.ins().select(is_pos, one, zero);
                        builder.ins().select(is_neg, neg_one, pos_or_zero)
                    }
                    // libm functions — call via C ABI
                    Sin | Cos | Tan | Asin | Acos | Atan | Log | Log2 | Exp | Cbrt => {
                        let libm_name = match op {
                            Sin => "sin",
                            Cos => "cos",
                            Tan => "tan",
                            Asin => "asin",
                            Acos => "acos",
                            Atan => "atan",
                            Log => "log",
                            Log2 => "log2",
                            Exp => "exp",
                            Cbrt => "cbrt",
                            _ => unreachable!(),
                        };
                        // Declare f64 -> f64 libm function
                        let mut sig = module.make_signature();
                        sig.params.push(AbiParam::new(types::F64));
                        sig.returns.push(AbiParam::new(types::F64));
                        let fid = module
                            .declare_function(libm_name, Linkage::Import, &sig)
                            .map_err(|e| e.to_string())?;
                        let fref = module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[val]);
                        builder.inst_results(call)[0]
                    }
                };
                Ok(Some(result))
            }
            Instruction::MathBinaryF64(op, a, b) => {
                use crate::mir::MathBinaryOp::*;
                let va = get(a);
                let vb = get(b);
                let result = match op {
                    // Cranelift native
                    Min => builder.ins().fmin(va, vb),
                    Max => builder.ins().fmax(va, vb),
                    // libm functions
                    Pow | Atan2 => {
                        let libm_name = match op {
                            Pow => "pow",
                            Atan2 => "atan2",
                            _ => unreachable!(),
                        };
                        let mut sig = module.make_signature();
                        sig.params.push(AbiParam::new(types::F64));
                        sig.params.push(AbiParam::new(types::F64));
                        sig.returns.push(AbiParam::new(types::F64));
                        let fid = module
                            .declare_function(libm_name, Linkage::Import, &sig)
                            .map_err(|e| e.to_string())?;
                        let fref = module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[va, vb]);
                        builder.inst_results(call)[0]
                    }
                };
                Ok(Some(result))
            }

            // === Constant strings ===
            Instruction::ConstString(idx) => {
                let f = get_runtime_fn(module, builder, "wren_const_string", 1)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Static self-calls ===
            Instruction::CallStaticSelf { args } => {
                // In f64 mode: call inner function directly with f64 args
                // (no box/unbox roundtrip — args are already f64).
                // In i64 mode: call self with i64 args, prepending receiver.
                let target_id = if let Some(inner_id) = f64_self_id {
                    inner_id
                } else {
                    cranelift_module::FuncId::from_u32(match builder.func.name {
                        cranelift_codegen::ir::UserFuncName::User(ref u) => u.index,
                        _ => 0,
                    })
                };
                let self_func_ref = module.declare_func_in_func(target_id, builder.func);
                let mut call_args: Vec<Value> = Vec::with_capacity(1 + args.len());
                // Prepend receiver (param #0) for self-calls to match arity.
                // f64 inner functions don't need the receiver (they use
                // a reduced signature).
                if f64_self_id.is_none() {
                    if let Some(recv) = receiver_val {
                        call_args.push(recv);
                    }
                }
                for a in args {
                    call_args.push(get(a));
                }
                let result = builder.ins().call(self_func_ref, &call_args);
                Ok(Some(builder.inst_results(result)[0]))
            }
        };
        if let Some(vid) = err_sink.get() {
            return Err(format!("undefined value {:?}", vid));
        }
        result
    }

    /// Lower a MIR terminator to Cranelift IR.
    fn lower_terminator(
        term: &Terminator,
        builder: &mut FunctionBuilder,
        val_map: &HashMap<ValueId, Value>,
        block_map: &HashMap<BlockId, cranelift_codegen::ir::Block>,
        raw_bools: &std::collections::HashSet<ValueId>,
    ) -> Result<(), String> {
        // Surface undefined-value lookups as `Err` instead of
        // panicking the broker thread. The compile fails, the
        // function falls back to the interpreter, and the user
        // sees a slow-but-correct execution rather than a process
        // crash. Matches the `lower_instruction` handler.
        let undefined: std::cell::Cell<Option<ValueId>> = std::cell::Cell::new(None);
        let dummy_const = builder.ins().iconst(types::I64, 0);
        let get = |vid: &ValueId| -> Value {
            match val_map.get(vid) {
                Some(v) => *v,
                None => {
                    if undefined.get().is_none() {
                        undefined.set(Some(*vid));
                    }
                    dummy_const
                }
            }
        };

        match term {
            Terminator::Return(val) => {
                let v = get(val);
                // Coerce to the function's declared return type. The outer
                // JIT calling convention is i64 (NaN-boxed); the f64
                // inner-specialized helpers return f64. When the live value's
                // Cranelift type doesn't match, bit-reinterpret it — an f64
                // is its own valid NaN box and vice versa, so a bitcast is
                // the correct coercion either way.
                let return_ty = builder.func.signature.returns[0].value_type;
                let v_ty = builder.func.dfg.value_type(v);
                let v = if v_ty != return_ty {
                    builder.ins().bitcast(return_ty, MemFlags::new(), v)
                } else {
                    v
                };
                builder.ins().return_(&[v]);
            }
            Terminator::ReturnNull => {
                let null = builder.ins().iconst(types::I64, TAG_NULL as i64);
                builder.ins().return_(&[null]);
            }
            Terminator::Branch { target, args } => {
                let cl_block = block_map[target];
                let cl_args: Vec<BlockArg> = args.iter().map(|a| BlockArg::Value(get(a))).collect();
                builder.ins().jump(cl_block, &cl_args);
            }
            Terminator::CondBranch {
                condition,
                true_target,
                true_args,
                false_target,
                false_args,
            } => {
                let cond = get(condition);
                let t_block = block_map[true_target];
                let f_block = block_map[false_target];
                let t_args: Vec<BlockArg> =
                    true_args.iter().map(|a| BlockArg::Value(get(a))).collect();
                let f_args: Vec<BlockArg> =
                    false_args.iter().map(|a| BlockArg::Value(get(a))).collect();

                // If the condition is a raw boolean (from CmpLtF64 etc.),
                // use it directly — no NaN-box truthiness check needed.
                // This turns 8 instructions into 1 for typed comparisons.
                let is_truthy = if raw_bools.contains(condition) {
                    cond // Already a Cranelift i8 boolean
                } else {
                    // NaN-boxed truthiness: val != TAG_FALSE && val != TAG_NULL
                    let tag_false = builder.ins().iconst(types::I64, TAG_FALSE as i64);
                    let tag_null = builder.ins().iconst(types::I64, TAG_NULL as i64);
                    let not_false = builder.ins().icmp(IntCC::NotEqual, cond, tag_false);
                    let not_null = builder.ins().icmp(IntCC::NotEqual, cond, tag_null);
                    builder.ins().band(not_false, not_null)
                };

                builder
                    .ins()
                    .brif(is_truthy, t_block, &t_args, f_block, &f_args);
            }
            Terminator::Unreachable => {
                builder
                    .ins()
                    .trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
            }
        }
        if let Some(vid) = undefined.get() {
            return Err(format!("undefined value {:?} in terminator", vid));
        }
        Ok(())
    }

    /// Compute reverse post-order of MIR blocks starting from bb0.
    /// Guarantees dominators are visited before the blocks they dominate.
    fn compute_rpo(mir: &MirFunction) -> Vec<usize> {
        let n = mir.blocks.len();
        let mut visited = vec![false; n];
        let mut post_order = Vec::with_capacity(n);

        fn dfs(idx: usize, mir: &MirFunction, visited: &mut [bool], post_order: &mut Vec<usize>) {
            if visited[idx] {
                return;
            }
            visited[idx] = true;
            for succ in mir.blocks[idx].terminator.successors() {
                dfs(succ.0 as usize, mir, visited, post_order);
            }
            post_order.push(idx);
        }

        dfs(0, mir, &mut visited, &mut post_order);

        post_order.reverse(); // reverse post-order — bb0 is now first

        // Add any unreachable blocks AFTER reversing so they come last
        for (i, &seen) in visited.iter().enumerate().take(n) {
            if !seen {
                post_order.push(i);
            }
        }

        post_order
    }

    fn compute_rpo_from(mir: &MirFunction, start: BlockId) -> Vec<usize> {
        let n = mir.blocks.len();
        let mut visited = vec![false; n];
        let mut post_order = Vec::with_capacity(n);

        fn dfs(idx: usize, mir: &MirFunction, visited: &mut [bool], post_order: &mut Vec<usize>) {
            if visited[idx] {
                return;
            }
            visited[idx] = true;
            for succ in mir.blocks[idx].terminator.successors() {
                dfs(succ.0 as usize, mir, visited, post_order);
            }
            post_order.push(idx);
        }

        dfs(start.0 as usize, mir, &mut visited, &mut post_order);
        post_order.reverse();
        post_order
    }
}
