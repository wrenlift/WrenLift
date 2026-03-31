/// Cranelift-based JIT backend.
///
/// Translates MIR directly to Cranelift IR, bypassing the custom MachInst layer.
/// This provides correct register allocation and instruction encoding for x86_64
/// without the SCRATCH_GP / spill-slot conflicts of the hand-written emitter.
#[cfg(feature = "cranelift")]
pub mod cl {
    use crate::intern::Interner;
    use crate::mir::{
        BlockId, Instruction, MathUnaryOp, MirFunction, MirType, Terminator, ValueId,
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
    use std::collections::HashMap;

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
        callsite_ic_ptrs: Option<&[usize]>,
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

        // Keep verifier ON to catch IR construction bugs
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
        let param_count = mir.blocks[0]
            .instructions
            .iter()
            .filter(|(_, inst)| matches!(inst, Instruction::BlockParam(_)))
            .count();

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

        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig,
        );

        // 5. Lower MIR to Cranelift IR
        if std::env::var_os("WLIFT_CL_MIR").is_some() {
            eprintln!("=== CL MIR input for {} ===", safe_name);
            eprintln!("{}", mir.pretty_print(interner));
            eprintln!("=== end ===");
        }
        {
            let mut fb_ctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);

            lower_mir_to_cranelift(mir, interner, &mut builder, &mut module, callsite_ic_ptrs)?;

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
        let mut ctx = Context::for_function(func);
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let fn_ptr = module.get_finalized_function(func_id);
        let code_size = ctx.compiled_code().unwrap().code_info().total_size as usize;

        Ok(CraneliftCompiledCode {
            _module: module,
            fn_ptr,
            code_size,
        })
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
        callsite_ic_ptrs: Option<&[usize]>,
    ) -> Result<(), String> {
        // Map MIR blocks to Cranelift blocks
        let mut block_map: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
        for (i, _) in mir.blocks.iter().enumerate() {
            let cl_block = builder.create_block();
            block_map.insert(BlockId(i as u32), cl_block);
        }

        // Map MIR values to Cranelift values
        let mut val_map: HashMap<ValueId, Value> = HashMap::new();

        // Track which MIR values are raw Cranelift booleans (i8) rather than
        // NaN-boxed TAG_TRUE/TAG_FALSE. Used to skip the expensive truthiness
        // check in CondBranch when the condition is a direct fcmp/icmp result.
        let mut raw_bools: std::collections::HashSet<ValueId> = std::collections::HashSet::new();

        // Cache for declared runtime functions
        let mut runtime_cache: HashMap<String, cranelift_codegen::ir::FuncRef> = HashMap::new();

        // Call site counter for IC lookup
        let mut call_site_idx: usize = 0;

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
        let rpo = compute_rpo(mir);
        for &block_idx in &rpo {
            let block = &mir.blocks[block_idx];
            let bid = BlockId(block_idx as u32);
            let cl_block = block_map[&bid];
            builder.switch_to_block(cl_block);

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
            if block_idx == 0 {
                // Count BlockParam instructions to know how many func args
                let bp_count = block
                    .instructions
                    .iter()
                    .filter(|(_, inst)| matches!(inst, Instruction::BlockParam(_)))
                    .count();
                // Append function argument params to the entry block
                for _ in 0..bp_count {
                    builder.append_block_param(cl_block, types::I64);
                }
                // Map them
                let entry_params = builder.block_params(cl_block);
                let mut param_idx = 0usize;
                for &(vid, ref inst) in &block.instructions {
                    if matches!(inst, Instruction::BlockParam(_)) {
                        val_map.insert(vid, entry_params[param_idx]);
                        param_idx += 1;
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
                    &mut call_site_idx,
                )?;
                if let Some(val) = result {
                    val_map.insert(vid, val);
                    if is_raw_bool {
                        raw_bools.insert(vid);
                    }
                }
            }

            // Lower terminator
            lower_terminator(&block.terminator, builder, &val_map, &block_map, &raw_bools);
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
        callsite_ic_ptrs: Option<&[usize]>,
        call_site_idx: &mut usize,
    ) -> Result<Option<Value>, String> {
        let get = |vid: &ValueId| -> Value {
            *val_map
                .get(vid)
                .unwrap_or_else(|| panic!("undefined value {:?}", vid))
        };

        match inst {
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
            } => {
                let r = get(receiver);
                let ic_idx = *call_site_idx;
                *call_site_idx += 1;

                // Try inline IC: if we have IC data for this call site,
                // emit a class-check + direct call fast path.
                let ic = callsite_ic_ptrs
                    .and_then(|ptrs| ptrs.get(ic_idx))
                    .map(|&ptr| unsafe { &*(ptr as *const crate::mir::bytecode::CallSiteIC) });

                // Kind=1 (JIT leaf) or kind=5 (getter): emit inline fast path
                if let Some(ic) = ic {
                    if (ic.kind == 1 || ic.kind == 5) && ic.class != 0 && !ic.jit_ptr.is_null() {
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

                        // Compare against cached class
                        let cached_class = builder.ins().iconst(types::I64, ic.class as i64);
                        let class_match =
                            builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
                        builder
                            .ins()
                            .brif(class_match, fast_block, &[], slow_block, &[]);

                        // Fast path: direct call or inline field load
                        builder.switch_to_block(fast_block);
                        let fast_result = if ic.kind == 5 {
                            // Getter: inline field load
                            let field_idx = ic.func_id as i32;
                            let fields_ptr = builder.ins().load(
                                types::I64,
                                MemFlags::trusted(),
                                obj_ptr,
                                INSTANCE_FIELDS,
                            );
                            let offset = field_idx * VALUE_SIZE;
                            builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), fields_ptr, offset)
                        } else {
                            // Kind=1: direct call to JIT function pointer
                            let jit_addr = builder.ins().iconst(types::I64, ic.jit_ptr as i64);
                            // Build call signature: (recv, args...) -> i64
                            let mut sig = module.make_signature();
                            sig.params.push(AbiParam::new(types::I64)); // recv
                            for _ in args {
                                sig.params.push(AbiParam::new(types::I64));
                            }
                            sig.returns.push(AbiParam::new(types::I64));
                            let sig_ref = builder.import_signature(sig);
                            let mut call_args = vec![r];
                            for a in args {
                                call_args.push(get(a));
                            }
                            let call = builder.ins().call_indirect(sig_ref, jit_addr, &call_args);
                            builder.inst_results(call)[0]
                        };
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(fast_result)]);

                        // Slow path: full dispatch via wren_call_N
                        builder.switch_to_block(slow_block);
                        let method_bits = method.index() as u64;
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
                let method_bits = method.index() as u64;
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

            // === Super calls ===
            Instruction::SuperCall { method, args } => {
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
                    let args: Vec<Value> = elems.iter().map(|e| get(e)).collect();
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
            Instruction::SubscriptGet { receiver, args } => {
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
                // i64 (NaN-boxed) → f64 bitcast
                Ok(Some(builder.ins().bitcast(
                    types::F64,
                    MemFlags::new(),
                    get(a),
                )))
            }
            Instruction::Box(a) => {
                // f64 → i64 (NaN-boxed) bitcast
                Ok(Some(builder.ins().bitcast(
                    types::I64,
                    MemFlags::new(),
                    get(a),
                )))
            }

            // === Guards ===
            Instruction::GuardNum(src) => {
                // (val >> 48) != 0x7FFC → pass (is a number)
                // On failure, return null (deopt)
                // For now, just pass through — guards are mainly for optimized tier
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
                // Direct recursive call to self — use module's func_id to get
                // a proper FuncRef, avoiding full dispatch overhead.
                // The func_id was declared in compile_mir; re-declare it in
                // this function so Cranelift can emit a direct call.
                let self_func_ref = module.declare_func_in_func(
                    cranelift_module::FuncId::from_u32(match builder.func.name {
                        cranelift_codegen::ir::UserFuncName::User(ref u) => u.index,
                        _ => 0,
                    }),
                    builder.func,
                );
                let call_args: Vec<Value> = args.iter().map(|a| get(a)).collect();
                let result = builder.ins().call(self_func_ref, &call_args);
                Ok(Some(builder.inst_results(result)[0]))
            }
        }
    }

    /// Lower a MIR terminator to Cranelift IR.
    fn lower_terminator(
        term: &Terminator,
        builder: &mut FunctionBuilder,
        val_map: &HashMap<ValueId, Value>,
        block_map: &HashMap<BlockId, cranelift_codegen::ir::Block>,
        raw_bools: &std::collections::HashSet<ValueId>,
    ) {
        let get = |vid: &ValueId| -> Value {
            *val_map
                .get(vid)
                .unwrap_or_else(|| panic!("undefined value {:?} in terminator", vid))
        };

        match term {
            Terminator::Return(val) => {
                let v = get(val);
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

        // Add any unreachable blocks at the end
        for i in 0..n {
            if !visited[i] {
                post_order.push(i);
            }
        }

        post_order.reverse(); // reverse post-order
        post_order
    }
}
