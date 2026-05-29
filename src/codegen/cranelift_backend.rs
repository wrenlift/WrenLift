/// Cranelift-based JIT backend.
///
/// Translates MIR directly to Cranelift IR, bypassing the custom MachInst layer.
/// This provides correct register allocation and instruction encoding for x86_64
/// without the SCRATCH_GP / spill-slot conflicts of the hand-written emitter.
///
/// # AOT GC contract
///
/// Every Wren `Value` held live across a runtime helper call must be visible
/// to the stack-map walker, otherwise the GC can promote (move) or sweep
/// (free) the object the slot points at and the lowering will dereference
/// stale memory after the call returns.
///
/// Two ways to satisfy this:
///
/// 1. **Blanket cover.** If the value is the result of an `Instruction::*`
///    MIR op, the generic dispatch at this file's instruction-lowering
///    site calls `declare_value_needs_stack_map` for you. You get this
///    for free — no per-op work needed.
///
/// 2. **Explicit declaration.** If the value is a lowering-internal
///    intermediate (e.g., a value built inline before a helper call —
///    `stack_load`s, computed pointers, NaN-mask results that feed
///    into a helper arg), Cranelift cannot guess it should appear in
///    the stack map. Call `builder.declare_value_needs_stack_map(v)`
///    on every such intermediate that may carry a heap pointer
///    across the next safepoint. Gated on `env_stack_maps()` so JIT
///    / non-stackmap modes don't pay the cost.
///
/// `runtime_fns::wren_write_barrier`, `wren_map_set`, `wren_list_add`,
/// and the SM cross-fn invoke helpers all rely on the runtime
/// scanning the caller's slot. If you skip the declare, the GC
/// validator (`WLIFT_VALIDATE_BARRIERS=1` /
/// `WLIFT_VALIDATE_STACKMAP=1`) will panic naming the function and
/// safepoint offset — much faster than a cold SIGSEGV hunt. Run the
/// `aot_gc_coverage` property suite whenever you change a lowering
/// that adds or moves a helper call.
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
    use cranelift_codegen::ir::{
        AbiParam, BlockArg, Function, InstBuilder, MemFlagsData as MemFlags, Signature, Type, Value,
    };
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_codegen::Context;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    use std::collections::{HashMap, HashSet};

    const QNAN: u64 = 0x7FFC_0000_0000_0000;
    const SIGN_BIT: u64 = 1u64 << 63;
    /// Top-16-bit pattern for an object NaN-box: `SIGN_BIT | QNAN`. A
    /// receiver Value is an object iff `(value & TAG_OBJ) == TAG_OBJ`
    /// — every other Wren Value (Number, Null, Bool, Undefined,
    /// String-box) clears at least one of those bits. JIT class-check
    /// sites must guard with this before reading `recv.class`,
    /// because the load offset (`HEADER_CLASS`) doesn't fail
    /// "safely" for a non-object: a Number's f64 bits, masked
    /// through PTR_MASK, can land at an unmapped page and SIGSEGV.
    const TAG_OBJ: u64 = SIGN_BIT | QNAN;
    const TAG_NULL: u64 = QNAN; // 0x7FFC_0000_0000_0000 — no extra bits
    const TAG_FALSE: u64 = QNAN | 1;
    const TAG_TRUE: u64 = QNAN | 2;
    // Note: QNAN | 3 = TAG_UNDEFINED (not null!)
    const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    /// Emit a guarded receiver-class load: returns `(obj_ptr,
    /// recv_class)` iff the receiver is a NaN-boxed object,
    /// branching to `not_object_block` otherwise. Safe to use on
    /// any Value; non-object receivers (Numbers, Null, Bool, ...)
    /// take the not-object branch instead of dereferencing garbage
    /// at HEADER_CLASS.
    fn emit_class_load_guarded(
        builder: &mut FunctionBuilder,
        recv: cranelift_codegen::ir::Value,
        not_object_block: cranelift_codegen::ir::Block,
    ) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
        use cranelift_codegen::ir::condcodes::IntCC;
        use cranelift_codegen::ir::{InstBuilder, MemFlagsData as MemFlags};
        let tag_obj = builder.ins().iconst(types::I64, TAG_OBJ as i64);
        let high = builder.ins().band(recv, tag_obj);
        let is_obj = builder.ins().icmp(IntCC::Equal, high, tag_obj);
        let object_block = builder.create_block();
        builder
            .ins()
            .brif(is_obj, object_block, &[], not_object_block, &[]);
        builder.switch_to_block(object_block);
        let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
        let obj_ptr = builder.ins().band(recv, ptr_mask);
        let recv_class = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), obj_ptr, HEADER_CLASS);
        (obj_ptr, recv_class)
    }

    fn emit_guarded_obj_ptr_with_type(
        builder: &mut FunctionBuilder,
        boxed: Value,
        expected_obj_type: i32,
        miss_block: cranelift_codegen::ir::Block,
    ) -> Value {
        let tag_obj = builder.ins().iconst(types::I64, TAG_OBJ as i64);
        let high = builder.ins().band(boxed, tag_obj);
        let is_obj = builder.ins().icmp(IntCC::Equal, high, tag_obj);
        let object_block = builder.create_block();
        builder
            .ins()
            .brif(is_obj, object_block, &[], miss_block, &[]);
        builder.switch_to_block(object_block);

        let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
        let obj_ptr = builder.ins().band(boxed, ptr_mask);
        let obj_type =
            builder
                .ins()
                .uload8(types::I64, MemFlags::trusted(), obj_ptr, HEADER_OBJ_TYPE);
        let expected = builder.ins().iconst(types::I64, expected_obj_type as i64);
        let type_ok = builder.ins().icmp(IntCC::Equal, obj_type, expected);
        let ok_block = builder.create_block();
        builder.ins().brif(type_ok, ok_block, &[], miss_block, &[]);
        builder.switch_to_block(ok_block);
        obj_ptr
    }

    fn emit_guarded_simd_vector(
        builder: &mut FunctionBuilder,
        boxed: Value,
        expected_kind: i32,
        vec_ty: Type,
        miss_block: cranelift_codegen::ir::Block,
    ) -> Value {
        let obj_ptr =
            emit_guarded_obj_ptr_with_type(builder, boxed, OBJ_TYPE_SIMD as i32, miss_block);
        let kind = builder
            .ins()
            .uload8(types::I64, MemFlags::trusted(), obj_ptr, SIMD_KIND);
        let expected = builder.ins().iconst(types::I64, expected_kind as i64);
        let kind_ok = builder.ins().icmp(IntCC::Equal, kind, expected);
        let ok_block = builder.create_block();
        builder.ins().brif(kind_ok, ok_block, &[], miss_block, &[]);
        builder.switch_to_block(ok_block);
        builder
            .ins()
            .load(vec_ty, MemFlags::trusted(), obj_ptr, SIMD_LANES)
    }

    fn bitcast_vector_if_needed(builder: &mut FunctionBuilder, value: Value, ty: Type) -> Value {
        if builder.func.dfg.value_type(value) == ty {
            value
        } else {
            builder.ins().bitcast(ty, MemFlags::new(), value)
        }
    }

    fn box_bool(builder: &mut FunctionBuilder, raw: Value) -> Value {
        let tag_true = builder.ins().iconst(types::I64, TAG_TRUE as i64);
        let tag_false = builder.ins().iconst(types::I64, TAG_FALSE as i64);
        builder.ins().select(raw, tag_true, tag_false)
    }

    fn box_u32_as_num(builder: &mut FunctionBuilder, raw: Value) -> Value {
        let raw64 = builder.ins().uextend(types::I64, raw);
        let num = builder.ins().fcvt_from_uint(types::F64, raw64);
        builder.ins().bitcast(types::I64, MemFlags::new(), num)
    }

    /// Emit a method dispatch through the `wren_call_*` runtime
    /// helpers. Picks the matching fixed-arity `wren_call_N` for
    /// 0..=8 user args, or routes through `wren_call_dynamic` with
    /// a stack-allocated `[u64; n]` buffer for 9+.
    ///
    /// A bug-prone shape this consolidates: every call site that
    /// branched on `args.len()` previously had its own `match`
    /// plus a `_ => "wren_call_N"` fallback that silently
    /// truncated higher-arity calls (the highest the table covered
    /// was 8, the lowest was 4). The JIT'd function's signature
    /// still expected the original arity, so the truncated args
    /// arrived as garbage in slots N..real_arity — load-bearing
    /// example: Renderer2D's
    /// `drawSprite_(texture, x, y, w, h, u0, v0, u1, v1, r, g, b, a)`
    /// (13 user args) lost `v1`/`r`/`g`/`b`/`a` on every call,
    /// surfacing as `Float32Array[head + k] = px` writes hitting
    /// null receivers.
    fn emit_wren_call<G>(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut G,
        receiver: Value,
        method_val: Value,
        args: &[Value],
    ) -> Result<Value, String>
    where
        G: FnMut(
                &mut dyn Module,
                &mut FunctionBuilder,
                &str,
                usize,
            ) -> Result<cranelift_codegen::ir::FuncRef, String>
            + ?Sized,
    {
        if std::env::var_os("WLIFT_EMIT_WREN_CALL_TRACE").is_some() {
            eprintln!(
                "emit_wren_call: args.len()={} routing={}",
                args.len(),
                if args.len() > 8 { "wren_call_dynamic" } else { "wren_call_N" }
            );
        }
        if args.len() > 8 {
            let f = get_runtime_fn(module, builder, "wren_call_dynamic", 4)?;
            let slot = builder.create_sized_stack_slot(
                cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    (args.len() as u32) * 8,
                    3, // 8-byte alignment (2^3)
                ),
            );
            let buf_ptr = builder.ins().stack_addr(types::I64, slot, 0);
            for (i, &arg) in args.iter().enumerate() {
                builder.ins().store(MemFlags::trusted(), arg, buf_ptr, (i as i32) * 8);
            }
            let count = builder.ins().iconst(types::I64, args.len() as i64);
            let call = builder.ins().call(f, &[receiver, method_val, count, buf_ptr]);
            return Ok(builder.inst_results(call)[0]);
        }
        let call_name = match args.len() {
            0 => "wren_call_0",
            1 => "wren_call_1",
            2 => "wren_call_2",
            3 => "wren_call_3",
            4 => "wren_call_4",
            5 => "wren_call_5",
            6 => "wren_call_6",
            7 => "wren_call_7",
            _ => "wren_call_8",
        };
        let f = get_runtime_fn(module, builder, call_name, 2 + args.len())?;
        let mut call_args = vec![receiver, method_val];
        call_args.extend_from_slice(args);
        let call = builder.ins().call(f, &call_args);
        Ok(builder.inst_results(call)[0])
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_method_call_slow<G>(
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut G,
        receiver: Value,
        method: crate::intern::SymbolId,
        args: &[Value],
        ic_idx: Option<usize>,
        aot_config: Option<&AotLoweringConfig>,
    ) -> Result<Value, String>
    where
        G: FnMut(
                &mut dyn Module,
                &mut FunctionBuilder,
                &str,
                usize,
            ) -> Result<cranelift_codegen::ir::FuncRef, String>
            + ?Sized,
    {
        let method_val = if let Some(cfg) = aot_config {
            let slot = aot_intern_symbol(cfg, method.index(), interner);
            let gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
            let base = builder.ins().global_value(types::I64, gv);
            builder
                .ins()
                .load(types::I64, MemFlags::trusted(), base, (slot as i32) * 8)
        } else {
            let mut method_bits = method.index() as u64;
            if let Some(ic_idx) = ic_idx.filter(|_| env_jit_callsite_ic()) {
                method_bits |= ((ic_idx as u64) + 1) << 32;
            }
            builder.ins().iconst(types::I64, method_bits as i64)
        };
        emit_wren_call(builder, module, get_runtime_fn, receiver, method_val, args)
    }

    fn emit_alloc_simd_vector<G>(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut G,
        helper_name: &str,
        vector: Value,
    ) -> Result<Value, String>
    where
        G: FnMut(
                &mut dyn Module,
                &mut FunctionBuilder,
                &str,
                usize,
            ) -> Result<cranelift_codegen::ir::FuncRef, String>
            + ?Sized,
    {
        let lane_ty = builder.func.dfg.value_type(vector).lane_type();
        let mut lane_args = Vec::with_capacity(4);
        for lane in 0..4 {
            let scalar = builder.ins().extractlane(vector, lane);
            let lane_bits = match lane_ty {
                types::F32 => {
                    let bits = builder.ins().bitcast(types::I32, MemFlags::new(), scalar);
                    builder.ins().uextend(types::I64, bits)
                }
                types::I32 => builder.ins().sextend(types::I64, scalar),
                other => {
                    return Err(format!(
                        "unsupported SIMD lane type {:?} for {}",
                        other, helper_name
                    ));
                }
            };
            lane_args.push(lane_bits);
        }
        let f = get_runtime_fn(module, builder, helper_name, 4)?;
        let call = builder.ins().call(f, &lane_args);
        Ok(builder.inst_results(call)[0])
    }

    #[allow(clippy::too_many_arguments)]
    fn try_lower_simd_intrinsic_call<G>(
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut G,
        receiver: Value,
        method: crate::intern::SymbolId,
        args: &[Value],
        ic_idx: usize,
        aot_config: Option<&AotLoweringConfig>,
    ) -> Result<Option<Value>, String>
    where
        G: FnMut(
                &mut dyn Module,
                &mut FunctionBuilder,
                &str,
                usize,
            ) -> Result<cranelift_codegen::ir::FuncRef, String>
            + ?Sized,
    {
        let method_sig = interner.resolve(method);
        let supported = matches!(
            (method_sig, args.len()),
            ("+(_)", 1)
                | ("-(_)", 1)
                | ("*(_)", 1)
                | ("/(_)", 1)
                | ("min(_)", 1)
                | ("max(_)", 1)
                | ("==(_)", 1)
                | ("!=(_)", 1)
                | ("<(_)", 1)
                | ("<=(_)", 1)
                | (">(_)", 1)
                | (">=(_)", 1)
                | ("-", 0)
                | ("abs", 0)
                | ("sqrt", 0)
                | ("reinterpretAsInt", 0)
                | ("reinterpretAsFloat", 0)
                | ("&(_)", 1)
                | ("|(_)", 1)
                | ("^(_)", 1)
                | ("~", 0)
                | ("allTrue", 0)
                | ("anyTrue", 0)
                | ("bitmask", 0)
        );
        if !supported {
            return Ok(None);
        }

        let slow_block = builder.create_block();
        let merge_block = builder.create_block();
        builder.append_block_param(merge_block, types::I64);

        let recv_obj =
            emit_guarded_obj_ptr_with_type(builder, receiver, OBJ_TYPE_SIMD as i32, slow_block);
        let recv_kind = builder
            .ins()
            .uload8(types::I64, MemFlags::trusted(), recv_obj, SIMD_KIND);
        let f32_kind = builder.ins().iconst(types::I64, SIMD_KIND_F32X4 as i64);
        let i32_kind = builder.ins().iconst(types::I64, SIMD_KIND_I32X4 as i64);

        macro_rules! dual_same_kind_binary {
            ($f32_emit:expr, $i32_emit:expr) => {{
                let f32_block = builder.create_block();
                let i32_check_block = builder.create_block();
                let i32_block = builder.create_block();
                let is_f32 = builder.ins().icmp(IntCC::Equal, recv_kind, f32_kind);
                builder
                    .ins()
                    .brif(is_f32, f32_block, &[], i32_check_block, &[]);

                builder.switch_to_block(f32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::F32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_F32X4 as i32,
                    types::F32X4,
                    slow_block,
                );
                let out = $f32_emit(builder, lhs, rhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4f",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);

                builder.switch_to_block(i32_check_block);
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);

                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_I32X4 as i32,
                    types::I32X4,
                    slow_block,
                );
                let out = $i32_emit(builder, lhs, rhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }};
        }

        macro_rules! dual_compare {
            ($f32_emit:expr, $i32_emit:expr) => {{
                let f32_block = builder.create_block();
                let i32_check_block = builder.create_block();
                let i32_block = builder.create_block();
                let is_f32 = builder.ins().icmp(IntCC::Equal, recv_kind, f32_kind);
                builder
                    .ins()
                    .brif(is_f32, f32_block, &[], i32_check_block, &[]);

                builder.switch_to_block(f32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::F32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_F32X4 as i32,
                    types::F32X4,
                    slow_block,
                );
                let out = $f32_emit(builder, lhs, rhs);
                let out = bitcast_vector_if_needed(builder, out, types::I32X4);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);

                builder.switch_to_block(i32_check_block);
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);

                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_I32X4 as i32,
                    types::I32X4,
                    slow_block,
                );
                let out = $i32_emit(builder, lhs, rhs);
                let out = bitcast_vector_if_needed(builder, out, types::I32X4);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }};
        }

        macro_rules! dual_same_kind_unary {
            ($f32_emit:expr, $i32_emit:expr) => {{
                let f32_block = builder.create_block();
                let i32_check_block = builder.create_block();
                let i32_block = builder.create_block();
                let is_f32 = builder.ins().icmp(IntCC::Equal, recv_kind, f32_kind);
                builder
                    .ins()
                    .brif(is_f32, f32_block, &[], i32_check_block, &[]);

                builder.switch_to_block(f32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::F32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let out = $f32_emit(builder, lhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4f",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);

                builder.switch_to_block(i32_check_block);
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);

                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let out = $i32_emit(builder, lhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }};
        }

        match (method_sig, args.len()) {
            ("+(_)", 1) => dual_same_kind_binary!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fadd(lhs, rhs),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().iadd(lhs, rhs)
            ),
            ("-(_)", 1) => dual_same_kind_binary!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fsub(lhs, rhs),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().isub(lhs, rhs)
            ),
            ("*(_)", 1) => dual_same_kind_binary!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fmul(lhs, rhs),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().imul(lhs, rhs)
            ),
            ("/(_)", 1) => {
                let f32_block = builder.create_block();
                let is_f32 = builder.ins().icmp(IntCC::Equal, recv_kind, f32_kind);
                builder.ins().brif(is_f32, f32_block, &[], slow_block, &[]);
                builder.switch_to_block(f32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::F32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_F32X4 as i32,
                    types::F32X4,
                    slow_block,
                );
                let out = builder.ins().fdiv(lhs, rhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4f",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("min(_)", 1) => dual_same_kind_binary!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fmin(lhs, rhs),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().smin(lhs, rhs)
            ),
            ("max(_)", 1) => dual_same_kind_binary!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fmax(lhs, rhs),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().smax(lhs, rhs)
            ),
            ("==(_)", 1) => dual_compare!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fcmp(
                    FloatCC::Equal,
                    lhs,
                    rhs
                ),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().icmp(
                    IntCC::Equal,
                    lhs,
                    rhs
                )
            ),
            ("!=(_)", 1) => dual_compare!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fcmp(
                    FloatCC::NotEqual,
                    lhs,
                    rhs
                ),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().icmp(
                    IntCC::NotEqual,
                    lhs,
                    rhs
                )
            ),
            ("<(_)", 1) => dual_compare!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fcmp(
                    FloatCC::LessThan,
                    lhs,
                    rhs
                ),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().icmp(
                    IntCC::SignedLessThan,
                    lhs,
                    rhs
                )
            ),
            ("<=(_)", 1) => dual_compare!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fcmp(
                    FloatCC::LessThanOrEqual,
                    lhs,
                    rhs
                ),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().icmp(
                    IntCC::SignedLessThanOrEqual,
                    lhs,
                    rhs
                )
            ),
            (">(_)", 1) => dual_compare!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fcmp(
                    FloatCC::GreaterThan,
                    lhs,
                    rhs
                ),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().icmp(
                    IntCC::SignedGreaterThan,
                    lhs,
                    rhs
                )
            ),
            (">=(_)", 1) => dual_compare!(
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().fcmp(
                    FloatCC::GreaterThanOrEqual,
                    lhs,
                    rhs
                ),
                |builder: &mut FunctionBuilder, lhs, rhs| builder.ins().icmp(
                    IntCC::SignedGreaterThanOrEqual,
                    lhs,
                    rhs
                )
            ),
            ("-", 0) => dual_same_kind_unary!(
                |builder: &mut FunctionBuilder, lhs| builder.ins().fneg(lhs),
                |builder: &mut FunctionBuilder, lhs| builder.ins().ineg(lhs)
            ),
            ("abs", 0) => dual_same_kind_unary!(
                |builder: &mut FunctionBuilder, lhs| builder.ins().fabs(lhs),
                |builder: &mut FunctionBuilder, lhs| builder.ins().iabs(lhs)
            ),
            ("sqrt", 0) => {
                let f32_block = builder.create_block();
                let is_f32 = builder.ins().icmp(IntCC::Equal, recv_kind, f32_kind);
                builder.ins().brif(is_f32, f32_block, &[], slow_block, &[]);
                builder.switch_to_block(f32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::F32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let out = builder.ins().sqrt(lhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4f",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("reinterpretAsInt", 0) => {
                let f32_block = builder.create_block();
                let is_f32 = builder.ins().icmp(IntCC::Equal, recv_kind, f32_kind);
                builder.ins().brif(is_f32, f32_block, &[], slow_block, &[]);
                builder.switch_to_block(f32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::F32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    lhs,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("reinterpretAsFloat", 0) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4f",
                    lhs,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("&(_)", 1) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_I32X4 as i32,
                    types::I32X4,
                    slow_block,
                );
                let out = builder.ins().band(lhs, rhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("|(_)", 1) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_I32X4 as i32,
                    types::I32X4,
                    slow_block,
                );
                let out = builder.ins().bor(lhs, rhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("^(_)", 1) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let rhs = emit_guarded_simd_vector(
                    builder,
                    args[0],
                    SIMD_KIND_I32X4 as i32,
                    types::I32X4,
                    slow_block,
                );
                let out = builder.ins().bxor(lhs, rhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("~", 0) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let out = builder.ins().bnot(lhs);
                let result = emit_alloc_simd_vector(
                    builder,
                    module,
                    get_runtime_fn,
                    "wren_alloc_simd4i",
                    out,
                )?;
                builder.ins().jump(merge_block, &[BlockArg::Value(result)]);
            }
            ("allTrue", 0) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let raw = builder.ins().vall_true(lhs);
                let boxed = box_bool(builder, raw);
                builder.ins().jump(merge_block, &[BlockArg::Value(boxed)]);
            }
            ("anyTrue", 0) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let raw = builder.ins().vany_true(lhs);
                let boxed = box_bool(builder, raw);
                builder.ins().jump(merge_block, &[BlockArg::Value(boxed)]);
            }
            ("bitmask", 0) => {
                let i32_block = builder.create_block();
                let is_i32 = builder.ins().icmp(IntCC::Equal, recv_kind, i32_kind);
                builder.ins().brif(is_i32, i32_block, &[], slow_block, &[]);
                builder.switch_to_block(i32_block);
                let lhs =
                    builder
                        .ins()
                        .load(types::I32X4, MemFlags::trusted(), recv_obj, SIMD_LANES);
                let raw = builder.ins().vhigh_bits(types::I32, lhs);
                let boxed = box_u32_as_num(builder, raw);
                builder.ins().jump(merge_block, &[BlockArg::Value(boxed)]);
            }
            _ => return Ok(None),
        }

        builder.switch_to_block(slow_block);
        let slow_result = emit_method_call_slow(
            interner,
            builder,
            module,
            get_runtime_fn,
            receiver,
            method,
            args,
            Some(ic_idx),
            aot_config,
        )?;
        builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(slow_result)]);

        builder.switch_to_block(merge_block);
        Ok(Some(builder.block_params(merge_block)[0]))
    }

    // ---------------------------------------------------------------------
    // Cached process-wide env flags for the JIT compile path
    // ---------------------------------------------------------------------
    //
    // The Cranelift lowering reads several `WLIFT_*` flags every time it
    // emits a call site (`WLIFT_ENABLE_JIT_CALLSITE_IC`,
    // `WLIFT_ENABLE_PURE_LEAF_DIRECT`) or a function (`WLIFT_ENABLE_STACK_MAPS`).
    // `std::env::var_os` acquires a global mutex on every call — fine for
    // one-shot startup probes, but the broker thread compiles dozens of
    // functions during warmup with hundreds of call sites between them.
    // Cache once into a `OnceLock<bool>`.

    #[inline]
    fn env_jit_callsite_ic() -> bool {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("WLIFT_ENABLE_JIT_CALLSITE_IC").is_some())
    }

    #[inline]
    fn env_pure_leaf_direct() -> bool {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("WLIFT_ENABLE_PURE_LEAF_DIRECT").is_some())
    }

    /// Stack maps are now ON by default. JIT-compiled Wren methods
    /// keep boxed `Value`s in CPU registers across calls, but only
    /// the args passed *into* `wren_call_N` get pushed as JIT roots
    /// — the caller's other live values are register-resident and
    /// invisible to the GC scanner. Without stack maps, any minor GC
    /// triggered by a callee's allocation strands those pointers in
    /// freed/reused nursery memory, and the next use produces
    /// "instance of Object" miscompiles or segfaults. The hatch site's
    /// template parser hit this within ~10 requests under live load.
    ///
    /// `WLIFT_DISABLE_STACK_MAPS=1` is preserved as a kill switch so
    /// a regression here can be bisected without a rebuild.
    #[inline]
    fn env_stack_maps() -> bool {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("WLIFT_DISABLE_STACK_MAPS").is_none())
    }

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
        /// Per-safepoint live-root metadata derived from Cranelift's
        /// user stack maps. Populated for the main function body
        /// (OSR entries are not yet covered). The runtime GC scanner
        /// in `crate::runtime::vm::VM::scan_native_stack_roots`
        /// keys off the safepoint `code_offset` (return address
        /// minus function start) to find the live boxed slots in the
        /// JIT frame.
        pub native_meta: Option<crate::codegen::native_meta::NativeFrameMetadata>,
    }

    // SAFETY: The JITModule's memory is self-contained and the fn_ptr
    // points into it. Safe to send across threads for installation.
    unsafe impl Send for CraneliftCompiledCode {}
    unsafe impl Sync for CraneliftCompiledCode {}

    /// Compile a MIR function to native code using Cranelift.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_mir(
        mir: &MirFunction,
        interner: &Interner,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
        inline_bodies: Option<
            std::sync::Arc<std::collections::HashMap<u32, std::sync::Arc<MirFunction>>>,
        >,
        cha_by_method: crate::runtime::engine::SharedCha,
    ) -> Result<CraneliftCompiledCode, String> {
        // 1. Create Cranelift ISA for the host
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;

        // Frame-pointer preservation is gated behind the same env
        // var as the stack-map marking it supports — when marks are
        // default-off, leave FP behaviour at Cranelift's default
        // (no enforcement) so we don't perturb the JIT'd prologue
        // shape that the existing naked `wren_call_N` stubs rely
        // on. Flip both together with `WLIFT_ENABLE_STACK_MAPS=1`
        // when iterating on the GC root work.
        let pfp = if env_stack_maps() { "true" } else { "false" };
        flag_builder
            .set("preserve_frame_pointers", pfp)
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
                    None, // f64 inner is JIT-only
                    None, // f64 inner has no method calls — nothing to inline
                    None, // and no method dispatch — nothing for CHA
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
            let compiled_code = ctx.compiled_code().unwrap();
            let code_size = compiled_code.code_info().total_size as usize;
            let native_meta = native_meta_from_cranelift(compiled_code);
            return Ok(CraneliftCompiledCode {
                _module: module,
                fn_ptr,
                osr_entries: Vec::new(),
                code_size,
                native_meta,
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
                inline_bodies.clone(),
                cha_by_method.clone(),
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
                inline_bodies.clone(),
                cha_by_method.clone(),
            )
        } else {
            Vec::new()
        };
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let fn_ptr = module.get_finalized_function(func_id);
        let compiled_code = ctx.compiled_code().unwrap();
        let code_size = compiled_code.code_info().total_size as usize;
        let native_meta = native_meta_from_cranelift(compiled_code);
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
            native_meta,
        })
    }

    /// Translate Cranelift's user stack maps (SP-relative root offsets at
    /// each call safepoint) into the GC scanner's `NativeFrameMetadata`
    /// shape (FP-relative spill-slot offsets). Returns `None` if the
    /// compiled code has no frame layout (shouldn't happen for normal
    /// functions, but `frame_layout` is `Option`-typed so we guard
    /// defensively rather than panic).
    pub fn native_meta_from_cranelift(
        compiled: &cranelift_codegen::CompiledCode,
    ) -> Option<crate::codegen::native_meta::NativeFrameMetadata> {
        use crate::codegen::native_meta::{
            LiveRootMetadata, NativeFrameMetadata, RootLocation, SafepointKind, SafepointMetadata,
        };

        let layout = compiled.buffer.frame_layout()?;
        // Cranelift's `frame_to_fp_offset` is the offset, in bytes,
        // from the bottom of the frame (= SP at safepoint, after the
        // prologue has dropped SP) up to FP. So at a safepoint:
        //
        //   FP = SP + frame_to_fp_offset
        //
        // and a root at `SP + entry_offset` lives at
        //
        //   FP - frame_to_fp_offset + entry_offset
        //   = FP + (entry_offset as i64 - frame_to_fp_offset as i64)
        //
        // which is what the GC scanner ([`scan_native_stack_roots`])
        // computes via `(jit_fp + spill_offset)`.
        let fp_anchor = layout.frame_to_fp_offset as i64;

        let mut safepoints = Vec::new();
        let mut ordinal = 0u32;
        for (code_offset, _span, map) in compiled.buffer.user_stack_maps() {
            let mut live_roots: Vec<LiveRootMetadata> = Vec::new();
            for (_ty, sp_offset) in map.entries() {
                let fp_relative = sp_offset as i64 - fp_anchor;
                // The runtime currently models spill offsets as i32.
                // Stack frames > 2GB are nonsensical for our workload;
                // out-of-range values would be a Cranelift miscompile,
                // so we drop the root rather than mask it. Live roots
                // we drop here become missed roots — the GC may free
                // an object that's still in use. Logging would be
                // nice but `eprintln!` from inside compile is noisy
                // for normal operation.
                let Ok(fp_relative_i32) = i32::try_from(fp_relative) else {
                    continue;
                };
                live_roots.push(LiveRootMetadata {
                    slot: live_roots.len() as u16,
                    location: RootLocation::Spill(fp_relative_i32),
                });
            }
            safepoints.push(SafepointMetadata {
                ordinal,
                inst_index: 0,
                code_offset: *code_offset,
                kind: SafepointKind::CallRuntime,
                live_roots,
            });
            ordinal += 1;
        }

        if safepoints.is_empty() {
            return None;
        }

        Some(NativeFrameMetadata {
            boxed_values: Vec::new(),
            safepoints,
            spill_safe_nonleaf: true,
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

    #[allow(clippy::too_many_arguments)]
    fn compile_osr_entries(
        mir: &MirFunction,
        interner: &Interner,
        module: &mut dyn Module,
        safe_name: &str,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
        inline_bodies: Option<
            std::sync::Arc<std::collections::HashMap<u32, std::sync::Arc<MirFunction>>>,
        >,
        cha_by_method: crate::runtime::engine::SharedCha,
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
                    None, // OSR-entry path is JIT-only
                    inline_bodies.clone(),
                    cha_by_method.clone(),
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
            "wren_call_5",
            "wren_call_6",
            "wren_call_7",
            "wren_call_8",
            "wren_call_dynamic",
            "wren_load_jit_ptr",
            "wren_load_jit_closure",
            "wren_jit_roots_snapshot",
            "wren_jit_roots_restore",
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
            "wren_make_closure_5",
            "wren_make_closure_6",
            "wren_make_closure_7",
            "wren_make_closure_8",
            "wren_make_closure_n",
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
            "wren_alloc_simd4f",
            "wren_alloc_simd4i",
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
        module: &mut dyn Module,
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

    /// Whole-program method table built once at AOT-build time.
    /// Maps each method signature text to every implementation
    /// across all walked modules. The Call-site lowering consults
    /// this to devirtualize: a sig with one impl becomes a direct
    /// call (guarded by a class check); a sig with several emits
    /// a class-dispatch tree branching to each implementation.
    /// Trivial getters get inlined as a single field load instead
    /// of a call.
    pub struct AotCha {
        pub by_sig: std::collections::HashMap<String, Vec<AotMethodImpl>>,
    }

    pub struct AotMethodImpl {
        pub class_name: String,
        pub fn_symbol: String,
        pub arity: u8,
        /// `Some(idx)` if the method body is `return _field` —
        /// lets the call site emit a direct load instead of a
        /// function call. Mirrors the JIT's IC kind=5 inline.
        pub trivial_getter_field: Option<u16>,
        /// Symbol of the modvars data array that holds this
        /// class's pointer (defining module's modvars). Used at
        /// the dispatch site to load `&class` for the class
        /// check.
        pub class_modvars_symbol: String,
        pub class_slot: u32,
        /// `true` when this body was emitted with the SM poll
        /// ABI (`(fiber, resume_v) -> i64`) instead of the
        /// regular `(receiver, ...args) -> i64`. The CHA
        /// dispatch tree skips the direct-call fast block for
        /// these and routes through `wren_call_N`, which
        /// detects SM in `dispatch_method` and drives the poll
        /// fn through `call_closure_jit_or_sync`.
        pub is_state_machine: bool,
    }

    /// Per-emit defining-class context for static-field
    /// lowering. The AOT driver sets this on the cell before
    /// emitting each class method (and clears it again
    /// afterwards), so `Instruction::GetStaticField` /
    /// `SetStaticField` can load the receiver class from the
    /// defining-class slot in modvars and pass it explicitly to
    /// the helper instead of relying on `JitContext.defining_class`
    /// being threaded through TLS — which `wlift_aot_enter` does
    /// not populate.
    #[derive(Clone, Debug)]
    pub struct AotDefiningClass {
        /// Linker symbol of the defining class's owning module's
        /// modvars data array. Same names the CHA dispatch tree
        /// uses (`wlift_modvars_<n>`).
        pub modvars_symbol: String,
        /// Slot index inside that modvars array holding the
        /// `*mut ObjClass` (NaN-boxed). Populated at startup by
        /// `wlift_aot_install_class`.
        pub slot: u32,
    }

    /// Per-module data the AOT lowering needs at every emit site
    /// that today calls a TLS-routed runtime helper. Passing this
    /// switches the lowering off the JIT-shaped fast/slow paths
    /// onto direct data-section addressing — the produced object
    /// stops needing a runtime to resolve module-var slots, string
    /// constants, etc. `None` keeps JIT semantics unchanged.
    pub struct AotLoweringConfig {
        /// `cranelift_module::DataId` for this module's per-module
        /// var array (`wlift_modvars_<n>`). The lowering replaces
        /// `wren_get_module_var(slot)` / `wren_set_module_var(slot,
        /// val)` with a `global_value` + load/store at offset
        /// `slot * 8`, killing both helper calls and the TLS read.
        pub modvars_data: cranelift_module::DataId,

        /// `DataId` for this module's string-constant slot array
        /// (`wlift_consts_<n>`). The body addresses
        /// `slots[dedup[sym_idx]]` to load a `*mut ObjString` —
        /// the slot itself is populated once at startup by the
        /// per-module init pass (step 7). Lowering pushes new
        /// `(sym_idx → slot, text)` entries into `const_strings`
        /// the first time it sees a given symbol; the AOT driver
        /// reads back the populated map after lowering to size +
        /// describe the slot array's data section.
        pub consts_data: cranelift_module::DataId,
        /// Dedup table for `Instruction::ConstString` lowering.
        /// Keyed by the SymbolId (`u32`) the MIR carries; the
        /// stored value is the slot index assigned the first time
        /// the lowering encountered that symbol. Driving the slot
        /// numbering through the lowering keeps the source-of-
        /// truth for "how many slots does this module need" in
        /// the same pass that emits the loads against them.
        pub const_strings: std::cell::RefCell<Vec<(u32, String)>>,

        /// `DataId` for this module's symbol-remap table
        /// (`wlift_symbols_<n>`). Each slot is a `u64`-padded
        /// `SymbolId` re-interned in the VM's interner at
        /// startup — necessary because the MIR's SymbolIds are
        /// indices into the source's per-parse interner, but
        /// runtime helpers (`wren_call_*`, `wren_is_type`, …)
        /// expect VM-interner indices.
        ///
        /// Lowering replaces every `iconst <method_sym>` against
        /// a runtime-helper arg with `load symbols_data[slot]`,
        /// where `slot` comes from `symbol_remap` keyed by the
        /// source SymbolId.
        pub symbols_data: cranelift_module::DataId,
        /// Dedup table for symbol-remap slot allocation. Same
        /// shape as `const_strings`; the strings get re-interned
        /// in the VM's interner at startup, populating the slot
        /// array.
        pub symbol_remap: std::cell::RefCell<Vec<(u32, String)>>,

        /// `DataId` for this module's closure FuncId slot
        /// table (`wlift_closures_<n>`). MIR's
        /// `Instruction::MakeClosure { fn_id, .. }` carries a
        /// build-time-relative index into `ModuleMir::closures`
        /// — the JIT path patches these to engine FuncIds at
        /// install time (`patch_closure_ids` in vm.rs); AOT
        /// can't bake real FuncIds, so it loads them from this
        /// slot array, populated at startup by
        /// `wlift_aot_register_closure`.
        pub closures_data: cranelift_module::DataId,

        /// Whole-program method table — `None` falls back to the
        /// pre-CHA behaviour where every Call site routes
        /// through `wren_call_N`. With CHA wired, every Call to a
        /// signature with at least one user-defined
        /// implementation skips the helper entirely: 1 impl
        /// becomes a class-checked direct call (or trivial-getter
        /// inline); 2+ impls become a class-dispatch tree.
        pub cha: Option<*const AotCha>,

        /// Defining-class context for the function currently
        /// being emitted. The AOT driver flips this between
        /// emissions so each class method's
        /// `Instruction::GetStaticField` / `SetStaticField` can
        /// resolve its owning class via modvars without
        /// per-frame TLS setup. `None` for top-level bodies and
        /// closures (where static-field access is unreachable
        /// or already returns null in the legacy helper).
        pub current_defining_class: std::cell::RefCell<Option<AotDefiningClass>>,

        /// Function-scoped closure pointer for the body
        /// currently being lowered. When the MIR contains any
        /// `Instruction::GetUpvalue` / `SetUpvalue` the lowering
        /// reads `JitContext.closure` once at function entry into
        /// this Cranelift `Variable`, then every upvalue access
        /// becomes inline pointer chasing against the stored
        /// pointer — no per-access TLS read, no per-access helper
        /// call, and the value survives nested calls that
        /// re-mutate the TLS context. `None` when the function
        /// body has no upvalue access (skip the load entirely).
        /// Cleared to `None` between emissions by `emit_aot_function`.
        pub current_closure_ptr_var: std::cell::RefCell<Option<cranelift_frontend::Variable>>,

        /// Function-scoped JIT-roots snapshot. AOT lowering emits
        /// `wren_jit_roots_snapshot()` at function entry and
        /// `wren_jit_roots_restore(snapshot)` before every return,
        /// so any roots leaked into `JIT_ROOTS_STORE` by the
        /// function's allocations get released at the function
        /// boundary. Mirrors the snapshot/restore pattern
        /// `wren_call_N_inner` already uses for its arg roots —
        /// applies it function-wide for AOT bodies because they
        /// leak roots under finish_alloc's "push but don't pop"
        /// model. Cleared between emissions by `emit_aot_function`.
        pub current_jit_roots_snapshot_var:
            std::cell::RefCell<Option<cranelift_frontend::Variable>>,

        /// Shared abort-exit block emitted once at function entry.
        /// Each MIR block's lowering polls `wren_aot_check_error()`
        /// at its top and branches here when the VM is mid-error;
        /// the block restores `wren_jit_roots_snapshot_var` and
        /// returns null so the AOT-stub fast path's `has_error`
        /// route in `vm_interp::run_fiber` picks the error up. The
        /// per-opcode `has_error` check the BC interpreter does
        /// has no direct analogue in straight-line Cranelift code,
        /// so without this branch a `Fiber.abort` inside a
        /// `while (true) { … }` body keeps iterating after the
        /// abort fires. Cleared between emissions by
        /// `emit_aot_function`.
        pub current_abort_exit_block: std::cell::RefCell<Option<cranelift_codegen::ir::Block>>,

        /// State-machine layout for the function currently being
        /// lowered. `Some(_)` flips the lowering onto the stackless-
        /// coroutine shape: function signature becomes `(fiber: i64,
        /// resume_v: i64) -> i64`; a synthetic dispatch block calls
        /// `wlift_aot_sm_load_state(fiber)` and `br_table`s to one of
        /// `layout.resume_entries`; every `Terminator::Return(v)` in
        /// `layout.yield_blocks` is preceded by `wlift_aot_sm_yield(
        /// fiber, next_state)` (suspension semantics) while every
        /// other return is preceded by `wlift_aot_sm_done(fiber)`
        /// (final-value semantics). This is the AOT analogue of what
        /// the BC interp already does via `pending_fiber_action +
        /// run_fiber`. Cleared between emissions by
        /// `emit_aot_function`.
        #[cfg(feature = "aot")]
        pub current_state_machine_layout:
            std::cell::RefCell<Option<crate::codegen::aot_state_machine::StateMachineLayout>>,

        /// Function-scoped fiber-pointer variable for the
        /// state-machine body currently being lowered. Defined
        /// once at function entry from the first block param;
        /// every yield-/done-terminator emit reads it to pass
        /// `fiber` to the runtime helper. `None` when the function
        /// isn't a state machine. Cleared between emissions.
        pub current_fiber_ptr_var: std::cell::RefCell<Option<cranelift_frontend::Variable>>,

        /// Function-scoped resume-value variable. Holds the
        /// `resume_v` parameter of the state-machine poll
        /// signature so cross-fn call sites can thread it
        /// through to `wlift_aot_invoke_sm_method`. Without
        /// this, every nested cross-fn call would pass `0` as
        /// the resume value and a re-entered child state
        /// machine would lose the value the user passed to
        /// `fiber.call`.
        pub current_resume_v_var: std::cell::RefCell<Option<cranelift_frontend::Variable>>,
    }

    /// AOT entry point — populate `builder.func` with the CLIF
    /// translation of `mir`, against any `cranelift_module::Module`
    /// impl. The JIT path drives the same code via
    /// `lower_mir_to_cranelift` below; AOT's
    /// `codegen::aot::compile_to_object` calls this directly with
    /// an `ObjectModule`.
    ///
    /// `aot_config = None` keeps the JIT-shaped lowering — emits
    /// runtime helpers for TLS-routed state. `Some(&cfg)` flips
    /// the lowering to direct data-section addressing per the
    /// fields on [`AotLoweringConfig`].
    ///
    /// IC pointers, code-base, and OSR layout are JIT-only signals
    /// (runtime self-call patching, on-stack replacement entries,
    /// inline-cache snapshots) — pass `None` for all three from
    /// AOT, since `ObjectModule` outputs go through a static
    /// linker rather than the JIT's mutate-in-place finalisation.
    pub fn lower_mir_to_module(
        mir: &MirFunction,
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        aot_config: Option<&AotLoweringConfig>,
    ) -> Result<(), String> {
        lower_mir_impl(
            mir, interner, builder, module, None, None, None, None, None, aot_config, None, None,
        )
    }

    /// Lower a MIR function into Cranelift IR using the FunctionBuilder.
    #[allow(clippy::too_many_arguments)]
    fn lower_mir_to_cranelift(
        mir: &MirFunction,
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
        inline_bodies: Option<
            std::sync::Arc<std::collections::HashMap<u32, std::sync::Arc<MirFunction>>>,
        >,
        cha_by_method: crate::runtime::engine::SharedCha,
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
            None,
            inline_bodies,
            cha_by_method,
        )
    }

    /// Inner lowering with optional f64 specialization.
    /// When `f64_self_id` is Some, this function is the f64→f64 inner version:
    /// - BlockParam types are f64 (not i64)
    /// - Unbox/Box of params/returns are no-ops
    /// - CallStaticSelf calls the inner function directly with f64 args
    #[allow(clippy::too_many_arguments)] // Lowering context is inherently wide (IC, OSR, f64 inner, AOT mode, ...).
    #[allow(clippy::blocks_in_conditions)]
    fn lower_mir_impl(
        mir: &MirFunction,
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        callsite_ic_ptrs: Option<&[crate::mir::bytecode::CallSiteIC]>,
        callsite_ic_live_ptrs: Option<&[usize]>,
        jit_code_base: Option<*const *const u8>,
        f64_self_id: Option<cranelift_module::FuncId>,
        osr_entry: Option<OsrEntryLayout>,
        aot_config: Option<&AotLoweringConfig>,
        inline_bodies: Option<
            std::sync::Arc<std::collections::HashMap<u32, std::sync::Arc<MirFunction>>>,
        >,
        cha_by_method: crate::runtime::engine::SharedCha,
    ) -> Result<(), String> {
        // Map MIR blocks to Cranelift blocks
        let mut block_map: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
        for (i, _) in mir.blocks.iter().enumerate() {
            let cl_block = builder.create_block();
            block_map.insert(BlockId(i as u32), cl_block);
        }

        // Map MIR values to Cranelift values.
        let mut val_map: HashMap<ValueId, Value> = HashMap::new();
        // Per-fresh_vid Cranelift Variables for state-machine
        // resume-load values. The dispatcher's `br_table` jumps
        // directly into a resume entry, so the same fresh ValueId
        // can be defined by multiple emit sites (the call_check
        // synthetic, the post-yield block, etc.). val_map is a
        // single-binding HashMap and can't merge those defs;
        // routing them through Variables lets Cranelift's
        // SSA construction pick the right value at each use.
        // Each block iteration refreshes val_map[fresh_vid] to
        // `use_var(var)` so downstream val_map lookups still
        // return one Value but the Variable carries the
        // multi-block convergence.
        #[cfg(feature = "aot")]
        let mut var_map: HashMap<ValueId, cranelift_frontend::Variable> = HashMap::new();

        // GC stack-map marking: when an SSA value holds a Wren `Value`
        // (NaN-boxed pointer-or-scalar), we tell Cranelift to keep it
        // on the stack across safepoints so the GC can find live
        // heap pointers held in JIT frames. Without this, Cranelift's
        // regalloc is free to keep a heap pointer in a callee-saved
        // register across a call; if that call's callee allocates
        // and triggers GC, the heap pointer is invisible and the
        // object underneath it gets freed (mark-sweep) or moved
        // (generational).
        //
        // The f64 inner function never carries Wren Values across
        // calls — its parameters and locals are unboxed `f64`s, and
        // the only outbound calls are recursive self-calls that take
        // and return `f64`. Skip marking there.
        //
        // Off by default — opt in via `WLIFT_ENABLE_STACK_MAPS=1`.
        // Force-spilling every Wren value across every safepoint
        // currently amplifies an unrelated UNDEF-leak in the JIT;
        // re-enable once that init bug is fixed.
        let mark_stack_map = f64_self_id.is_none() && env_stack_maps();
        let value_types = if mark_stack_map {
            infer_osr_value_types(mir)
        } else {
            Vec::new()
        };
        let is_wren_value = |vid: ValueId, value_types: &[MirType]| -> bool {
            value_types
                .get(vid.0 as usize)
                .map(|ty| matches!(ty, MirType::Value))
                .unwrap_or(false)
        };

        if let Some(ref layout) = osr_entry {
            let osr_entry = builder.create_block();
            builder.switch_to_block(osr_entry);
            for vid in &layout.external_args {
                let param = builder.append_block_param(osr_entry, types::I64);
                val_map.insert(*vid, param);
                if mark_stack_map && is_wren_value(*vid, &value_types) {
                    builder.declare_value_needs_stack_map(param);
                }
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
        let mut get_runtime_fn = |module: &mut dyn Module,
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

        // Pre-scan: if this is an AOT body that touches upvalues, hoist
        // a `closure_ptr` Variable that we'll define once at function
        // entry and read on every GetUpvalue / SetUpvalue. Skipped when
        // the body has no upvalue ops (no overhead) and when running
        // outside AOT (the JIT helper path keeps its own TLS state).
        if let Some(cfg) = aot_config {
            let needs_closure_ptr = osr_entry.is_none()
                && f64_self_id.is_none()
                && mir.blocks.iter().any(|b| {
                    b.instructions.iter().any(|(_, i)| {
                        matches!(i, Instruction::GetUpvalue(_) | Instruction::SetUpvalue(..))
                    })
                });
            if needs_closure_ptr {
                // FunctionBuilder mints the variable index for us;
                // we just stash the returned Variable so the
                // entry-block setup + every upvalue lowering site
                // can look it up.
                //
                // NOTE: closure_ptr_var holds a raw closure pointer
                // (output of `wren_load_jit_closure`), not a
                // NaN-boxed Wren Value, so `declare_var_needs_stack_map`
                // wouldn't help — the GC walker skips spill slots
                // whose bits don't match the NaN-box tag. Refresh
                // logic for the cached pointer across GC lives in
                // the JitContext save/restore in `wlift_aot_invoke_sm_method`
                // (see `root_saved_jit_context` / `restore_rooted_jit_context`).
                let var = builder.declare_var(types::I64);
                *cfg.current_closure_ptr_var.borrow_mut() = Some(var);
            }

            // Always declare a JIT-roots snapshot Variable for AOT
            // bodies (skipping OSR + f64-inner emit paths, which
            // don't allocate Wren values). Defined at function
            // entry, read at every Return instruction's lowering
            // to restore JIT_ROOTS_STORE to its entry length —
            // releases any roots leaked into the global stack by
            // finish_alloc's "push but don't pop" model.
            if osr_entry.is_none() && f64_self_id.is_none() {
                let snap_var = builder.declare_var(types::I64);
                *cfg.current_jit_roots_snapshot_var.borrow_mut() = Some(snap_var);
            }

            // State-machine bodies need a Variable to carry the
            // fiber pointer across the function so each Return's
            // yield/done helper call can pass it. `resume_v_var`
            // similarly carries the second poll-fn param so
            // nested cross-fn invocations can thread it through.
            #[cfg(feature = "aot")]
            if cfg.current_state_machine_layout.borrow().is_some() {
                let fiber_var = builder.declare_var(types::I64);
                *cfg.current_fiber_ptr_var.borrow_mut() = Some(fiber_var);
                // Fibers are allocated straight to old gen and never
                // FORWARDED, so the pointer is stable — no stack-map
                // declaration needed on `fiber_var`.
                let resume_v_var = builder.declare_var(types::I64);
                // `resume_v` is the second poll-fn param: an
                // arbitrary Wren Value the resumer passed via
                // `fiber.call(value)`. It can be a List / Map / String
                // and therefore FORWARDED across a minor GC.
                if env_stack_maps() {
                    builder.declare_var_needs_stack_map(resume_v_var);
                }
                *cfg.current_resume_v_var.borrow_mut() = Some(resume_v_var);
            }
        }

        // State-machine prologue. Emit a synthetic dispatch block
        // BEFORE the RPO walk so it becomes Cranelift's function
        // entry (entry = first block switched to). It holds the
        // 2-param signature (`fiber`, `resume_v`), reads the
        // saved state ID via a runtime helper, and `br_table`s
        // to the right resume entry. Subsequent blocks are
        // emitted in their normal RPO order.
        #[cfg(feature = "aot")]
        if let Some(cfg) = aot_config {
            let layout = cfg.current_state_machine_layout.borrow().clone();
            if let Some(layout) = layout {
                let dispatch_block = builder.create_block();
                builder.switch_to_block(dispatch_block);
                let fiber_param = builder.append_block_param(dispatch_block, types::I64);
                let resume_v_param = builder.append_block_param(dispatch_block, types::I64);
                if let Some(var) = *cfg.current_fiber_ptr_var.borrow() {
                    builder.def_var(var, fiber_param);
                }
                if let Some(var) = *cfg.current_resume_v_var.borrow() {
                    builder.def_var(var, resume_v_param);
                }
                // Define every function-entry Variable here at
                // the actual function entry (the dispatch
                // block) — the dispatcher's `br_table` can
                // jump straight to a resume_entry block (state
                // >= 1), bypassing bb0 where the same setup
                // would otherwise run. Variables left
                // undefined under that path produce panics
                // (snap_var → out-of-bounds in
                // `wren_jit_roots_restore`) or SIGSEGVs
                // (closure_ptr_var → null deref on every
                // GetUpvalue / SetUpvalue site).
                if let Some(snap_var) = *cfg.current_jit_roots_snapshot_var.borrow() {
                    let f = get_runtime_fn(module, builder, "wren_jit_roots_snapshot", 0)?;
                    let call = builder.ins().call(f, &[]);
                    let snap = builder.inst_results(call)[0];
                    builder.def_var(snap_var, snap);
                }
                if let Some(closure_var) = *cfg.current_closure_ptr_var.borrow() {
                    let f = get_runtime_fn(module, builder, "wren_load_jit_closure", 0)?;
                    let call = builder.ins().call(f, &[]);
                    let closure_bits = builder.inst_results(call)[0];
                    builder.def_var(closure_var, closure_bits);
                }
                // Materialise function args (BlockParam vids) HERE
                // in the dispatch block, not in bb0. The dispatcher's
                // `br_table` can jump to any resume entry without
                // going through bb0 — defining the arg vids only at
                // bb0 leaves them undefined on the resume paths
                // and Cranelift's verifier rejects with "uses value
                // vN from non-dominating instM" once any post-resume
                // code references a BlockParam-bound arg.
                //
                // Reading the args here in dispatch dominates every
                // resume entry (since dispatch is the function
                // entry), so subsequent val_map lookups for arg vids
                // are SSA-correct.
                //
                // Closure bodies use slot+1 (slot 0 holds the closure
                // receiver written by CrossFnCallInit's save_arg, not
                // a user param); methods use slot directly.
                {
                    let bb0 = mir.entry_block();
                    let entry = &mir.blocks[bb0.0 as usize];
                    let is_closure_body = interner.resolve(mir.name) == "<closure>";
                    let slot_offset = if is_closure_body { 1u16 } else { 0u16 };
                    let load_fn = get_runtime_fn(module, builder, "wlift_aot_sm_load_value", 2)?;
                    for &(vid, ref inst) in &entry.instructions {
                        if let Instruction::BlockParam(idx) = inst {
                            let slot = builder
                                .ins()
                                .iconst(types::I64, (*idx + slot_offset) as i64);
                            let call = builder.ins().call(load_fn, &[fiber_param, slot]);
                            let v = builder.inst_results(call)[0];
                            val_map.insert(vid, v);
                            if *idx == 0 {
                                receiver_val = Some(v);
                            }
                            // Function args loaded here are Wren
                            // Values that persist across the entire
                            // SM body's safepoint chain. Without
                            // declaring them, Cranelift's safepoint
                            // pass doesn't spill+reload them and
                            // every subsequent use reads a stale
                            // register copy. Declare unconditionally
                            // on the entry block — for SM-tainted
                            // bodies, function args are always
                            // i64-shaped Wren Value bits coming back
                            // from the slot save. (`is_wren_value`
                            // checks `value_types` keyed by ValueId,
                            // but the entry block's BlockParam ids
                            // can fall outside the populated range
                            // for SM bodies, dropping the gate
                            // unconditionally.)
                            if mark_stack_map {
                                builder.declare_value_needs_stack_map(v);
                            }
                        }
                    }
                }

                let load_state = get_runtime_fn(module, builder, "wlift_aot_sm_load_state", 1)?;
                let call = builder.ins().call(load_state, &[fiber_param]);
                let state_id_64 = builder.inst_results(call)[0];
                let state_id = builder.ins().ireduce(types::I32, state_id_64);
                // br_table over the resume entries. Cranelift
                // requires a default block for out-of-range
                // values; use the first resume entry as the
                // default since reaching an out-of-range state
                // ID means the state struct was corrupted (the
                // function would have set it to one of these).
                let default_block = block_map[&layout.resume_entries[0]];
                let mut jt_data = cranelift_codegen::ir::JumpTableData::new(
                    builder.func.dfg.block_call(default_block, &[]),
                    &layout
                        .resume_entries
                        .iter()
                        .map(|bid| builder.func.dfg.block_call(block_map[bid], &[]))
                        .collect::<Vec<_>>(),
                );
                let _ = &mut jt_data;
                let jt = builder.create_jump_table(jt_data);
                builder.ins().br_table(state_id, jt);
            }
        }

        // Process blocks in reverse post-order (dominance order).
        // The MIR block array may have preheader blocks (bb4) listed after
        // loop bodies (bb2), but Cranelift requires values to be defined
        // before use. RPO guarantees dominators come first.
        let rpo = match osr_entry.as_ref() {
            Some(layout) => compute_rpo_from(mir, layout.target_block),
            #[cfg(feature = "aot")]
            None => {
                // SM resume entries are reached only via the synthetic
                // dispatch's `br_table`, so a plain DFS-from-bb0 leaves
                // them and every block downstream of them off the RPO
                // walk — they fall into the "unreachable" tail in raw
                // block-index order. That order isn't a valid dominance
                // order: a CrossFnCallResume block emits the
                // `wlift_aot_sm_load_value` defs for its
                // saved-across-yield ValueIds, and any post-resume
                // block that uses those defs must come AFTER it in
                // RPO. Without seeding DFS from every resume entry,
                // the post-resume block lowers first and Cranelift
                // rejects the use of an undefined SSA value (e.g.
                // `App.serve_(_)`'s SSE branch references the load
                // for the `conn` save-slot before the load is
                // emitted, surfacing as `undefined value v47`).
                let mut roots: Vec<BlockId> = vec![BlockId(0)];
                if let Some(cfg) = aot_config {
                    if let Some(layout) = cfg.current_state_machine_layout.borrow().as_ref() {
                        roots.extend(layout.resume_entries.iter().copied());
                    }
                }
                compute_rpo_multi_root(mir, &roots)
            }
            #[cfg(not(feature = "aot"))]
            None => compute_rpo(mir),
        };
        // Determine reachability from bb0 (or osr_entry's
        // start) over the post-transform MIR. The SM transform's
        // tail-duplication can leave the original (pre-clone)
        // blocks unreachable. Emitting them via the regular
        // lowering path would try to bind operands to ValueIds
        // whose def was dropped during the split — and there's
        // no `val_map` entry for those ValueIds. Cranelift
        // requires every created block to have a terminator
        // though, so emit a `trap` for unreachable blocks
        // instead of just skipping them. SM resume entries are
        // reached only via the synthetic dispatch's `br_table`
        // (not from bb0), so seed the walk with them too.
        let reachable: std::collections::HashSet<usize> = {
            let start = osr_entry
                .as_ref()
                .map(|l| l.target_block.0 as usize)
                .unwrap_or(0);
            let mut seen = std::collections::HashSet::new();
            let mut stack = vec![start];
            #[cfg(feature = "aot")]
            if let Some(cfg) = aot_config {
                if let Some(layout) = cfg.current_state_machine_layout.borrow().as_ref() {
                    for entry in &layout.resume_entries {
                        stack.push(entry.0 as usize);
                    }
                }
            }
            while let Some(i) = stack.pop() {
                if !seen.insert(i) {
                    continue;
                }
                if i < mir.blocks.len() {
                    for s in mir.blocks[i].terminator.successors() {
                        stack.push(s.0 as usize);
                    }
                }
            }
            seen
        };

        #[cfg_attr(not(feature = "aot"), allow(unused_labels))]
        'block_loop: for &block_idx in &rpo {
            let block = &mir.blocks[block_idx];
            let bid = BlockId(block_idx as u32);
            let cl_block = block_map[&bid];
            builder.switch_to_block(cl_block);

            if !reachable.contains(&block_idx) {
                builder
                    .ins()
                    .trap(cranelift_codegen::ir::TrapCode::user(2).unwrap());
                continue 'block_loop;
            }

            // Synthetic CrossFnCallResume block: the block has
            // no MIR-level instructions; the lowering replaces
            // its body entirely with `invoke_sm_method + peek
            // + brif`. Reached from the matching CrossFnCallInit
            // block via a direct jump AND from the dispatch's
            // br_table when the caller resumes after a child
            // yield.
            #[cfg(feature = "aot")]
            'cross_fn_resume: {
                use crate::codegen::aot_state_machine::BlockKind;
                let cfg = match aot_config {
                    Some(c) => c,
                    None => break 'cross_fn_resume,
                };
                let kind = cfg
                    .current_state_machine_layout
                    .borrow()
                    .as_ref()
                    .and_then(|l| l.block_kinds.get(&bid).cloned());
                let Some(BlockKind::CrossFnCallResume {
                    done_block,
                    receiver,
                    args,
                    result,
                    method_sym,
                }) = kind
                else {
                    break 'cross_fn_resume;
                };
                let fiber = match *cfg.current_fiber_ptr_var.borrow() {
                    Some(v) => builder.use_var(v),
                    None => break 'cross_fn_resume,
                };
                // Emit the resume_loads prologue — the dispatcher's
                // `br_table` jumps directly to this synthetic
                // call_check block, so saved-across-yield values
                // need a dominating load HERE before any
                // post-call code reads them. The transform now
                // keys these on the call_check id (not the
                // done_block) for exactly this reason.
                {
                    let layout_clone = cfg.current_state_machine_layout.borrow().clone();
                    if let Some(layout) = layout_clone {
                        if let Some(loads) = layout.resume_loads.get(&bid) {
                            let load_fn =
                                get_runtime_fn(module, builder, "wlift_aot_sm_load_value", 2)?;
                            for (slot, fresh_vid) in loads {
                                let slot_v = builder.ins().iconst(types::I64, *slot as i64);
                                let call = builder.ins().call(load_fn, &[fiber, slot_v]);
                                let v = builder.inst_results(call)[0];
                                let var = *var_map
                                    .entry(*fresh_vid)
                                    .or_insert_with(|| builder.declare_var(types::I64));
                                builder.def_var(var, v);
                                val_map.insert(*fresh_vid, v);
                                // SM-generated fresh_vids never appear as
                                // BlockParam / Instruction defs in the
                                // MIR, so `infer_osr_value_types` leaves
                                // their entry at the default `MirType::Void`
                                // and the `is_wren_value` gate would skip
                                // declaration. Declare unconditionally —
                                // resume_loads always materialise Wren
                                // Value bits read from fiber.saved_values.
                                if mark_stack_map {
                                    builder.declare_value_needs_stack_map(v);
                                }
                            }
                        }
                    }
                }
                // 1) invoke. Receiver/args were saved into the
                // child frame slots either by the matching
                // Init block (initial entry) or by a previous
                // suspension's still-live frame (resume entry).
                let invoke_fn = get_runtime_fn(module, builder, "wlift_aot_invoke_sm_method", 6)?;
                let vm_fn = get_runtime_fn(module, builder, "wren_load_jit_vm", 0)?;
                let vm_call = builder.ins().call(vm_fn, &[]);
                let vm = builder.inst_results(vm_call)[0];
                // Always read the receiver from slot 0 of the
                // child (top) frame. The Init block stashed it
                // there, and on resume the dispatcher's br_table
                // enters this synthetic block without going
                // through Init — so the caller's val_map for
                // the original receiver ValueId may not even
                // dominate this block. Reading from the saved
                // slot keeps the Cranelift IR
                // dominance-correct.
                let _ = receiver;
                let load_arg_fn = get_runtime_fn(module, builder, "wlift_aot_sm_load_arg", 2)?;
                let zero = builder.ins().iconst(types::I64, 0);
                let recv_call = builder.ins().call(load_arg_fn, &[fiber, zero]);
                let recv_v = builder.inst_results(recv_call)[0];
                // recv_v lives across trace_load_fn (a no-op when
                // tracing disabled) and the actual invoke_call below.
                // Both can safepoint; without a stack-map declaration
                // here the spill of recv_v keeps the pre-GC pointer
                // and the invoke runs with a stale receiver. The
                // load returned a Wren Value, so it's always GC-relevant.
                if env_stack_maps() {
                    builder.declare_value_needs_stack_map(recv_v);
                }
                // Trace the loaded recv against the receiver MIR
                // ValueId so a save→load mismatch becomes visible.
                let trace_load_fn =
                    get_runtime_fn(module, builder, "wlift_aot_trace_init_load", 4)?;
                let recv_id = builder.ins().iconst(types::I64, receiver.0 as i64);
                let _ = builder
                    .ins()
                    .call(trace_load_fn, &[fiber, zero, recv_id, recv_v]);
                let symbols_gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
                let symbols_addr = builder.ins().global_value(types::I64, symbols_gv);
                let sym_slot = aot_intern_symbol(cfg, method_sym.index(), interner);
                let sym_v = builder.ins().load(
                    types::I64,
                    cranelift_codegen::ir::MemFlagsData::trusted(),
                    symbols_addr,
                    (sym_slot as i32) * 8,
                );
                let num_args_v = builder.ins().iconst(types::I64, args.len() as i64);
                let resume_v = if let Some(rv) = *cfg.current_resume_v_var.borrow() {
                    builder.use_var(rv)
                } else {
                    builder.ins().iconst(types::I64, 0)
                };
                let invoke_call = builder
                    .ins()
                    .call(invoke_fn, &[vm, fiber, recv_v, sym_v, num_args_v, resume_v]);
                let ret = builder.inst_results(invoke_call)[0];
                // ret is the cross-fn callee's return Value. It lives
                // across pop_frame / clear_poll_kind / save_value
                // below in the done branch (and across the brif's
                // bitcast in the propagate branch). Any of those can
                // safepoint, so declare it.
                if env_stack_maps() {
                    builder.declare_value_needs_stack_map(ret);
                }
                // 2) peek + brif.
                let peek_fn = get_runtime_fn(module, builder, "wlift_aot_sm_peek_poll_kind", 0)?;
                let peek_call = builder.ins().call(peek_fn, &[]);
                let kind_64 = builder.inst_results(peek_call)[0];
                let yield_const = builder.ins().iconst(types::I64, 1);
                let is_yield = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    kind_64,
                    yield_const,
                );
                let propagate_block = builder.create_block();
                let done_cl_block_local = builder.create_block();
                builder
                    .ins()
                    .brif(is_yield, propagate_block, &[], done_cl_block_local, &[]);
                // 3) propagate: just Return ret.
                builder.switch_to_block(propagate_block);
                let return_ty = builder.func.signature.returns[0].value_type;
                let ret_propagate = if builder.func.dfg.value_type(ret) != return_ty {
                    builder.ins().bitcast(
                        return_ty,
                        cranelift_codegen::ir::MemFlagsData::new(),
                        ret,
                    )
                } else {
                    ret
                };
                builder.ins().return_(&[ret_propagate]);
                // 4) done: pop frame, clear kind, jump to
                //    done_block. The MIR transform already
                //    rewrote downstream uses of `result` to
                //    a fresh ValueId loaded by the resume
                //    block's prologue (cap-1's resume_loads),
                //    so we only need the runtime to put the
                //    call's ret in the right slot. We use
                //    save_value (writes to caller's
                //    active-depth frame) at the slot the
                //    transform allocated for `result`.
                builder.switch_to_block(done_cl_block_local);
                let pop_fn = get_runtime_fn(module, builder, "wlift_aot_sm_pop_frame", 1)?;
                let _ = builder.ins().call(pop_fn, &[fiber]);
                let clear_fn = get_runtime_fn(module, builder, "wlift_aot_sm_clear_poll_kind", 0)?;
                let _ = builder.ins().call(clear_fn, &[]);
                // Bind the result. val_map.insert dominates the
                // immediate done_block jump; ALSO save to a
                // per-call slot so the done_block's prologue can
                // load_value it back. Without the slot, any
                // post-done block reached via a tail-duplicated
                // path (or any CFG join) would see `result` as
                // an undefined SSA value — the verifier rejects
                // with "uses value vN from non-dominating instM".
                val_map.insert(result, ret);
                let result_slot = cfg
                    .current_state_machine_layout
                    .borrow()
                    .as_ref()
                    .and_then(|l| l.cross_fn_results.get(&done_block).copied());
                if let Some((slot, _)) = result_slot {
                    let save_fn = get_runtime_fn(module, builder, "wlift_aot_sm_save_value", 3)?;
                    let slot_v = builder.ins().iconst(types::I64, slot as i64);
                    let _ = builder.ins().call(save_fn, &[fiber, slot_v, ret]);
                }
                let target = block_map[&done_block];
                builder.ins().jump(target, &[]);
                // Skip the regular per-instruction + terminator
                // emission for this block.
                continue 'block_loop;
            }

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
                if mark_stack_map && matches!(ty, MirType::Value) {
                    builder.declare_value_needs_stack_map(param);
                }
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
                } else if {
                    #[cfg(feature = "aot")]
                    {
                        aot_config
                            .map(|c| c.current_state_machine_layout.borrow().is_some())
                            .unwrap_or(false)
                    }
                    #[cfg(not(feature = "aot"))]
                    {
                        false
                    }
                } {
                    // State-machine entry: bb0 is reached via
                    // the dispatch block's `br_table`, which
                    // doesn't pass any block args. Don't append
                    // block params — the (fiber, resume_v)
                    // signature lives on the dispatch block.
                    //
                    // BlockParam(idx) vids are already bound in
                    // val_map by the dispatch block's load_value
                    // sequence; that one dominates every resume
                    // entry too, so we don't re-emit loads here.
                    // (Re-emitting would overwrite val_map[vid]
                    // with a Value defined inside bb0, which the
                    // resume-only paths don't dominate — exactly
                    // the bug we're fixing.)
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
                                if mark_stack_map && is_wren_value(vid, &value_types) {
                                    builder.declare_value_needs_stack_map(entry_params[idx]);
                                }
                            }
                        }
                    }
                }

                // AOT-only: define the function-scoped closure-pointer
                // variable from `JitContext.closure`. The Variable was
                // declared above; defining it here in the entry block
                // makes it available to every subsequent GetUpvalue /
                // SetUpvalue lowering site without re-reading TLS or
                // calling the per-access helper.
                if let Some(cfg) = aot_config {
                    if let Some(var) = *cfg.current_closure_ptr_var.borrow() {
                        let f = get_runtime_fn(module, builder, "wren_load_jit_closure", 0)?;
                        let call = builder.ins().call(f, &[]);
                        let closure_bits = builder.inst_results(call)[0];
                        builder.def_var(var, closure_bits);
                    }

                    // AOT-only: snapshot JIT_ROOTS_STORE.len() at
                    // function entry. Each Return restores to this
                    // length, releasing any roots leaked into the
                    // global stack by finish_alloc's "push but
                    // don't pop" mode. Mirrors wren_call_N_inner's
                    // root_base / jit_roots_restore_len pair.
                    if let Some(snap_var) = *cfg.current_jit_roots_snapshot_var.borrow() {
                        let f = get_runtime_fn(module, builder, "wren_jit_roots_snapshot", 0)?;
                        let call = builder.ins().call(f, &[]);
                        let snap = builder.inst_results(call)[0];
                        builder.def_var(snap_var, snap);
                    }
                }
            }

            // State-machine resume entry: emit
            // `wlift_aot_sm_load_value(fiber, slot)` for every
            // value live across the corresponding suspension,
            // and define each as the freshly-allocated ValueId
            // the transform inserted during MIR rewriting. The
            // remaining instructions in this block reference
            // those fresh ValueIds, not the originals.
            #[cfg(feature = "aot")]
            if let Some(cfg) = aot_config {
                if let Some(layout) = cfg.current_state_machine_layout.borrow().clone() {
                    if let Some(loads) = layout.resume_loads.get(&bid) {
                        if let Some(fiber_var) = *cfg.current_fiber_ptr_var.borrow() {
                            let fiber = builder.use_var(fiber_var);
                            let load_fn =
                                get_runtime_fn(module, builder, "wlift_aot_sm_load_value", 2)?;
                            for (slot, fresh_vid) in loads {
                                let slot_v = builder.ins().iconst(types::I64, *slot as i64);
                                let call = builder.ins().call(load_fn, &[fiber, slot_v]);
                                let v = builder.inst_results(call)[0];
                                let var = *var_map
                                    .entry(*fresh_vid)
                                    .or_insert_with(|| builder.declare_var(types::I64));
                                builder.def_var(var, v);
                                val_map.insert(*fresh_vid, v);
                                // Same reasoning as the CrossFnCallResume
                                // resume_loads above: SM-generated
                                // fresh_vids fall outside
                                // `infer_osr_value_types`'s populated
                                // range, so declare unconditionally.
                                if mark_stack_map {
                                    builder.declare_value_needs_stack_map(v);
                                }
                            }
                        }
                    }
                    // DirectYield resume: bind the suspension Call's
                    // result ValueId to the `resume_v` poll-fn
                    // parameter. The transform dropped the Call when
                    // splitting (so the ValueId has no MIR-level
                    // def), but post-yield code may reference it
                    // (e.g. `var x = Fiber.yield()` then uses `x`).
                    // The resumer's `fiber.call(value)` threaded
                    // `value` through to `resume_v`; that's exactly
                    // what the yield's result should bind to.
                    if let Some(call_dst) = layout.direct_yield_results.get(&bid).copied() {
                        if let Some(resume_var) = *cfg.current_resume_v_var.borrow() {
                            let v = builder.use_var(resume_var);
                            val_map.insert(call_dst, v);
                            // `call_dst` was the dropped Fiber.yield
                            // call's MIR result. The SM transform
                            // generated it via mir.new_value(), so
                            // `infer_osr_value_types` leaves the entry
                            // at MirType::Void and `is_wren_value`
                            // returns false. Declare unconditionally —
                            // a Fiber.yield result is always a Wren
                            // Value (whatever the resumer's
                            // `fiber.call(value)` passed back).
                            if mark_stack_map {
                                builder.declare_value_needs_stack_map(v);
                            }
                        }
                    }
                    // CrossFnCall done_block prologue: load the
                    // call's result from the slot the done branch
                    // saved into. Provides a dominating def for
                    // every downstream use of `result`, regardless
                    // of how the post-done CFG was duplicated or
                    // joined.
                    if let Some((slot, call_dst)) = layout.cross_fn_results.get(&bid).copied() {
                        if let Some(fiber_var) = *cfg.current_fiber_ptr_var.borrow() {
                            let fiber = builder.use_var(fiber_var);
                            let load_fn =
                                get_runtime_fn(module, builder, "wlift_aot_sm_load_value", 2)?;
                            let slot_v = builder.ins().iconst(types::I64, slot as i64);
                            let call = builder.ins().call(load_fn, &[fiber, slot_v]);
                            let v = builder.inst_results(call)[0];
                            val_map.insert(call_dst, v);
                            // `call_dst` is the original cross-fn call's
                            // result ValueId, generated by the SM
                            // transform's `mir.new_value()` and so out
                            // of `infer_osr_value_types`'s range.
                            // Declare unconditionally — the cross-fn
                            // call's runtime return is always a Wren
                            // Value.
                            if mark_stack_map {
                                builder.declare_value_needs_stack_map(v);
                            }
                        }
                    }
                }
            }

            // Poll `vm.has_error` at every block entry except bb0
            // (which sits behind the function-entry snapshot that
            // can't have observed an in-flight error). Without this,
            // a `Fiber.abort` mid-loop sets the flag but the AOT
            // body keeps iterating — the BC interp's per-opcode
            // check has no analogue in straight-line Cranelift code,
            // so a spec like `JSON.parse("{")` re-aborts forever
            // at successive offsets. Check + brif to a shared
            // abort-exit block, lazy-created on first need; the
            // post-loop emit fills in its body (roots-restore +
            // typed null return).
            if let Some(cfg) = aot_config {
                if block_idx != 0
                    && cfg.current_jit_roots_snapshot_var.borrow().is_some()
                    && f64_self_id.is_none()
                {
                    let existing = *cfg.current_abort_exit_block.borrow();
                    let abort_exit = match existing {
                        Some(b) => b,
                        None => {
                            let b = builder.create_block();
                            *cfg.current_abort_exit_block.borrow_mut() = Some(b);
                            b
                        }
                    };
                    let f = get_runtime_fn(module, builder, "wren_aot_check_error", 0)?;
                    let call = builder.ins().call(f, &[]);
                    let err = builder.inst_results(call)[0];
                    let cont = builder.create_block();
                    builder.ins().brif(err, abort_exit, &[], cont, &[]);
                    builder.switch_to_block(cont);
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
                    aot_config,
                    inline_bodies.as_ref(),
                    cha_by_method.as_ref(),
                )?;
                if let Some(val) = result {
                    val_map.insert(vid, val);
                    if is_raw_bool {
                        raw_bools.insert(vid);
                    }
                    if mark_stack_map && is_wren_value(vid, &value_types) {
                        builder.declare_value_needs_stack_map(val);
                    }
                }
            }

            // Pre-terminator hook: AOT bodies emit a JIT-roots
            // restore right before every Return so any roots
            // leaked into JIT_ROOTS_STORE by alloc helpers
            // (finish_alloc's "push but don't pop" mode) get
            // released at the function boundary. The snapshot
            // Variable was defined at function entry above.
            //
            // For state-machine cross-fn call sites we
            // override the terminator entirely (the propagate
            // branch returns, the done branch jumps); track
            // that here so the generic terminator emission
            // below skips the duplicate.
            #[cfg_attr(not(feature = "aot"), allow(unused_mut))]
            let mut skip_default_terminator = false;
            // The pre-terminator hook fires for both Return-
            // (DirectYield) and Branch- (CrossFnCallInit, where
            // the transform wired the yielding block to its
            // synthetic call_check via Branch so compute_rpo
            // walks them in the right order) terminated blocks
            // when the layout has a block_kind for them.
            #[cfg(feature = "aot")]
            let has_state_machine_kind = aot_config
                .and_then(|c| {
                    c.current_state_machine_layout
                        .borrow()
                        .as_ref()
                        .map(|l| l.block_kinds.contains_key(&bid))
                })
                .unwrap_or(false);
            #[cfg(not(feature = "aot"))]
            let has_state_machine_kind = false;
            if matches!(
                block.terminator,
                Terminator::Return(_) | Terminator::ReturnNull
            ) || (has_state_machine_kind
                && matches!(block.terminator, Terminator::Branch { .. }))
            {
                if let Some(cfg) = aot_config {
                    if let Some(snap_var) = *cfg.current_jit_roots_snapshot_var.borrow() {
                        let snap = builder.use_var(snap_var);
                        let f = get_runtime_fn(module, builder, "wren_jit_roots_restore", 1)?;
                        let _ = builder.ins().call(f, &[snap]);
                    }

                    // State-machine bodies stamp the
                    // suspension/done semantics into the runtime
                    // *before* the bare Return — so the dispatcher
                    // running this poll sees `kind=Yield` and
                    // resumes from the saved state on next call,
                    // or `kind=Done` and marks the fiber finished.
                    #[cfg(feature = "aot")]
                    if let Some(layout) = cfg.current_state_machine_layout.borrow().clone() {
                        if let Some(fiber_var) = *cfg.current_fiber_ptr_var.borrow() {
                            let fiber = builder.use_var(fiber_var);
                            if let Some(next_state) = layout.yield_blocks.get(&bid) {
                                // Save every live-across value
                                // before the suspension. The
                                // matching resume block prologue
                                // loads them back into fresh
                                // ValueIds the transform already
                                // wired into downstream
                                // instructions.
                                if let Some(saves) = layout.yield_saves.get(&bid) {
                                    let save_fn = get_runtime_fn(
                                        module,
                                        builder,
                                        "wlift_aot_sm_save_value",
                                        3,
                                    )?;
                                    for (slot, vid) in saves {
                                        let v = match val_map.get(vid) {
                                            Some(v) => *v,
                                            None => continue,
                                        };
                                        let slot_v = builder.ins().iconst(types::I64, *slot as i64);
                                        let _ = builder.ins().call(save_fn, &[fiber, slot_v, v]);
                                    }
                                }
                                // Branch on kind: DirectYield
                                // emits the yield-stamp + plain
                                // Return; CrossFnCall builds the
                                // push/save/invoke/peek/branch
                                // sequence and replaces the
                                // Return's value with the
                                // call's runtime result.
                                use crate::codegen::aot_state_machine::BlockKind;
                                let kind_record = layout.block_kinds.get(&bid).cloned();
                                match kind_record {
                                    Some(BlockKind::CrossFnCallInit {
                                        resume_check_block,
                                        receiver,
                                        args,
                                        result: _result,
                                        method_sym: _method_sym,
                                    }) => {
                                        // Init half: advance own
                                        // state to the call_check
                                        // block's state, push a
                                        // new frame for the
                                        // callee, save args, and
                                        // jump to the call_check
                                        // block. The actual
                                        // invoke + kind-check +
                                        // propagate is in the
                                        // CrossFnCallResume
                                        // lowering below, which
                                        // both this jump and the
                                        // dispatcher's `br_table`
                                        // land in.
                                        let set_state_fn = get_runtime_fn(
                                            module,
                                            builder,
                                            "wlift_aot_sm_set_state",
                                            2,
                                        )?;
                                        let ns =
                                            builder.ins().iconst(types::I64, *next_state as i64);
                                        let _ = builder.ins().call(set_state_fn, &[fiber, ns]);
                                        let push_fn = get_runtime_fn(
                                            module,
                                            builder,
                                            "wlift_aot_sm_push_frame",
                                            1,
                                        )?;
                                        let _ = builder.ins().call(push_fn, &[fiber]);
                                        let save_arg_fn = get_runtime_fn(
                                            module,
                                            builder,
                                            "wlift_aot_sm_save_arg",
                                            3,
                                        )?;
                                        let trace_init_fn = get_runtime_fn(
                                            module,
                                            builder,
                                            "wlift_aot_trace_init_save",
                                            4,
                                        )?;
                                        let recv_v = match val_map.get(&receiver) {
                                            Some(v) => *v,
                                            None => builder.ins().iconst(types::I64, 0),
                                        };
                                        let zero = builder.ins().iconst(types::I64, 0);
                                        let _ =
                                            builder.ins().call(save_arg_fn, &[fiber, zero, recv_v]);
                                        let recv_id =
                                            builder.ins().iconst(types::I64, receiver.0 as i64);
                                        let _ = builder
                                            .ins()
                                            .call(trace_init_fn, &[fiber, zero, recv_id, recv_v]);
                                        for (i, a) in args.iter().enumerate() {
                                            let av = match val_map.get(a) {
                                                Some(v) => *v,
                                                None => builder.ins().iconst(types::I64, 0),
                                            };
                                            let slot =
                                                builder.ins().iconst(types::I64, (i + 1) as i64);
                                            let _ =
                                                builder.ins().call(save_arg_fn, &[fiber, slot, av]);
                                            let arg_id =
                                                builder.ins().iconst(types::I64, a.0 as i64);
                                            let _ = builder
                                                .ins()
                                                .call(trace_init_fn, &[fiber, slot, arg_id, av]);
                                        }
                                        let target = block_map[&resume_check_block];
                                        builder.ins().jump(target, &[]);
                                        // Generic terminator
                                        // emission below would
                                        // try to lower the bare
                                        // Return(result) on the
                                        // already-terminated
                                        // block — switch to a
                                        // fresh dead block to
                                        // absorb its `return 0`.
                                        let dead_block = builder.create_block();
                                        builder.switch_to_block(dead_block);
                                        skip_default_terminator = true;
                                    }
                                    _ => {
                                        // DirectYield (or no
                                        // BlockKind for legacy
                                        // closures pre-cap-3):
                                        // existing yield-stamp +
                                        // bare Return treatment.
                                        let yield_fn = get_runtime_fn(
                                            module,
                                            builder,
                                            "wlift_aot_sm_yield",
                                            2,
                                        )?;
                                        let next_state_v =
                                            builder.ins().iconst(types::I64, *next_state as i64);
                                        let _ =
                                            builder.ins().call(yield_fn, &[fiber, next_state_v]);
                                    }
                                }
                            } else {
                                let done_fn =
                                    get_runtime_fn(module, builder, "wlift_aot_sm_done", 1)?;
                                let _ = builder.ins().call(done_fn, &[fiber]);
                            }
                        }
                    }
                }
            }

            // Lower terminator (unless the cross-fn lowering
            // already emitted the propagate / done branches and
            // switched to a dead block — see
            // `skip_default_terminator` above).
            if !skip_default_terminator {
                lower_terminator(&block.terminator, builder, &val_map, &block_map, &raw_bools)?;
            } else {
                // The dead-block terminator must still be
                // well-formed for Cranelift's verifier. A bare
                // `return 0` is unreachable but valid.
                let zero = builder.ins().iconst(types::I64, 0);
                builder.ins().return_(&[zero]);
            }
        }

        // Fill in the shared abort-exit block (if any block needed
        // it). Restores the function-entry roots snapshot, then
        // returns a typed null so the AOT-stub fast path's
        // `has_error` route in `vm_interp::run_fiber` picks the
        // error up. Returning the function's declared type avoids
        // a Cranelift verifier mismatch on f64-inner emit paths
        // (which never set this block in the first place).
        if let Some(cfg) = aot_config {
            if let Some(abort_exit) = *cfg.current_abort_exit_block.borrow() {
                builder.switch_to_block(abort_exit);
                if let Some(snap_var) = *cfg.current_jit_roots_snapshot_var.borrow() {
                    let snap = builder.use_var(snap_var);
                    let f = get_runtime_fn(module, builder, "wren_jit_roots_restore", 1)?;
                    let _ = builder.ins().call(f, &[snap]);
                }
                let return_ty = builder.func.signature.returns[0].value_type;
                let null = if return_ty == types::F64 {
                    let zero = builder.ins().iconst(types::I64, TAG_NULL as i64);
                    builder.ins().bitcast(types::F64, MemFlags::new(), zero)
                } else {
                    builder.ins().iconst(return_ty, TAG_NULL as i64)
                };
                builder.ins().return_(&[null]);
            }
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
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
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

    /// Look up a SymbolId in the per-module symbol-remap dedup
    /// table (or push a fresh entry) and return its slot index
    /// inside `wlift_symbols_<n>`. The bootstrap re-interns each
    /// stored name in the VM's interner at startup so the slot
    /// reads back the right VM-side SymbolId at runtime.
    fn aot_intern_symbol(cfg: &AotLoweringConfig, sym_id: u32, interner: &Interner) -> usize {
        let mut tbl = cfg.symbol_remap.borrow_mut();
        if let Some(idx) = tbl.iter().position(|(s, _)| *s == sym_id) {
            return idx;
        }
        let text = interner
            .resolve(crate::intern::SymbolId::from_raw(sym_id))
            .to_string();
        tbl.push((sym_id, text));
        tbl.len() - 1
    }

    /// Lower a single MIR instruction to Cranelift IR.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)] // Instruction lowering threads builder/module/val-map/IC/JIT-code-base — wide by design.
    fn lower_instruction(
        inst: &Instruction,
        _mir: &MirFunction,
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        val_map: &HashMap<ValueId, Value>,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
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
        aot_config: Option<&AotLoweringConfig>,
        inline_bodies: Option<
            &std::sync::Arc<std::collections::HashMap<u32, std::sync::Arc<MirFunction>>>,
        >,
        cha_by_method: Option<&std::sync::Arc<crate::runtime::engine::ChaMap>>,
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
                // Inline numeric fast path: if `a` is a number, fneg
                // and bitcast back; otherwise dispatch the user's
                // prefix `-` operator via the runtime helper.
                let la = get(a);
                let qnan = builder.ins().iconst(types::I64, QNAN as i64);
                let fast_block = builder.create_block();
                let slow_block = builder.create_block();
                let merge_block = builder.create_block();
                builder.append_block_param(merge_block, types::I64);

                let masked = builder.ins().band(la, qnan);
                let is_nan_box = builder.ins().icmp(IntCC::Equal, masked, qnan);
                builder
                    .ins()
                    .brif(is_nan_box, slow_block, &[], fast_block, &[]);

                builder.switch_to_block(fast_block);
                let fa = builder.ins().bitcast(types::F64, MemFlags::new(), la);
                let fneg = builder.ins().fneg(fa);
                let ineg = builder.ins().bitcast(types::I64, MemFlags::new(), fneg);
                builder.ins().jump(merge_block, &[BlockArg::Value(ineg)]);

                builder.switch_to_block(slow_block);
                let f = get_runtime_fn(module, builder, "wren_num_neg", 1)?;
                let call = builder.ins().call(f, &[la]);
                let slow_result = builder.inst_results(call)[0];
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(slow_result)]);

                builder.switch_to_block(merge_block);
                Ok(Some(builder.block_params(merge_block)[0]))
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
                // Inline numeric fast path: when both operands are nums,
                // f64-compare directly. `FloatCC::Equal` is ordered, so
                // NaN != NaN as IEEE / Wren both expect. Slow path
                // (wren_cmp_eq) handles object content equality (notably
                // String content compare) and operator overloads.
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Cmp(FloatCC::Equal),
                    "wren_cmp_eq",
                )
            }
            Instruction::CmpNe(a, b) => {
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Cmp(FloatCC::NotEqual),
                    "wren_cmp_ne",
                )
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
            //
            // AOT mode: each module owns a `wlift_modvars_<n>`
            // data symbol — a `[u64; var_count]` in `.bss` —
            // declared by the AOT driver and threaded through
            // here as a `DataId`. Get/Set become a `global_value`
            // load + offset, killing the runtime helper entirely.
            //
            // JIT mode keeps the `wren_get/set_module_var` dispatch
            // — those helpers consult the TLS `JitContext` which
            // is patched per-frame by the install path.
            Instruction::GetModuleVar(idx) => {
                if let Some(cfg) = aot_config {
                    let gv = module.declare_data_in_func(cfg.modvars_data, builder.func);
                    let base = builder.ins().global_value(types::I64, gv);
                    let result = builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        base,
                        (*idx as i32) * 8,
                    );
                    Ok(Some(result))
                } else {
                    let f = get_runtime_fn(module, builder, "wren_get_module_var", 1)?;
                    let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                    let result = builder.ins().call(f, &[idx_val]);
                    Ok(Some(builder.inst_results(result)[0]))
                }
            }
            Instruction::SetModuleVar(idx, val) => {
                if let Some(cfg) = aot_config {
                    let gv = module.declare_data_in_func(cfg.modvars_data, builder.func);
                    let base = builder.ins().global_value(types::I64, gv);
                    let store_val = get(val);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), store_val, base, (*idx as i32) * 8);
                    // SetModuleVar's MIR contract: result is the
                    // stored value (mirrors the helper's return).
                    Ok(Some(store_val))
                } else {
                    let f = get_runtime_fn(module, builder, "wren_set_module_var", 2)?;
                    let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                    let result = builder.ins().call(f, &[idx_val, get(val)]);
                    Ok(Some(builder.inst_results(result)[0]))
                }
            }

            // === Method calls — inline IC fast path + wren_call_N slow path ===
            Instruction::Call {
                receiver,
                method,
                args,
                pure_call: _,
            } => {
                // Calls with > 8 user args route through
                // `wren_call_dynamic(receiver, method, count, ptr)`
                // — Cranelift can't pass more than 8 i64s in
                // registers without spilling, so spill to a
                // stack-allocated `[u64; n]` buffer once and let
                // the dispatcher walk it. Skips CHA / IC fast
                // paths to keep the lowering simple; AOT bodies
                // with > 8-arg method calls are rare enough
                // (`@hatch:gpu`'s pipeline setup) that the loss
                // of devirt isn't a hot-path concern.
                if args.len() > 8 {
                    let r = get(receiver);
                    let method_val = if let Some(cfg) = aot_config {
                        let slot = aot_intern_symbol(cfg, method.index(), interner);
                        let sym_gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
                        let sym_base = builder.ins().global_value(types::I64, sym_gv);
                        builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            sym_base,
                            (slot as i32) * 8,
                        )
                    } else {
                        builder.ins().iconst(types::I64, method.index() as i64)
                    };
                    let buf_size = (args.len() * 8) as u32;
                    let stack_slot =
                        builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            buf_size,
                            8,
                        ));
                    for (i, a) in args.iter().enumerate() {
                        let v = get(a);
                        builder.ins().stack_store(v, stack_slot, (i * 8) as i32);
                    }
                    let buf = builder.ins().stack_addr(types::I64, stack_slot, 0);
                    let count = builder.ins().iconst(types::I64, args.len() as i64);
                    let f = get_runtime_fn(module, builder, "wren_call_dynamic", 4)?;
                    let call = builder.ins().call(f, &[r, method_val, count, buf]);
                    return Ok(Some(builder.inst_results(call)[0]));
                }
                let r = get(receiver);
                let ic_idx = *call_site_idx;
                *call_site_idx += 1;
                let arg_vals: Vec<Value> = args.iter().map(get).collect();

                if let Some(simd_result) = try_lower_simd_intrinsic_call(
                    interner,
                    builder,
                    module,
                    get_runtime_fn,
                    r,
                    *method,
                    &arg_vals,
                    ic_idx,
                    aot_config,
                )? {
                    return Ok(Some(simd_result));
                }

                // ============================================================
                // AOT devirtualization (CHA-driven).
                //
                // Look up `method`'s signature in the whole-program method
                // table. For every signature with at least one user-defined
                // implementation we emit a class-checked direct call —
                // optionally inlining trivial getters as a single field
                // load. The slow `wren_call_N` path stays as the fallback
                // when the receiver's class doesn't match any known impl
                // (e.g. the call is actually on a prelude class). With CHA
                // wired, monomorphic dispatch in AOT bodies costs `mask +
                // load + icmp + brif + load`, not a runtime helper call.
                // ============================================================
                if let Some(cfg) = aot_config {
                    if let Some(cha_ptr) = cfg.cha {
                        let cha = unsafe { &*cha_ptr };
                        let sig_text = interner.resolve(*method).to_string();
                        if let Some(impls) = cha.by_sig.get(&sig_text) {
                            if !impls.is_empty() {
                                let merge_block = builder.create_block();
                                builder.append_block_param(merge_block, types::I64);

                                // Slow-path block — `wren_call_N` with
                                // the remapped symbol. Reached either
                                // when the receiver isn't an object at
                                // all (Number / Null / Bool) or when no
                                // CHA-known class matched. Created up
                                // front so the is-object guard can
                                // branch straight here without a class
                                // load that would fault on non-object
                                // receivers.
                                let slow_block = builder.create_block();

                                // Is-object guard: short-circuit to the
                                // slow path before the receiver-class
                                // load, since masking a Number Value
                                // off the bottom 48 bits and reading
                                // `+HEADER_CLASS` lands in unmapped
                                // memory.
                                let tag_obj_const =
                                    builder.ins().iconst(types::I64, TAG_OBJ as i64);
                                let high = builder.ins().band(r, tag_obj_const);
                                let is_obj = builder.ins().icmp(IntCC::Equal, high, tag_obj_const);
                                let object_block = builder.create_block();
                                builder
                                    .ins()
                                    .brif(is_obj, object_block, &[], slow_block, &[]);
                                builder.switch_to_block(object_block);

                                // Receiver's class header field.
                                let mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                                let recv_obj = builder.ins().band(r, mask);
                                let recv_class_field = builder.ins().load(
                                    types::I64,
                                    MemFlags::trusted(),
                                    recv_obj,
                                    HEADER_CLASS,
                                );

                                // Chain a class-check per impl. Match
                                // → emit body (direct call or inline
                                // trivial-getter load), jump to merge.
                                // Miss → fall to the next check or the
                                // final `wren_call_N` slow block.
                                for impl_ in impls {
                                    // SM bodies have the
                                    // `(fiber, resume_v) -> i64`
                                    // poll ABI; a direct
                                    // Cranelift call would pack
                                    // `(receiver, ...args)` into
                                    // those slots and SIGSEGV
                                    // on the first
                                    // `wlift_aot_sm_load_state`.
                                    // Skip the fast block — the
                                    // shared `wren_call_N` slow
                                    // block at the bottom routes
                                    // through `dispatch_method`,
                                    // which detects SM and uses
                                    // `call_closure_jit_or_sync`'s
                                    // SM-aware path.
                                    if impl_.is_state_machine {
                                        continue;
                                    }
                                    let next_check = builder.create_block();
                                    let fast_block = builder.create_block();

                                    let class_data_id = module
                                        .declare_data(
                                            &impl_.class_modvars_symbol,
                                            Linkage::Export,
                                            true,
                                            false,
                                        )
                                        .map_err(|e| e.to_string())?;
                                    let gv =
                                        module.declare_data_in_func(class_data_id, builder.func);
                                    let modvars_addr = builder.ins().global_value(types::I64, gv);
                                    let boxed_cls = builder.ins().load(
                                        types::I64,
                                        MemFlags::trusted(),
                                        modvars_addr,
                                        (impl_.class_slot as i32) * 8,
                                    );
                                    let cls_mask =
                                        builder.ins().iconst(types::I64, PTR_MASK as i64);
                                    let expected_cls = builder.ins().band(boxed_cls, cls_mask);
                                    let eq = builder.ins().icmp(
                                        IntCC::Equal,
                                        recv_class_field,
                                        expected_cls,
                                    );
                                    builder.ins().brif(eq, fast_block, &[], next_check, &[]);

                                    builder.switch_to_block(fast_block);
                                    let fast_result = if let Some(field_idx) =
                                        impl_.trivial_getter_field
                                    {
                                        // Inline trivial getter: load
                                        // recv.fields[field_idx].
                                        let fields_ptr = builder.ins().load(
                                            types::I64,
                                            MemFlags::trusted(),
                                            recv_obj,
                                            INSTANCE_FIELDS,
                                        );
                                        builder.ins().load(
                                            types::I64,
                                            MemFlags::trusted(),
                                            fields_ptr,
                                            (field_idx as i32) * 8,
                                        )
                                    } else {
                                        // Direct call to the AOT'd
                                        // method body. The body's
                                        // MIR arity ALREADY counts
                                        // the receiver — so the
                                        // signature has `arity`
                                        // params total (recv + N-1
                                        // user args), and the call
                                        // passes `[r, args...]` of
                                        // matching length.
                                        let mut sig = Signature::new(
                                            module.target_config().default_call_conv,
                                        );
                                        for _ in 0..(impl_.arity as usize) {
                                            sig.params.push(AbiParam::new(types::I64));
                                        }
                                        sig.returns.push(AbiParam::new(types::I64));
                                        let body_id = module
                                            .declare_function(
                                                &impl_.fn_symbol,
                                                Linkage::Import,
                                                &sig,
                                            )
                                            .map_err(|e| e.to_string())?;
                                        let fn_ref =
                                            module.declare_func_in_func(body_id, builder.func);
                                        let user_arity = (impl_.arity as usize).saturating_sub(1);
                                        let mut call_args = vec![r];
                                        for a in args.iter().take(user_arity) {
                                            call_args.push(get(a));
                                        }
                                        // Pad with nulls when MIR
                                        // site arg count is below
                                        // the body's declared
                                        // arity. Defensive — the
                                        // resolver should have
                                        // matched arities, but a
                                        // mismatched signature
                                        // would otherwise fault
                                        // Cranelift's verifier.
                                        while call_args.len() < impl_.arity as usize {
                                            let null =
                                                builder.ins().iconst(types::I64, TAG_NULL as i64);
                                            call_args.push(null);
                                        }
                                        let call = builder.ins().call(fn_ref, &call_args);
                                        builder.inst_results(call)[0]
                                    };
                                    builder
                                        .ins()
                                        .jump(merge_block, &[BlockArg::Value(fast_result)]);

                                    builder.switch_to_block(next_check);
                                }

                                // Last next_check falls through here;
                                // route it to the shared slow_block.
                                builder.ins().jump(slow_block, &[]);
                                builder.switch_to_block(slow_block);

                                // Fallback: wren_call_N with the
                                // remapped symbol — used when none of
                                // the known impl class checks matched
                                // (receiver was a prelude type or an
                                // unrelated class).
                                let slot = aot_intern_symbol(cfg, method.index(), interner);
                                let sym_gv =
                                    module.declare_data_in_func(cfg.symbols_data, builder.func);
                                let sym_base = builder.ins().global_value(types::I64, sym_gv);
                                let method_val = builder.ins().load(
                                    types::I64,
                                    MemFlags::trusted(),
                                    sym_base,
                                    (slot as i32) * 8,
                                );
                                // Pick the helper whose user-arg
                                // count matches the call site. The
                                // `wren_call_N` family ranges
                                // 0..=8; arity > 8 routes through
                                // `wren_call_dynamic` with a
                                // stack-allocated args buffer so we
                                // never silently truncate (which
                                // surfaced as `Float32Array[_]=:
                                // value must be a number` on the
                                // sprite-batch path).
                                let slow_result = if args.len() > 8 {
                                    let f = get_runtime_fn(module, builder, "wren_call_dynamic", 4)?;
                                    let slot = builder.create_sized_stack_slot(
                                        cranelift_codegen::ir::StackSlotData::new(
                                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                            (args.len() as u32) * 8,
                                            3,
                                        ),
                                    );
                                    let buf_ptr = builder.ins().stack_addr(types::I64, slot, 0);
                                    for (i, a) in args.iter().enumerate() {
                                        builder.ins().store(
                                            MemFlags::trusted(),
                                            get(a),
                                            buf_ptr,
                                            (i as i32) * 8,
                                        );
                                    }
                                    let count = builder.ins().iconst(types::I64, args.len() as i64);
                                    let call =
                                        builder.ins().call(f, &[r, method_val, count, buf_ptr]);
                                    builder.inst_results(call)[0]
                                } else {
                                    let call_name = match args.len() {
                                        0 => "wren_call_0",
                                        1 => "wren_call_1",
                                        2 => "wren_call_2",
                                        3 => "wren_call_3",
                                        4 => "wren_call_4",
                                        5 => "wren_call_5",
                                        6 => "wren_call_6",
                                        7 => "wren_call_7",
                                        _ => "wren_call_8",
                                    };
                                    let f = get_runtime_fn(module, builder, call_name, 2 + args.len())?;
                                    let mut slow_args = vec![r, method_val];
                                    for a in args.iter() {
                                        slow_args.push(get(a));
                                    }
                                    let slow_call = builder.ins().call(f, &slow_args);
                                    builder.inst_results(slow_call)[0]
                                };
                                builder
                                    .ins()
                                    .jump(merge_block, &[BlockArg::Value(slow_result)]);

                                builder.switch_to_block(merge_block);
                                return Ok(Some(builder.block_params(merge_block)[0]));
                            }
                        }
                    }
                }

                // === JIT-CHA multi-class dispatch tree ===
                // For every (class, func_id) pair the engine's CHA
                // discovered for this method, emit a class check.
                // On match, splice the callee body inline if it's
                // small enough; otherwise route through the existing
                // `wren_known_call_N_nocheck` helper, which still
                // skips the polymorphism re-check the slow path
                // does. Receivers that aren't in CHA fall through
                // to `wren_call_N`. This subsumes the kind=1 IC
                // fast path while extending coverage to call sites
                // that see multiple receiver classes (where the IC
                // alone keeps thrashing).
                if let Some(cha) = cha_by_method {
                    if args.len() <= 4 {
                        let impls: Vec<(usize, u32, usize)> =
                            cha.get(method).cloned().unwrap_or_default();
                        if !impls.is_empty() {
                            let merge_block = builder.create_block();
                            builder.append_block_param(merge_block, types::I64);

                            // Non-objects (Numbers, Null, Bool, ...)
                            // skip every class check and route to
                            // wren_call_N; reading `recv.class` from
                            // their NaN-box bits would dereference
                            // garbage. The slow_block is the merge
                            // target for both "no impl matched" and
                            // "receiver isn't an object".
                            let slow_block = builder.create_block();
                            let (_obj_ptr, recv_class) =
                                emit_class_load_guarded(builder, r, slow_block);

                            for (class_ptr, fid, _closure_ptr) in &impls {
                                let next_check = builder.create_block();
                                let fast_block = builder.create_block();
                                let cached_class =
                                    builder.ins().iconst(types::I64, *class_ptr as i64);
                                let class_match =
                                    builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
                                builder
                                    .ins()
                                    .brif(class_match, fast_block, &[], next_check, &[]);

                                builder.switch_to_block(fast_block);
                                let inlinable_body =
                                    inline_bodies.as_ref().and_then(|b| b.get(fid)).cloned();
                                let mut emitted_inline = false;
                                if let Some(callee_mir) = inlinable_body {
                                    let mut callee_vals: HashMap<ValueId, Value> = HashMap::new();
                                    let mut callee_args: Vec<Value> =
                                        Vec::with_capacity(args.len() + 1);
                                    callee_args.push(r);
                                    for a in args.iter() {
                                        callee_args.push(get(a));
                                    }
                                    let callee_block = &callee_mir.blocks[0];
                                    let mut inline_call_idx = 0usize;
                                    let mut inline_failed = false;
                                    for (vid, callee_inst) in &callee_block.instructions {
                                        match callee_inst {
                                            Instruction::BlockParam(idx) => {
                                                let i = *idx as usize;
                                                if i < callee_args.len() {
                                                    callee_vals.insert(*vid, callee_args[i]);
                                                } else {
                                                    inline_failed = true;
                                                    break;
                                                }
                                            }
                                            _ => {
                                                let res = lower_instruction(
                                                    callee_inst,
                                                    &callee_mir,
                                                    interner,
                                                    builder,
                                                    module,
                                                    &callee_vals,
                                                    get_runtime_fn,
                                                    None,
                                                    None,
                                                    jit_code_base,
                                                    &mut inline_call_idx,
                                                    f64_self_id,
                                                    Some(callee_args[0]),
                                                    aot_config,
                                                    None,
                                                    None,
                                                )?;
                                                if let Some(v) = res {
                                                    callee_vals.insert(*vid, v);
                                                }
                                            }
                                        }
                                    }
                                    let return_val = if inline_failed {
                                        None
                                    } else {
                                        match &callee_block.terminator {
                                            Terminator::Return(v) => callee_vals.get(v).copied(),
                                            Terminator::ReturnNull => Some(
                                                builder.ins().iconst(types::I64, TAG_NULL as i64),
                                            ),
                                            _ => None,
                                        }
                                    };
                                    if let Some(rv) = return_val {
                                        builder.ins().jump(merge_block, &[BlockArg::Value(rv)]);
                                        emitted_inline = true;
                                    }
                                }

                                if !emitted_inline {
                                    // Helper-based fast path for
                                    // non-inlinable callees. Mirrors
                                    // the kind=1 IC fast block (class
                                    // check + nocheck helper). Args
                                    // capped at 3 since the *_nocheck
                                    // family only goes up to arity 3.
                                    if args.len() <= 3 {
                                        let packed =
                                            (*fid as u64) | ((method.index() as u64) << 32);
                                        let fid_val =
                                            builder.ins().iconst(types::I64, packed as i64);
                                        let fast_name = match args.len() {
                                            0 => "wren_known_call_0_nocheck",
                                            1 => "wren_known_call_1_nocheck",
                                            2 => "wren_known_call_2_nocheck",
                                            _ => "wren_known_call_3_nocheck",
                                        };
                                        let fast_arg_count = 2 + args.len();
                                        let fast_f = get_runtime_fn(
                                            module,
                                            builder,
                                            fast_name,
                                            fast_arg_count,
                                        )?;
                                        let mut fast_args = vec![fid_val, r];
                                        for a in args.iter() {
                                            fast_args.push(get(a));
                                        }
                                        let fast_call = builder.ins().call(fast_f, &fast_args);
                                        let fast_result = builder.inst_results(fast_call)[0];
                                        builder
                                            .ins()
                                            .jump(merge_block, &[BlockArg::Value(fast_result)]);
                                    } else {
                                        // Arity 4: fall through to
                                        // wren_call_N at the tail.
                                        builder.ins().jump(next_check, &[]);
                                    }
                                }

                                builder.switch_to_block(next_check);
                            }

                            // After all class checks miss, fall into
                            // the shared slow_block (also reached by
                            // the is-object guard for non-object
                            // receivers).
                            builder.ins().jump(slow_block, &[]);

                            builder.switch_to_block(slow_block);
                            // No class matched — full dispatch
                            // through `emit_wren_call`, which picks
                            // the right `wren_call_N` (0..=8) or
                            // routes 9+ through `wren_call_dynamic`.
                            let method_bits = method.index() as u64;
                            let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                            let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                            let slow_result = emit_wren_call(
                                builder,
                                module,
                                get_runtime_fn,
                                r,
                                method_val,
                                &arg_vals,
                            )?;
                            builder
                                .ins()
                                .jump(merge_block, &[BlockArg::Value(slow_result)]);

                            builder.switch_to_block(merge_block);
                            return Ok(Some(builder.block_params(merge_block)[0]));
                        }
                    }
                }

                // Try inline IC: emit class-check + fast path.
                // Kind=5 (getter): inline field load (class baked as constant).
                // Kind=1: currently only used for IC index encoding in slow path.
                let ic = callsite_ic_ptrs.and_then(|ics| ics.get(ic_idx));
                let _live_ptr = callsite_ic_live_ptrs.and_then(|ptrs| ptrs.get(ic_idx).copied());

                // AOT mode: ICs are JIT-only (mutable code memory).
                // Skip the kind=5 inline-getter fast path entirely
                // and use the slow path so the symbol-remap table
                // indirection emits cleanly.
                let ic = if aot_config.is_some() { None } else { ic };

                if let Some(ic) = ic {
                    // Only emit IC fast path for kind=5 (getter inline).
                    // Kind=1 uses the slow path with IC index encoding so
                    // dispatch_call_rooted can use cached method lookups.
                    if ic.kind == 5 && ic.class != 0 {
                        let fast_block = builder.create_block();
                        let slow_block = builder.create_block();
                        let merge_block = builder.create_block();
                        builder.append_block_param(merge_block, types::I64);

                        // is-object guard + class load. Non-objects
                        // (Numbers, Null, Bool, ...) skip straight
                        // to the slow path; reading their NaN-box
                        // bits at +16 would dereference garbage.
                        let (obj_ptr, recv_class) = emit_class_load_guarded(builder, r, slow_block);

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
                        if env_jit_callsite_ic() {
                            method_bits |= ((ic_idx as u64) + 1) << 32;
                        }
                        let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                        let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                        let slow_result = emit_wren_call(
                            builder,
                            module,
                            get_runtime_fn,
                            r,
                            method_val,
                            &arg_vals,
                        )?;
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(slow_result)]);

                        // Merge
                        builder.switch_to_block(merge_block);
                        return Ok(Some(builder.block_params(merge_block)[0]));
                    }
                }

                // No IC or unsupported IC kind: full dispatch.
                //
                // AOT mode: re-key the method symbol through the
                // per-module symbol-remap table — `method.index()`
                // is an index into the source's per-parse interner;
                // the runtime helper expects a VM-interner index.
                // The bootstrap populates `wlift_symbols_<n>` at
                // startup via `wlift_aot_intern_symbols`, so the
                // remap is just a `load` here.
                //
                // JIT mode: bake the source SymbolId directly —
                // `install_module_mir_*` already remaps the MIR's
                // symbols into the VM's interner before the JIT
                // ever sees them.
                let method_val = if let Some(cfg) = aot_config {
                    let slot = aot_intern_symbol(cfg, method.index(), interner);
                    let gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
                    let base = builder.ins().global_value(types::I64, gv);
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), base, (slot as i32) * 8)
                } else {
                    let mut method_bits = method.index() as u64;
                    if env_jit_callsite_ic() {
                        method_bits |= ((ic_idx as u64) + 1) << 32;
                    }
                    builder.ins().iconst(types::I64, method_bits as i64)
                };
                // Route arity > 8 through wren_call_dynamic to
                // avoid the silent .min(8) truncation that
                // corrupted Renderer2D's drawSprite_(receiver +
                // 13 args) call frame.
                let result_val = if args.len() > 8 {
                    let f = get_runtime_fn(module, builder, "wren_call_dynamic", 4)?;
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            (args.len() as u32) * 8,
                            3,
                        ),
                    );
                    let buf_ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    for (i, a) in args.iter().enumerate() {
                        builder.ins().store(
                            MemFlags::trusted(),
                            get(a),
                            buf_ptr,
                            (i as i32) * 8,
                        );
                    }
                    let count = builder.ins().iconst(types::I64, args.len() as i64);
                    let call = builder.ins().call(f, &[r, method_val, count, buf_ptr]);
                    builder.inst_results(call)[0]
                } else {
                    let call_name = match args.len() {
                        0 => "wren_call_0",
                        1 => "wren_call_1",
                        2 => "wren_call_2",
                        3 => "wren_call_3",
                        4 => "wren_call_4",
                        5 => "wren_call_5",
                        6 => "wren_call_6",
                        7 => "wren_call_7",
                        _ => "wren_call_8",
                    };
                    let f = get_runtime_fn(module, builder, call_name, 2 + args.len())?;
                    let mut call_args = vec![r, method_val];
                    for a in args.iter() {
                        call_args.push(get(a));
                    }
                    let result = builder.ins().call(f, &call_args);
                    builder.inst_results(result)[0]
                };
                Ok(Some(result_val))
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
                // High-arity fallbacks (>8 user args) route through
                // `wren_call_dynamic` via `emit_wren_call`. The
                // earlier hard-error here was load-bearing only for
                // truncation safety; with the dynamic helper in
                // place every call site is arity-correct.
                let r = get(receiver);

                // AOT mode: short-circuit every JIT-specific fast
                // path (pure-leaf direct, inline getter, baked-class
                // IC, jit_code_base lookups) to the standard
                // wren_call_N slow path with the symbol remapped
                // through the per-module table. Those fast paths
                // depend on `expected_class` being a JIT-allocated
                // class pointer and `jit_code_base` being live —
                // neither holds for static-linked AOT output.
                if let Some(cfg) = aot_config {
                    let slot = aot_intern_symbol(cfg, method.index(), interner);
                    let gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
                    let base = builder.ins().global_value(types::I64, gv);
                    let method_val = builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        base,
                        (slot as i32) * 8,
                    );
                    let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                    let result = emit_wren_call(
                        builder,
                        module,
                        get_runtime_fn,
                        r,
                        method_val,
                        &arg_vals,
                    )?;
                    return Ok(Some(result));
                }

                // === CHA-driven body inlining ===
                // When the engine flagged this callee as a small,
                // single-block, dispatch-free body, splice the body
                // straight into the caller's Cranelift function
                // behind the same class-check guard the kind=1 IC
                // would emit. A receiver whose class doesn't match
                // the speculation falls through to `wren_call_N`,
                // matching the existing CallKnownFunc fallback.
                //
                // This subsumes the trivial-getter / pure-leaf-direct
                // paths for any callee the inliner can lower, which
                // is most short Wren methods (field load + arithmetic
                // + field store + return). The body emits with no
                // helper hop, no jit_ctx swap, no jit_roots push, no
                // depth tracking — i.e. the AOT-shape direct call,
                // but with a per-callsite speculation guard.
                if let Some(bodies) = inline_bodies {
                    if *expected_class != 0 && args.len() <= 4 {
                        if let Some(callee_mir) = bodies.get(func_id).cloned() {
                            let fast_block = builder.create_block();
                            let slow_block = builder.create_block();
                            let merge_block = builder.create_block();
                            builder.append_block_param(merge_block, types::I64);

                            // is-object guard + class load. Non-objects
                            // skip the inlined body and route to the
                            // slow path.
                            let (_obj_ptr, recv_class) =
                                emit_class_load_guarded(builder, r, slow_block);
                            let cached_class =
                                builder.ins().iconst(types::I64, *expected_class as i64);
                            let class_match =
                                builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
                            builder
                                .ins()
                                .brif(class_match, fast_block, &[], slow_block, &[]);

                            // Fast block: walk the callee's single
                            // block, lowering each instruction into the
                            // caller's function with a fresh local
                            // val_map. BlockParam(i) maps to the
                            // caller-supplied receiver/arg values.
                            builder.switch_to_block(fast_block);
                            let mut callee_vals: HashMap<ValueId, Value> = HashMap::new();
                            // arg slot 0 = receiver, 1.. = user args.
                            let mut callee_args: Vec<Value> = Vec::with_capacity(args.len() + 1);
                            callee_args.push(r);
                            for a in args.iter() {
                                callee_args.push(get(a));
                            }
                            let callee_block = &callee_mir.blocks[0];
                            let mut inline_call_idx = 0usize;
                            let mut inline_failed = false;
                            for (vid, callee_inst) in &callee_block.instructions {
                                match callee_inst {
                                    Instruction::BlockParam(idx) => {
                                        let i = *idx as usize;
                                        if i < callee_args.len() {
                                            callee_vals.insert(*vid, callee_args[i]);
                                        } else {
                                            // Eligibility check should have
                                            // matched arity, but stay defensive.
                                            inline_failed = true;
                                            break;
                                        }
                                    }
                                    _ => {
                                        let res = lower_instruction(
                                            callee_inst,
                                            &callee_mir,
                                            interner,
                                            builder,
                                            module,
                                            &callee_vals,
                                            get_runtime_fn,
                                            None,
                                            None,
                                            jit_code_base,
                                            &mut inline_call_idx,
                                            f64_self_id,
                                            Some(callee_args[0]),
                                            aot_config,
                                            None,
                                            None,
                                        )?;
                                        if let Some(v) = res {
                                            callee_vals.insert(*vid, v);
                                        }
                                    }
                                }
                            }
                            let return_val = if inline_failed {
                                None
                            } else {
                                match &callee_block.terminator {
                                    Terminator::Return(v) => callee_vals.get(v).copied(),
                                    Terminator::ReturnNull => {
                                        Some(builder.ins().iconst(types::I64, TAG_NULL as i64))
                                    }
                                    _ => None,
                                }
                            };
                            if let Some(rv) = return_val {
                                builder.ins().jump(merge_block, &[BlockArg::Value(rv)]);
                            } else {
                                // Inline fell through (unsupported terminator
                                // or missing param) — collapse the fast block
                                // into the slow path so emit stays sound.
                                builder.ins().jump(slow_block, &[]);
                            }

                            // Slow path: full dispatch via emit_wren_call.
                            builder.switch_to_block(slow_block);
                            let method_bits = method.index() as u64;
                            let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                            let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                            let slow_result = emit_wren_call(
                                builder,
                                module,
                                get_runtime_fn,
                                r,
                                method_val,
                                &arg_vals,
                            )?;
                            builder
                                .ins()
                                .jump(merge_block, &[BlockArg::Value(slow_result)]);

                            builder.switch_to_block(merge_block);
                            return Ok(Some(builder.block_params(merge_block)[0]));
                        }
                    }
                }

                // === Pure-leaf direct call (ZERO FFI) ===
                // Callee has no internal method calls, so no context
                // setup is needed. Emit: class check + load callee ptr
                // + call_indirect. The JIT code slot address is stable
                // because engine.jit_code doesn't reallocate post-load.
                //
                // Gated off by default: this path emits a raw
                // `call_indirect` and passes args through CPU registers
                // without rooting them or emitting Cranelift stack
                // maps. "Pure leaf" means "no Wren-level calls" — the
                // body can still allocate (string concat, list grow,
                // list-of-num alloc), and without rooting the GC
                // scanner can't see the args. Set
                // `WLIFT_ENABLE_PURE_LEAF_DIRECT=1` once Cranelift
                // stack maps for the call_indirect args are wired up;
                // until then fall through to `wren_known_call_N_nocheck`,
                // which roots args before dispatching.
                let pure_leaf_enabled = env_pure_leaf_direct();
                if pure_leaf_enabled
                    && inline_getter_field.is_none()
                    && *pure_leaf
                    && *expected_class != 0
                    && args.len() <= 4
                {
                    if let Some(jit_base_ptr) = jit_code_base {
                        let fast_block = builder.create_block();
                        let slow_block = builder.create_block();
                        let merge_block = builder.create_block();
                        builder.append_block_param(merge_block, types::I64);

                        // is-object guard + class load.
                        let (_obj_ptr, recv_class) =
                            emit_class_load_guarded(builder, r, slow_block);
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

                        // Slow path: full dispatch via emit_wren_call.
                        builder.switch_to_block(slow_block);
                        let method_bits = method.index() as u64;
                        let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                        let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                        let slow_result = emit_wren_call(
                            builder,
                            module,
                            get_runtime_fn,
                            r,
                            method_val,
                            &arg_vals,
                        )?;
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

                    // is-object guard + class load.
                    let (obj_ptr, recv_class) = emit_class_load_guarded(builder, r, slow_block);
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

                    // Slow path: class mismatch → emit_wren_call.
                    builder.switch_to_block(slow_block);
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                    let slow_result = emit_wren_call(
                        builder,
                        module,
                        get_runtime_fn,
                        r,
                        method_val,
                        &arg_vals,
                    )?;
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(slow_result)]);

                    builder.switch_to_block(merge_block);
                    return Ok(Some(builder.block_params(merge_block)[0]));
                }

                // The cached-class fast path uses the `wren_known_call_N_nocheck`
                // helpers, which only exist for N ∈ 0..=3. Higher-arity
                // call sites must skip this branch entirely and fall
                // through to `emit_wren_call` — otherwise the dispatch
                // truncates args 4..N to whatever the helper signature
                // expects, leaving the callee to read garbage. Hot
                // example: Renderer2D's `pushVertex_(head, px, py, u, v,
                // r, g, b, a)` (9 user args) had `u, v, r, g, b, a`
                // silently dropped, surfacing as NaN sprite vertices.
                if *expected_class != 0 && args.len() <= 3 {
                    let fast_block = builder.create_block();
                    let slow_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);

                    // is-object guard + class load. Non-object
                    // receivers (Numbers, Null, Bool, ...) skip
                    // straight to wren_call_N — a Number's f64
                    // bits masked through PTR_MASK can land at an
                    // unmapped page on macOS aarch64, so the
                    // pre-existing "comparison fails safely"
                    // assumption did not hold.
                    let (_obj_ptr, recv_class) = emit_class_load_guarded(builder, r, slow_block);
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
                    let fast_arg_count = 2 + args.len();
                    let fast_f = get_runtime_fn(module, builder, fast_name, fast_arg_count)?;
                    let mut fast_args = vec![fid_val, r];
                    for a in args.iter() {
                        fast_args.push(get(a));
                    }
                    let fast_call = builder.ins().call(fast_f, &fast_args);
                    let fast_result = builder.inst_results(fast_call)[0];
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(fast_result)]);

                    // Slow path: class mismatch → emit_wren_call.
                    builder.switch_to_block(slow_block);
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                    let slow_result = emit_wren_call(
                        builder,
                        module,
                        get_runtime_fn,
                        r,
                        method_val,
                        &arg_vals,
                    )?;
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(slow_result)]);

                    builder.switch_to_block(merge_block);
                    return Ok(Some(builder.block_params(merge_block)[0]));
                }

                // No cached class (or arity > 3): route through the
                // generic method-dispatch helpers. `emit_wren_call`
                // picks `wren_call_N` for 0..=8 and `wren_call_dynamic`
                // for 9+. The packed func_id hint isn't usable here —
                // `wren_known_call_N` only covers 0..=3 — and the
                // generic dispatcher resolves the same target from
                // the receiver's class anyway.
                if args.len() <= 3 {
                    let packed = (*func_id as u64) | ((method.index() as u64) << 32);
                    let fid_val = builder.ins().iconst(types::I64, packed as i64);
                    let call_name = match args.len() {
                        0 => "wren_known_call_0",
                        1 => "wren_known_call_1",
                        2 => "wren_known_call_2",
                        _ => "wren_known_call_3",
                    };
                    let arg_count = 2 + args.len();
                    let f = get_runtime_fn(module, builder, call_name, arg_count)?;
                    let mut call_args = vec![fid_val, r];
                    for a in args.iter() {
                        call_args.push(get(a));
                    }
                    let result = builder.ins().call(f, &call_args);
                    Ok(Some(builder.inst_results(result)[0]))
                } else {
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                    let result = emit_wren_call(
                        builder,
                        module,
                        get_runtime_fn,
                        r,
                        method_val,
                        &arg_vals,
                    )?;
                    Ok(Some(result))
                }
            }

            // === Super calls ===
            Instruction::SuperCall { method, args } => {
                if args.len() > 4 {
                    return Err(format!(
                        "SuperCall with arity {} not supported by JIT",
                        args.len()
                    ));
                }
                // AOT mode: re-key the method symbol through the
                // per-module symbol-remap table — same fix Call's
                // slow path does. Without this, super dispatch
                // hits whatever VM-interner symbol happens to
                // match the source-interner index.
                let method_val = if let Some(cfg) = aot_config {
                    let slot = aot_intern_symbol(cfg, method.index(), interner);
                    let gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
                    let base = builder.ins().global_value(types::I64, gv);
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), base, (slot as i32) * 8)
                } else {
                    builder.ins().iconst(types::I64, method.index() as i64)
                };
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
                    // >4 elements: create empty + add each.
                    //
                    // The intermediate `list` value lives across every
                    // `wren_list_add` call below. The outer
                    // `lower_mir_impl` only declares the *result* of
                    // this MakeList — i.e. the post-loop value — so
                    // without an explicit declaration here, a GC fired
                    // from inside any `wren_list_add` strands the
                    // register-held `list` pointer and the next
                    // iteration writes through a stale ObjList header.
                    let f_make = get_runtime_fn(module, builder, "wren_make_list", 0)?;
                    let make_result = builder.ins().call(f_make, &[]);
                    let list = builder.inst_results(make_result)[0];
                    if env_stack_maps() {
                        builder.declare_value_needs_stack_map(list);
                    }

                    let f_add = get_runtime_fn(module, builder, "wren_list_add", 2)?;
                    for e in elems {
                        builder.ins().call(f_add, &[list, get(e)]);
                    }
                    Ok(Some(list))
                }
            }

            Instruction::MakeMap(pairs) => {
                // Same reasoning as MakeList: `map` lives across
                // every `wren_map_set` call below, so declare it
                // explicitly to keep the GC's stack-map walker
                // aware of the live receiver pointer across each
                // helper safepoint.
                let f_make = get_runtime_fn(module, builder, "wren_make_map", 0)?;
                let make_result = builder.ins().call(f_make, &[]);
                let map = builder.inst_results(make_result)[0];
                if env_stack_maps() {
                    builder.declare_value_needs_stack_map(map);
                }

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
                    // Declare each intermediate concat result as
                    // needing stack-map coverage. Without this,
                    // a multi-part concat's intermediate strings
                    // (held only in this register-passed chain)
                    // aren't tracked across the next call's
                    // safepoint, so a GC fired from inside the
                    // next concat helper sweeps them. The
                    // surrounding `lower_mir_impl` only declares
                    // top-level instruction results; chains
                    // built inside this lowering need explicit
                    // declaration.
                    if env_stack_maps() {
                        builder.declare_value_needs_stack_map(result);
                    }
                }
                Ok(Some(result))
            }
            Instruction::ToString(a) => {
                let f = get_runtime_fn(module, builder, "wren_to_string", 1)?;
                let result = builder.ins().call(f, &[get(a)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Upvalues ===
            //
            // AOT mode: when the body has any upvalue access, the
            // entry block stashed `JitContext.closure` into a
            // function-scoped Cranelift Variable. Lower each access
            // to inline pointer chasing against that local — saves
            // the per-access TLS read + helper-call overhead, and
            // the local survives nested calls that re-mutate TLS.
            //
            // Layout: ObjClosure.upvalues is a `Vec<*mut ObjUpvalue>`,
            // so the data pointer lives at +CLOSURE_UPVALUES_DATA;
            // each ObjUpvalue's value is reached through its
            // `location` field at +UPVALUE_LOCATION (open upvalues
            // point at a stack slot, closed ones at the upvalue's
            // own `closed` storage — the indirection is essential).
            //
            // JIT mode keeps the helper call (the dispatch path
            // already populates `ctx.closure` and the helper
            // amortises away under tier-up perf budgets).
            Instruction::GetUpvalue(idx) => {
                if let Some(cfg) = aot_config {
                    if let Some(var) = *cfg.current_closure_ptr_var.borrow() {
                        let closure_ptr = builder.use_var(var);
                        let upvalues_data = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            closure_ptr,
                            CLOSURE_UPVALUES_DATA,
                        );
                        let upvalue_ptr = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            upvalues_data,
                            (*idx as i32) * 8,
                        );
                        let location_ptr = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            upvalue_ptr,
                            UPVALUE_LOCATION,
                        );
                        let value =
                            builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), location_ptr, 0);
                        return Ok(Some(value));
                    }
                }
                let f = get_runtime_fn(module, builder, "wren_get_upvalue", 1)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SetUpvalue(idx, val) => {
                if let Some(cfg) = aot_config {
                    if let Some(var) = *cfg.current_closure_ptr_var.borrow() {
                        let closure_ptr = builder.use_var(var);
                        let upvalues_data = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            closure_ptr,
                            CLOSURE_UPVALUES_DATA,
                        );
                        let upvalue_ptr = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            upvalues_data,
                            (*idx as i32) * 8,
                        );
                        let location_ptr = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            upvalue_ptr,
                            UPVALUE_LOCATION,
                        );
                        let v = get(val);
                        builder.ins().store(MemFlags::trusted(), v, location_ptr, 0);
                        // Generational write barrier: the upvalue
                        // header is the slot's owner; a young value
                        // stored into an old upvalue must surface to
                        // the major GC's remembered set.
                        let barrier = get_runtime_fn(module, builder, "wren_write_barrier", 2)?;
                        builder.ins().call(barrier, &[upvalue_ptr, v]);
                        return Ok(Some(v));
                    }
                }
                let f = get_runtime_fn(module, builder, "wren_set_upvalue", 2)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val, get(val)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Static fields ===
            //
            // AOT mode under a class-method emit: load the
            // defining class from `wlift_modvars_<n>[slot]` and
            // call `wlift_aot_get/set_static_field(class, sym)`.
            // Bypasses `JitContext.defining_class` — `wlift_aot_enter`
            // never populates it, so the JIT helper's TLS read
            // would return null in every AOT frame.
            //
            // JIT mode (or AOT top-level / closure where no
            // defining class is in scope) keeps the legacy
            // `wren_get/set_static_field` call.
            Instruction::GetStaticField(sym) => {
                if let Some(cfg) = aot_config {
                    if let Some(defining) = cfg.current_defining_class.borrow().as_ref() {
                        let class_data_id = module
                            .declare_data(&defining.modvars_symbol, Linkage::Export, true, false)
                            .map_err(|e| e.to_string())?;
                        let gv = module.declare_data_in_func(class_data_id, builder.func);
                        let modvars_addr = builder.ins().global_value(types::I64, gv);
                        let class_bits = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            modvars_addr,
                            (defining.slot as i32) * 8,
                        );
                        let f = get_runtime_fn(module, builder, "wlift_aot_get_static_field", 2)?;
                        let sym_val = builder.ins().iconst(types::I64, sym.index() as i64);
                        let result = builder.ins().call(f, &[class_bits, sym_val]);
                        return Ok(Some(builder.inst_results(result)[0]));
                    }
                }
                let f = get_runtime_fn(module, builder, "wren_get_static_field", 1)?;
                let idx_val = builder.ins().iconst(types::I64, sym.index() as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SetStaticField(sym, val) => {
                if let Some(cfg) = aot_config {
                    if let Some(defining) = cfg.current_defining_class.borrow().as_ref() {
                        let class_data_id = module
                            .declare_data(&defining.modvars_symbol, Linkage::Export, true, false)
                            .map_err(|e| e.to_string())?;
                        let gv = module.declare_data_in_func(class_data_id, builder.func);
                        let modvars_addr = builder.ins().global_value(types::I64, gv);
                        let class_bits = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            modvars_addr,
                            (defining.slot as i32) * 8,
                        );
                        let f = get_runtime_fn(module, builder, "wlift_aot_set_static_field", 3)?;
                        let sym_val = builder.ins().iconst(types::I64, sym.index() as i64);
                        let result = builder.ins().call(f, &[class_bits, sym_val, get(val)]);
                        return Ok(Some(builder.inst_results(result)[0]));
                    }
                }
                let f = get_runtime_fn(module, builder, "wren_set_static_field", 2)?;
                let idx_val = builder.ins().iconst(types::I64, sym.index() as i64);
                let result = builder.ins().call(f, &[idx_val, get(val)]);
                Ok(Some(builder.inst_results(result)[0]))
            }

            // === Closures ===
            //
            // AOT mode: MIR's `fn_id` is a build-time-relative
            // index into `ModuleMir::closures` (the JIT path
            // patches these into engine FuncIds at install time
            // via `patch_closure_ids`; AOT skips that pass).
            // Read the runtime FuncId from
            // `wlift_closures_<n>[fn_id]` — the bootstrap calls
            // `wlift_aot_register_closure` once per closure at
            // startup, populating the slot.
            Instruction::MakeClosure { fn_id, upvalues } => {
                let n = upvalues.len();
                let fn_id_val = if let Some(cfg) = aot_config {
                    let gv = module.declare_data_in_func(cfg.closures_data, builder.func);
                    let base = builder.ins().global_value(types::I64, gv);
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), base, (*fn_id as i32) * 8)
                } else {
                    builder.ins().iconst(types::I64, *fn_id as i64)
                };
                if n <= 8 {
                    let name = match n {
                        0 => "wren_make_closure_0",
                        1 => "wren_make_closure_1",
                        2 => "wren_make_closure_2",
                        3 => "wren_make_closure_3",
                        4 => "wren_make_closure_4",
                        5 => "wren_make_closure_5",
                        6 => "wren_make_closure_6",
                        7 => "wren_make_closure_7",
                        _ => "wren_make_closure_8",
                    };
                    let f = get_runtime_fn(module, builder, name, 1 + n)?;
                    let mut args = vec![fn_id_val];
                    for uv in upvalues.iter() {
                        args.push(get(uv));
                    }
                    let result = builder.ins().call(f, &args);
                    Ok(Some(builder.inst_results(result)[0]))
                } else {
                    // > 8 upvalues: spill the captured values into a
                    // stack-allocated `[u64; n]` buffer and route through
                    // `wren_make_closure_n(fn_id, n, ptr)` so every
                    // upvalue reaches the closure's Vec. Truncating
                    // past index 7 was what made `Session.cookie`'s
                    // 7-upvalue middleware (well within range, but
                    // we hit the same path on bigger captures) miss
                    // its trailing slots and crash at first access.
                    let slot =
                        builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            (n * 8) as u32,
                            8,
                        ));
                    for (i, uv) in upvalues.iter().enumerate() {
                        let v = get(uv);
                        builder.ins().stack_store(v, slot, (i * 8) as i32);
                    }
                    let buf = builder.ins().stack_addr(types::I64, slot, 0);
                    let count = builder.ins().iconst(types::I64, n as i64);
                    let f = get_runtime_fn(module, builder, "wren_make_closure_n", 3)?;
                    let result = builder.ins().call(f, &[fn_id_val, count, buf]);
                    Ok(Some(builder.inst_results(result)[0]))
                }
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
                let typed_array_block = builder.create_block();
                let simd_block = builder.create_block();
                let type_miss_block = builder.create_block();
                let in_bounds_block = builder.create_block();
                let simd_in_bounds_block = builder.create_block();
                let check_i32_block = builder.create_block();
                let check_f32_block = builder.create_block();
                let get_u8_block = builder.create_block();
                let get_i32_block = builder.create_block();
                let get_f32_block = builder.create_block();
                let get_f64_block = builder.create_block();
                let simd_get_i32_block = builder.create_block();
                let simd_get_f32_block = builder.create_block();
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
                //    TypedArray / Simd tags.
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
                builder
                    .ins()
                    .brif(is_ta, typed_array_block, &[], type_miss_block, &[]);
                builder.switch_to_block(type_miss_block);
                let simd_tag = builder.ins().iconst(types::I64, OBJ_TYPE_SIMD as i64);
                let is_simd = builder.ins().icmp(IntCC::Equal, obj_type_byte, simd_tag);
                builder
                    .ins()
                    .brif(is_simd, simd_block, &[], slow_block, &[]);

                // 3. TypedArray fast path: convert NaN-boxed Num
                //    index to i64, bounds-check against element
                //    count. Negative indices fall to the slow path
                //    to preserve Wren semantics.
                builder.switch_to_block(typed_array_block);
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
                    .brif(is_u8, get_u8_block, &[], check_i32_block, &[]);
                builder.switch_to_block(check_i32_block);
                let k_i32_const = builder.ins().iconst(types::I64, TA_KIND_I32 as i64);
                let is_i32 = builder.ins().icmp(IntCC::Equal, kind, k_i32_const);
                builder
                    .ins()
                    .brif(is_i32, get_i32_block, &[], check_f32_block, &[]);
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

                // 5b. I32: 4-byte signed load → f64 → box.
                builder.switch_to_block(get_i32_block);
                let four_i32 = builder.ins().iconst(types::I64, 4);
                let i32_offset = builder.ins().imul(idx_i, four_i32);
                let i32_addr = builder.ins().iadd(data, i32_offset);
                let i32_val = builder
                    .ins()
                    .load(types::I32, MemFlags::trusted(), i32_addr, 0);
                let i32_as_i64 = builder.ins().sextend(types::I64, i32_val);
                let i32_as_f64 = builder.ins().fcvt_from_sint(types::F64, i32_as_i64);
                let i32_bits = builder
                    .ins()
                    .bitcast(types::I64, MemFlags::new(), i32_as_f64);
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(i32_bits)]);

                // 5c. F32: 4-byte float load → f64 promote → box.
                builder.switch_to_block(get_f32_block);
                let four_f32 = builder.ins().iconst(types::I64, 4);
                let f32_offset = builder.ins().imul(idx_i, four_f32);
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

                // 5d. F64: direct 8-byte float load → box.
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

                // 6. Simd fast path: fixed 4-lane bounds check then
                //    lane-kind dispatch.
                builder.switch_to_block(simd_block);
                let simd_idx_f = builder.ins().bitcast(types::F64, MemFlags::new(), idx);
                let simd_idx_i = builder.ins().fcvt_to_sint(types::I64, simd_idx_f);
                let zero = builder.ins().iconst(types::I64, 0);
                let four_lanes = builder.ins().iconst(types::I64, 4);
                let simd_in_range_low =
                    builder
                        .ins()
                        .icmp(IntCC::SignedGreaterThanOrEqual, simd_idx_i, zero);
                let simd_in_range_high =
                    builder
                        .ins()
                        .icmp(IntCC::SignedLessThan, simd_idx_i, four_lanes);
                let simd_in_range = builder.ins().band(simd_in_range_low, simd_in_range_high);
                builder
                    .ins()
                    .brif(simd_in_range, simd_in_bounds_block, &[], slow_block, &[]);

                builder.switch_to_block(simd_in_bounds_block);
                let simd_kind =
                    builder
                        .ins()
                        .uload8(types::I64, MemFlags::trusted(), obj_ptr, SIMD_KIND);
                let simd_f32_const = builder.ins().iconst(types::I64, SIMD_KIND_F32X4 as i64);
                let simd_is_f32 = builder.ins().icmp(IntCC::Equal, simd_kind, simd_f32_const);
                builder.ins().brif(
                    simd_is_f32,
                    simd_get_f32_block,
                    &[],
                    simd_get_i32_block,
                    &[],
                );

                builder.switch_to_block(simd_get_f32_block);
                let simd_data_base = builder.ins().iadd_imm(obj_ptr, SIMD_LANES as i64);
                let four_simd_f32 = builder.ins().iconst(types::I64, 4);
                let simd_f32_offset = builder.ins().imul(simd_idx_i, four_simd_f32);
                let simd_f32_addr = builder.ins().iadd(simd_data_base, simd_f32_offset);
                let simd_f32_val =
                    builder
                        .ins()
                        .load(types::F32, MemFlags::trusted(), simd_f32_addr, 0);
                let simd_f32_as_f64 = builder.ins().fpromote(types::F64, simd_f32_val);
                let simd_f32_bits =
                    builder
                        .ins()
                        .bitcast(types::I64, MemFlags::new(), simd_f32_as_f64);
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(simd_f32_bits)]);

                builder.switch_to_block(simd_get_i32_block);
                let simd_data_base = builder.ins().iadd_imm(obj_ptr, SIMD_LANES as i64);
                let four_simd_i32 = builder.ins().iconst(types::I64, 4);
                let simd_i32_offset = builder.ins().imul(simd_idx_i, four_simd_i32);
                let simd_i32_addr = builder.ins().iadd(simd_data_base, simd_i32_offset);
                let simd_i32_val =
                    builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), simd_i32_addr, 0);
                let simd_i32_as_i64 = builder.ins().sextend(types::I64, simd_i32_val);
                let simd_i32_as_f64 = builder.ins().fcvt_from_sint(types::F64, simd_i32_as_i64);
                let simd_i32_bits =
                    builder
                        .ins()
                        .bitcast(types::I64, MemFlags::new(), simd_i32_as_f64);
                builder
                    .ins()
                    .jump(merge_block, &[BlockArg::Value(simd_i32_bits)]);

                // 7. Slow path: existing runtime dispatch.
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
            //
            // AOT mode: dedup against the per-module string table
            // and emit a `load` against `wlift_consts_<n>[slot]` —
            // the slot's `*mut ObjString` is populated once by the
            // per-module init pass at startup, so the body's use
            // site is a single load, no helper call.
            //
            // JIT mode keeps `wren_const_string(sym)` — the helper
            // resolves through the live VM's interner + GC and
            // caches the resulting `ObjString` per call. Cheaper
            // for the JIT, which can't pre-bake a per-module
            // table without ahead-of-time knowledge of the program.
            Instruction::ConstString(idx) => {
                if let Some(cfg) = aot_config {
                    let slot = {
                        let mut tbl = cfg.const_strings.borrow_mut();
                        match tbl.iter().position(|(s, _)| *s == *idx) {
                            Some(s) => s,
                            None => {
                                let text = interner
                                    .resolve(crate::intern::SymbolId::from_raw(*idx))
                                    .to_string();
                                tbl.push((*idx, text));
                                tbl.len() - 1
                            }
                        }
                    };
                    let gv = module.declare_data_in_func(cfg.consts_data, builder.func);
                    let base = builder.ins().global_value(types::I64, gv);
                    let result = builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        base,
                        (slot as i32) * 8,
                    );
                    Ok(Some(result))
                } else {
                    let f = get_runtime_fn(module, builder, "wren_const_string", 1)?;
                    let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                    let result = builder.ins().call(f, &[idx_val]);
                    Ok(Some(builder.inst_results(result)[0]))
                }
            }

            // === Static self-calls ===
            Instruction::CallStaticSelf { args } => {
                // SM-tainted bodies are lowered with a fixed
                // `(fiber, resume_v) -> i64` poll signature, so a
                // direct recursive call would feed (receiver, args...)
                // into the 2-arg slot and trip Cranelift's verifier
                // with "mismatched argument count". Route through
                // `wren_call_N` instead, mirroring the SM-skip guard
                // in the dispatch_call CHA fast block — dispatch_method
                // sets up a fresh SM poll cycle with the right ABI.
                #[cfg(feature = "aot")]
                if let Some(cfg) = aot_config {
                    if cfg.current_state_machine_layout.borrow().is_some() {
                        let recv = receiver_val.ok_or_else(|| {
                            "CallStaticSelf in SM body without receiver_val".to_string()
                        })?;
                        let slot = aot_intern_symbol(cfg, _mir.name.index(), interner);
                        let gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
                        let base = builder.ins().global_value(types::I64, gv);
                        let method_val = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            base,
                            (slot as i32) * 8,
                        );
                        let arg_vals: Vec<_> = args.iter().map(|a| get(a)).collect();
                        let result = emit_wren_call(
                            builder,
                            module,
                            get_runtime_fn,
                            recv,
                            method_val,
                            &arg_vals,
                        )?;
                        return Ok(Some(result));
                    }
                }
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

    /// Convert a raw-bool (Cranelift i8/i64 0-or-1) into the
    /// NaN-boxed Wren bool the rest of the lowering expects when a
    /// value flows through a block param. The receiving block param
    /// is typed i64 and downstream uses assume Wren-Value semantics;
    /// passing a raw 0 misclassifies as truthy under the standard
    /// `v != TAG_FALSE && v != TAG_NULL` truthiness check.
    fn box_raw_bool_if_needed(builder: &mut FunctionBuilder, v: Value, is_raw_bool: bool) -> Value {
        if !is_raw_bool {
            return v;
        }
        let tag_true = builder.ins().iconst(types::I64, TAG_TRUE as i64);
        let tag_false = builder.ins().iconst(types::I64, TAG_FALSE as i64);
        builder.ins().select(v, tag_true, tag_false)
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
                // Box raw-bool args into NaN-boxed Wren bools before
                // passing as block params. Block params are typed
                // i64 (Wren Value) and downstream uses (e.g.
                // CondBranch's truthiness check) assume that. Without
                // boxing, a raw 0 (raw false) passes the
                // "v != TAG_FALSE && v != TAG_NULL" check and gets
                // misclassified as truthy — turning `while (cond)`
                // into an infinite loop after the second function
                // call when `cond` flows through a phi-style block
                // param. Surfaced as Reader.readLine hanging on the
                // second call once _lineBuf had any leftover bytes.
                let cl_block = block_map[target];
                let cl_args: Vec<BlockArg> = args
                    .iter()
                    .map(|a| {
                        BlockArg::Value(box_raw_bool_if_needed(
                            builder,
                            get(a),
                            raw_bools.contains(a),
                        ))
                    })
                    .collect();
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
                let t_args: Vec<BlockArg> = true_args
                    .iter()
                    .map(|a| {
                        BlockArg::Value(box_raw_bool_if_needed(
                            builder,
                            get(a),
                            raw_bools.contains(a),
                        ))
                    })
                    .collect();
                let f_args: Vec<BlockArg> = false_args
                    .iter()
                    .map(|a| {
                        BlockArg::Value(box_raw_bool_if_needed(
                            builder,
                            get(a),
                            raw_bools.contains(a),
                        ))
                    })
                    .collect();

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
    #[allow(dead_code)]
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

    /// Reverse post-order with multiple DFS roots. Each root is
    /// visited only once across all roots so a block reachable
    /// from several roots ends up in a single, dominance-respecting
    /// position. Unreachable blocks are appended in block-index
    /// order so every block has a slot (Cranelift requires a
    /// terminator on every created block; the lowering emits a
    /// `trap` for blocks that didn't make the `reachable` set).
    ///
    /// AOT state-machine bodies need this: the synthetic dispatch
    /// `br_table` is the only predecessor of each resume entry, so
    /// a plain DFS-from-bb0 misses them. Seeding DFS from `bb0`
    /// plus every resume entry recovers the dominance order
    /// Cranelift expects between a CrossFnCallResume block (which
    /// emits the `load_value` defs for its saved values) and any
    /// downstream block that uses those defs.
    #[cfg(feature = "aot")]
    fn compute_rpo_multi_root(mir: &MirFunction, roots: &[BlockId]) -> Vec<usize> {
        let n = mir.blocks.len();
        let mut visited = vec![false; n];
        let mut post_order = Vec::with_capacity(n);

        fn dfs(idx: usize, mir: &MirFunction, visited: &mut [bool], post_order: &mut Vec<usize>) {
            if idx >= mir.blocks.len() || visited[idx] {
                return;
            }
            visited[idx] = true;
            for succ in mir.blocks[idx].terminator.successors() {
                dfs(succ.0 as usize, mir, visited, post_order);
            }
            post_order.push(idx);
        }

        for root in roots {
            dfs(root.0 as usize, mir, &mut visited, &mut post_order);
        }
        post_order.reverse();
        for (i, &seen) in visited.iter().enumerate().take(n) {
            if !seen {
                post_order.push(i);
            }
        }
        post_order
    }
}
