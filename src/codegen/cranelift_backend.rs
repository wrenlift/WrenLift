/// Cranelift-based JIT backend.
///
/// Translates MIR directly to Cranelift IR, bypassing the custom MachInst layer.
/// This provides correct register allocation and instruction encoding for x86_64
/// without the SCRATCH_GP / spill-slot conflicts of the hand-written emitter.
#[cfg(feature = "cranelift")]
pub mod cl {
    use crate::intern::Interner;
    use crate::mir::{
        BlockId, DeoptReg, Instruction, MirFunction, MirType, Terminator, ValueId,
        osr_external_live_values, osr_reachable_blocks, osr_rematerializable_defs,
    };
    use crate::runtime::object_layout::*;
    use cranelift_codegen::Context;
    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{
        AbiParam, BlockArg, Function, InstBuilder, MemFlagsData as MemFlags, Signature, Type, Value,
    };
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    use std::collections::{HashMap, HashSet};

    pub(crate) const QNAN: u64 = 0x7FFC_0000_0000_0000;
    pub(crate) const SIGN_BIT: u64 = 1u64 << 63;
    /// Top-16-bit pattern for an object NaN-box: `SIGN_BIT | QNAN`. A
    /// receiver Value is an object iff `(value & TAG_OBJ) == TAG_OBJ`
    /// — every other Wren Value (Number, Null, Bool, Undefined,
    /// String-box) clears at least one of those bits. JIT class-check
    /// sites must guard with this before reading `recv.class`,
    /// because the load offset (`HEADER_CLASS`) doesn't fail
    /// "safely" for a non-object: a Number's f64 bits, masked
    /// through PTR_MASK, can land at an unmapped page and SIGSEGV.
    pub(crate) const TAG_OBJ: u64 = SIGN_BIT | QNAN;
    pub(crate) const TAG_NULL: u64 = QNAN; // 0x7FFC_0000_0000_0000 — no extra bits
    pub(crate) const TAG_FALSE: u64 = QNAN | 1;
    pub(crate) const TAG_TRUE: u64 = QNAN | 2;
    // Note: QNAN | 3 = TAG_UNDEFINED (not null!)
    pub(crate) const PTR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    /// Emit a guarded receiver-class load: returns `(obj_ptr,
    /// recv_class)` iff the receiver is a NaN-boxed object,
    /// branching to `not_object_block` otherwise. Safe to use on
    /// any Value; non-object receivers (Numbers, Null, Bool, ...)
    /// take the not-object branch instead of dereferencing garbage
    /// at HEADER_CLASS.
    /// Whether AArch64 `fmov` encodes the value as an immediate
    /// (zero, or ±(16..=31)/16 × 2^(-3..=4)); x86-64 folds the rest
    /// cheaply enough that the same rule is used for both.
    pub(crate) fn f64_is_fmov_immediate(n: f64) -> bool {
        if n == 0.0 {
            return true;
        }
        let bits = n.to_bits();
        let frac = bits & ((1u64 << 52) - 1);
        if frac & ((1u64 << 48) - 1) != 0 {
            return false;
        }
        let exp = ((bits >> 52) & 0x7ff) as i32 - 1023;
        (-3..=4).contains(&exp)
    }

    pub(crate) fn const_f64_of(mir: &MirFunction, vid: ValueId) -> Option<f64> {
        mir.blocks.iter().find_map(|b| {
            b.instructions.iter().find_map(|(d, i)| match i {
                Instruction::ConstF64(c) if *d == vid => Some(*c),
                _ => None,
            })
        })
    }

    pub(crate) fn is_positive_power_of_two(c: f64) -> bool {
        c > 0.0
            && c.is_finite()
            && (c.to_bits() & ((1u64 << 52) - 1)) == 0
            && c >= f64::MIN_POSITIVE
    }

    /// A fresh instance of the class object `class_val`: bumped out of
    /// the Immix small region when the compile knows it, its fields
    /// null, the start byte written; the helper when the region is out
    /// of room, the object exceeds a line, or there is no region.
    #[allow(clippy::type_complexity)]
    fn emit_alloc_instance(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        class_val: Value,
    ) -> Result<Value, String> {
        use crate::runtime::gc_immix_heap::{BUMP_CODES, BUMP_CUR, BUMP_LIMIT, BUMP_PLAIN_FLAG};
        let bump = crate::codegen::jit_bump_region();
        let helper = get_runtime_fn(module, builder, "wren_alloc_instance", 1)?;
        if bump == 0 {
            let call = builder.ins().call(helper, &[class_val]);
            return Ok(builder.inst_results(call)[0]);
        }
        let slow_block = builder.create_block();
        let sized_block = builder.create_block();
        let fast_block = builder.create_block();
        let loop_block = builder.create_block();
        let done_block = builder.create_block();
        let merge_block = builder.create_block();
        builder.append_block_param(loop_block, types::I64);
        builder.append_block_param(merge_block, types::I64);

        let mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
        let class = builder.ins().band(class_val, mask);
        let nf16 = builder
            .ins()
            .load(types::I16, MemFlags::trusted(), class, CLASS_NUM_FIELDS);
        let nf = builder.ins().uextend(types::I64, nf16);
        // size = (40 + 8 * nf + 15) & !15
        let fb = builder.ins().imul_imm_u(nf, VALUE_SIZE as i64);
        let raw = builder.ins().iadd_imm_u(fb, INSTANCE_SIZE as i64 + 15);
        let size = builder.ins().band_imm_s(raw, !15i64);
        let fits = builder
            .ins()
            .icmp_imm_u(IntCC::UnsignedLessThanOrEqual, size, 128);
        builder.ins().brif(fits, sized_block, &[], slow_block, &[]);

        builder.switch_to_block(sized_block);
        let bump_v = builder.ins().iconst(types::I64, bump as i64);
        let cur = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), bump_v, BUMP_CUR);
        let limit = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), bump_v, BUMP_LIMIT);
        // Never straddle a line: start at the next line if the object
        // would.
        let off = builder.ins().band_imm_u(cur, 127);
        let end_in_line = builder.ins().iadd(off, size);
        let straddles = builder
            .ins()
            .icmp_imm_u(IntCC::UnsignedGreaterThan, end_in_line, 128);
        let c127 = builder.ins().iadd_imm_u(cur, 127);
        let aligned = builder.ins().band_imm_s(c127, !127i64);
        let p = builder.ins().select(straddles, aligned, cur);
        let np = builder.ins().iadd(p, size);
        let room = builder
            .ins()
            .icmp(IntCC::UnsignedLessThanOrEqual, np, limit);
        builder.ins().brif(room, fast_block, &[], slow_block, &[]);

        builder.switch_to_block(fast_block);
        builder
            .ins()
            .store(MemFlags::trusted(), np, bump_v, BUMP_CUR);
        let codes = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), bump_v, BUMP_CODES);
        let q = builder.ins().ushr_imm_u(p, 4);
        let code_p = builder.ins().iadd(codes, q);
        let sq = builder.ins().ushr_imm_u(size, 4);
        let code = builder.ins().bor_imm_u(sq, BUMP_PLAIN_FLAG as i64);
        let code8 = builder.ins().ireduce(types::I8, code);
        builder.ins().store(MemFlags::trusted(), code8, code_p, 0);
        // Header: type byte, clear mark and flags, the class, the
        // field count, no owned fields, the fields right after the
        // header.
        let type_word = builder.ins().iconst(types::I64, OBJ_TYPE_INSTANCE as i64);
        builder.ins().store(MemFlags::trusted(), type_word, p, 0);
        let zero = builder.ins().iconst(types::I64, 0);
        builder
            .ins()
            .store(MemFlags::trusted(), class, p, HEADER_CLASS);
        builder
            .ins()
            .store(MemFlags::trusted(), nf, p, INSTANCE_NUM_FIELDS);
        let fields = builder.ins().iadd_imm_u(p, INSTANCE_SIZE as i64);
        let has_fields = builder.ins().icmp_imm_u(IntCC::NotEqual, nf, 0);
        let fields_or_null = builder.ins().select(has_fields, fields, zero);
        builder
            .ins()
            .store(MemFlags::trusted(), fields_or_null, p, INSTANCE_FIELDS);
        builder.ins().brif(
            has_fields,
            loop_block,
            &[BlockArg::Value(zero)],
            done_block,
            &[],
        );

        // Null every field.
        builder.switch_to_block(loop_block);
        let i = builder.block_params(loop_block)[0];
        let io = builder.ins().imul_imm_u(i, 8);
        let slot = builder.ins().iadd(fields, io);
        let null_v = builder.ins().iconst(types::I64, TAG_NULL as i64);
        builder.ins().store(MemFlags::trusted(), null_v, slot, 0);
        let next = builder.ins().iadd_imm_u(i, 1);
        let more = builder.ins().icmp(IntCC::UnsignedLessThan, next, nf);
        builder
            .ins()
            .brif(more, loop_block, &[BlockArg::Value(next)], done_block, &[]);

        builder.switch_to_block(done_block);
        let tag = builder.ins().iconst(types::I64, TAG_OBJ as i64);
        let boxed = builder.ins().bor(p, tag);
        builder.ins().jump(merge_block, &[BlockArg::Value(boxed)]);

        builder.switch_to_block(slow_block);
        let call = builder.ins().call(helper, &[class_val]);
        let sv = builder.inst_results(call)[0];
        builder.ins().jump(merge_block, &[BlockArg::Value(sv)]);

        builder.switch_to_block(merge_block);
        Ok(builder.block_params(merge_block)[0])
    }

    /// Fold the class of `value` into the element-class word of the
    /// list at `obj`, as `ObjList::note_element` does: unseen takes the
    /// class, the same class keeps it, anything else is mixed.
    fn emit_note_list_element(builder: &mut FunctionBuilder, obj: Value, value: Value) {
        use crate::runtime::object::ELEM_CLASS_MIXED;
        let tag_obj = builder.ins().iconst(types::I64, TAG_OBJ as i64);
        let high = builder.ins().band(value, tag_obj);
        let is_obj = builder.ins().icmp(IntCC::Equal, high, tag_obj);
        let mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
        let masked = builder.ins().band(value, mask);
        let null_obj = builder.ins().iconst(
            types::I64,
            crate::codegen::runtime_fns::JIT_NULL_OBJECT.as_ptr() as i64,
        );
        let ptr = builder.ins().select(is_obj, masked, null_obj);
        let class = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), ptr, HEADER_CLASS);
        let mixed = builder.ins().iconst(types::I64, ELEM_CLASS_MIXED as i64);
        // The null object's class is zero: not an object, so mixed.
        let has_class = builder.ins().icmp_imm_u(IntCC::NotEqual, class, 0);
        let class = builder.ins().select(has_class, class, mixed);
        let cur = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), obj, LIST_ELEM_CLASS);
        let unseen = builder.ins().icmp_imm_u(IntCC::Equal, cur, 0);
        let same = builder.ins().icmp(IntCC::Equal, cur, class);
        let keep = builder.ins().bor(unseen, same);
        let new = builder.ins().select(keep, class, mixed);
        builder
            .ins()
            .store(MemFlags::trusted(), new, obj, LIST_ELEM_CLASS);
    }

    /// Or the kind of `value` into the class's field-kind byte for
    /// field `idx` of the instance at `obj_ptr`, as
    /// `ObjInstance::note_field_kind` does; a class without the bytes
    /// is skipped.
    fn emit_note_field_kind(builder: &mut FunctionBuilder, obj_ptr: Value, idx: u16, value: Value) {
        let class = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), obj_ptr, HEADER_CLASS);
        let kinds = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), class, CLASS_FIELD_KINDS);
        let note = builder.create_block();
        let done = builder.create_block();
        builder.ins().brif(kinds, note, &[], done, &[]);
        builder.switch_to_block(note);
        emit_store_kind_bit(builder, kinds, idx as i32, value, done);
        builder.switch_to_block(done);
    }

    /// `emit_note_field_kind` for an instance of a class known at
    /// compile time: the byte has a fixed address, and one that already
    /// records another kind, or the kind being stored, never changes
    /// again.
    fn emit_note_field_kind_static(
        builder: &mut FunctionBuilder,
        class: usize,
        idx: u16,
        value: Value,
    ) {
        use crate::runtime::object::FIELD_OTHER;
        let class = class as *const crate::runtime::object::ObjClass;
        let kinds = unsafe { (*class).field_kinds_ptr };
        let len = unsafe { (*class).field_kinds.len() };
        if kinds.is_null() || idx as usize >= len {
            return;
        }
        // The main thread may be or'ing this byte while the compile
        // reads it; the compiled code reads it again at run time.
        let seen = unsafe { std::ptr::read_volatile(kinds.add(idx as usize)) };
        if seen & FIELD_OTHER != 0 {
            return;
        }
        let p = builder.ins().iconst(types::I64, kinds as i64);
        let done = builder.create_block();
        emit_store_kind_bit(builder, p, idx as i32, value, done);
        builder.switch_to_block(done);
    }

    /// Or the kind of `value` into the byte at `kinds + idx`, storing
    /// only when that changes it, then continue in `done`.
    fn emit_store_kind_bit(
        builder: &mut FunctionBuilder,
        kinds: Value,
        idx: i32,
        value: Value,
        done: cranelift_codegen::ir::Block,
    ) {
        use crate::runtime::object::{FIELD_NUM, FIELD_OTHER};
        let seen = builder
            .ins()
            .load(types::I8, MemFlags::trusted(), kinds, idx);
        let qnan = builder.ins().iconst(types::I64, QNAN as i64);
        let masked = builder.ins().band(value, qnan);
        let is_num = builder.ins().icmp(IntCC::NotEqual, masked, qnan);
        let num_bit = builder.ins().iconst(types::I8, FIELD_NUM as i64);
        let other_bit = builder.ins().iconst(types::I8, FIELD_OTHER as i64);
        let bit = builder.ins().select(is_num, num_bit, other_bit);
        let new = builder.ins().bor(seen, bit);
        let changed = builder.ins().icmp(IntCC::NotEqual, new, seen);
        let store = builder.create_block();
        builder.ins().brif(changed, store, &[], done, &[]);
        builder.switch_to_block(store);
        builder.ins().store(MemFlags::trusted(), new, kinds, idx);
        builder.ins().jump(done, &[]);
    }

    /// Or the kind of `result` into the result profile byte at `slot`.
    fn emit_note_call_result(builder: &mut FunctionBuilder, slot: usize, result: Value) {
        use crate::mir::bytecode::{RESULT_NUM, RESULT_OTHER};
        let p = builder.ins().iconst(types::I64, slot as i64);
        let seen = builder.ins().load(types::I8, MemFlags::trusted(), p, 0);
        let qnan = builder.ins().iconst(types::I64, QNAN as i64);
        let masked = builder.ins().band(result, qnan);
        let is_num = builder.ins().icmp(IntCC::NotEqual, masked, qnan);
        let num_bit = builder.ins().iconst(types::I8, RESULT_NUM as i64);
        let other_bit = builder.ins().iconst(types::I8, RESULT_OTHER as i64);
        let bit = builder.ins().select(is_num, num_bit, other_bit);
        let seen = builder.ins().bor(seen, bit);
        builder.ins().store(MemFlags::trusted(), seen, p, 0);
    }

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
                if args.len() > 8 {
                    "wren_call_dynamic"
                } else {
                    "wren_call_N"
                }
            );
        }
        if args.len() > 8 {
            let f = get_runtime_fn(module, builder, "wren_call_dynamic", 4)?;
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                (args.len() as u32) * 8,
                3, // 8-byte alignment (2^3)
            ));
            let buf_ptr = builder.ins().stack_addr(types::I64, slot, 0);
            for (i, &arg) in args.iter().enumerate() {
                builder
                    .ins()
                    .store(MemFlags::trusted(), arg, buf_ptr, (i as i32) * 8);
            }
            let count = builder.ins().iconst(types::I64, args.len() as i64);
            emit_cur_frame(builder);
            let call = builder
                .ins()
                .call(f, &[receiver, method_val, count, buf_ptr]);
            emit_error_poll(builder, module, get_runtime_fn)?;
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
        emit_cur_frame(builder);
        let call = builder.ins().call(f, &call_args);
        emit_error_poll(builder, module, get_runtime_fn)?;
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
            let base = builder.ins().symbol_value(types::I64, gv);
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

    /// `iterate(_)`, `iteratorValue(_)` and `add(_)` on a list, inline:
    /// the protocol steps a Num index against the count and ends with
    /// `false`, the value is the element at a Num index within the
    /// count, and an add with room stores and counts. Anything else
    /// takes the call. Returns `None` when `method` is not one of them.
    fn try_lower_list_protocol<G>(
        interner: &Interner,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut G,
        receiver: Value,
        method: crate::intern::SymbolId,
        args: &[Value],
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
        if args.len() != 1 {
            return Ok(None);
        }
        let which = match interner.resolve(method) {
            "iterate(_)" => 0,
            "iteratorValue(_)" => 1,
            "add(_)" => 2,
            _ => return Ok(None),
        };
        let arg = args[0];
        let slow = builder.create_block();
        let merge = builder.create_block();
        builder.append_block_param(merge, types::I64);
        let obj = emit_guarded_obj_ptr_with_type(
            builder,
            receiver,
            crate::runtime::object::ObjType::List as i32,
            slow,
        );
        let count32 = builder.ins().uload32(MemFlags::trusted(), obj, LIST_COUNT);
        let qnan = builder.ins().iconst(types::I64, QNAN as i64);
        let masked = builder.ins().band(arg, qnan);
        let is_box = builder.ins().icmp(IntCC::Equal, masked, qnan);
        match which {
            0 => {
                let countf = builder.ins().fcvt_from_uint(types::F64, count32);
                let tag_null = builder.ins().iconst(types::I64, TAG_NULL as i64);
                let is_null = builder.ins().icmp(IntCC::Equal, arg, tag_null);
                let num_block = builder.create_block();
                let step_block = builder.create_block();
                let cand_block = builder.create_block();
                builder.append_block_param(cand_block, types::F64);
                let zero = builder.ins().f64const(0.0);
                builder.ins().brif(
                    is_null,
                    cand_block,
                    &[BlockArg::Value(zero)],
                    num_block,
                    &[],
                );
                builder.switch_to_block(num_block);
                builder.ins().brif(is_box, slow, &[], step_block, &[]);
                builder.switch_to_block(step_block);
                let f = builder.ins().bitcast(types::F64, MemFlags::new(), arg);
                let one = builder.ins().f64const(1.0);
                let next = builder.ins().fadd(f, one);
                builder.ins().jump(cand_block, &[BlockArg::Value(next)]);
                builder.switch_to_block(cand_block);
                let cand = builder.block_params(cand_block)[0];
                let in_range = builder.ins().fcmp(FloatCC::LessThan, cand, countf);
                let bits = builder.ins().bitcast(types::I64, MemFlags::new(), cand);
                let tag_false = builder.ins().iconst(types::I64, TAG_FALSE as i64);
                let res = builder.ins().select(in_range, bits, tag_false);
                builder.ins().jump(merge, &[BlockArg::Value(res)]);
            }
            1 => {
                let num_block = builder.create_block();
                let load_block = builder.create_block();
                builder.ins().brif(is_box, slow, &[], num_block, &[]);
                builder.switch_to_block(num_block);
                let f = builder.ins().bitcast(types::F64, MemFlags::new(), arg);
                let idx = builder.ins().fcvt_to_sint_sat(types::I64, f);
                let in_range = builder.ins().icmp(IntCC::UnsignedLessThan, idx, count32);
                builder.ins().brif(in_range, load_block, &[], slow, &[]);
                builder.switch_to_block(load_block);
                let elements =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj, LIST_ELEMENTS);
                let off = builder.ins().ishl_imm_u(idx, 3);
                let ea = builder.ins().iadd(elements, off);
                let v = builder.ins().load(types::I64, MemFlags::trusted(), ea, 0);
                builder.ins().jump(merge, &[BlockArg::Value(v)]);
            }
            _ => {
                let cap32 = builder
                    .ins()
                    .uload32(MemFlags::trusted(), obj, LIST_CAPACITY);
                let room = builder.ins().icmp(IntCC::UnsignedLessThan, count32, cap32);
                let store_block = builder.create_block();
                let grow_block = builder.create_block();
                builder.ins().brif(room, store_block, &[], grow_block, &[]);
                builder.switch_to_block(grow_block);
                let f = get_runtime_fn(module, builder, "wren_list_add", 2)?;
                builder.ins().call(f, &[receiver, arg]);
                builder.ins().jump(merge, &[BlockArg::Value(receiver)]);
                builder.switch_to_block(store_block);
                emit_note_list_element(builder, obj, arg);
                let elements =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj, LIST_ELEMENTS);
                let off = builder.ins().ishl_imm_u(count32, 3);
                let ea = builder.ins().iadd(elements, off);
                builder.ins().store(MemFlags::trusted(), arg, ea, 0);
                let next = builder.ins().iadd_imm_u(count32, 1);
                let next32 = builder.ins().ireduce(types::I32, next);
                builder
                    .ins()
                    .store(MemFlags::trusted(), next32, obj, LIST_COUNT);
                builder.ins().jump(merge, &[BlockArg::Value(receiver)]);
            }
        }
        builder.switch_to_block(slow);
        let method_val = builder.ins().iconst(types::I64, method.index() as i64);
        let sv = emit_wren_call(builder, module, get_runtime_fn, receiver, method_val, args)?;
        builder.ins().jump(merge, &[BlockArg::Value(sv)]);
        builder.switch_to_block(merge);
        Ok(Some(builder.block_params(merge)[0]))
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
        ic_idx: Option<usize>,
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
            ic_idx,
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
    // `WLIFT_ENABLE_PURE_LEAF_DIRECT`).
    // `std::env::var_os` acquires a global mutex on every call — fine for
    // one-shot startup probes, but the broker thread compiles dozens of
    // functions during warmup with hundreds of call sites between them.
    // Cache once into a `OnceLock<bool>`.

    #[inline]
    pub(crate) fn env_jit_callsite_ic() -> bool {
        use std::sync::OnceLock;
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("WLIFT_ENABLE_JIT_CALLSITE_IC").is_some())
    }

    thread_local! {
        /// Address of the compiling function's module variable cell
        /// (`engine::ModuleVarsCell`), set by the engine around each JIT
        /// compile on this thread. Zero means "unknown, use the helper".
        static JIT_MODVARS_CELL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Set the module variable cell JIT lowering bakes for this thread's
    /// next compile; pass 0 to clear.
    pub fn set_jit_modvars_cell(addr: usize) {
        JIT_MODVARS_CELL.with(|c| c.set(addr));
    }

    thread_local! {
        /// The receiver of a body being spliced behind its class check,
        /// with that class.
        static INLINE_CLASS: std::cell::Cell<Option<(Value, usize)>> =
            const { std::cell::Cell::new(None) };
    }

    use crate::codegen::COLD_LOOP_EXIT_AFTER;

    /// What compiled code does for the tier above it: count its
    /// entries and loop iterations in `cell`, calling `wren_tier_tick`
    /// when the count reaches the cell's next tick, and poll the
    /// cell's re-tier word at each header in `retier_headers`
    /// (outermost loops only), handing the header's live-ins to
    /// `wren_retier` when the word is set. Every loop header in
    /// `tick_headers` counts, so a body that lives in one long outer
    /// loop still reaches its proposal: baseline code counts each down
    /// in the cell, an optimised body in a register it debits the cell
    /// from every `LOOP_TICK` iterations, so the count costs its loops
    /// no store. Every call's result kind is
    /// or'd into the byte at `result_kinds + register` when that base
    /// is non-zero.
    #[derive(Clone, Default)]
    pub struct TierHook {
        pub func_id: u32,
        pub cell: usize,
        /// The body's optimised generation, 0 for baseline code; it
        /// polls for a newer one.
        pub generation: u32,
        pub retier_headers: HashSet<BlockId>,
        pub tick_headers: HashSet<BlockId>,
        pub result_kinds: usize,
        /// Bytes at `result_kinds`: one per bytecode register. Values
        /// the compile clone adds beyond them have no byte.
        pub result_kinds_len: usize,
    }

    thread_local! {
        static JIT_TIER_HOOK: std::cell::RefCell<Option<TierHook>> =
            const { std::cell::RefCell::new(None) };
    }

    /// Set the tier hook for this thread's next compile; `None` clears it.
    pub fn set_jit_tier_hook(hook: Option<TierHook>) {
        JIT_TIER_HOOK.with(|c| *c.borrow_mut() = hook);
    }

    thread_local! {
        /// The function's tier cell and the body's generation for this
        /// thread's next top-tier compile: a cold loop polls the cell's
        /// re-tier word for a newer generation.
        static JIT_RETIER_CELL: std::cell::Cell<(usize, u32)> = const { std::cell::Cell::new((0, 0)) };
    }

    pub fn set_jit_retier_cell(cell: usize, generation: u32) {
        JIT_RETIER_CELL.with(|c| c.set((cell, generation)));
    }

    fn jit_retier_cell() -> (usize, u32) {
        JIT_RETIER_CELL.with(|c| c.get())
    }

    /// The re-tier word a body of generation `generation` polls.
    fn retier_word_offset(generation: u32) -> i32 {
        if generation == 0 {
            TIER_CELL_RETIER
        } else {
            TIER_CELL_RETIER_TOP
        }
    }

    /// The header word `wren_retier` decodes: the caller's generation
    /// above the block id.
    fn retier_header_word(header: BlockId, generation: u32) -> i64 {
        header.0 as i64 | ((generation as i64) << 32)
    }

    thread_local! {
        /// Whether this thread's next compile notes the kind of every
        /// field it stores, for the LLVM tier's speculation: every
        /// native body must, or a byte goes stale.
        static JIT_NOTE_FIELD_KINDS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    pub fn set_jit_note_field_kinds(on: bool) {
        JIT_NOTE_FIELD_KINDS.with(|c| c.set(on));
    }

    fn jit_note_field_kinds() -> bool {
        JIT_NOTE_FIELD_KINDS.with(|c| c.get())
    }

    thread_local! {
        /// The function this thread is compiling, for code that names
        /// itself to the runtime (guard deopts).
        static JIT_FUNC_ID: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };
    }

    /// Set the function id for this thread's next JIT compile.
    pub fn set_jit_func_id(id: u32) {
        JIT_FUNC_ID.with(|c| c.set(id));
    }

    pub(crate) fn jit_func_id() -> u32 {
        JIT_FUNC_ID.with(|c| c.get())
    }

    fn jit_tier_hook() -> Option<TierHook> {
        JIT_TIER_HOOK.with(|c| c.borrow().clone())
    }

    /// Byte offsets inside `engine::TierCell`.
    const TIER_CELL_COUNTDOWN: i32 = 0;
    const TIER_CELL_RETIER: i32 = 4;
    const TIER_CELL_RETIER_TOP: i32 = 16;

    /// Load a word from the safepoint page: unreadable while a
    /// collector waits, so the load faults and the fault handler parks
    /// the thread. The load may trap, so nothing removes it.
    fn emit_safepoint_poll(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        aot_config: Option<&AotLoweringConfig>,
    ) {
        let addr = match aot_config {
            // An AOT binary finds the runtime's page by symbol.
            Some(cfg) => {
                let gv = module.declare_data_in_func(cfg.safepoint_data, builder.func);
                builder.ins().symbol_value(types::I64, gv)
            }
            None => {
                let page = crate::codegen::jit_safepoint_page();
                if page == 0 {
                    return;
                }
                builder.ins().iconst(types::I64, page as i64)
            }
        };
        builder.ins().load(types::I32, MemFlags::new(), addr, 0);
    }

    /// Count down the cell and call `wren_tier_tick` when it reaches
    /// zero.
    #[allow(clippy::type_complexity)] // the runtime-fn resolver closure type is shared verbatim
    fn emit_tier_tick(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        hook: &TierHook,
    ) -> Result<(), String> {
        let cell = builder.ins().iconst(types::I64, hook.cell as i64);
        let c = builder
            .ins()
            .uload32(MemFlags::trusted(), cell, TIER_CELL_COUNTDOWN);
        let c1 = builder.ins().iadd_imm_s(c, -1);
        builder
            .ins()
            .istore32(MemFlags::trusted(), c1, cell, TIER_CELL_COUNTDOWN);
        let tick = builder.ins().icmp_imm_u(IntCC::Equal, c1, 0);
        let tick_block = builder.create_block();
        let cont_block = builder.create_block();
        builder.set_cold_block(tick_block);
        builder.ins().brif(tick, tick_block, &[], cont_block, &[]);
        builder.switch_to_block(tick_block);
        let fid = builder.ins().iconst(types::I64, hook.func_id as i64);
        let f = get_runtime_fn(module, builder, "wren_tier_tick", 1)?;
        builder.ins().call(f, &[fid]);
        builder.ins().jump(cont_block, &[]);
        builder.switch_to_block(cont_block);
        Ok(())
    }

    /// The i64 a boxed value was converted from: `Box(I64ToF64(i))`.
    fn int_source(mir: &MirFunction, v: &ValueId) -> Option<ValueId> {
        let def = |x: &ValueId| {
            mir.blocks
                .iter()
                .flat_map(|b| b.instructions.iter())
                .find(|(d, _)| d == x)
                .map(|(_, i)| i)
        };
        match def(v)? {
            Instruction::Box(f) => match def(f)? {
                Instruction::I64ToF64(i) => Some(*i),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether every block parameter typed Value only ever receives a
    /// Num the inner f64 body carries raw: an entry parameter, a box, a
    /// constant, a recursive result, or such a parameter. Optimistic
    /// over the parameters, to a fixed point.
    fn value_params_carry_nums(mir: &MirFunction) -> bool {
        use crate::mir::Terminator;
        let defs: HashMap<ValueId, &Instruction> = mir
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter().map(|(v, i)| (*v, i)))
            .collect();
        let mut nums: HashSet<ValueId> = mir
            .blocks
            .iter()
            .skip(1)
            .flat_map(|b| b.params.iter())
            .filter(|(_, t)| *t == MirType::Value)
            .map(|(p, _)| *p)
            .collect();
        let carries = |v: ValueId, nums: &HashSet<ValueId>| -> bool {
            let mut v = v;
            for _ in 0..64 {
                if nums.contains(&v) {
                    return true;
                }
                match defs.get(&v) {
                    Some(Instruction::Move(a)) => v = *a,
                    Some(
                        Instruction::BlockParam(_)
                        | Instruction::Box(_)
                        | Instruction::ConstNum(_)
                        | Instruction::CallStaticSelf { .. },
                    ) => return true,
                    _ => return false,
                }
            }
            false
        };
        loop {
            let mut dropped = false;
            for b in &mir.blocks {
                let edges: Vec<(BlockId, &[ValueId])> = match &b.terminator {
                    Terminator::Branch { target, args } => vec![(*target, args.as_slice())],
                    Terminator::CondBranch {
                        true_target,
                        true_args,
                        false_target,
                        false_args,
                        ..
                    } => vec![
                        (*true_target, true_args.as_slice()),
                        (*false_target, false_args.as_slice()),
                    ],
                    _ => Vec::new(),
                };
                for (target, args) in edges {
                    let params = &mir.blocks[target.0 as usize].params;
                    for (i, a) in args.iter().enumerate() {
                        let Some((p, _)) = params.get(i) else {
                            continue;
                        };
                        if nums.contains(p) && !carries(*a, &nums) {
                            nums.remove(p);
                            dropped = true;
                        }
                    }
                }
            }
            if !dropped {
                break;
            }
        }
        mir.blocks
            .iter()
            .skip(1)
            .flat_map(|b| b.params.iter())
            .filter(|(_, t)| *t == MirType::Value)
            .all(|(p, _)| nums.contains(p))
    }

    thread_local! {
        /// Whether the body this thread lowers polls for errors.
        static ERROR_POLL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Leave the function with null when an error is pending, as the
    /// interpreter would have unwound, instead of running on and
    /// raising another over it. Emitted after every call that can
    /// raise — a dispatch, a compiled callee, a helper's slow path — so
    /// an inline fast path never pays for it: the pending word first,
    /// the helper only when it is set.
    fn emit_error_poll<G>(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut G,
    ) -> Result<(), String>
    where
        G: FnMut(
                &mut dyn Module,
                &mut FunctionBuilder,
                &str,
                usize,
            ) -> Result<cranelift_codegen::ir::FuncRef, String>
            + ?Sized,
    {
        if !ERROR_POLL.get() {
            return Ok(());
        }
        let word = builder.ins().iconst(
            types::I64,
            crate::codegen::runtime_fns::ERROR_PENDING.0.as_ptr() as i64,
        );
        let pending = builder.ins().uload32(MemFlags::trusted(), word, 0);
        let check = builder.create_block();
        let leave = builder.create_block();
        let cont = builder.create_block();
        builder.set_cold_block(check);
        builder.set_cold_block(leave);
        builder.ins().brif(pending, check, &[], cont, &[]);
        builder.switch_to_block(check);
        let f = get_runtime_fn(module, builder, "wren_aot_check_error", 0)?;
        let call = builder.ins().call(f, &[]);
        let err = builder.inst_results(call)[0];
        builder.ins().brif(err, leave, &[], cont, &[]);
        builder.switch_to_block(leave);
        let return_ty = builder.func.signature.returns[0].value_type;
        let null = builder.ins().iconst(types::I64, TAG_NULL as i64);
        let null = if return_ty == types::F64 {
            builder.ins().bitcast(types::F64, MemFlags::new(), null)
        } else {
            null
        };
        builder.ins().return_(&[null]);
        builder.switch_to_block(cont);
        Ok(())
    }

    thread_local! {
        /// Address of the current frame pair (`JitThread::cur`) of the
        /// thread the compiling body will run on, set by the engine
        /// around each JIT compile; 0 leaves the body without stores.
        static JIT_CUR_CELL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// Site word of the instruction being lowered: its source
        /// offset plus one, 0 when it has none.
        static FRAME_SITE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// Set the frame pair cell JIT lowering bakes for this thread's
    /// next compile; pass 0 to clear.
    pub fn set_jit_cur_cell(addr: usize) {
        JIT_CUR_CELL.with(|c| c.set(addr));
    }

    /// Before a call into the runtime that can raise or run code: the
    /// thread's current frame pair is this frame and the site. A trace
    /// walks up from here; the direct calls above need nothing, their
    /// return addresses name their sites.
    fn emit_cur_frame(builder: &mut FunctionBuilder) {
        let cell = JIT_CUR_CELL.with(|c| c.get());
        if cell == 0 {
            return;
        }
        let key = jit_func_id() as i64 | ((FRAME_SITE.get() as i64) << 32);
        let cellv = builder.ins().iconst(types::I64, cell as i64);
        let fp = builder.ins().get_frame_pointer(types::I64);
        builder.ins().store(MemFlags::trusted(), fp, cellv, 0);
        let key = builder.ins().iconst(types::I64, key);
        builder.ins().store(MemFlags::trusted(), key, cellv, 8);
    }

    /// The site tables of the functions just defined in `module`, for
    /// the engine to register: each function's code range and the site
    /// each of its instructions was lowered under.
    fn code_sites(
        module: &JITModule,
        defined: &[(cranelift_module::FuncId, &cranelift_codegen::CompiledCode)],
    ) -> Vec<crate::codegen::CodeSites> {
        defined
            .iter()
            .map(|(id, code)| {
                let start = module.get_finalized_function(*id) as usize;
                let ranges = code
                    .buffer
                    .get_srclocs_sorted()
                    .iter()
                    .map(|l| (l.start, l.end, l.loc.bits()))
                    .collect();
                crate::codegen::CodeSites {
                    start,
                    end: start + code.code_info().total_size as usize,
                    func_id: jit_func_id(),
                    sites: crate::codegen::SiteTable::Ranges(ranges),
                }
            })
            .collect()
    }

    /// Iterations an optimised body's outermost loop runs between
    /// debits of its cell.
    const LOOP_TICK: i64 = 256;

    /// Count an iteration in `counter`, a register the body carries
    /// around the loop; every `LOOP_TICK` of them the cell's countdown
    /// is debited by as many and `wren_tier_tick` called when it runs
    /// out.
    #[allow(clippy::type_complexity)] // the runtime-fn resolver closure type is shared verbatim
    fn emit_loop_tick(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        hook: &TierHook,
        counter: cranelift_frontend::Variable,
    ) -> Result<(), String> {
        // A body entered at a loop header starts the counter at zero.
        let c = builder.use_var(counter);
        let c1 = builder.ins().iadd_imm_s(c, 1);
        builder.def_var(counter, c1);
        let debit = builder.ins().icmp_imm_s(IntCC::Equal, c1, LOOP_TICK);
        let debit_block = builder.create_block();
        let tick_block = builder.create_block();
        let cont_block = builder.create_block();
        builder.set_cold_block(debit_block);
        builder.set_cold_block(tick_block);
        builder.ins().brif(debit, debit_block, &[], cont_block, &[]);
        builder.switch_to_block(debit_block);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(counter, zero);
        let cell = builder.ins().iconst(types::I64, hook.cell as i64);
        let left = builder
            .ins()
            .uload32(MemFlags::trusted(), cell, TIER_CELL_COUNTDOWN);
        let left1 = builder.ins().iadd_imm_s(left, -LOOP_TICK);
        builder
            .ins()
            .istore32(MemFlags::trusted(), left1, cell, TIER_CELL_COUNTDOWN);
        // The countdown is unsigned; running out wraps it past the
        // interval it started from.
        let out = builder
            .ins()
            .icmp_imm_u(IntCC::UnsignedLessThan, left, LOOP_TICK);
        builder.ins().brif(out, tick_block, &[], cont_block, &[]);
        builder.switch_to_block(tick_block);
        let fid = builder.ins().iconst(types::I64, hook.func_id as i64);
        let f = get_runtime_fn(module, builder, "wren_tier_tick", 1)?;
        builder.ins().call(f, &[fid]);
        builder.ins().jump(cont_block, &[]);
        builder.switch_to_block(cont_block);
        Ok(())
    }

    pub(crate) fn jit_modvars_cell() -> usize {
        JIT_MODVARS_CELL.with(|c| c.get())
    }

    /// The frame pair cell baked into this thread's next compile.
    #[cfg(feature = "llvm")]
    pub(crate) fn jit_cur_cell() -> usize {
        JIT_CUR_CELL.with(|c| c.get())
    }

    /// A deopt helper's function id argument.
    fn plain_fid(func_id: u32) -> i64 {
        func_id as i64
    }

    /// Whether slot `idx` of the compiling function's module is within
    /// the module's variable vector. The vector is sized to the module's
    /// declarations at load and never shrinks, so a slot inside it now
    /// stays inside it, and the access needs no bounds check.
    pub(crate) fn jit_modvar_in_range(idx: u16) -> bool {
        let cell = jit_modvars_cell();
        if cell == 0 {
            return false;
        }
        // SAFETY: the cell is leaked per module and only ever read here.
        let cell = unsafe { &*(cell as *const crate::runtime::engine::ModuleVarsCell) };
        (idx as usize) < cell.len.load(std::sync::atomic::Ordering::Acquire)
    }

    #[inline]
    pub(crate) fn direct_calls_enabled() -> bool {
        crate::codegen::direct_calls_enabled()
    }

    /// Address space reserved per compiled body for its code and data;
    /// only the pages used are ever committed.
    #[cfg(not(windows))]
    const JIT_ARENA_BYTES: usize = 4 << 20;

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
        /// The code ranges and call sites of the functions defined, for
        /// traces.
        pub sites: Vec<crate::codegen::CodeSites>,
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

        flag_builder
            .set("preserve_frame_pointers", "true")
            .map_err(|e| format!("Failed to set preserve_frame_pointers: {}", e))?;

        // Disable probestack — macOS aarch64 inline probestack can cause
        // false SIGSEGV (interpreted as stack overflow by the Rust runtime).
        flag_builder
            .set("enable_probestack", "false")
            .map_err(|e| format!("Failed to set enable_probestack: {}", e))?;

        // The verifier is a third of a compile; a release build runs it
        // only under `WLIFT_CL_VERIFY` (safe to run with).
        let verify = cfg!(debug_assertions) || std::env::var_os("WLIFT_CL_VERIFY").is_some();
        flag_builder
            .set("enable_verifier", if verify { "true" } else { "false" })
            .map_err(|e| e.to_string())?;
        let isa = cranelift_native::builder()
            .map_err(|e| e.to_string())?
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        // 2. Create JIT module with runtime symbol resolution
        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        // One reservation holds the body's code and its data (the loop
        // entry request word, f64 constants): the code reaches them
        // pc-relative, which has a 2 GB reach, so they must not be
        // mapped on the far side of whatever the host reserved. Not on
        // Windows, where a reservation is committed up front.
        #[cfg(not(windows))]
        {
            let arena = cranelift_jit::ArenaMemoryProvider::new_with_size(JIT_ARENA_BYTES)
                .map_err(|e| e.to_string())?;
            jit_builder.memory_provider(Box::new(arena));
        }

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
        // A mid-body guard needs the boxed register file the inner
        // f64 body does not carry.
        let has_mid_body_guards = mir.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|(_, inst)| matches!(inst, Instruction::GuardNumAt { .. }))
        });
        let has_self_calls = mir.blocks.iter().any(|b| {
            b.instructions
                .iter()
                .any(|(_, inst)| matches!(inst, Instruction::CallStaticSelf { .. }))
        });
        // The inner body carries its parameters raw, which only its
        // own recursion knows how to pass; any other call would need
        // them boxed.
        let has_other_calls = mir.blocks.iter().any(|b| {
            b.instructions.iter().any(|(_, inst)| {
                matches!(
                    inst,
                    Instruction::Call { .. }
                        | Instruction::CallKnownFunc { .. }
                        | Instruction::SuperCall { .. }
                        | Instruction::SubscriptGet { .. }
                        | Instruction::SubscriptSet { .. }
                        | Instruction::MakeClosure { .. }
                        | Instruction::MakeList(..)
                        | Instruction::MakeMap(..)
                        | Instruction::MakeRange { .. }
                        | Instruction::StringConcat(..)
                        | Instruction::ToString(..)
                )
            })
        });
        // The inner body carries every Num raw, so a block parameter
        // typed Value must only ever receive one: a parameter, a box, a
        // constant, a recursive result, or such a parameter again.
        let use_f64_inner = has_num_guards
            && has_self_calls
            && param_count > 0
            && !has_mid_body_guards
            && !has_other_calls
            && value_params_carry_nums(mir);

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
                builder.finalize(module.target_config());
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
                // The inner body assumes its parameters are numbers; the
                // wrapper checks the guarded ones and hands anything else
                // to the interpreter.
                let guarded: Vec<usize> =
                    mir.blocks[0]
                        .instructions
                        .iter()
                        .filter_map(|(_, inst)| match inst {
                            Instruction::GuardNum(src) => mir.blocks[0]
                                .instructions
                                .iter()
                                .find_map(|(v, i)| match i {
                                    Instruction::BlockParam(idx) if v == src => Some(*idx as usize),
                                    _ => None,
                                }),
                            _ => None,
                        })
                        .collect();
                if !guarded.is_empty() {
                    let mut runtime_cache: HashMap<String, cranelift_codegen::ir::FuncRef> =
                        HashMap::new();
                    let mut get_runtime_fn =
                        |module: &mut dyn Module,
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
                    let qnan = builder.ins().iconst(types::I64, QNAN as i64);
                    let mut fails: Option<Value> = None;
                    for &idx in &guarded {
                        let Some(&p) = entry_params.get(idx) else {
                            continue;
                        };
                        let masked = builder.ins().band(p, qnan);
                        let is_box = builder.ins().icmp(IntCC::Equal, masked, qnan);
                        fails = Some(match fails {
                            Some(f) => builder.ins().bor(f, is_box),
                            None => is_box,
                        });
                    }
                    if let Some(fails) = fails {
                        emit_guard_deopt(
                            &mut builder,
                            &mut module,
                            &mut get_runtime_fn,
                            fails,
                            jit_func_id(),
                        )?;
                    }
                }
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
                builder.finalize(module.target_config());
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
            let sites = code_sites(
                &module,
                &[
                    (func_id, compiled_code),
                    (inner_id, inner_ctx.compiled_code().unwrap()),
                ],
            );
            return Ok(CraneliftCompiledCode {
                _module: module,
                fn_ptr,
                osr_entries: Vec::new(),
                code_size,
                sites,
            });
        }

        // Standard path (no f64 specialization): one body carrying the
        // entries for its loops.
        let layouts: Vec<OsrEntryLayout> = if should_compile_osr_entries(mir, interner) {
            collect_osr_targets(mir)
                .into_iter()
                .filter_map(|target| {
                    let layout = osr_entry_layout(mir, target);
                    if layout.is_none() && std::env::var_os("WLIFT_OSR_TRACE").is_some() {
                        eprintln!(
                            "osr-trace: skip {} bb{} unsupported live-in layout",
                            safe_name, target.0
                        );
                    }
                    layout
                })
                .collect()
        } else {
            Vec::new()
        };
        let entries = if layouts.is_empty() {
            None
        } else {
            let request = module
                .declare_data(&format!("{safe_name}_osr"), Linkage::Local, true, false)
                .map_err(|e| e.to_string())?;
            let mut desc = cranelift_module::DataDescription::new();
            desc.define_zeroinit(8);
            desc.set_align(8);
            module
                .define_data(request, &desc)
                .map_err(|e| e.to_string())?;
            Some(OsrEntries { layouts, request })
        };
        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig.clone(),
        );
        {
            let mut fb_ctx = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);

            lower_mir_impl(
                mir,
                interner,
                &mut builder,
                &mut module,
                callsite_ic_ptrs,
                callsite_ic_live_ptrs,
                jit_code_base,
                None,
                entries.as_ref(),
                None,
                inline_bodies.clone(),
                cha_by_method.clone(),
            )?;

            builder.seal_all_blocks();
            builder.finalize(module.target_config());
        }

        // Dump Cranelift IR if requested
        if std::env::var_os("WLIFT_CL_IR").is_some() {
            eprintln!("=== Cranelift IR for {} ===", safe_name);
            eprintln!("{}", func.display());
            eprintln!("=== end ===");
        }

        // 6. Compile
        if std::env::var_os("WLIFT_CL_VERIFY").is_some()
            && let Err(errs) = cranelift_codegen::verify_function(&func, module.isa())
        {
            eprintln!(
                "cl-verify: {} (FuncId u0:{}) failed:\n{}\nIR:\n{}",
                safe_name,
                func_id.as_u32(),
                errs,
                func.display()
            );
        }
        let mut ctx = Context::for_function(func);
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        let osr_defs = match &entries {
            Some(entries) => {
                define_osr_stubs(mir, &mut module, &safe_name, &sig, func_id, entries)?
            }
            None => Vec::new(),
        };
        module.finalize_definitions().map_err(|e| e.to_string())?;

        let fn_ptr = module.get_finalized_function(func_id);
        let compiled_code = ctx.compiled_code().unwrap();
        let code_size = compiled_code.code_info().total_size as usize;
        let sites = code_sites(&module, &[(func_id, compiled_code)]);
        let osr_entries = osr_defs
            .into_iter()
            .map(|def| crate::codegen::NativeOsrEntry {
                target_block: def.target_block,
                param_count: def.param_count,
                ptr: module.get_finalized_function(def.func_id),
                live_in_regs: def.live_in_regs,
                live_in_num: def.live_in_num,
                live_in_field: def.live_in_field,
                live_in_int: def.live_in_int,
                live_in_modvar: def.live_in_modvar,
            })
            .collect();

        Ok(CraneliftCompiledCode {
            _module: module,
            fn_ptr,
            osr_entries,
            code_size,
            sites,
        })
    }

    struct PendingOsrDefinition {
        target_block: BlockId,
        param_count: u16,
        func_id: cranelift_module::FuncId,
        live_in_regs: Vec<u32>,
        live_in_num: Vec<bool>,
        live_in_field: Vec<Option<u16>>,
        live_in_int: Vec<bool>,
        live_in_modvar: Vec<Option<u16>>,
    }

    #[derive(Clone)]
    pub(crate) struct OsrEntryLayout {
        pub(crate) target_block: BlockId,
        pub(crate) external_args: Vec<ValueId>,
        pub(crate) param_count: u16,
    }

    pub(crate) fn should_compile_osr_entries(mir: &MirFunction, interner: &Interner) -> bool {
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

    /// One stub per loop entry: it posts the frame for the calling
    /// thread and calls the body, which takes it at its entry.
    fn define_osr_stubs(
        mir: &MirFunction,
        module: &mut dyn Module,
        safe_name: &str,
        body_sig: &cranelift_codegen::ir::Signature,
        body: cranelift_module::FuncId,
        entries: &OsrEntries,
    ) -> Result<Vec<PendingOsrDefinition>, String> {
        let i64_params: HashSet<ValueId> = mir
            .blocks
            .iter()
            .flat_map(|b| b.params.iter())
            .filter(|(_, t)| *t == MirType::I64)
            .map(|(p, _)| *p)
            .collect();
        let mut defs = Vec::with_capacity(entries.layouts.len());
        for (i, layout) in entries.layouts.iter().enumerate() {
            let target_block = layout.target_block;
            if std::env::var_os("WLIFT_OSR_TRACE").is_some() {
                eprintln!(
                    "osr-trace: layout {} bb{} externals={:?}",
                    safe_name, target_block.0, layout.external_args
                );
            }
            // One parameter: pointer to the live-in Values in block order
            // (externals first, then the target block's params).
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(types::I64));
            sig.returns.push(AbiParam::new(types::I64));
            let osr_name = format!("{}_osr_bb{}", safe_name, target_block.0);
            let func_id = module
                .declare_function(&osr_name, Linkage::Local, &sig)
                .map_err(|e| e.to_string())?;
            let mut func = Function::with_name_signature(
                cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
                sig,
            );
            let mut fb_ctx = FunctionBuilderContext::new();
            {
                let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                let frame = builder.block_params(entry)[0];
                let kind = builder.ins().iconst(types::I64, i as i64 + 1);
                // The entry index goes in the word the caller keeps
                // before the live-ins; the frame is posted for this
                // thread alone, and the body's pending count tells its
                // entry to look.
                builder
                    .ins()
                    .store(MemFlags::trusted(), kind, frame, -VALUE_SIZE);
                let gv = module.declare_data_in_func(entries.request, builder.func);
                let request = builder.ins().symbol_value(types::I64, gv);
                let post = declare_runtime_fn(module, &mut builder, "wren_osr_post", 2)?;
                builder.ins().call(post, &[request, frame]);
                let zero = builder.ins().iconst(types::I64, 0);
                let args: Vec<Value> = body_sig.params.iter().map(|_| zero).collect();
                let body_ref = module.declare_func_in_func(body, builder.func);
                let call = builder.ins().call(body_ref, &args);
                let result = builder.inst_results(call)[0];
                builder.ins().return_(&[result]);
                builder.seal_all_blocks();
                builder.finalize(module.target_config());
            }
            let mut ctx = Context::for_function(func);
            module
                .define_function(func_id, &mut ctx)
                .map_err(|e| e.to_string())?;
            let live: Vec<ValueId> = layout
                .external_args
                .iter()
                .copied()
                .chain(
                    mir.blocks[target_block.0 as usize]
                        .params
                        .iter()
                        .map(|(p, _)| *p),
                )
                .collect();
            defs.push(PendingOsrDefinition {
                target_block,
                param_count: layout.param_count,
                func_id,
                // A live-in that scalar replacement split out of an
                // object parameter is read from that object's field.
                live_in_regs: live
                    .iter()
                    .map(|v| {
                        mir.scalar_param_sources
                            .get(v)
                            .map(|(o, _)| o.0)
                            .unwrap_or(v.0)
                    })
                    .collect(),
                live_in_num: live
                    .iter()
                    .map(|v| mir.speculated_num_params.contains(v))
                    .collect(),
                live_in_field: live
                    .iter()
                    .map(|v| mir.scalar_param_sources.get(v).map(|(_, f)| *f))
                    .collect(),
                live_in_int: live.iter().map(|v| i64_params.contains(v)).collect(),
                live_in_modvar: live
                    .iter()
                    .map(|v| mir.promoted_modvar_params.get(v).copied())
                    .collect(),
            });
        }
        Ok(defs)
    }

    /// Loop headers of the function: targets of edges whose source they
    /// dominate. Block ids are no guide once passes append blocks out of
    /// order, so this uses dominators.
    pub fn collect_osr_targets(mir: &MirFunction) -> Vec<BlockId> {
        use crate::mir::opt::licm::{compute_dominators, compute_rpo};
        let mut with_preds = mir.clone();
        with_preds.compute_predecessors();
        let rpo = compute_rpo(&with_preds);
        let idom = compute_dominators(&with_preds, &rpo);
        let dominates = |a: usize, b: usize| {
            let mut cur = b;
            loop {
                if cur == a {
                    return true;
                }
                let next = idom[cur];
                if next == usize::MAX || next == cur {
                    return false;
                }
                cur = next;
            }
        };
        let mut seen = HashSet::new();
        let mut targets = Vec::new();
        for block in &mir.blocks {
            if idom[block.id.0 as usize] == usize::MAX && block.id.0 != 0 {
                continue;
            }
            for target in block.terminator.successors() {
                if dominates(target.0 as usize, block.id.0 as usize)
                    && !mir.osr_excluded.contains(&target)
                    && seen.insert(target)
                {
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

    pub(crate) fn osr_entry_layout(mir: &MirFunction, target: BlockId) -> Option<OsrEntryLayout> {
        let target_idx = target.0 as usize;
        let target_block = mir.blocks.get(target_idx)?;
        // Live-ins arrive boxed; the entry unboxes a parameter carried
        // as f64. Any other f64 live-in has no interpreter register.
        if target_block
            .params
            .iter()
            .any(|(_, ty)| !matches!(ty, MirType::Value | MirType::F64 | MirType::I64))
        {
            return None;
        }
        let f64_params: HashSet<ValueId> = mir
            .blocks
            .iter()
            .flat_map(|b| b.params.iter())
            .filter(|(_, t)| matches!(t, MirType::F64 | MirType::I64))
            .map(|(p, _)| *p)
            .collect();

        let value_types = infer_osr_value_types(mir);
        let external_args = osr_external_live_values(mir, target);
        if external_args.iter().any(|vid| {
            !matches!(
                value_types.get(vid.0 as usize).copied(),
                Some(MirType::Value)
            ) && !f64_params.contains(vid)
        }) {
            return None;
        }

        // Live-ins arrive through one pointer to an array of Values, so
        // any count is fine as long as it fits the descriptor.
        let param_count = external_args.len() + target_block.params.len();
        if param_count > u16::MAX as usize {
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

    pub(crate) fn infer_osr_value_types(mir: &MirFunction) -> Vec<MirType> {
        crate::mir::infer_value_types(mir)
    }

    /// A module variable write: through the module's data in an AOT
    /// body, the module's cell in a JIT body. The value is the result.
    #[allow(clippy::type_complexity)] // the runtime-fn resolver closure type is shared verbatim
    fn emit_set_module_var(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        idx: u16,
        val: Value,
        aot_config: Option<&AotLoweringConfig>,
    ) -> Result<Value, String> {
        if let Some(cfg) = aot_config {
            let gv = module.declare_data_in_func(cfg.modvars_data, builder.func);
            let base = builder.ins().symbol_value(types::I64, gv);
            let store_val = val;
            builder
                .ins()
                .store(MemFlags::trusted(), store_val, base, (idx as i32) * 8);
            // SetModuleVar's MIR contract: result is the
            // stored value (mirrors the helper's return).
            Ok(store_val)
        } else if jit_modvar_in_range(idx) {
            let store_val = val;
            let cell = builder.ins().iconst(types::I64, jit_modvars_cell() as i64);
            let base = builder.ins().load(types::I64, MemFlags::trusted(), cell, 0);
            builder
                .ins()
                .store(MemFlags::trusted(), store_val, base, (idx as i32) * 8);
            Ok(store_val)
        } else if jit_modvars_cell() != 0 {
            // In-range stores go straight to the vector; an
            // index past the current length takes the helper,
            // which owns growth.
            let store_val = val;
            let cell = builder.ins().iconst(types::I64, jit_modvars_cell() as i64);
            let base = builder.ins().load(types::I64, MemFlags::trusted(), cell, 0);
            let len = builder.ins().load(types::I64, MemFlags::trusted(), cell, 8);
            let idx_val = builder.ins().iconst(types::I64, idx as i64);
            let in_range = builder.ins().icmp(IntCC::UnsignedLessThan, idx_val, len);
            let hit = builder.create_block();
            let miss = builder.create_block();
            let merge = builder.create_block();
            builder.ins().brif(in_range, hit, &[], miss, &[]);
            builder.switch_to_block(hit);
            builder.seal_block(hit);
            builder
                .ins()
                .store(MemFlags::trusted(), store_val, base, (idx as i32) * 8);
            builder.ins().jump(merge, &[]);
            builder.switch_to_block(miss);
            builder.seal_block(miss);
            let f = get_runtime_fn(module, builder, "wren_set_module_var", 2)?;
            builder.ins().call(f, &[idx_val, store_val]);
            builder.ins().jump(merge, &[]);
            builder.switch_to_block(merge);
            builder.seal_block(merge);
            Ok(store_val)
        } else {
            let f = get_runtime_fn(module, builder, "wren_set_module_var", 2)?;
            let idx_val = builder.ins().iconst(types::I64, idx as i64);
            let result = builder.ins().call(f, &[idx_val, val]);
            Ok(builder.inst_results(result)[0])
        }
    }

    /// A module variable read: through the module's data in an AOT
    /// body, the module's cell in a JIT body.
    #[allow(clippy::type_complexity)] // the runtime-fn resolver closure type is shared verbatim
    fn emit_get_module_var(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        idx: u16,
        aot_config: Option<&AotLoweringConfig>,
    ) -> Result<Value, String> {
        if let Some(cfg) = aot_config {
            let gv = module.declare_data_in_func(cfg.modvars_data, builder.func);
            let base = builder.ins().symbol_value(types::I64, gv);
            let result =
                builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), base, (idx as i32) * 8);
            Ok(result)
        } else if jit_modvar_in_range(idx) {
            let cell = builder.ins().iconst(types::I64, jit_modvars_cell() as i64);
            let base = builder.ins().load(types::I64, MemFlags::trusted(), cell, 0);
            Ok(builder
                .ins()
                .load(types::I64, MemFlags::trusted(), base, (idx as i32) * 8))
        } else if jit_modvars_cell() != 0 {
            // Three loads through the module's stable cell; an
            // index past the current length reads null, as the
            // helper does.
            let cell = builder.ins().iconst(types::I64, jit_modvars_cell() as i64);
            let base = builder.ins().load(types::I64, MemFlags::trusted(), cell, 0);
            let len = builder.ins().load(types::I64, MemFlags::trusted(), cell, 8);
            let idx_val = builder.ins().iconst(types::I64, idx as i64);
            let in_range = builder.ins().icmp(IntCC::UnsignedLessThan, idx_val, len);
            let hit = builder.create_block();
            let miss = builder.create_block();
            let merge = builder.create_block();
            builder.append_block_param(merge, types::I64);
            builder.ins().brif(in_range, hit, &[], miss, &[]);
            builder.switch_to_block(hit);
            builder.seal_block(hit);
            let v = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), base, (idx as i32) * 8);
            builder.ins().jump(merge, &[BlockArg::Value(v)]);
            builder.switch_to_block(miss);
            builder.seal_block(miss);
            let null = builder.ins().iconst(types::I64, TAG_NULL as i64);
            builder.ins().jump(merge, &[BlockArg::Value(null)]);
            builder.switch_to_block(merge);
            builder.seal_block(merge);
            Ok(builder.block_params(merge)[0])
        } else {
            let f = get_runtime_fn(module, builder, "wren_get_module_var", 1)?;
            let idx_val = builder.ins().iconst(types::I64, idx as i64);
            let result = builder.ins().call(f, &[idx_val]);
            Ok(builder.inst_results(result)[0])
        }
    }

    /// The values a loop entry rebuilds rather than loads: constants,
    /// and module variables read just before the loop.
    #[allow(clippy::type_complexity)] // the runtime-fn resolver closure type is shared verbatim
    fn emit_osr_external_constants(
        mir: &MirFunction,
        target: BlockId,
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        aot_config: Option<&AotLoweringConfig>,
    ) -> Result<Vec<(ValueId, Value)>, String> {
        let mut out = Vec::new();
        for (vid, inst) in osr_rematerializable_defs(mir, target) {
            let value = match inst {
                Instruction::GetModuleVar(idx) => {
                    emit_get_module_var(builder, module, get_runtime_fn, idx, aot_config)?
                }
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
            out.push((vid, value));
        }
        Ok(out)
    }

    /// The loop entries a body carries: one layout per header, and the
    /// body's pending count, raised by a stub before it calls the body.
    pub(crate) struct OsrEntries {
        pub layouts: Vec<OsrEntryLayout>,
        pub request: cranelift_module::DataId,
    }

    /// Every runtime function name with its address, for the JIT
    /// module's symbol table.
    fn runtime_symbols() -> Vec<(&'static str, usize)> {
        crate::codegen::runtime_fns::RUNTIME_FN_NAMES
            .iter()
            .filter_map(|name| crate::codegen::runtime_fns::resolve(name).map(|a| (*name, a)))
            .collect()
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
        /// val)` with a `symbol_value` + load/store at offset
        /// `slot * 8`, killing both helper calls and the TLS read.
        pub modvars_data: cranelift_module::DataId,

        /// `DataId` of the runtime's safepoint page
        /// (`wlift_safepoint_page`), imported: loop headers load
        /// from it.
        pub safepoint_data: cranelift_module::DataId,

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
        entries: Option<&OsrEntries>,
        aot_config: Option<&AotLoweringConfig>,
        inline_bodies: Option<
            std::sync::Arc<std::collections::HashMap<u32, std::sync::Arc<MirFunction>>>,
        >,
        cha_by_method: crate::runtime::engine::SharedCha,
    ) -> Result<(), String> {
        // A splice that failed to lower may have left its receiver here.
        INLINE_CLASS.set(None);
        // A JIT body polls for a pending error after each helper that
        // can raise; AOT polls at block entries.
        ERROR_POLL.set(aot_config.is_none());
        FRAME_SITE.set(0);
        // Map MIR blocks to Cranelift blocks
        let mut block_map: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
        for (i, _) in mir.blocks.iter().enumerate() {
            let cl_block = builder.create_block();
            block_map.insert(BlockId(i as u32), cl_block);
        }

        // Map MIR values to Cranelift values.
        let mut val_map: HashMap<ValueId, Value> = HashMap::new();

        // A value the loops' entries load from the interpreter's frame,
        // or rebuild from a constant, is a Cranelift variable: its own
        // definition and each entry's define it, and the frontend
        // merges them where the paths meet.
        let mut osr_vars: HashMap<ValueId, cranelift_frontend::Variable> = HashMap::new();
        // Block parameters carried as raw f64.
        let f64_params: std::collections::HashSet<ValueId> = mir
            .blocks
            .iter()
            .flat_map(|b| b.params.iter())
            .filter(|(_, t)| *t == MirType::F64)
            .map(|(p, _)| *p)
            .collect();
        let i64_params: std::collections::HashSet<ValueId> = mir
            .blocks
            .iter()
            .flat_map(|b| b.params.iter())
            .filter(|(_, t)| *t == MirType::I64)
            .map(|(p, _)| *p)
            .collect();
        let live_in = if entries.is_some() {
            Some(crate::mir::live_in_sets(mir))
        } else {
            None
        };
        if let Some(entries) = entries {
            for layout in &entries.layouts {
                for vid in &layout.external_args {
                    if osr_vars.contains_key(vid) {
                        continue;
                    }
                    let ty = if f64_params.contains(vid) {
                        types::F64
                    } else {
                        types::I64
                    };
                    let var = builder.declare_var(ty);
                    osr_vars.insert(*vid, var);
                }
            }
        }
        // Constants a loop entry would otherwise rebuild are defined
        // once at the function's entry, ahead of every path.
        let mut pre_defined: std::collections::HashSet<ValueId> = std::collections::HashSet::new();

        // Cold loop headers get an iteration counter and a buffer for
        // their live-ins; only OSR-entered code can hand a loop back to
        // the interpreter, because only then does a frame exist to resume.
        // Counters for the loops the body leaves once they prove hot,
        // one per `ColdLoopExit`, zeroed at every entry.
        let cold_counters: HashMap<BlockId, cranelift_codegen::ir::StackSlot> = mir
            .blocks
            .iter()
            .filter(|b| {
                b.instructions
                    .iter()
                    .any(|(_, i)| matches!(i, Instruction::ColdLoopExit { .. }))
            })
            .map(|b| {
                let counter =
                    builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        8,
                        3,
                    ));
                (b.id, counter)
            })
            .collect();
        // Re-tier polls at outermost loop headers, JIT bodies only.
        let tier_hook = if aot_config.is_none() && f64_self_id.is_none() {
            jit_tier_hook()
        } else {
            None
        };
        // An optimised body's loop-iteration counter, a register.
        let loop_tick_var = builder.declare_var(types::I64);
        let mut retier_polls: HashMap<BlockId, (cranelift_codegen::ir::StackSlot, Vec<ValueId>)> =
            HashMap::new();
        if let Some(hook) = &tier_hook {
            for header in &hook.retier_headers {
                if header.0 as usize >= mir.blocks.len() {
                    continue;
                }
                // The body's own live-ins at the header; the top tier's
                // entry names the same registers when it compiled the
                // same shape, and the transfer declines otherwise.
                let live: Vec<ValueId> = osr_external_live_values(mir, *header)
                    .into_iter()
                    .chain(mir.blocks[header.0 as usize].params.iter().map(|(p, _)| *p))
                    .collect();
                if live
                    .iter()
                    .any(|v| mir.scalar_param_sources.contains_key(v))
                {
                    continue;
                }
                let buf =
                    builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        (live.len().max(1) * 16) as u32,
                        3,
                    ));
                retier_polls.insert(*header, (buf, live));
            }
        }
        // Also the table every guard's snapshot for the interpreter
        // boxes by, so a body with guards has it whether or not it polls.
        let has_deopts = mir.blocks.iter().any(|b| {
            b.instructions.iter().any(|(_, inst)| {
                matches!(
                    inst,
                    Instruction::GuardNumAt { .. }
                        | Instruction::GuardClassAt { .. }
                        | Instruction::SlowPathExit { .. }
                )
            })
        });
        let exit_value_types = if cold_counters.is_empty() && retier_polls.is_empty() && !has_deopts
        {
            Vec::new()
        } else {
            infer_osr_value_types(mir)
        };

        // The function's entry: its arguments, then a jump to the body
        // or, when a loop entry stub posted a frame for this thread, to
        // that loop's header with the interpreter's frame loaded.
        let mut entry_params: Option<Vec<Value>> = None;
        if let Some(entries) = entries {
            let dispatch = builder.create_block();
            builder.switch_to_block(dispatch);
            for _ in 0..mir.arity {
                builder.append_block_param(dispatch, types::I64);
            }
            let params = builder.block_params(dispatch).to_vec();
            for counter in cold_counters.values() {
                let zero = builder.ins().iconst(types::I64, 0);
                builder.ins().stack_store(types::I64, zero, *counter, 0);
            }
            for layout in &entries.layouts {
                for (vid, v) in emit_osr_external_constants(
                    mir,
                    layout.target_block,
                    builder,
                    module,
                    &mut |m, b, name, n| declare_runtime_fn(m, b, name, n),
                    aot_config,
                )? {
                    if pre_defined.insert(vid) {
                        val_map.insert(vid, v);
                    }
                }
            }
            // The pending count is the only word threads share; a
            // thread that finds it set but has no frame of its own
            // takes the normal entry.
            let gv = module.declare_data_in_func(entries.request, builder.func);
            let request = builder.ins().symbol_value(types::I64, gv);
            let pending = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), request, 0);
            let look = builder.create_block();
            let osr = builder.create_block();
            builder.set_cold_block(look);
            builder.set_cold_block(osr);
            builder
                .ins()
                .brif(pending, look, &[], block_map[&mir.entry_block()], &[]);
            builder.switch_to_block(look);
            let take = declare_runtime_fn(module, builder, "wren_osr_take", 1)?;
            let call = builder.ins().call(take, &[request]);
            let args_ptr = builder.inst_results(call)[0];
            builder
                .ins()
                .brif(args_ptr, osr, &[], block_map[&mir.entry_block()], &[]);
            builder.switch_to_block(osr);
            let kind = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), args_ptr, -VALUE_SIZE);
            let prologues: Vec<cranelift_codegen::ir::Block> = entries
                .layouts
                .iter()
                .map(|_| builder.create_block())
                .collect();
            let index = builder.ins().iadd_imm_s(kind, -1);
            let index = builder.ins().ireduce(types::I32, index);
            let jt = cranelift_codegen::ir::JumpTableData::new(
                builder.func.dfg.block_call(prologues[0], &[]),
                &prologues
                    .iter()
                    .map(|b| builder.func.dfg.block_call(*b, &[]))
                    .collect::<Vec<_>>(),
            );
            let jt = builder.create_jump_table(jt);
            builder.ins().br_table(index, jt);
            for (layout, prologue) in entries.layouts.iter().zip(prologues) {
                builder.switch_to_block(prologue);
                let mut slot = 0i32;
                for vid in &layout.external_args {
                    let v = builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        args_ptr,
                        slot * VALUE_SIZE,
                    );
                    slot += 1;
                    let v = if f64_params.contains(vid) {
                        builder.ins().bitcast(types::F64, MemFlags::new(), v)
                    } else if i64_params.contains(vid) {
                        let f = builder.ins().bitcast(types::F64, MemFlags::new(), v);
                        builder.ins().fcvt_to_sint(types::I64, f)
                    } else {
                        v
                    };
                    builder.def_var(osr_vars[vid], v);
                }
                let target_block = &mir.blocks[layout.target_block.0 as usize];
                let mut args: Vec<BlockArg> = Vec::with_capacity(target_block.params.len());
                for (_, ty) in &target_block.params {
                    let v = builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        args_ptr,
                        slot * VALUE_SIZE,
                    );
                    slot += 1;
                    let v = match ty {
                        MirType::F64 => builder.ins().bitcast(types::F64, MemFlags::new(), v),
                        MirType::I64 => {
                            let f = builder.ins().bitcast(types::F64, MemFlags::new(), v);
                            builder.ins().fcvt_to_sint(types::I64, f)
                        }
                        _ => v,
                    };
                    args.push(BlockArg::Value(v));
                }
                builder.ins().jump(block_map[&layout.target_block], &args);
            }
            entry_params = Some(params);
        }

        // Receiver (entry_params[0]) saved for CallStaticSelf
        let mut receiver_val: Option<Value> = None;

        // Track which MIR values are raw Cranelift booleans (i8) rather than
        // NaN-boxed TAG_TRUE/TAG_FALSE. Used to skip the expensive truthiness
        // check in CondBranch when the condition is a direct fcmp/icmp result.
        let mut raw_bools: std::collections::HashSet<ValueId> = std::collections::HashSet::new();

        // Cache for declared runtime functions
        let mut runtime_cache: HashMap<String, cranelift_codegen::ir::FuncRef> = HashMap::new();

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
            let needs_closure_ptr = f64_self_id.is_none()
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
                // NaN-boxed Wren Value. Refresh logic for the cached
                // pointer across GC lives in the JitContext
                // save/restore in `wlift_aot_invoke_sm_method`
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
            if f64_self_id.is_none() {
                let snap_var = builder.declare_var(types::I64);
                *cfg.current_jit_roots_snapshot_var.borrow_mut() = Some(snap_var);
            }
        }

        // Process blocks in reverse post-order (dominance order).
        // The MIR block array may have preheader blocks (bb4) listed after
        // loop bodies (bb2), but Cranelift requires values to be defined
        // before use. RPO guarantees dominators come first.
        let rpo = compute_rpo(mir);
        // Determine reachability from bb0 over the post-transform MIR. The SM transform's
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
            let mut seen = std::collections::HashSet::new();
            let mut stack = vec![0usize];
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

        // Loop headers: any block H with a predecessor P where
        // `P.id >= H.id`. The MIR builder lowers `while` / `for-in`
        // / `continue` so the back-edge always jumps to a header
        // whose id is less-than-or-equal to the body's id; this
        // single CFG check identifies them without a dominator
        // pass.
        //
        // Used by the back-edge `wren_jit_roots_restore` emit
        // below: long-running functions (the canonical case is
        // `App.listen`'s accept loop in @hatch:web — never returns)
        // would otherwise pin one `JIT_ROOTS_STORE` entry per
        // allocation forever because `finish_alloc` pushes
        // unconditionally and the only existing release site is
        // the exit-time restore. Releasing back to the function-
        // entry snapshot at every loop header drops accumulated
        // entries each iteration; the conservative stack scan
        // covers anything still live across the back-edge.
        let loop_headers: std::collections::HashSet<BlockId> = {
            let mut headers = std::collections::HashSet::new();
            for block in &mir.blocks {
                for &pred in &block.predecessors {
                    if pred.0 >= block.id.0 {
                        headers.insert(block.id);
                        break;
                    }
                }
            }
            headers
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

            // Add block parameters (from loop back-edges / CondBranch args)
            for (vid, ty) in &block.params {
                let cl_type = match ty {
                    MirType::F64 => types::F64,
                    // Every Num is raw in the inner f64 body.
                    MirType::Value if f64_self_id.is_some() => types::F64,
                    _ => types::I64,
                };
                let param = builder.append_block_param(cl_block, cl_type);
                val_map.insert(*vid, param);
                if let Some(var) = osr_vars.get(vid) {
                    builder.def_var(*var, param);
                }
            }
            // Materialise the current value of every OSR variable at
            // block entry so plain lookups in this block see the value
            // flowing in along the edge that was taken.
            for (vid, var) in &osr_vars {
                if live_in
                    .as_ref()
                    .is_some_and(|l| !l.contains(block_idx, *vid))
                {
                    continue;
                }
                let v = builder.use_var(*var);
                val_map.insert(*vid, v);
            }

            if let Some((_, live)) = retier_polls.get(&bid)
                && std::env::var_os("WLIFT_OSR_TRACE").is_some()
            {
                let missing: Vec<ValueId> = live
                    .iter()
                    .copied()
                    .filter(|v| !val_map.contains_key(v))
                    .collect();
                if !missing.is_empty() {
                    eprintln!(
                        "osr-trace: retier poll skipped at bb{} live-ins undefined: {:?}",
                        bid.0, missing
                    );
                }
            }
            if let Some(hook) = tier_hook
                .as_ref()
                .filter(|h| h.cell != 0 && h.tick_headers.contains(&bid))
            {
                if hook.generation == 0 {
                    emit_tier_tick(builder, module, &mut get_runtime_fn, hook)?;
                } else {
                    emit_loop_tick(builder, module, &mut get_runtime_fn, hook, loop_tick_var)?;
                }
            }
            if let (Some(hook), Some((buf, live))) = (
                &tier_hook,
                retier_polls
                    .get(&bid)
                    .filter(|(_, live)| live.iter().all(|v| val_map.contains_key(v))),
            ) {
                if hook.generation == 0 && !hook.tick_headers.contains(&bid) {
                    emit_tier_tick(builder, module, &mut get_runtime_fn, hook)?;
                }
                let cell = builder.ins().iconst(types::I64, hook.cell as i64);
                let word = builder.ins().uload32(
                    MemFlags::trusted(),
                    cell,
                    retier_word_offset(hook.generation),
                );
                let exit_block = builder.create_block();
                let cont_block = builder.create_block();
                builder.set_cold_block(exit_block);
                builder.ins().brif(word, exit_block, &[], cont_block, &[]);
                builder.switch_to_block(exit_block);
                // A module variable this loop carries in a parameter
                // goes back to the module first; the body entered
                // reads it from there.
                for (p, _) in &mir.blocks[bid.0 as usize].params {
                    if let (Some(slot), Some(v)) =
                        (mir.promoted_modvar_params.get(p), val_map.get(p))
                    {
                        let boxed =
                            box_for_snapshot(builder, *v, *p, &raw_bools, &exit_value_types);
                        emit_set_module_var(
                            builder,
                            module,
                            &mut get_runtime_fn,
                            *slot,
                            boxed,
                            aot_config,
                        )?;
                    }
                }
                emit_live_snapshot(builder, live, &val_map, &raw_bools, &exit_value_types, *buf)?;
                let buf_ptr = builder.ins().stack_addr(types::I64, *buf, 0);
                let fid = builder.ins().iconst(types::I64, plain_fid(hook.func_id));
                let header_id = builder
                    .ins()
                    .iconst(types::I64, retier_header_word(bid, hook.generation));
                let n = builder.ins().iconst(types::I64, live.len() as i64);
                let retier_fn = get_runtime_fn(module, builder, "wren_retier", 4)?;
                emit_cur_frame(builder);
                let call = builder.ins().call(retier_fn, &[fid, header_id, buf_ptr, n]);
                let result = builder.inst_results(call)[0];
                let declined_bits = builder.ins().iconst(
                    types::I64,
                    crate::codegen::runtime_fns::RETIER_DECLINED as i64,
                );
                let declined = builder.ins().icmp(IntCC::Equal, result, declined_bits);
                let ret_block = builder.create_block();
                builder
                    .ins()
                    .brif(declined, cont_block, &[], ret_block, &[]);
                builder.switch_to_block(ret_block);
                builder.ins().return_(&[result]);
                builder.switch_to_block(cont_block);
            }

            // For the entry block (first in RPO = bb0), map BlockParam
            // instructions to Cranelift's function parameters.
            // Cranelift adds signature params to the first switched-to block.
            // For the entry block, add function params as block params
            // THEN map BlockParam instructions to those params.
            if block_idx == 0 {
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
                    // (includes receiver even if dead), unless the
                    // dispatch block already holds them.
                    let entry_params = match entry_params.take() {
                        Some(params) => params,
                        None => {
                            let arity = mir.arity as usize;
                            for _ in 0..arity {
                                builder.append_block_param(cl_block, types::I64);
                            }
                            builder.block_params(cl_block).to_vec()
                        }
                    };
                    if !entry_params.is_empty() {
                        receiver_val = Some(entry_params[0]);
                    }
                    // Map BlockParam(idx) → entry_params[idx]
                    for &(vid, ref inst) in &block.instructions {
                        if let Instruction::BlockParam(idx) = inst {
                            let idx = *idx as usize;
                            if idx < entry_params.len() {
                                val_map.insert(vid, entry_params[idx]);
                                // A parameter a loop entry also loads
                                // is read through its variable.
                                if let Some(var) = osr_vars.get(&vid) {
                                    builder.def_var(*var, entry_params[idx]);
                                }
                                // A promotable baseline body profiles
                                // its arguments the way it does call
                                // results; the receiver is never a Num.
                                if idx > 0
                                    && let Some(hook) = tier_hook.as_ref().filter(|h| {
                                        h.result_kinds != 0 && (vid.0 as usize) < h.result_kinds_len
                                    })
                                {
                                    emit_note_call_result(
                                        builder,
                                        hook.result_kinds + vid.0 as usize,
                                        entry_params[idx],
                                    );
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
                if let Some(hook) = tier_hook.as_ref().filter(|h| h.cell != 0) {
                    emit_tier_tick(builder, module, &mut get_runtime_fn, hook)?;
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.def_var(loop_tick_var, zero);
                }
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
            if let Some(cfg) = aot_config
                && block_idx != 0
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

            // Back-edge JIT-roots release. Emitted at the top of
            // every loop header (after block-params + snap_var def
            // + has_error check) so each iteration starts at the
            // function-entry snapshot length. Long-running
            // functions like `App.listen`'s accept loop would
            // otherwise pin one `JIT_ROOTS_STORE` entry per
            // allocation forever; releasing back to the entry
            // snapshot every iteration drops accumulated entries;
            // the conservative stack scan covers anything still
            // live across the back-edge.
            // A loop that allocates nothing and never ticks would
            // otherwise keep a collector on another thread waiting.
            if loop_headers.contains(&bid) {
                emit_safepoint_poll(builder, module, aot_config);
            }
            #[cfg(feature = "aot")]
            if loop_headers.contains(&bid)
                && let Some(cfg) = aot_config
                && let Some(snap_var) = *cfg.current_jit_roots_snapshot_var.borrow()
            {
                let snap = builder.use_var(snap_var);
                let f = get_runtime_fn(module, builder, "wren_jit_roots_restore", 1)?;
                let _ = builder.ins().call(f, &[snap]);
            }

            // Lower each instruction
            for &(vid, ref inst) in &block.instructions {
                if pre_defined.contains(&vid) {
                    continue;
                }
                // A loop compiled cold: its generic calls fill their
                // caches as it runs; the 256th iteration asks for the
                // function to be compiled again from them, and every
                // iteration polls for that body to transfer into.
                if let (Instruction::ColdLoopExit { header }, None) = (inst, aot_config) {
                    let live: Vec<ValueId> = osr_external_live_values(mir, *header)
                        .into_iter()
                        .chain(mir.blocks[header.0 as usize].params.iter().map(|(p, _)| *p))
                        .collect();
                    let (cell_addr, generation) = jit_retier_cell();
                    if let Some(counter) = cold_counters.get(header)
                        && cell_addr != 0
                        && live.iter().all(|v| val_map.contains_key(v))
                        && !live
                            .iter()
                            .any(|v| mir.scalar_param_sources.contains_key(v))
                    {
                        let c = builder
                            .ins()
                            .stack_load(types::I64, types::I64, *counter, 0);
                        let c1 = builder.ins().iadd_imm_s(c, 1);
                        builder.ins().stack_store(types::I64, c1, *counter, 0);
                        let hot = builder
                            .ins()
                            .icmp_imm_s(IntCC::Equal, c1, COLD_LOOP_EXIT_AFTER);
                        let ask_block = builder.create_block();
                        let poll_block = builder.create_block();
                        builder.set_cold_block(ask_block);
                        let fid = builder.ins().iconst(types::I64, jit_func_id() as i64);
                        builder.ins().brif(hot, ask_block, &[], poll_block, &[]);
                        builder.switch_to_block(ask_block);
                        let f = get_runtime_fn(module, builder, "wren_cold_loop_hot", 1)?;
                        builder.ins().call(f, &[fid]);
                        builder.ins().jump(poll_block, &[]);
                        builder.switch_to_block(poll_block);
                        let cell = builder.ins().iconst(types::I64, cell_addr as i64);
                        let word = builder.ins().uload32(
                            MemFlags::trusted(),
                            cell,
                            retier_word_offset(generation),
                        );
                        let exit_block = builder.create_block();
                        let cont_block = builder.create_block();
                        builder.set_cold_block(exit_block);
                        builder.ins().brif(word, exit_block, &[], cont_block, &[]);
                        builder.switch_to_block(exit_block);
                        let buf = builder.create_sized_stack_slot(
                            cranelift_codegen::ir::StackSlotData::new(
                                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                                (live.len().max(1) * 16) as u32,
                                3,
                            ),
                        );
                        emit_live_snapshot(
                            builder,
                            &live,
                            &val_map,
                            &raw_bools,
                            &exit_value_types,
                            buf,
                        )?;
                        let buf_ptr = builder.ins().stack_addr(types::I64, buf, 0);
                        let header_id = builder
                            .ins()
                            .iconst(types::I64, retier_header_word(*header, generation));
                        let n = builder.ins().iconst(types::I64, live.len() as i64);
                        let retier_fn = get_runtime_fn(module, builder, "wren_retier", 4)?;
                        let fid = builder.ins().iconst(types::I64, plain_fid(jit_func_id()));
                        emit_cur_frame(builder);
                        let call = builder.ins().call(retier_fn, &[fid, header_id, buf_ptr, n]);
                        let result = builder.inst_results(call)[0];
                        let declined_bits = builder.ins().iconst(
                            types::I64,
                            crate::codegen::runtime_fns::RETIER_DECLINED as i64,
                        );
                        let declined = builder.ins().icmp(IntCC::Equal, result, declined_bits);
                        let ret_block = builder.create_block();
                        builder
                            .ins()
                            .brif(declined, cont_block, &[], ret_block, &[]);
                        builder.switch_to_block(ret_block);
                        builder.ins().return_(&[result]);
                        builder.switch_to_block(cont_block);
                    } else if std::env::var_os("WLIFT_OSR_TRACE").is_some() {
                        eprintln!("osr-trace: cold loop poll skipped at bb{}", header.0);
                    }
                    continue;
                }
                // In a block that ends unreachable, an exit is the
                // block: the guard that led here has already failed.
                if let (true, Instruction::SlowPathExit { pc, live }, None) = (
                    matches!(block.terminator, Terminator::Unreachable),
                    inst,
                    aot_config,
                ) {
                    emit_deopt_at(
                        builder,
                        module,
                        &mut get_runtime_fn,
                        jit_func_id(),
                        *pc,
                        live,
                        &val_map,
                        &raw_bools,
                        &exit_value_types,
                    )?;
                    let dead = builder.create_block();
                    builder.switch_to_block(dead);
                    continue;
                }
                // Track raw booleans from f64 comparisons
                let is_raw_bool = matches!(
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
                );
                let site = mir
                    .span_map
                    .get(&vid)
                    .map(|sp| sp.start as u32 + 1)
                    .unwrap_or(0);
                FRAME_SITE.set(site);
                // The code emitted for the instruction carries its site,
                // so a return address into it names the site.
                builder.set_srcloc(cranelift_codegen::ir::SourceLoc::new(site));
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
                    mir.ic_sites.get(&vid).map(|i| *i as usize),
                    f64_self_id,
                    receiver_val,
                    aot_config,
                    inline_bodies.as_ref(),
                    cha_by_method.as_ref(),
                    Some((&raw_bools, &exit_value_types)),
                )?;
                if let Some(val) = result {
                    val_map.insert(vid, val);
                    // A promotable baseline body profiles what each
                    // call returns for the top tier to speculate on.
                    if let Some(ref hook) = tier_hook
                        && hook.result_kinds != 0
                        && (vid.0 as usize) < hook.result_kinds_len
                        && matches!(
                            inst,
                            Instruction::Call { .. }
                                | Instruction::CallKnownFunc { .. }
                                | Instruction::SuperCall { .. }
                        )
                    {
                        emit_note_call_result(builder, hook.result_kinds + vid.0 as usize, val);
                    }
                    if let Some(var) = osr_vars.get(&vid) {
                        builder.def_var(*var, val);
                    }
                    if is_raw_bool {
                        raw_bools.insert(vid);
                    }
                }
            }

            // Pre-terminator hook: AOT bodies emit a JIT-roots
            // restore right before every Return so any roots
            // leaked into JIT_ROOTS_STORE by alloc helpers
            // (finish_alloc's "push but don't pop" mode) get
            // released at the function boundary. The snapshot
            // Variable was defined at function entry above.
            if matches!(
                block.terminator,
                Terminator::Return(_) | Terminator::ReturnNull
            ) && let Some(cfg) = aot_config
                && let Some(snap_var) = *cfg.current_jit_roots_snapshot_var.borrow()
            {
                let snap = builder.use_var(snap_var);
                let f = get_runtime_fn(module, builder, "wren_jit_roots_restore", 1)?;
                let _ = builder.ins().call(f, &[snap]);
            }

            lower_terminator(&block.terminator, builder, &val_map, &block_map, &raw_bools)?;
        }

        // Fill in the shared abort-exit block (if any block needed
        // it). Restores the function-entry roots snapshot, then
        // returns a typed null so the AOT-stub fast path's
        // `has_error` route in `vm_interp::run_fiber` picks the
        // error up. Returning the function's declared type avoids
        // a Cranelift verifier mismatch on f64-inner emit paths
        // (which never set this block in the first place).
        if let Some(cfg) = aot_config
            && let Some(abort_exit) = *cfg.current_abort_exit_block.borrow()
        {
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

        Ok(())
    }

    /// Describes what the fast-path of an inline boxed binary operation does.
    /// Branch on `fails` to a cold block that re-executes the call in
    /// the interpreter through `wren_deopt_*` with the entry parameters
    /// and returns its result; lowering continues on the other edge.
    #[allow(clippy::type_complexity)] // the runtime-fn resolver closure type is shared verbatim
    fn emit_guard_deopt(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        fails: Value,
        func_id: u32,
    ) -> Result<(), String> {
        let entry = builder
            .func
            .layout
            .entry_block()
            .ok_or("guard outside a function body")?;
        let params: Vec<Value> = builder.block_params(entry).to_vec();
        let deopt_block = builder.create_block();
        let cont_block = builder.create_block();
        builder.set_cold_block(deopt_block);
        builder.ins().brif(fails, deopt_block, &[], cont_block, &[]);
        builder.switch_to_block(deopt_block);
        let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            (params.len().max(1) * 8) as u32,
            3,
        ));
        for (i, p) in params.iter().enumerate() {
            builder
                .ins()
                .stack_store(types::I64, *p, slot, (i * 8) as i32);
        }
        let buf = builder.ins().stack_addr(types::I64, slot, 0);
        let fid = builder.ins().iconst(types::I64, plain_fid(func_id));
        let n = builder.ins().iconst(types::I64, params.len() as i64);
        let f = get_runtime_fn(module, builder, "wren_deopt_n", 3)?;
        emit_cur_frame(builder);
        let call = builder.ins().call(f, &[fid, n, buf]);
        let result = builder.inst_results(call)[0];
        builder.ins().return_(&[result]);
        builder.switch_to_block(cont_block);
        Ok(())
    }

    /// Store `live` into `buf` as `(register, boxed value)` pairs, the
    /// layout `wren_retier` reads.
    fn emit_live_snapshot(
        builder: &mut FunctionBuilder,
        live: &[ValueId],
        val_map: &HashMap<ValueId, Value>,
        raw_bools: &HashSet<ValueId>,
        value_types: &[MirType],
        buf: cranelift_codegen::ir::StackSlot,
    ) -> Result<(), String> {
        for (i, vid) in live.iter().enumerate() {
            let Some(&v) = val_map.get(vid) else {
                return Err(format!("snapshot live-in {:?} undefined", vid));
            };
            let boxed = box_for_snapshot(builder, v, *vid, raw_bools, value_types);
            let reg = builder.ins().iconst(types::I64, vid.0 as i64);
            builder
                .ins()
                .stack_store(types::I64, reg, buf, (i * 16) as i32);
            builder
                .ins()
                .stack_store(types::I64, boxed, buf, (i * 16 + 8) as i32);
        }
        Ok(())
    }

    /// `v` as a NaN-boxed word whatever representation `vid` is
    /// carried in.
    fn box_for_snapshot(
        builder: &mut FunctionBuilder,
        v: Value,
        vid: ValueId,
        raw_bools: &HashSet<ValueId>,
        value_types: &[MirType],
    ) -> Value {
        match value_types.get(vid.0 as usize) {
            Some(MirType::F64) => builder.ins().bitcast(types::I64, MemFlags::new(), v),
            Some(MirType::I64) => {
                let f = builder.ins().fcvt_from_sint(types::F64, v);
                builder.ins().bitcast(types::I64, MemFlags::new(), f)
            }
            Some(MirType::Bool) if raw_bools.contains(&vid) => {
                let t = builder.ins().iconst(types::I64, TAG_TRUE as i64);
                let f = builder.ins().iconst(types::I64, TAG_FALSE as i64);
                builder.ins().select(v, t, f)
            }
            _ => v,
        }
    }

    /// A mid-body guard: when `fails`, store the `live` registers in
    /// the word layout `wren_deopt_at` reads and hand the function to
    /// it; it resumes the interpreter at `pc` and returns the
    /// function's result.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn emit_guard_deopt_at(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        fails: Value,
        func_id: u32,
        pc: u32,
        live: &[DeoptReg],
        val_map: &HashMap<ValueId, Value>,
        raw_bools: &HashSet<ValueId>,
        value_types: &[MirType],
    ) -> Result<(), String> {
        let deopt_block = builder.create_block();
        let cont_block = builder.create_block();
        builder.set_cold_block(deopt_block);
        builder.ins().brif(fails, deopt_block, &[], cont_block, &[]);
        builder.switch_to_block(deopt_block);
        emit_deopt_at(
            builder,
            module,
            get_runtime_fn,
            func_id,
            pc,
            live,
            val_map,
            raw_bools,
            value_types,
        )?;
        builder.switch_to_block(cont_block);
        Ok(())
    }

    /// Store the `live` registers in the word layout `wren_deopt_at`
    /// reads, in a fresh stack slot: its address and word count.
    fn emit_deopt_words(
        builder: &mut FunctionBuilder,
        live: &[DeoptReg],
        val_map: &HashMap<ValueId, Value>,
        raw_bools: &HashSet<ValueId>,
        value_types: &[MirType],
    ) -> Result<(Value, usize), String> {
        let words = live
            .iter()
            .map(crate::codegen::runtime_fns::deopt_words)
            .sum::<usize>();
        let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            (words.max(1) * 8) as u32,
            3,
        ));
        let mut at = 0i32;
        for r in live {
            let tag = builder
                .ins()
                .iconst(types::I64, crate::codegen::runtime_fns::deopt_tag(r) as i64);
            builder.ins().stack_store(types::I64, tag, slot, at * 8);
            at += 1;
            for c in crate::codegen::runtime_fns::deopt_consts(r) {
                let c = builder.ins().iconst(types::I64, c as i64);
                builder.ins().stack_store(types::I64, c, slot, at * 8);
                at += 1;
            }
            for vid in r.source.operands() {
                let Some(&v) = val_map.get(&vid) else {
                    return Err(format!("deopt live value {:?} undefined", vid));
                };
                let boxed = box_for_snapshot(builder, v, vid, raw_bools, value_types);
                builder.ins().stack_store(types::I64, boxed, slot, at * 8);
                at += 1;
            }
        }
        let buf = builder.ins().stack_addr(types::I64, slot, 0);
        Ok((buf, words))
    }

    /// Leave the function from the current block: store the `live`
    /// registers in the word layout `wren_deopt_at` reads, hand the
    /// function to it and return its result.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn emit_deopt_at(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        get_runtime_fn: &mut dyn FnMut(
            &mut dyn Module,
            &mut FunctionBuilder,
            &str,
            usize,
        ) -> Result<cranelift_codegen::ir::FuncRef, String>,
        func_id: u32,
        pc: u32,
        live: &[DeoptReg],
        val_map: &HashMap<ValueId, Value>,
        raw_bools: &HashSet<ValueId>,
        value_types: &[MirType],
    ) -> Result<(), String> {
        let (buf, words) = emit_deopt_words(builder, live, val_map, raw_bools, value_types)?;
        let fid = builder.ins().iconst(types::I64, plain_fid(func_id));
        let pc = builder.ins().iconst(types::I64, pc as i64);
        let n = builder.ins().iconst(types::I64, words as i64);
        let f = get_runtime_fn(module, builder, "wren_deopt_at", 4)?;
        emit_cur_frame(builder);
        let call = builder.ins().call(f, &[fid, pc, n, buf]);
        let result = builder.inst_results(call)[0];
        builder.ins().return_(&[result]);
        Ok(())
    }

    enum InlineBinOp {
        /// f64 arithmetic: "fadd", "fsub", "fmul", "fdiv", "frem"
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
    /// Wren's `%` on two f64s: C fmod, a truncated remainder with the
    /// dividend's sign. Integral operands below 2^53 take an exact
    /// integer remainder inline; anything else goes to libm so large
    /// quotients and fractions stay exact.
    fn emit_f64_rem(
        builder: &mut FunctionBuilder,
        module: &mut dyn Module,
        av: Value,
        bv: Value,
    ) -> Result<Value, String> {
        let fast = builder.create_block();
        let slow = builder.create_block();
        let merge = builder.create_block();
        builder.append_block_param(merge, types::F64);

        let ai = builder.ins().fcvt_to_sint_sat(types::I64, av);
        let bi = builder.ins().fcvt_to_sint_sat(types::I64, bv);
        let a_back = builder.ins().fcvt_from_sint(types::F64, ai);
        let b_back = builder.ins().fcvt_from_sint(types::F64, bi);
        let a_int = builder.ins().fcmp(FloatCC::Equal, a_back, av);
        let b_int = builder.ins().fcmp(FloatCC::Equal, b_back, bv);
        let limit = builder.ins().f64const(9007199254740992.0);
        let a_abs = builder.ins().fabs(av);
        let b_abs = builder.ins().fabs(bv);
        let a_small = builder.ins().fcmp(FloatCC::LessThan, a_abs, limit);
        let b_small = builder.ins().fcmp(FloatCC::LessThan, b_abs, limit);
        let zero = builder.ins().iconst(types::I64, 0);
        let b_nz = builder.ins().icmp(IntCC::NotEqual, bi, zero);
        let ok1 = builder.ins().band(a_int, b_int);
        let ok2 = builder.ins().band(a_small, b_small);
        let ok3 = builder.ins().band(ok1, ok2);
        let ok = builder.ins().band(ok3, b_nz);
        builder.ins().brif(ok, fast, &[], slow, &[]);

        builder.switch_to_block(fast);
        builder.seal_block(fast);
        let r = builder.ins().srem(ai, bi);
        let rf = builder.ins().fcvt_from_sint(types::F64, r);
        // A zero remainder keeps the dividend's sign, as fmod does.
        let rf = builder.ins().fcopysign(rf, av);
        builder.ins().jump(merge, &[BlockArg::Value(rf)]);

        builder.switch_to_block(slow);
        builder.seal_block(slow);
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));
        let fid = module
            .declare_function("fmod", Linkage::Import, &sig)
            .map_err(|e| e.to_string())?;
        let fref = module.declare_func_in_func(fid, builder.func);
        let call = builder.ins().call(fref, &[av, bv]);
        let slow_r = builder.inst_results(call)[0];
        builder.ins().jump(merge, &[BlockArg::Value(slow_r)]);

        builder.switch_to_block(merge);
        builder.seal_block(merge);
        Ok(builder.block_params(merge)[0])
    }

    #[allow(clippy::type_complexity)] // the runtime-fn resolver closure type is shared verbatim
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
                    "frem" => emit_f64_rem(builder, module, fa, fb)?,
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
        // Equality on anything but a Num is identity unless the left
        // operand's class says otherwise, which the helper decides.
        let identity = match op {
            InlineBinOp::Cmp(FloatCC::Equal) => Some(true),
            InlineBinOp::Cmp(FloatCC::NotEqual) => Some(false),
            _ => None,
        };
        if let Some(eq) = identity {
            let ideq_block = builder.create_block();
            let call_block = builder.create_block();
            let obj_block = builder.create_block();
            let tag_obj = builder.ins().iconst(types::I64, TAG_OBJ as i64);
            let high = builder.ins().band(la, tag_obj);
            let is_obj = builder.ins().icmp(IntCC::Equal, high, tag_obj);
            builder.ins().brif(is_obj, obj_block, &[], ideq_block, &[]);
            builder.switch_to_block(obj_block);
            let mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
            let ptr = builder.ins().band(la, mask);
            let class = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), ptr, HEADER_CLASS);
            let flags = builder
                .ins()
                .load(types::I8, MemFlags::trusted(), class, CLASS_FLAGS);
            let eq_bit = builder.ins().iconst(types::I8, CLASS_FLAG_EQ as i64);
            let custom = builder.ins().band(flags, eq_bit);
            builder.ins().brif(custom, call_block, &[], ideq_block, &[]);
            builder.switch_to_block(ideq_block);
            let same = builder.ins().icmp(IntCC::Equal, la, lb);
            let true_val = builder.ins().iconst(types::I64, TAG_TRUE as i64);
            let false_val = builder.ins().iconst(types::I64, TAG_FALSE as i64);
            let r = if eq {
                builder.ins().select(same, true_val, false_val)
            } else {
                builder.ins().select(same, false_val, true_val)
            };
            builder.ins().jump(merge_block, &[BlockArg::Value(r)]);
            builder.switch_to_block(call_block);
        }
        let f = get_runtime_fn(module, builder, slow_fn, 2)?;
        emit_cur_frame(builder);
        let call = builder.ins().call(f, &[la, lb]);
        let slow_result = builder.inst_results(call)[0];
        emit_error_poll(builder, module, get_runtime_fn)?;
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
        mir: &MirFunction,
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
        // The inline-cache entry of this call, when it has one.
        ic_site: Option<usize>,
        f64_self_id: Option<cranelift_module::FuncId>,
        receiver_val: Option<Value>,
        aot_config: Option<&AotLoweringConfig>,
        inline_bodies: Option<
            &std::sync::Arc<std::collections::HashMap<u32, std::sync::Arc<MirFunction>>>,
        >,
        cha_by_method: Option<&std::sync::Arc<crate::runtime::engine::ChaMap>>,
        deopt_state: Option<(&HashSet<ValueId>, &[MirType])>,
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
                if f64_self_id.is_some() {
                    return Ok(Some(builder.ins().f64const(*n)));
                }
                let bits = n.to_bits() as i64;
                Ok(Some(builder.ins().iconst(types::I64, bits)))
            }
            Instruction::ConstBool(b) => {
                let bits = if *b { TAG_TRUE } else { TAG_FALSE } as i64;
                Ok(Some(builder.ins().iconst(types::I64, bits)))
            }
            Instruction::ConstNull => Ok(Some(builder.ins().iconst(types::I64, TAG_NULL as i64))),
            Instruction::ConstF64(n) => {
                if f64_is_fmov_immediate(*n) {
                    return Ok(Some(builder.ins().f64const(*n)));
                }
                // A constant the optimiser would otherwise rebuild from
                // integer moves at every use is loaded from a data slot
                // instead, so it stays hoisted out of loops.
                let mut desc = cranelift_module::DataDescription::new();
                desc.define(n.to_bits().to_le_bytes().to_vec().into_boxed_slice());
                let data_id = module
                    .declare_anonymous_data(false, false)
                    .map_err(|e| e.to_string())?;
                module
                    .define_data(data_id, &desc)
                    .map_err(|e| e.to_string())?;
                let gv = module.declare_data_in_func(data_id, builder.func);
                let addr = builder.ins().symbol_value(types::I64, gv);
                let mut flags = MemFlags::trusted();
                flags.set_readonly();
                Ok(Some(builder.ins().load(types::F64, flags, addr, 0)))
            }
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
                let la = get(a);
                let lb = get(b);
                emit_inline_boxed_binop(
                    builder,
                    module,
                    get_runtime_fn,
                    la,
                    lb,
                    InlineBinOp::Arith("frem"),
                    "wren_num_mod",
                )
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
                emit_cur_frame(builder);
                let call = builder.ins().call(f, &[la]);
                let slow_result = builder.inst_results(call)[0];
                emit_error_poll(builder, module, get_runtime_fn)?;
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
                // The fields follow the header.
                let fields_ptr = builder.ins().iadd_imm_u(obj_ptr, INSTANCE_SIZE as i64);
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
                let fields_ptr = builder.ins().iadd_imm_u(obj_ptr, INSTANCE_SIZE as i64);
                // Store field value
                let offset = (*idx as i32) * VALUE_SIZE;
                builder
                    .ins()
                    .store(MemFlags::trusted(), store_val, fields_ptr, offset);
                // Only the LLVM tier reads the field kinds.
                if aot_config.is_none() && jit_note_field_kinds() {
                    match INLINE_CLASS.get() {
                        Some((r, class)) if r == recv_val => {
                            emit_note_field_kind_static(builder, class, *idx, store_val);
                        }
                        _ => emit_note_field_kind(builder, obj_ptr, *idx, store_val),
                    }
                }
                // SetField result is the stored value
                Ok(Some(store_val))
            }

            // === Module variables ===
            //
            // AOT mode: each module owns a `wlift_modvars_<n>`
            // data symbol — a `[u64; var_count]` in `.bss` —
            // declared by the AOT driver and threaded through
            // here as a `DataId`. Get/Set become a `symbol_value`
            // load + offset, killing the runtime helper entirely.
            //
            // JIT mode keeps the `wren_get/set_module_var` dispatch
            // — those helpers consult the TLS `JitContext` which
            // is patched per-frame by the install path.
            Instruction::GetModuleVar(idx) => Ok(Some(emit_get_module_var(
                builder,
                module,
                get_runtime_fn,
                *idx,
                aot_config,
            )?)),
            Instruction::SetModuleVar(idx, val) => Ok(Some(emit_set_module_var(
                builder,
                module,
                get_runtime_fn,
                *idx,
                get(val),
                aot_config,
            )?)),
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
                        let sym_base = builder.ins().symbol_value(types::I64, sym_gv);
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
                        builder
                            .ins()
                            .stack_store(types::I64, v, stack_slot, (i * 8) as i32);
                    }
                    let buf = builder.ins().stack_addr(types::I64, stack_slot, 0);
                    let count = builder.ins().iconst(types::I64, args.len() as i64);
                    let f = get_runtime_fn(module, builder, "wren_call_dynamic", 4)?;
                    emit_cur_frame(builder);
                    let call = builder.ins().call(f, &[r, method_val, count, buf]);
                    emit_error_poll(builder, module, get_runtime_fn)?;
                    return Ok(Some(builder.inst_results(call)[0]));
                }
                let r = get(receiver);
                let arg_vals: Vec<Value> = args.iter().map(get).collect();

                if let Some(simd_result) = try_lower_simd_intrinsic_call(
                    interner,
                    builder,
                    module,
                    get_runtime_fn,
                    r,
                    *method,
                    &arg_vals,
                    ic_site,
                    aot_config,
                )? {
                    return Ok(Some(simd_result));
                }
                if aot_config.is_none()
                    && let Some(v) = try_lower_list_protocol(
                        interner,
                        builder,
                        module,
                        get_runtime_fn,
                        r,
                        *method,
                        &arg_vals,
                    )?
                {
                    return Ok(Some(v));
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
                if let Some(cfg) = aot_config
                    && let Some(cha_ptr) = cfg.cha
                {
                    let cha = unsafe { &*cha_ptr };
                    let sig_text = interner.resolve(*method).to_string();
                    if let Some(impls) = cha.by_sig.get(&sig_text)
                        && !impls.is_empty()
                    {
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
                        let tag_obj_const = builder.ins().iconst(types::I64, TAG_OBJ as i64);
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
                            let gv = module.declare_data_in_func(class_data_id, builder.func);
                            let modvars_addr = builder.ins().symbol_value(types::I64, gv);
                            let boxed_cls = builder.ins().load(
                                types::I64,
                                MemFlags::trusted(),
                                modvars_addr,
                                (impl_.class_slot as i32) * 8,
                            );
                            let cls_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                            let expected_cls = builder.ins().band(boxed_cls, cls_mask);
                            let eq =
                                builder
                                    .ins()
                                    .icmp(IntCC::Equal, recv_class_field, expected_cls);
                            builder.ins().brif(eq, fast_block, &[], next_check, &[]);

                            builder.switch_to_block(fast_block);
                            let fast_result = if let Some(field_idx) = impl_.trivial_getter_field {
                                // Inline trivial getter: load
                                // recv.fields[field_idx].
                                let fields_ptr =
                                    builder.ins().iadd_imm_u(recv_obj, INSTANCE_SIZE as i64);
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
                                let mut sig =
                                    Signature::new(module.target_config().default_call_conv);
                                for _ in 0..(impl_.arity as usize) {
                                    sig.params.push(AbiParam::new(types::I64));
                                }
                                sig.returns.push(AbiParam::new(types::I64));
                                let body_id = module
                                    .declare_function(&impl_.fn_symbol, Linkage::Import, &sig)
                                    .map_err(|e| e.to_string())?;
                                let fn_ref = module.declare_func_in_func(body_id, builder.func);
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
                                    let null = builder.ins().iconst(types::I64, TAG_NULL as i64);
                                    call_args.push(null);
                                }
                                emit_cur_frame(builder);
                                let call = builder.ins().call(fn_ref, &call_args);
                                emit_error_poll(builder, module, get_runtime_fn)?;
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
                        let sym_gv = module.declare_data_in_func(cfg.symbols_data, builder.func);
                        let sym_base = builder.ins().symbol_value(types::I64, sym_gv);
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
                            emit_cur_frame(builder);
                            let call = builder.ins().call(f, &[r, method_val, count, buf_ptr]);
                            emit_error_poll(builder, module, get_runtime_fn)?;
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
                            emit_cur_frame(builder);
                            let slow_call = builder.ins().call(f, &slow_args);
                            emit_error_poll(builder, module, get_runtime_fn)?;
                            builder.inst_results(slow_call)[0]
                        };
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(slow_result)]);

                        builder.switch_to_block(merge_block);
                        return Ok(Some(builder.block_params(merge_block)[0]));
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
                // A site whose cache was empty at this compile makes
                // the generic call, which fills the cache for the next
                // compile to inline from; the class hierarchy would
                // dispatch it without ever recording what it sees.
                let cold_site = ic_site
                    .and_then(|i| callsite_ic_ptrs.and_then(|ics| ics.get(i)))
                    .is_some_and(|ic| ic.kind == 0);
                if let Some(cha) = cha_by_method
                    && args.len() <= 4
                    && !cold_site
                {
                    let impls: Vec<crate::runtime::engine::ChaImpl> =
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

                        for crate::runtime::engine::ChaImpl {
                            class: class_ptr,
                            fid,
                            ..
                        } in &impls
                        {
                            let next_check = builder.create_block();
                            let fast_block = builder.create_block();
                            let cached_class = builder.ins().iconst(types::I64, *class_ptr as i64);
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
                                let mut inline_failed = false;
                                let outer_class = INLINE_CLASS.replace(Some((r, *class_ptr)));
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
                                                None,
                                                f64_self_id,
                                                Some(callee_args[0]),
                                                aot_config,
                                                None,
                                                None,
                                                None,
                                            )?;
                                            if let Some(v) = res {
                                                callee_vals.insert(*vid, v);
                                            }
                                        }
                                    }
                                }
                                INLINE_CLASS.set(outer_class);
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
                                    let packed = (*fid as u64) | ((method.index() as u64) << 32);
                                    let fid_val = builder.ins().iconst(types::I64, packed as i64);
                                    let fast_name = match args.len() {
                                        0 => "wren_known_call_0_nocheck",
                                        1 => "wren_known_call_1_nocheck",
                                        2 => "wren_known_call_2_nocheck",
                                        _ => "wren_known_call_3_nocheck",
                                    };
                                    let fast_arg_count = 2 + args.len();
                                    let fast_f =
                                        get_runtime_fn(module, builder, fast_name, fast_arg_count)?;
                                    let mut fast_args = vec![fid_val, r];
                                    for a in args.iter() {
                                        fast_args.push(get(a));
                                    }
                                    emit_cur_frame(builder);
                                    let fast_call = builder.ins().call(fast_f, &fast_args);
                                    emit_error_poll(builder, module, get_runtime_fn)?;
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
                        let arg_vals: Vec<_> = args.iter().map(&get).collect();
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

                // Try inline IC: emit class-check + fast path.
                // Kind=5 (getter): inline field load (class baked as constant).
                // Kind=1: currently only used for IC index encoding in slow path.
                let ic = ic_site.and_then(|i| callsite_ic_ptrs.and_then(|ics| ics.get(i)));
                let _ = callsite_ic_live_ptrs;

                // AOT mode: ICs are JIT-only (mutable code memory).
                // Skip the kind=5 inline-getter fast path entirely
                // and use the slow path so the symbol-remap table
                // indirection emits cleanly.
                let ic = if aot_config.is_some() { None } else { ic };

                if let Some(ic) = ic {
                    // A constructor on a resolved class: allocate and run
                    // the initialiser directly when the receiver is that
                    // class object.
                    if ic.kind == 3
                        && ic.class != 0
                        && ic.func_id != 0
                        && args.len() <= 3
                        && std::env::var_os("X_NO3").is_none()
                    {
                        let fast_block = builder.create_block();
                        let slow_block = builder.create_block();
                        let merge_block = builder.create_block();
                        builder.append_block_param(merge_block, types::I64);
                        let class_bits = builder
                            .ins()
                            .iconst(types::I64, (TAG_OBJ | ic.class as u64) as i64);
                        let hit = builder.ins().icmp(IntCC::Equal, r, class_bits);
                        builder.ins().brif(hit, fast_block, &[], slow_block, &[]);
                        builder.switch_to_block(fast_block);
                        let packed = ic.func_id | ((method.index() as u64) << 32);
                        let packed_val = builder.ins().iconst(types::I64, packed as i64);
                        let arg_vals: Vec<_> = args.iter().map(&get).collect();
                        // The initialiser is called straight through its
                        // slot when it is compiled; the helper otherwise.
                        let helper_block = builder.create_block();
                        if let (true, Some(jit_base_ptr)) = (direct_calls_enabled(), jit_code_base)
                        {
                            let slot_addr = unsafe { jit_base_ptr.add(ic.func_id as usize) as i64 };
                            let slot_addr_val = builder.ins().iconst(types::I64, slot_addr);
                            let jit_ptr =
                                builder
                                    .ins()
                                    .load(types::I64, MemFlags::new(), slot_addr_val, 0);
                            let depth_block = builder.create_block();
                            let call_block = builder.create_block();
                            let zero = builder.ins().iconst(types::I64, 0);
                            let has_jit = builder.ins().icmp(IntCC::NotEqual, jit_ptr, zero);
                            builder
                                .ins()
                                .brif(has_jit, depth_block, &[], helper_block, &[]);
                            builder.switch_to_block(depth_block);
                            let depth_addr = builder.ins().iconst(
                                types::I64,
                                &crate::codegen::runtime_fns::JIT_DIRECT_DEPTH
                                    as *const std::sync::atomic::AtomicU32
                                    as i64,
                            );
                            let depth =
                                builder
                                    .ins()
                                    .load(types::I32, MemFlags::trusted(), depth_addr, 0);
                            let room = builder.ins().icmp_imm_u(
                                IntCC::UnsignedLessThan,
                                depth,
                                crate::codegen::runtime_fns::MAX_JIT_DEPTH as i64,
                            );
                            builder.ins().brif(room, call_block, &[], helper_block, &[]);
                            builder.switch_to_block(call_block);
                            let inst = emit_alloc_instance(builder, module, get_runtime_fn, r)?;
                            let deeper = builder.ins().iadd_imm_u(depth, 1);
                            builder
                                .ins()
                                .store(MemFlags::trusted(), deeper, depth_addr, 0);
                            let mut sig = module.make_signature();
                            sig.params.push(AbiParam::new(types::I64));
                            for _ in args.iter() {
                                sig.params.push(AbiParam::new(types::I64));
                            }
                            sig.returns.push(AbiParam::new(types::I64));
                            let sig_ref = builder.import_signature(sig);
                            let mut call_args = vec![inst];
                            call_args.extend(arg_vals.iter().copied());
                            let _ = builder.ins().call_indirect(sig_ref, jit_ptr, &call_args);
                            builder
                                .ins()
                                .store(MemFlags::trusted(), depth, depth_addr, 0);
                            builder.ins().jump(merge_block, &[BlockArg::Value(inst)]);
                        } else {
                            builder.ins().jump(helper_block, &[]);
                        }
                        builder.switch_to_block(helper_block);
                        let name = [
                            "wren_construct_0",
                            "wren_construct_1",
                            "wren_construct_2",
                            "wren_construct_3",
                        ][args.len()];
                        let f = get_runtime_fn(module, builder, name, 2 + args.len())?;
                        let mut call_args = vec![packed_val, r];
                        call_args.extend(arg_vals.iter().copied());
                        emit_cur_frame(builder);
                        let call = builder.ins().call(f, &call_args);
                        emit_error_poll(builder, module, get_runtime_fn)?;
                        let fast_result = builder.inst_results(call)[0];
                        builder
                            .ins()
                            .jump(merge_block, &[BlockArg::Value(fast_result)]);
                        builder.switch_to_block(slow_block);
                        let method_val = builder.ins().iconst(types::I64, method.index() as i64);
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
                        let fields_ptr = builder.ins().iadd_imm_u(obj_ptr, INSTANCE_SIZE as i64);
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
                        let method_bits = crate::codegen::runtime_fns::pack_method_word(
                            method.index(),
                            ic_site
                                .filter(|_| env_jit_callsite_ic())
                                .map(|i| (i, jit_func_id())),
                        );
                        let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                        let arg_vals: Vec<_> = args.iter().map(&get).collect();
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
                    let base = builder.ins().symbol_value(types::I64, gv);
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), base, (slot as i32) * 8)
                } else {
                    let method_bits = crate::codegen::runtime_fns::pack_method_word(
                        method.index(),
                        ic_site
                            .filter(|_| env_jit_callsite_ic() || cold_site)
                            .map(|i| (i, jit_func_id())),
                    );
                    builder.ins().iconst(types::I64, method_bits as i64)
                };
                // Route arity > 8 through wren_call_dynamic to
                // avoid the silent .min(8) truncation that
                // corrupted Renderer2D's drawSprite_(receiver +
                // 13 args) call frame.
                let result_val = if args.len() > 8 {
                    let f = get_runtime_fn(module, builder, "wren_call_dynamic", 4)?;
                    let slot =
                        builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            (args.len() as u32) * 8,
                            3,
                        ));
                    let buf_ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    for (i, a) in args.iter().enumerate() {
                        builder
                            .ins()
                            .store(MemFlags::trusted(), get(a), buf_ptr, (i as i32) * 8);
                    }
                    let count = builder.ins().iconst(types::I64, args.len() as i64);
                    emit_cur_frame(builder);
                    let call = builder.ins().call(f, &[r, method_val, count, buf_ptr]);
                    emit_error_poll(builder, module, get_runtime_fn)?;
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
                    emit_cur_frame(builder);
                    let result = builder.ins().call(f, &call_args);
                    emit_error_poll(builder, module, get_runtime_fn)?;
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
                direct,
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
                    let base = builder.ins().symbol_value(types::I64, gv);
                    let method_val = builder.ins().load(
                        types::I64,
                        MemFlags::trusted(),
                        base,
                        (slot as i32) * 8,
                    );
                    let arg_vals: Vec<_> = args.iter().map(&get).collect();
                    let result =
                        emit_wren_call(builder, module, get_runtime_fn, r, method_val, &arg_vals)?;
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
                if let Some(bodies) = inline_bodies
                    && *expected_class != 0
                    && args.len() <= 4
                    && let Some(callee_mir) = bodies.get(func_id).cloned()
                {
                    let fast_block = builder.create_block();
                    let slow_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);

                    // is-object guard + class load. Non-objects
                    // skip the inlined body and route to the
                    // slow path.
                    let (_obj_ptr, recv_class) = emit_class_load_guarded(builder, r, slow_block);
                    let cached_class = builder.ins().iconst(types::I64, *expected_class as i64);
                    let class_match = builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
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
                    let mut inline_failed = false;
                    let outer_class = INLINE_CLASS.replace(Some((r, *expected_class)));
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
                                    None,
                                    f64_self_id,
                                    Some(callee_args[0]),
                                    aot_config,
                                    None,
                                    None,
                                    None,
                                )?;
                                if let Some(v) = res {
                                    callee_vals.insert(*vid, v);
                                }
                            }
                        }
                    }
                    INLINE_CLASS.set(outer_class);
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
                    let arg_vals: Vec<_> = args.iter().map(&get).collect();
                    let slow_result =
                        emit_wren_call(builder, module, get_runtime_fn, r, method_val, &arg_vals)?;
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(slow_result)]);

                    builder.switch_to_block(merge_block);
                    return Ok(Some(builder.block_params(merge_block)[0]));
                }

                // === Pure-leaf direct call (ZERO FFI) ===
                // Callee has no internal method calls, so no context
                // setup is needed. Emit: class check + load callee ptr
                // + call_indirect. The JIT code slot address is stable
                // because engine.jit_code doesn't reallocate post-load.
                //
                // Gated off by default: this path emits a raw
                // `call_indirect` and passes args through CPU registers
                // without rooting them. "Pure leaf" means "no Wren-level
                // calls" — the body can still allocate (string concat,
                // list grow, list-of-num alloc). Set
                // `WLIFT_ENABLE_PURE_LEAF_DIRECT=1` to take it; until
                // then fall through to `wren_known_call_N_nocheck`,
                // which roots args before dispatching.
                if direct_calls_enabled()
                    && inline_getter_field.is_none()
                    && *direct
                    && *expected_class != 0
                    && args.len() <= 4
                    && let Some(jit_base_ptr) = jit_code_base
                {
                    let fast_block = builder.create_block();
                    let slow_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, types::I64);

                    // is-object guard + class load.
                    let (_obj_ptr, recv_class) = emit_class_load_guarded(builder, r, slow_block);
                    let cached_class = builder.ins().iconst(types::I64, *expected_class as i64);
                    let class_match = builder.ins().icmp(IntCC::Equal, recv_class, cached_class);
                    builder
                        .ins()
                        .brif(class_match, fast_block, &[], slow_block, &[]);

                    // Fast path: load the callee's JIT slot and call_indirect.
                    // slot_addr = jit_code_base + func_id * 8
                    builder.switch_to_block(fast_block);
                    let slot_addr = unsafe { jit_base_ptr.add(*func_id as usize) as i64 };
                    let slot_addr_val = builder.ins().iconst(types::I64, slot_addr);
                    let jit_ptr = builder
                        .ins()
                        .load(types::I64, MemFlags::new(), slot_addr_val, 0);
                    // Guard: if slot is null (callee not yet compiled),
                    // fall to slow path.
                    let zero = builder.ins().iconst(types::I64, 0);
                    let has_jit = builder.ins().icmp(IntCC::NotEqual, jit_ptr, zero);
                    let depth_block = builder.create_block();
                    let pure_call_block = builder.create_block();
                    builder
                        .ins()
                        .brif(has_jit, depth_block, &[], slow_block, &[]);
                    builder.switch_to_block(depth_block);
                    let depth_addr = builder.ins().iconst(
                        types::I64,
                        &crate::codegen::runtime_fns::JIT_DIRECT_DEPTH
                            as *const std::sync::atomic::AtomicU32 as i64,
                    );
                    let depth = builder
                        .ins()
                        .load(types::I32, MemFlags::trusted(), depth_addr, 0);
                    let room = builder.ins().icmp_imm_u(
                        IntCC::UnsignedLessThan,
                        depth,
                        crate::codegen::runtime_fns::MAX_JIT_DEPTH as i64,
                    );
                    builder
                        .ins()
                        .brif(room, pure_call_block, &[], slow_block, &[]);

                    builder.switch_to_block(pure_call_block);
                    let deeper = builder.ins().iadd_imm_u(depth, 1);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), deeper, depth_addr, 0);
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
                    emit_cur_frame(builder);
                    let call = builder.ins().call_indirect(sig_ref, jit_ptr, &call_args);
                    emit_error_poll(builder, module, get_runtime_fn)?;
                    let fast_result = builder.inst_results(call)[0];
                    builder
                        .ins()
                        .store(MemFlags::trusted(), depth, depth_addr, 0);
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(fast_result)]);

                    // Slow path: full dispatch via emit_wren_call.
                    builder.switch_to_block(slow_block);
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let arg_vals: Vec<_> = args.iter().map(&get).collect();
                    let slow_result =
                        emit_wren_call(builder, module, get_runtime_fn, r, method_val, &arg_vals)?;
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(slow_result)]);

                    builder.switch_to_block(merge_block);
                    return Ok(Some(builder.block_params(merge_block)[0]));
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
                    let fields_ptr = builder.ins().iadd_imm_u(obj_ptr, INSTANCE_SIZE as i64);
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
                    let arg_vals: Vec<_> = args.iter().map(&get).collect();
                    let slow_result =
                        emit_wren_call(builder, module, get_runtime_fn, r, method_val, &arg_vals)?;
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
                    emit_cur_frame(builder);
                    let fast_call = builder.ins().call(fast_f, &fast_args);
                    emit_error_poll(builder, module, get_runtime_fn)?;
                    let fast_result = builder.inst_results(fast_call)[0];
                    builder
                        .ins()
                        .jump(merge_block, &[BlockArg::Value(fast_result)]);

                    // Slow path: class mismatch → emit_wren_call.
                    builder.switch_to_block(slow_block);
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let arg_vals: Vec<_> = args.iter().map(&get).collect();
                    let slow_result =
                        emit_wren_call(builder, module, get_runtime_fn, r, method_val, &arg_vals)?;
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
                    emit_cur_frame(builder);
                    let result = builder.ins().call(f, &call_args);
                    emit_error_poll(builder, module, get_runtime_fn)?;
                    Ok(Some(builder.inst_results(result)[0]))
                } else {
                    let method_bits = method.index() as u64;
                    let method_val = builder.ins().iconst(types::I64, method_bits as i64);
                    let arg_vals: Vec<_> = args.iter().map(&get).collect();
                    let result =
                        emit_wren_call(builder, module, get_runtime_fn, r, method_val, &arg_vals)?;
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
                    let base = builder.ins().symbol_value(types::I64, gv);
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), base, (slot as i32) * 8)
                } else {
                    builder.ins().iconst(types::I64, method.index() as i64)
                };
                // A JIT body knows its method's class at compile time
                // and passes it: the context's may be a direct
                // caller's.
                let defining = if aot_config.is_none() {
                    crate::codegen::jit_defining_class()
                } else {
                    0
                };
                let mut call_args = Vec::with_capacity(2 + args.len());
                let call_name = if defining != 0 && !args.is_empty() {
                    call_args.push(builder.ins().iconst(types::I64, defining as i64));
                    match args.len() {
                        1 => "wren_super_call_from_1",
                        2 => "wren_super_call_from_2",
                        3 => "wren_super_call_from_3",
                        _ => "wren_super_call_from_4",
                    }
                } else {
                    match args.len() {
                        0 => "wren_super_call_0",
                        1 => "wren_super_call_1",
                        2 => "wren_super_call_2",
                        3 => "wren_super_call_3",
                        _ => "wren_super_call_4",
                    }
                };
                call_args.push(method_val);
                for a in args.iter().take(4) {
                    call_args.push(get(a));
                }
                let f = get_runtime_fn(module, builder, call_name, call_args.len())?;
                emit_cur_frame(builder);
                let result = builder.ins().call(f, &call_args);
                emit_error_poll(builder, module, get_runtime_fn)?;
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
                }
                Ok(Some(result))
            }
            Instruction::ToString(a) => {
                let f = get_runtime_fn(module, builder, "wren_to_string", 1)?;
                emit_cur_frame(builder);
                let result = builder.ins().call(f, &[get(a)]);
                emit_error_poll(builder, module, get_runtime_fn)?;
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
                if let Some(cfg) = aot_config
                    && let Some(var) = *cfg.current_closure_ptr_var.borrow()
                {
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
                let f = get_runtime_fn(module, builder, "wren_get_upvalue", 1)?;
                let idx_val = builder.ins().iconst(types::I64, *idx as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SetUpvalue(idx, val) => {
                if let Some(cfg) = aot_config
                    && let Some(var) = *cfg.current_closure_ptr_var.borrow()
                {
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
                    return Ok(Some(v));
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
                if let Some(cfg) = aot_config
                    && let Some(defining) = cfg.current_defining_class.borrow().as_ref()
                {
                    let class_data_id = module
                        .declare_data(&defining.modvars_symbol, Linkage::Export, true, false)
                        .map_err(|e| e.to_string())?;
                    let gv = module.declare_data_in_func(class_data_id, builder.func);
                    let modvars_addr = builder.ins().symbol_value(types::I64, gv);
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
                let f = get_runtime_fn(module, builder, "wren_get_static_field", 1)?;
                let idx_val = builder.ins().iconst(types::I64, sym.index() as i64);
                let result = builder.ins().call(f, &[idx_val]);
                Ok(Some(builder.inst_results(result)[0]))
            }
            Instruction::SetStaticField(sym, val) => {
                if let Some(cfg) = aot_config
                    && let Some(defining) = cfg.current_defining_class.borrow().as_ref()
                {
                    let class_data_id = module
                        .declare_data(&defining.modvars_symbol, Linkage::Export, true, false)
                        .map_err(|e| e.to_string())?;
                    let gv = module.declare_data_in_func(class_data_id, builder.func);
                    let modvars_addr = builder.ins().symbol_value(types::I64, gv);
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
                // Under AOT a method's closure records the class whose
                // static fields its body names; the JIT helper reads it
                // from the context instead.
                let stamp_class = aot_config
                    .and_then(|cfg| cfg.current_defining_class.borrow().clone())
                    .map(|defining| -> Result<_, String> {
                        let class_data_id = module
                            .declare_data(&defining.modvars_symbol, Linkage::Export, true, false)
                            .map_err(|e| e.to_string())?;
                        let gv = module.declare_data_in_func(class_data_id, builder.func);
                        let modvars_addr = builder.ins().symbol_value(types::I64, gv);
                        let class_bits = builder.ins().load(
                            types::I64,
                            MemFlags::trusted(),
                            modvars_addr,
                            (defining.slot as i32) * 8,
                        );
                        let f = get_runtime_fn(module, builder, "wlift_aot_set_closure_class", 2)?;
                        Ok((f, class_bits))
                    })
                    .transpose()?;
                let fn_id_val = if let Some(cfg) = aot_config {
                    let gv = module.declare_data_in_func(cfg.closures_data, builder.func);
                    let base = builder.ins().symbol_value(types::I64, gv);
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
                    let closure = builder.inst_results(result)[0];
                    Ok(Some(match stamp_class {
                        Some((f, class_bits)) => {
                            let r = builder.ins().call(f, &[closure, class_bits]);
                            builder.inst_results(r)[0]
                        }
                        None => closure,
                    }))
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
                        builder
                            .ins()
                            .stack_store(types::I64, v, slot, (i * 8) as i32);
                    }
                    let buf = builder.ins().stack_addr(types::I64, slot, 0);
                    let count = builder.ins().iconst(types::I64, n as i64);
                    let f = get_runtime_fn(module, builder, "wren_make_closure_n", 3)?;
                    let result = builder.ins().call(f, &[fn_id_val, count, buf]);
                    let closure = builder.inst_results(result)[0];
                    Ok(Some(match stamp_class {
                        Some((f, class_bits)) => {
                            let r = builder.ins().call(f, &[closure, class_bits]);
                            builder.inst_results(r)[0]
                        }
                        None => closure,
                    }))
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
                let shr48 = builder.ins().ushr_imm_u(r, 48);
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
                let list_block = builder.create_block();
                let list_tag = builder
                    .ins()
                    .iconst(types::I64, crate::runtime::object::ObjType::List as i64);
                let is_list = builder.ins().icmp(IntCC::Equal, obj_type_byte, list_tag);
                let not_list_block = builder.create_block();
                builder
                    .ins()
                    .brif(is_list, list_block, &[], not_list_block, &[]);
                builder.switch_to_block(not_list_block);
                let simd_tag = builder.ins().iconst(types::I64, OBJ_TYPE_SIMD as i64);
                let is_simd = builder.ins().icmp(IntCC::Equal, obj_type_byte, simd_tag);
                builder
                    .ins()
                    .brif(is_simd, simd_block, &[], slow_block, &[]);

                // 2b. List: an integral Num index within the count reads
                //     the element; anything else (a negative index, a
                //     range) is the helper's. An index boxed from an
                //     i64 is read as that integer.
                builder.switch_to_block(list_block);
                let list_idx = if let Some(i) =
                    int_source(mir, &args[0]).and_then(|i| val_map.get(&i).copied())
                {
                    let count = builder
                        .ins()
                        .uload32(MemFlags::trusted(), obj_ptr, LIST_COUNT);
                    let in_range = builder.ins().icmp(IntCC::UnsignedLessThan, i, count);
                    let load_block = builder.create_block();
                    builder
                        .ins()
                        .brif(in_range, load_block, &[], slow_block, &[]);
                    builder.switch_to_block(load_block);
                    i
                } else {
                    let qnan = builder.ins().iconst(types::I64, QNAN as i64);
                    let masked = builder.ins().band(idx, qnan);
                    let is_box = builder.ins().icmp(IntCC::Equal, masked, qnan);
                    let num_block = builder.create_block();
                    builder.ins().brif(is_box, slow_block, &[], num_block, &[]);
                    builder.switch_to_block(num_block);
                    let f = builder.ins().bitcast(types::F64, MemFlags::new(), idx);
                    let i = builder.ins().fcvt_to_sint_sat(types::I64, f);
                    let back = builder.ins().fcvt_from_sint(types::F64, i);
                    let integral = builder.ins().fcmp(FloatCC::Equal, back, f);
                    let count = builder
                        .ins()
                        .uload32(MemFlags::trusted(), obj_ptr, LIST_COUNT);
                    let in_range = builder.ins().icmp(IntCC::UnsignedLessThan, i, count);
                    let ok = builder.ins().band(integral, in_range);
                    let load_block = builder.create_block();
                    builder.ins().brif(ok, load_block, &[], slow_block, &[]);
                    builder.switch_to_block(load_block);
                    i
                };
                let elements =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj_ptr, LIST_ELEMENTS);
                let off = builder.ins().imul_imm_s(list_idx, VALUE_SIZE as i64);
                let addr = builder.ins().iadd(elements, off);
                let elem = builder.ins().load(types::I64, MemFlags::trusted(), addr, 0);
                builder.ins().jump(merge_block, &[BlockArg::Value(elem)]);

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
                let simd_data_base = builder.ins().iadd_imm_u(obj_ptr, SIMD_LANES as i64);
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
                let simd_data_base = builder.ins().iadd_imm_u(obj_ptr, SIMD_LANES as i64);
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
                emit_cur_frame(builder);
                let slow_call = builder.ins().call(slow_fn, &[r, idx]);
                let slow_result = builder.inst_results(slow_call)[0];
                emit_error_poll(builder, module, get_runtime_fn)?;
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
                emit_cur_frame(builder);
                let result = builder.ins().call(f, &call_args);
                emit_error_poll(builder, module, get_runtime_fn)?;
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
                let shr48 = builder.ins().ushr_imm_u(r, 48);
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
                emit_cur_frame(builder);
                let slow_call = builder.ins().call(slow_fn, &[r, idx, val]);
                let slow_result = builder.inst_results(slow_call)[0];
                emit_error_poll(builder, module, get_runtime_fn)?;
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
                emit_cur_frame(builder);
                let result = builder.ins().call(f, &call_args);
                emit_error_poll(builder, module, get_runtime_fn)?;
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
            // Raw i8 results: an object test, then the class (or the
            // closure's function) compared against the baked pointer.
            Instruction::ClassIs(a, class_ptr) => {
                let v = get(a);
                let tag_obj = builder.ins().iconst(types::I64, TAG_OBJ as i64);
                let high = builder.ins().band(v, tag_obj);
                let is_obj = builder.ins().icmp(IntCC::Equal, high, tag_obj);
                let object_block = builder.create_block();
                let merge_block = builder.create_block();
                builder.append_block_param(merge_block, types::I8);
                let no = builder.ins().iconst(types::I8, 0);
                builder.ins().brif(
                    is_obj,
                    object_block,
                    &[],
                    merge_block,
                    &[BlockArg::Value(no)],
                );
                builder.switch_to_block(object_block);
                let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                let obj_ptr = builder.ins().band(v, ptr_mask);
                let class =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj_ptr, HEADER_CLASS);
                let expected = builder.ins().iconst(types::I64, *class_ptr as i64);
                let hit = builder.ins().icmp(IntCC::Equal, class, expected);
                builder.ins().jump(merge_block, &[BlockArg::Value(hit)]);
                builder.switch_to_block(merge_block);
                Ok(Some(builder.block_params(merge_block)[0]))
            }
            Instruction::GuardClassAt {
                value,
                class,
                pc,
                live,
            } => {
                let v = get(value);
                if aot_config.is_some() {
                    return Ok(Some(v));
                }
                let Some((raw_bools, value_types)) = deopt_state else {
                    return Err("mid-body guard inside an inlined body".into());
                };
                let deopt_block = builder.create_block();
                let cont_block = builder.create_block();
                builder.set_cold_block(deopt_block);
                let (_, recv_class) = emit_class_load_guarded(builder, v, deopt_block);
                let expected = builder.ins().iconst(types::I64, *class as i64);
                let hit = builder.ins().icmp(IntCC::Equal, recv_class, expected);
                builder.ins().brif(hit, cont_block, &[], deopt_block, &[]);
                builder.switch_to_block(deopt_block);
                emit_deopt_at(
                    builder,
                    module,
                    get_runtime_fn,
                    jit_func_id(),
                    *pc,
                    live,
                    val_map,
                    raw_bools,
                    value_types,
                )?;
                builder.switch_to_block(cont_block);
                Ok(Some(v))
            }
            // === Integer arithmetic on proven-integral values ===
            Instruction::AddI64(a, b) => Ok(Some(builder.ins().iadd(get(a), get(b)))),
            Instruction::SubI64(a, b) => Ok(Some(builder.ins().isub(get(a), get(b)))),
            Instruction::MulI64(a, b) => Ok(Some(builder.ins().imul(get(a), get(b)))),
            Instruction::RemI64(a, b) => Ok(Some(builder.ins().srem(get(a), get(b)))),
            Instruction::BandI64(a, b) => Ok(Some(builder.ins().band(get(a), get(b)))),
            Instruction::NegI64(a) => Ok(Some(builder.ins().ineg(get(a)))),
            Instruction::CmpLtI64(a, b) => Ok(Some(builder.ins().icmp(
                IntCC::SignedLessThan,
                get(a),
                get(b),
            ))),
            Instruction::CmpGtI64(a, b) => Ok(Some(builder.ins().icmp(
                IntCC::SignedGreaterThan,
                get(a),
                get(b),
            ))),
            Instruction::CmpLeI64(a, b) => Ok(Some(builder.ins().icmp(
                IntCC::SignedLessThanOrEqual,
                get(a),
                get(b),
            ))),
            Instruction::CmpGeI64(a, b) => Ok(Some(builder.ins().icmp(
                IntCC::SignedGreaterThanOrEqual,
                get(a),
                get(b),
            ))),
            Instruction::I64ToF64(a) => Ok(Some(builder.ins().fcvt_from_sint(types::F64, get(a)))),
            Instruction::F64ToI64(a) => {
                Ok(Some(builder.ins().fcvt_to_sint_sat(types::I64, get(a))))
            }
            // The count of a List the guard before it established.
            Instruction::ListCount(recv) => {
                let mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                let obj = builder.ins().band(get(recv), mask);
                let count32 = builder.ins().uload32(MemFlags::trusted(), obj, LIST_COUNT);
                let countf = builder.ins().fcvt_from_uint(types::F64, count32);
                if f64_self_id.is_some() {
                    Ok(Some(countf))
                } else {
                    Ok(Some(builder.ins().bitcast(
                        types::I64,
                        MemFlags::new(),
                        countf,
                    )))
                }
            }
            Instruction::IsNum(a) => {
                let v = get(a);
                let qnan = builder.ins().iconst(types::I64, QNAN as i64);
                let masked = builder.ins().band(v, qnan);
                Ok(Some(builder.ins().icmp(IntCC::NotEqual, masked, qnan)))
            }
            Instruction::ObjectIs(a, obj_ptr) => {
                let v = get(a);
                let expected = builder
                    .ins()
                    .iconst(types::I64, (TAG_OBJ | (*obj_ptr as u64 & PTR_MASK)) as i64);
                Ok(Some(builder.ins().icmp(IntCC::Equal, v, expected)))
            }
            Instruction::NewInstance { class, .. } => {
                let class_val = builder
                    .ins()
                    .iconst(types::I64, (TAG_OBJ | (*class as u64 & PTR_MASK)) as i64);
                Ok(Some(emit_alloc_instance(
                    builder,
                    module,
                    get_runtime_fn,
                    class_val,
                )?))
            }
            Instruction::ClosureFnIs(a, fn_ptr) => {
                let v = get(a);
                let tag_obj = builder.ins().iconst(types::I64, TAG_OBJ as i64);
                let high = builder.ins().band(v, tag_obj);
                let is_obj = builder.ins().icmp(IntCC::Equal, high, tag_obj);
                let object_block = builder.create_block();
                let closure_block = builder.create_block();
                let merge_block = builder.create_block();
                builder.append_block_param(merge_block, types::I8);
                let no = builder.ins().iconst(types::I8, 0);
                builder.ins().brif(
                    is_obj,
                    object_block,
                    &[],
                    merge_block,
                    &[BlockArg::Value(no)],
                );
                builder.switch_to_block(object_block);
                let ptr_mask = builder.ins().iconst(types::I64, PTR_MASK as i64);
                let obj_ptr = builder.ins().band(v, ptr_mask);
                let obj_type =
                    builder
                        .ins()
                        .uload8(types::I64, MemFlags::trusted(), obj_ptr, HEADER_OBJ_TYPE);
                let closure_tag = builder
                    .ins()
                    .iconst(types::I64, crate::runtime::object::ObjType::Closure as i64);
                let is_closure = builder.ins().icmp(IntCC::Equal, obj_type, closure_tag);
                builder.ins().brif(
                    is_closure,
                    closure_block,
                    &[],
                    merge_block,
                    &[BlockArg::Value(no)],
                );
                builder.switch_to_block(closure_block);
                let function =
                    builder
                        .ins()
                        .load(types::I64, MemFlags::trusted(), obj_ptr, CLOSURE_FUNCTION);
                let expected = builder.ins().iconst(types::I64, *fn_ptr as i64);
                let hit = builder.ins().icmp(IntCC::Equal, function, expected);
                builder.ins().jump(merge_block, &[BlockArg::Value(hit)]);
                builder.switch_to_block(merge_block);
                Ok(Some(builder.block_params(merge_block)[0]))
            }
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
                let av = get(a);
                let bv = get(b);
                // A positive power-of-two divisor: `a - trunc(a / b) * b`
                // is exact for every finite `a` because each step only
                // moves or clears the low bits of `a`; the sign of a zero
                // result follows the dividend as fmod's does.
                if let Some(c) = const_f64_of(mir, *b).filter(|c| is_positive_power_of_two(*c)) {
                    let inv = builder.ins().f64const(1.0 / c);
                    let q = builder.ins().fmul(av, inv);
                    let q = builder.ins().trunc(q);
                    let m = builder.ins().fmul(q, bv);
                    let r = builder.ins().fsub(av, m);
                    return Ok(Some(builder.ins().fcopysign(r, av)));
                }
                Ok(Some(emit_f64_rem(builder, module, av, bv)?))
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
            // A speculative guard at the entry: the body was specialised
            // on the profiled type, so a value of another type hands the
            // call to the interpreter and the function back to baseline.
            Instruction::GuardNum(src) => {
                let v = get(src);
                if f64_self_id.is_some() || aot_config.is_some() {
                    return Ok(Some(v));
                }
                let qnan = builder.ins().iconst(types::I64, QNAN as i64);
                let masked = builder.ins().band(v, qnan);
                let is_box = builder.ins().icmp(IntCC::Equal, masked, qnan);
                emit_guard_deopt(builder, module, get_runtime_fn, is_box, jit_func_id())?;
                Ok(Some(v))
            }
            // The Cranelift top tier keeps its slow paths.
            Instruction::SlowPathExit { .. } | Instruction::ColdLoopExit { .. } => Ok(None),
            Instruction::GuardNumAt {
                value, pc, live, ..
            } => {
                let v = get(value);
                if aot_config.is_some() {
                    return Ok(Some(v));
                }
                let Some((raw_bools, value_types)) = deopt_state else {
                    return Err("mid-body guard inside an inlined body".into());
                };
                let qnan = builder.ins().iconst(types::I64, QNAN as i64);
                let masked = builder.ins().band(v, qnan);
                let is_box = builder.ins().icmp(IntCC::Equal, masked, qnan);
                emit_guard_deopt_at(
                    builder,
                    module,
                    get_runtime_fn,
                    is_box,
                    jit_func_id(),
                    *pc,
                    live,
                    val_map,
                    raw_bools,
                    value_types,
                )?;
                Ok(Some(v))
            }
            Instruction::GuardBool(src) => {
                let v = get(src);
                if f64_self_id.is_some() || aot_config.is_some() {
                    return Ok(Some(v));
                }
                let t = builder.ins().iconst(types::I64, TAG_TRUE as i64);
                let f = builder.ins().iconst(types::I64, TAG_FALSE as i64);
                let is_t = builder.ins().icmp(IntCC::Equal, v, t);
                let is_f = builder.ins().icmp(IntCC::Equal, v, f);
                let is_bool = builder.ins().bor(is_t, is_f);
                let one = builder.ins().iconst(types::I8, 1);
                let fails = builder.ins().bxor(is_bool, one);
                emit_guard_deopt(builder, module, get_runtime_fn, fails, jit_func_id())?;
                Ok(Some(v))
            }
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
                    let base = builder.ins().symbol_value(types::I64, gv);
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
                if f64_self_id.is_none()
                    && let Some(recv) = receiver_val
                {
                    call_args.push(recv);
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
    pub(crate) fn compute_rpo(mir: &MirFunction) -> Vec<usize> {
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
}
