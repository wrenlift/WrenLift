/// Threaded-code interpreter for MIR functions.
///
/// Instead of decoding flat bytecode on every instruction, this lowers
/// MIR into an array of `ThreadedOp` entries — each entry has a handler
/// function pointer and pre-decoded operands. The interpreter loop is:
///
/// ```text
/// loop {
///     let op = &code[pc];
///     pc = (op.handler)(state, op);
/// }
/// ```
///
/// This eliminates: bytecode decode (~15ns), match dispatch (~10ns), and
/// gives better branch prediction (each handler is a unique call target).
/// Expected: ~40ns/method-call vs ~75ns for bytecode interpreter.

use crate::intern::SymbolId;
use crate::mir::bytecode::CallSiteIC;
use crate::mir::{BlockId, Instruction, MirFunction, Terminator, ValueId};
use crate::runtime::value::Value;
use std::cell::UnsafeCell;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ThreadedOp: one pre-decoded instruction
// ---------------------------------------------------------------------------

/// Handler function: takes the interpreter state + current op, returns next pc.
pub type Handler = fn(&mut ThreadedState, &ThreadedOp) -> usize;

/// A single threaded instruction. All operands are pre-decoded.
/// Sized for cache efficiency (32 bytes).
#[repr(C)]
pub struct ThreadedOp {
    /// Function pointer to execute this instruction.
    pub handler: Handler,
    /// Destination register (ValueId index).
    pub dst: u16,
    /// First operand register or immediate.
    pub a: u16,
    /// Second operand register or immediate.
    pub b: u16,
    /// Extended operand (method symbol, field index, branch target, etc.).
    pub c: u32,
    /// Extra data (IC entry pointer, constant bits, etc.).
    pub extra: u64,
}

// ---------------------------------------------------------------------------
// ThreadedCode: the compiled threaded representation
// ---------------------------------------------------------------------------

/// Threaded code for a single function.
pub struct ThreadedCode {
    /// The instruction array.
    pub ops: Vec<ThreadedOp>,
    /// Constant pool (NaN-boxed values).
    pub consts: Vec<u64>,
    /// Inline cache table (same layout as bytecode IC table).
    pub ic_table: UnsafeCell<Vec<CallSiteIC>>,
    /// Number of registers needed.
    pub reg_count: usize,
}

// ---------------------------------------------------------------------------
// ThreadedState: interpreter state for threaded execution
// ---------------------------------------------------------------------------

/// Mutable state for the threaded interpreter. Passed to every handler.
pub struct ThreadedState {
    /// Register file (indexed by ValueId).
    pub regs: Vec<Value>,
    /// Program counter (index into ThreadedCode.ops).
    pub pc: usize,
    /// Pointer to the current function's IC table.
    pub ic_table: *mut Vec<CallSiteIC>,
    /// Module variable pointer (from JIT context).
    pub module_vars: *mut u64,
    pub module_var_count: u32,
    /// Pointer to the VM (for runtime calls).
    pub vm: *mut u8,
    /// Return value (set by Return handler).
    pub return_value: Option<Value>,
}

// ---------------------------------------------------------------------------
// Instruction handlers
// ---------------------------------------------------------------------------

#[inline(always)]
fn get(state: &ThreadedState, reg: u16) -> Value {
    unsafe { *state.regs.get_unchecked(reg as usize) }
}

#[inline(always)]
fn set(state: &mut ThreadedState, reg: u16, val: Value) {
    unsafe { *state.regs.get_unchecked_mut(reg as usize) = val }
}

// -- Constants --

fn op_const_null(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    set(state, op.dst, Value::null());
    state.pc + 1
}

fn op_const_true(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    set(state, op.dst, Value::bool(true));
    state.pc + 1
}

fn op_const_false(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    set(state, op.dst, Value::bool(false));
    state.pc + 1
}

fn op_const_num(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    set(state, op.dst, Value::from_bits(op.extra));
    state.pc + 1
}

fn op_move(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    set(state, op.dst, get(state, op.a));
    state.pc + 1
}

// -- Arithmetic (boxed) --

fn op_add(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = get(state, op.a);
    let b = get(state, op.b);
    let r = crate::codegen::runtime_fns::wren_num_add(a.to_bits(), b.to_bits());
    set(state, op.dst, Value::from_bits(r));
    state.pc + 1
}

fn op_sub(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = get(state, op.a);
    let b = get(state, op.b);
    let r = crate::codegen::runtime_fns::wren_num_sub(a.to_bits(), b.to_bits());
    set(state, op.dst, Value::from_bits(r));
    state.pc + 1
}

fn op_mul(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = get(state, op.a);
    let b = get(state, op.b);
    let r = crate::codegen::runtime_fns::wren_num_mul(a.to_bits(), b.to_bits());
    set(state, op.dst, Value::from_bits(r));
    state.pc + 1
}

// -- Arithmetic (f64 unboxed) --

fn op_add_f64(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = f64::from_bits(get(state, op.a).to_bits());
    let b = f64::from_bits(get(state, op.b).to_bits());
    set(state, op.dst, Value::from_bits((a + b).to_bits()));
    state.pc + 1
}

fn op_sub_f64(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = f64::from_bits(get(state, op.a).to_bits());
    let b = f64::from_bits(get(state, op.b).to_bits());
    set(state, op.dst, Value::from_bits((a - b).to_bits()));
    state.pc + 1
}

fn op_mul_f64(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = f64::from_bits(get(state, op.a).to_bits());
    let b = f64::from_bits(get(state, op.b).to_bits());
    set(state, op.dst, Value::from_bits((a * b).to_bits()));
    state.pc + 1
}

// -- Comparisons --

fn op_cmp_lt(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let r = crate::codegen::runtime_fns::wren_cmp_lt(
        get(state, op.a).to_bits(),
        get(state, op.b).to_bits(),
    );
    set(state, op.dst, Value::from_bits(r));
    state.pc + 1
}

fn op_cmp_ge(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let r = crate::codegen::runtime_fns::wren_cmp_ge(
        get(state, op.a).to_bits(),
        get(state, op.b).to_bits(),
    );
    set(state, op.dst, Value::from_bits(r));
    state.pc + 1
}

fn op_cmp_lt_f64(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = f64::from_bits(get(state, op.a).to_bits());
    let b = f64::from_bits(get(state, op.b).to_bits());
    set(state, op.dst, Value::bool(a < b));
    state.pc + 1
}

fn op_cmp_ge_f64(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let a = f64::from_bits(get(state, op.a).to_bits());
    let b = f64::from_bits(get(state, op.b).to_bits());
    set(state, op.dst, Value::bool(a >= b));
    state.pc + 1
}

fn op_cmp_eq(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let r = crate::codegen::runtime_fns::wren_cmp_eq(
        get(state, op.a).to_bits(),
        get(state, op.b).to_bits(),
    );
    set(state, op.dst, Value::from_bits(r));
    state.pc + 1
}

fn op_cmp_ne(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let r = crate::codegen::runtime_fns::wren_cmp_ne(
        get(state, op.a).to_bits(),
        get(state, op.b).to_bits(),
    );
    set(state, op.dst, Value::from_bits(r));
    state.pc + 1
}

// -- Logical --

fn op_not(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let v = get(state, op.a);
    set(state, op.dst, Value::bool(v.is_falsy()));
    state.pc + 1
}

// -- Field access --

fn op_get_field(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let recv = get(state, op.a);
    let idx = op.b as usize;
    let val = unsafe {
        let obj_ptr = recv.as_object().unwrap_unchecked();
        let header = obj_ptr as *const crate::runtime::object::ObjHeader;
        let inst = obj_ptr as *const crate::runtime::object::ObjInstance;
        let fields = (*inst).fields;
        *fields.add(idx)
    };
    set(state, op.dst, val);
    state.pc + 1
}

fn op_set_field(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let recv = get(state, op.a);
    let idx = op.b as usize;
    let val = get(state, op.c as u16);
    unsafe {
        let obj_ptr = recv.as_object().unwrap_unchecked();
        let inst = obj_ptr as *mut crate::runtime::object::ObjInstance;
        let fields = (*inst).fields;
        *fields.add(idx) = val;
    }
    set(state, op.dst, val);
    state.pc + 1
}

// -- Module vars --

fn op_get_module_var(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let idx = op.a as usize;
    let val = if !state.module_vars.is_null() && idx < state.module_var_count as usize {
        Value::from_bits(unsafe { *state.module_vars.add(idx) })
    } else {
        Value::null()
    };
    set(state, op.dst, val);
    state.pc + 1
}

fn op_set_module_var(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let idx = op.a as usize;
    let val = get(state, op.b);
    if !state.module_vars.is_null() && idx < state.module_var_count as usize {
        unsafe { *state.module_vars.add(idx) = val.to_bits() };
    }
    set(state, op.dst, val);
    state.pc + 1
}

// -- Control flow --

fn op_branch(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    op.c as usize // target pc
}

fn op_cond_branch(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    let cond = get(state, op.a);
    if !cond.is_falsy() {
        op.c as usize // true target
    } else {
        op.extra as usize // false target
    }
}

fn op_return(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    state.return_value = Some(get(state, op.a));
    usize::MAX // sentinel: stop execution
}

fn op_return_null(state: &mut ThreadedState, _op: &ThreadedOp) -> usize {
    state.return_value = Some(Value::null());
    usize::MAX
}

// -- Method call (IC fast path) --
// This is the HOT handler. Pre-decoded: no bytecode parsing.

fn op_call(state: &mut ThreadedState, op: &ThreadedOp) -> usize {
    // op.a = recv reg, op.b = method_sym (u16), op.c = ic_idx,
    // op.extra = packed arg register info (up to 4 args)
    let recv = get(state, op.a);
    let method = SymbolId::from_raw(op.b as u32);
    let ic_idx = op.c as usize;
    let argc = (op.extra & 0xFF) as usize;

    // IC fast path
    let ic_table = unsafe { &*state.ic_table };
    if ic_idx < ic_table.len() {
        let ic = &ic_table[ic_idx];
        if ic.kind == 1 && recv.is_object() {
            let obj_ptr = unsafe { recv.as_object().unwrap_unchecked() };
            let recv_class = unsafe {
                (*(obj_ptr as *const crate::runtime::object::ObjHeader)).class as usize
            };
            if recv_class == ic.class {
                let jit_ptr = ic.jit_ptr;
                if !jit_ptr.is_null() {
                    // Direct JIT call — no bytecode decode overhead
                    let recv_bits = recv.to_bits();
                    let result = unsafe {
                        match argc {
                            0 => {
                                let f: extern "C" fn(u64) -> u64 =
                                    std::mem::transmute(jit_ptr);
                                f(recv_bits)
                            }
                            1 => {
                                let a0_reg = ((op.extra >> 8) & 0xFFFF) as u16;
                                let f: extern "C" fn(u64, u64) -> u64 =
                                    std::mem::transmute(jit_ptr);
                                f(recv_bits, get(state, a0_reg).to_bits())
                            }
                            2 => {
                                let a0_reg = ((op.extra >> 8) & 0xFFFF) as u16;
                                let a1_reg = ((op.extra >> 24) & 0xFFFF) as u16;
                                let f: extern "C" fn(u64, u64, u64) -> u64 =
                                    std::mem::transmute(jit_ptr);
                                f(
                                    recv_bits,
                                    get(state, a0_reg).to_bits(),
                                    get(state, a1_reg).to_bits(),
                                )
                            }
                            _ => {
                                let a0_reg = ((op.extra >> 8) & 0xFFFF) as u16;
                                let a1_reg = ((op.extra >> 24) & 0xFFFF) as u16;
                                let a2_reg = ((op.extra >> 40) & 0xFFFF) as u16;
                                let f: extern "C" fn(u64, u64, u64, u64) -> u64 =
                                    std::mem::transmute(jit_ptr);
                                f(
                                    recv_bits,
                                    get(state, a0_reg).to_bits(),
                                    get(state, a1_reg).to_bits(),
                                    get(state, a2_reg).to_bits(),
                                )
                            }
                        }
                    };
                    set(state, op.dst, Value::from_bits(result));
                    return state.pc + 1;
                }
            }
        }
    }

    // Slow path: full wren_call_N dispatch
    let recv_bits = recv.to_bits();
    let method_bits = method.index() as u64;
    let result = unsafe { match argc {
        0 => crate::codegen::runtime_fns::wren_call_0(recv_bits, method_bits),
        1 => {
            let a0 = get(state, ((op.extra >> 8) & 0xFFFF) as u16).to_bits();
            crate::codegen::runtime_fns::wren_call_1(recv_bits, method_bits, a0)
        }
        2 => {
            let a0 = get(state, ((op.extra >> 8) & 0xFFFF) as u16).to_bits();
            let a1 = get(state, ((op.extra >> 24) & 0xFFFF) as u16).to_bits();
            crate::codegen::runtime_fns::wren_call_2(recv_bits, method_bits, a0, a1)
        }
        _ => {
            let a0 = get(state, ((op.extra >> 8) & 0xFFFF) as u16).to_bits();
            let a1 = get(state, ((op.extra >> 24) & 0xFFFF) as u16).to_bits();
            let a2 = get(state, ((op.extra >> 40) & 0xFFFF) as u16).to_bits();
            crate::codegen::runtime_fns::wren_call_3(recv_bits, method_bits, a0, a1, a2)
        }
    } };
    set(state, op.dst, Value::from_bits(result));
    state.pc + 1
}

// -- Noop/placeholder --

fn op_noop(state: &mut ThreadedState, _op: &ThreadedOp) -> usize {
    state.pc + 1
}

fn op_unreachable(_state: &mut ThreadedState, _op: &ThreadedOp) -> usize {
    panic!("hit unreachable in threaded code");
}

// ---------------------------------------------------------------------------
// MIR → ThreadedCode lowering
// ---------------------------------------------------------------------------

/// Lower a MIR function into threaded code for fast interpretation.
pub fn lower_mir_to_threaded(mir: &MirFunction) -> ThreadedCode {
    let mut ops: Vec<ThreadedOp> = Vec::new();
    let mut block_offsets: HashMap<BlockId, usize> = HashMap::new();
    // Deferred branch fixups: (op_index, field=c or extra, target_block)
    let mut fixups: Vec<(usize, bool, BlockId)> = Vec::new(); // (idx, is_extra, block)
    let mut ic_entries: Vec<CallSiteIC> = Vec::new();
    let mut ic_idx: usize = 0;
    let mut max_reg: u16 = 0;

    // First pass: compute block offsets
    let mut offset = 0usize;
    for (block_idx, block) in mir.blocks.iter().enumerate() {
        block_offsets.insert(BlockId(block_idx as u32), offset);
        offset += block.instructions.len();
        offset += 1; // terminator
    }

    // Second pass: emit ops
    for (block_idx, block) in mir.blocks.iter().enumerate() {
        for &(vid, ref inst) in &block.instructions {
            let dst = vid.0 as u16;
            if dst > max_reg {
                max_reg = dst;
            }
            let threaded_op = match inst {
                Instruction::ConstNull => ThreadedOp {
                    handler: op_const_null,
                    dst,
                    a: 0, b: 0, c: 0, extra: 0,
                },
                Instruction::ConstBool(true) => ThreadedOp {
                    handler: op_const_true,
                    dst,
                    a: 0, b: 0, c: 0, extra: 0,
                },
                Instruction::ConstBool(false) => ThreadedOp {
                    handler: op_const_false,
                    dst,
                    a: 0, b: 0, c: 0, extra: 0,
                },
                Instruction::ConstNum(f) => ThreadedOp {
                    handler: op_const_num,
                    dst,
                    a: 0, b: 0, c: 0,
                    extra: Value::num(*f).to_bits(),
                },
                Instruction::ConstI64(i) => ThreadedOp {
                    handler: op_const_num,
                    dst,
                    a: 0, b: 0, c: 0,
                    extra: *i as u64,
                },
                Instruction::Move(src) => ThreadedOp {
                    handler: op_move,
                    dst,
                    a: src.0 as u16, b: 0, c: 0, extra: 0,
                },
                Instruction::BlockParam(idx) => ThreadedOp {
                    handler: op_move,
                    dst,
                    a: *idx, b: 0, c: 0, extra: 0,
                },
                // Arithmetic
                Instruction::Add(a, b) => ThreadedOp {
                    handler: op_add,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::Sub(a, b) => ThreadedOp {
                    handler: op_sub,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::Mul(a, b) => ThreadedOp {
                    handler: op_mul,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::AddF64(a, b) => ThreadedOp {
                    handler: op_add_f64,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::SubF64(a, b) => ThreadedOp {
                    handler: op_sub_f64,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::MulF64(a, b) => ThreadedOp {
                    handler: op_mul_f64,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                // Comparisons
                Instruction::CmpLt(a, b) => ThreadedOp {
                    handler: op_cmp_lt,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::CmpGe(a, b) => ThreadedOp {
                    handler: op_cmp_ge,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::CmpEq(a, b) => ThreadedOp {
                    handler: op_cmp_eq,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::CmpNe(a, b) => ThreadedOp {
                    handler: op_cmp_ne,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::CmpLtF64(a, b) => ThreadedOp {
                    handler: op_cmp_lt_f64,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                Instruction::CmpGeF64(a, b) => ThreadedOp {
                    handler: op_cmp_ge_f64,
                    dst,
                    a: a.0 as u16, b: b.0 as u16, c: 0, extra: 0,
                },
                // Logical
                Instruction::Not(a) => ThreadedOp {
                    handler: op_not,
                    dst,
                    a: a.0 as u16, b: 0, c: 0, extra: 0,
                },
                // Field access
                Instruction::GetField(recv, idx) => ThreadedOp {
                    handler: op_get_field,
                    dst,
                    a: recv.0 as u16, b: *idx, c: 0, extra: 0,
                },
                Instruction::SetField(recv, idx, val) => ThreadedOp {
                    handler: op_set_field,
                    dst,
                    a: recv.0 as u16, b: *idx, c: val.0 as u32, extra: 0,
                },
                // Module vars
                Instruction::GetModuleVar(slot) => ThreadedOp {
                    handler: op_get_module_var,
                    dst,
                    a: *slot, b: 0, c: 0, extra: 0,
                },
                Instruction::SetModuleVar(slot, val) => ThreadedOp {
                    handler: op_set_module_var,
                    dst,
                    a: *slot, b: val.0 as u16, c: 0, extra: 0,
                },
                // Method calls
                Instruction::Call { receiver, method, args } => {
                    let cur_ic = ic_idx;
                    ic_idx += 1;
                    ic_entries.push(CallSiteIC::default());
                    // Pack arg registers into extra: argc(8) | a0(16) | a1(16) | a2(16)
                    let argc = args.len();
                    let mut packed: u64 = argc as u64;
                    for (i, arg) in args.iter().take(3).enumerate() {
                        packed |= (arg.0 as u64) << (8 + i * 16);
                    }
                    ThreadedOp {
                        handler: op_call,
                        dst,
                        a: receiver.0 as u16,
                        b: method.index() as u16,
                        c: cur_ic as u32,
                        extra: packed,
                    }
                }
                // Everything else: noop placeholder (will be expanded)
                _ => ThreadedOp {
                    handler: op_noop,
                    dst,
                    a: 0, b: 0, c: 0, extra: 0,
                },
            };
            ops.push(threaded_op);
        }

        // Terminator
        let term_op = match &block.terminator {
            Terminator::Return(val) => ThreadedOp {
                handler: op_return,
                dst: 0,
                a: val.0 as u16, b: 0, c: 0, extra: 0,
            },
            Terminator::ReturnNull => ThreadedOp {
                handler: op_return_null,
                dst: 0,
                a: 0, b: 0, c: 0, extra: 0,
            },
            Terminator::Branch { target, args: _ } => {
                let idx = ops.len();
                fixups.push((idx, false, *target));
                ThreadedOp {
                    handler: op_branch,
                    dst: 0,
                    a: 0, b: 0,
                    c: 0, // fixup later
                    extra: 0,
                }
            }
            Terminator::CondBranch {
                condition,
                true_target,
                true_args: _,
                false_target,
                false_args: _,
            } => {
                let idx = ops.len();
                fixups.push((idx, false, *true_target));
                fixups.push((idx, true, *false_target));
                ThreadedOp {
                    handler: op_cond_branch,
                    dst: 0,
                    a: condition.0 as u16,
                    b: 0,
                    c: 0,     // true target, fixup
                    extra: 0, // false target, fixup
                }
            }
            Terminator::Unreachable => ThreadedOp {
                handler: op_unreachable,
                dst: 0,
                a: 0, b: 0, c: 0, extra: 0,
            },
        };
        ops.push(term_op);
    }

    // Apply fixups
    for (idx, is_extra, target_block) in fixups {
        let target_pc = block_offsets[&target_block];
        if is_extra {
            ops[idx].extra = target_pc as u64;
        } else {
            ops[idx].c = target_pc as u32;
        }
    }

    ThreadedCode {
        ops,
        consts: Vec::new(),
        ic_table: UnsafeCell::new(ic_entries),
        reg_count: (max_reg as usize) + 1,
    }
}

/// Execute threaded code and return the result.
pub fn execute_threaded(code: &ThreadedCode, args: &[Value], module_vars: *mut u64, module_var_count: u32, vm: *mut u8) -> Value {
    let mut state = ThreadedState {
        regs: vec![Value::null(); code.reg_count.max(args.len())],
        pc: 0,
        ic_table: code.ic_table.get(),
        module_vars,
        module_var_count,
        vm,
        return_value: None,
    };

    // Load arguments into registers (BlockParam slots)
    for (i, arg) in args.iter().enumerate() {
        if i < state.regs.len() {
            state.regs[i] = *arg;
        }
    }

    // Main loop: zero decode, one indirect call per instruction
    loop {
        if state.pc >= code.ops.len() {
            break;
        }
        let op = &code.ops[state.pc];
        let next_pc = (op.handler)(&mut state, op);
        if next_pc == usize::MAX {
            break; // Return
        }
        state.pc = next_pc;
    }

    state.return_value.unwrap_or(Value::null())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{MirFunction, Instruction, Terminator};
    use crate::intern::Interner;

    /// Helper: emit instruction into block, return ValueId
    fn emit(f: &mut MirFunction, bb: BlockId, inst: Instruction) -> ValueId {
        let vid = f.new_value();
        f.block_mut(bb).instructions.push((vid, inst));
        vid
    }

    #[test]
    fn test_simple_return_42() {
        let mut interner = Interner::new();
        let name = interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = emit(&mut f, bb, Instruction::ConstNum(42.0));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let code = lower_mir_to_threaded(&f);
        let result = execute_threaded(&code, &[], std::ptr::null_mut(), 0, std::ptr::null_mut());
        assert_eq!(result.as_num(), Some(42.0));
    }

    #[test]
    fn test_add() {
        let mut interner = Interner::new();
        let name = interner.intern("test_add");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = emit(&mut f, bb, Instruction::ConstNum(10.0));
        let v1 = emit(&mut f, bb, Instruction::ConstNum(32.0));
        let v2 = emit(&mut f, bb, Instruction::Add(v0, v1));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let code = lower_mir_to_threaded(&f);
        let result = execute_threaded(&code, &[], std::ptr::null_mut(), 0, std::ptr::null_mut());
        assert_eq!(result.as_num(), Some(42.0));
    }

    #[test]
    fn test_not() {
        let mut interner = Interner::new();
        let name = interner.intern("test_not");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v0 = emit(&mut f, bb, Instruction::ConstBool(true));
        let v1 = emit(&mut f, bb, Instruction::Not(v0));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let code = lower_mir_to_threaded(&f);
        let result = execute_threaded(&code, &[], std::ptr::null_mut(), 0, std::ptr::null_mut());
        // not(true) = false
        assert!(result.is_falsy());
    }

    #[test]
    fn test_lowering_produces_ops() {
        let mut interner = Interner::new();
        let name = interner.intern("test");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let code = lower_mir_to_threaded(&f);
        assert_eq!(code.ops.len(), 1); // just the ReturnNull terminator
    }
}
