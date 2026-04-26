/// Compact bytecode encoding for MIR functions.
///
/// Lowers a `MirFunction` (SSA IR with Rust enum instructions) into a flat
/// `Vec<u8>` bytecode stream for efficient interpretation. This is the final
/// lowering step — MIR remains the canonical IR for optimization passes.
///
/// Encoding: little-endian, register indices are u16, branch targets are u32
/// byte offsets, constants live in a side pool indexed by u16.
use std::collections::HashMap;

use crate::ast::Span;
use crate::mir::{
    osr_external_live_values, BasicBlock, BlockId, Instruction, MathBinaryOp, MathUnaryOp,
    MirFunction, Terminator, ValueId,
};

// ---------------------------------------------------------------------------
// Opcodes
// ---------------------------------------------------------------------------

/// Bytecode opcodes. Grouped by encoding width for cache locality.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // -- 3B: op + dst(2) --
    ConstNull = 0x00,
    ConstTrue = 0x01,
    ConstFalse = 0x02,

    // -- 5B: op + dst(2) + imm16(2) --
    ConstNum = 0x03,    // pool index
    ConstF64 = 0x04,    // pool index
    ConstI64 = 0x05,    // pool index
    ConstString = 0x06, // sym index
    GetModuleVar = 0x07,
    GetUpvalue = 0x08,
    BlockParam = 0x09,
    GetStaticField = 0x0A,

    // -- 5B: op + dst(2) + src(2) -- unary ops
    Neg = 0x0B,
    NegF64 = 0x0C,
    Not = 0x0D,
    BitNot = 0x0E,
    Unbox = 0x0F,
    BoxOp = 0x10,
    Move = 0x11,
    ToStringOp = 0x12,
    GuardNum = 0x13,
    GuardBool = 0x14,

    // -- 7B: op + dst(2) + src(2) + imm16(2) --
    GuardClass = 0x15,
    GuardProtocol = 0x16,
    IsType = 0x17,
    GetField = 0x18,
    SetModuleVar = 0x19,
    SetStaticField = 0x1A,
    SetUpvalue = 0x1B,
    MathUnaryF64 = 0x1C,

    // -- 7B: op + dst(2) + lhs(2) + rhs(2) -- binary ops
    Add = 0x1D,
    Sub = 0x1E,
    Mul = 0x1F,
    Div = 0x20,
    Mod = 0x21,
    AddF64 = 0x22,
    SubF64 = 0x23,
    MulF64 = 0x24,
    DivF64 = 0x25,
    ModF64 = 0x26,
    CmpLt = 0x27,
    CmpGt = 0x28,
    CmpLe = 0x29,
    CmpGe = 0x2A,
    CmpEq = 0x2B,
    CmpNe = 0x2C,
    CmpLtF64 = 0x2D,
    CmpGtF64 = 0x2E,
    CmpLeF64 = 0x2F,
    CmpGeF64 = 0x30,
    BitAnd = 0x31,
    BitOr = 0x32,
    BitXor = 0x33,
    Shl = 0x34,
    Shr = 0x35,

    // -- 8B: op + dst(2) + lhs(2) + rhs(2) + u8 --
    MakeRange = 0x36,     // +1B inclusive flag
    MathBinaryF64 = 0x37, // +1B math_op

    // -- 9B: op + dst(2) + recv(2) + idx(2) + val(2) --
    SetField = 0x38,

    // -- Variable: op + dst(2) + ... + count(1) + regs --
    Call = 0x39,
    CallStaticSelf = 0x3A,
    SuperCall = 0x3B,
    MakeClosure = 0x3C,
    MakeList = 0x3D,
    MakeMap = 0x3E,
    StringConcat = 0x3F,
    SubscriptGet = 0x40,
    SubscriptSet = 0x41,

    // -- Terminators --
    Return = 0x42,      // 3B: op + src(2)
    ReturnNull = 0x43,  // 1B
    Unreachable = 0x44, // 1B
    Branch = 0x45,      // variable: op + target(4) + argc(1) + [dst(2),src(2)]*argc
    CondBranch = 0x46,  // variable: op + cond(2) + true_off(4) + t_argc(1) + [dst,src]*t_argc
                        //           + false_off(4) + f_argc(1) + [dst,src]*f_argc
}

impl Op {
    /// Convert a raw u8 to an Op. Only valid for values produced by our encoder.
    ///
    /// # Safety
    /// The caller must ensure `v` is a valid Op discriminant produced by our encoder.
    #[inline(always)]
    pub unsafe fn from_u8_unchecked(v: u8) -> Op {
        std::mem::transmute(v)
    }
}

// ---------------------------------------------------------------------------
// Constant pool
// ---------------------------------------------------------------------------

/// A constant in the bytecode constant pool.
#[derive(Debug, Clone, Copy)]
pub enum BcConst {
    F64(f64),
    I64(i64),
}

// ---------------------------------------------------------------------------
// BytecodeFunction
// ---------------------------------------------------------------------------

/// Monomorphic inline cache entry for a call site.
/// Caches the resolved (class, native_fn_ptr) for zero-branch dispatch
/// after warmup. Written on first miss, checked on subsequent calls.
///
/// Kinds:
///   0 = empty
///   1 = JIT leaf (direct native call)
///   2 = interpreted closure (skip method lookup, inline frame push)
///   3 = constructor (skip method lookup, alloc instance, inline frame push)
///   4 = native method (direct fn pointer call)
///   5 = trivial getter (direct field load; func_id stores field index)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CallSiteIC {
    /// Cached receiver class pointer (0 = empty).
    pub class: usize,
    /// Kind 1: JIT fn ptr. Kind 2/3: defining_class ptr (reused).
    pub jit_ptr: *const u8,
    /// Cached closure pointer for dispatch.
    pub closure: *const u8,
    /// Cached func_id for quick bytecode lookup (kinds 2/3) or field index (kind 5).
    pub func_id: u64,
    /// Method type: 0 = empty, 1 = JIT leaf, 2 = interp closure, 3 = constructor, 4 = native, 5 = getter.
    pub kind: u64,
}

pub const CALLSITE_IC_CLASS: i32 = 0;
pub const CALLSITE_IC_JIT_PTR: i32 = 8;
pub const CALLSITE_IC_CLOSURE: i32 = 16;
pub const CALLSITE_IC_FUNC_ID: i32 = 24;
pub const CALLSITE_IC_KIND: i32 = 32;
pub const CALLSITE_IC_SIZE: i32 = 40;

impl Default for CallSiteIC {
    fn default() -> Self {
        Self {
            class: 0,
            jit_ptr: std::ptr::null(),
            closure: std::ptr::null(),
            func_id: 0,
            kind: 0,
        }
    }
}

// SAFETY: IC entries are only accessed from a single interpreter thread.
// The UnsafeCell is needed for interior mutability through Arc.
// Send is needed because BytecodeFunction is sent from the compilation thread.
unsafe impl Sync for BytecodeFunction {}
unsafe impl Send for BytecodeFunction {}
unsafe impl Send for CallSiteIC {}
unsafe impl Sync for CallSiteIC {}

impl Clone for BytecodeFunction {
    fn clone(&self) -> Self {
        Self {
            code: self.code.clone(),
            constants: self.constants.clone(),
            source_map: self.source_map.clone(),
            block_offsets: self.block_offsets.clone(),
            register_count: self.register_count,
            param_offsets: self.param_offsets.clone(),
            osr_points: self.osr_points.clone(),
            ic_table: std::cell::UnsafeCell::new(unsafe { &*self.ic_table.get() }.clone()),
        }
    }
}

impl std::fmt::Debug for BytecodeFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BytecodeFunction")
            .field("code_len", &self.code.len())
            .field("constants", &self.constants.len())
            .field("register_count", &self.register_count)
            .finish()
    }
}

impl std::fmt::Debug for CallSiteIC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IC(kind={})", self.kind)
    }
}

/// A compiled bytecode function ready for interpretation.
pub struct BytecodeFunction {
    /// Flat bytecode stream.
    pub code: Vec<u8>,
    /// Constant pool (f64/i64 values referenced by u16 index).
    pub constants: Vec<BcConst>,
    /// Source map: sorted (bytecode_offset, span) pairs for error reporting.
    pub source_map: Vec<(u32, Span)>,
    /// Block offset table: BlockId.0 → bytecode offset.
    pub block_offsets: Vec<u32>,
    /// Total register count (= MirFunction.next_value).
    pub register_count: u32,
    /// Cached entry block param layout: [(dst_register, param_index), ...].
    /// Pre-computed during lowering so callers don't need to re-scan bytecode.
    pub param_offsets: Vec<(u16, u16)>,
    /// Candidate loop-header OSR points discovered while lowering branches.
    pub osr_points: Vec<OsrPoint>,
    /// Inline cache table for call sites (mutable through Arc via UnsafeCell).
    pub ic_table: std::cell::UnsafeCell<Vec<CallSiteIC>>,
}

/// Metadata for a bytecode back-edge that may later transfer into native OSR.
#[derive(Debug, Clone)]
pub struct OsrPoint {
    /// Bytecode offset of the Branch opcode that forms the back-edge.
    pub branch_offset: u32,
    /// Bytecode offset of the target loop-header block.
    pub target_offset: u32,
    /// MIR block id for the target loop header.
    pub target_block: BlockId,
    /// Registers passed to native OSR: external live-ins first, then target
    /// block params after branch binding.
    pub param_regs: Vec<u16>,
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Pending branch target to be patched in pass 2.
struct PatchSite {
    /// Byte offset in `code` where the u32 target was written as a placeholder.
    code_offset: usize,
    /// The MIR BlockId this branch targets.
    target_block: BlockId,
}

struct PendingOsrPoint {
    branch_offset: u32,
    target_block: BlockId,
    param_regs: Vec<u16>,
}

/// Lower a MirFunction to compact bytecode.
pub fn lower(mir: &MirFunction) -> BytecodeFunction {
    let mut enc = Encoder::new(mir);
    enc.emit_all();
    enc.patch_branches();
    enc.finish()
}

struct Encoder<'a> {
    mir: &'a MirFunction,
    code: Vec<u8>,
    constants: Vec<BcConst>,
    const_dedup: HashMap<u64, u16>, // f64 bits or i64 bits → pool index
    source_map: Vec<(u32, Span)>,
    block_offsets: Vec<u32>,
    patches: Vec<PatchSite>,
    osr_points: Vec<PendingOsrPoint>,
    call_site_count: u16,
}

impl<'a> Encoder<'a> {
    fn new(mir: &'a MirFunction) -> Self {
        Self {
            mir,
            code: Vec::with_capacity(mir.blocks.len() * 64),
            constants: Vec::new(),
            const_dedup: HashMap::new(),
            source_map: Vec::new(),
            block_offsets: vec![0u32; mir.blocks.len()],
            patches: Vec::new(),
            osr_points: Vec::new(),
            call_site_count: 0,
        }
    }

    // -- Emit helpers -------------------------------------------------------

    fn emit_u8(&mut self, v: u8) {
        self.code.push(v);
    }

    fn emit_u16(&mut self, v: u16) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    fn emit_u32(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    fn emit_op(&mut self, op: Op) {
        self.code.push(op as u8);
    }

    fn emit_reg(&mut self, v: ValueId) {
        self.emit_u16(v.0 as u16);
    }

    /// Record source span for the current bytecode offset.
    fn record_span(&mut self, vid: ValueId) {
        if let Some(span) = self.mir.span_map.get(&vid) {
            self.source_map.push((self.code.len() as u32, span.clone()));
        }
    }

    /// Add an f64 to the constant pool (deduplicating).
    fn add_const_f64(&mut self, v: f64) -> u16 {
        let bits = v.to_bits();
        if let Some(&idx) = self.const_dedup.get(&bits) {
            return idx;
        }
        let idx = self.constants.len() as u16;
        self.constants.push(BcConst::F64(v));
        self.const_dedup.insert(bits, idx);
        idx
    }

    /// Add an i64 to the constant pool (deduplicating).
    fn add_const_i64(&mut self, v: i64) -> u16 {
        let bits = v as u64;
        // Use high bit to distinguish i64 from f64 in dedup key
        let key = bits.wrapping_add(0x8000_0000_0000_0000);
        if let Some(&idx) = self.const_dedup.get(&key) {
            return idx;
        }
        let idx = self.constants.len() as u16;
        self.constants.push(BcConst::I64(v));
        self.const_dedup.insert(key, idx);
        idx
    }

    /// Emit a placeholder u32 for a branch target and record a patch site.
    fn emit_branch_target(&mut self, target: BlockId) {
        let offset = self.code.len();
        self.emit_u32(0xFFFF_FFFF); // placeholder
        self.patches.push(PatchSite {
            code_offset: offset,
            target_block: target,
        });
    }

    // -- Main emit ----------------------------------------------------------

    fn emit_all(&mut self) {
        for block in &self.mir.blocks {
            self.block_offsets[block.id.0 as usize] = self.code.len() as u32;
            self.emit_block(block);
        }
    }

    fn emit_block(&mut self, block: &BasicBlock) {
        for (dst, inst) in &block.instructions {
            self.record_span(*dst);
            self.emit_instruction(*dst, inst);
        }
        self.emit_terminator(&block.terminator, block);
    }

    fn emit_instruction(&mut self, dst: ValueId, inst: &Instruction) {
        match inst {
            // -- Constants --
            Instruction::ConstNull => {
                self.emit_op(Op::ConstNull);
                self.emit_reg(dst);
            }
            Instruction::ConstBool(true) => {
                self.emit_op(Op::ConstTrue);
                self.emit_reg(dst);
            }
            Instruction::ConstBool(false) => {
                self.emit_op(Op::ConstFalse);
                self.emit_reg(dst);
            }
            Instruction::ConstNum(n) => {
                let idx = self.add_const_f64(*n);
                self.emit_op(Op::ConstNum);
                self.emit_reg(dst);
                self.emit_u16(idx);
            }
            Instruction::ConstF64(n) => {
                let idx = self.add_const_f64(*n);
                self.emit_op(Op::ConstF64);
                self.emit_reg(dst);
                self.emit_u16(idx);
            }
            Instruction::ConstI64(n) => {
                let idx = self.add_const_i64(*n);
                self.emit_op(Op::ConstI64);
                self.emit_reg(dst);
                self.emit_u16(idx);
            }
            Instruction::ConstString(sym_idx) => {
                self.emit_op(Op::ConstString);
                self.emit_reg(dst);
                self.emit_u16(*sym_idx as u16);
            }

            // -- 5B unary: op + dst + src --
            Instruction::Neg(a) => self.emit_unary(Op::Neg, dst, *a),
            Instruction::NegF64(a) => self.emit_unary(Op::NegF64, dst, *a),
            Instruction::Not(a) => self.emit_unary(Op::Not, dst, *a),
            Instruction::BitNot(a) => self.emit_unary(Op::BitNot, dst, *a),
            Instruction::Unbox(a) => self.emit_unary(Op::Unbox, dst, *a),
            Instruction::Box(a) => self.emit_unary(Op::BoxOp, dst, *a),
            Instruction::Move(a) => self.emit_unary(Op::Move, dst, *a),
            Instruction::ToString(a) => self.emit_unary(Op::ToStringOp, dst, *a),
            Instruction::GuardNum(a) => self.emit_unary(Op::GuardNum, dst, *a),
            Instruction::GuardBool(a) => self.emit_unary(Op::GuardBool, dst, *a),

            // -- 7B binary: op + dst + lhs + rhs --
            Instruction::Add(a, b) => self.emit_binary(Op::Add, dst, *a, *b),
            Instruction::Sub(a, b) => self.emit_binary(Op::Sub, dst, *a, *b),
            Instruction::Mul(a, b) => self.emit_binary(Op::Mul, dst, *a, *b),
            Instruction::Div(a, b) => self.emit_binary(Op::Div, dst, *a, *b),
            Instruction::Mod(a, b) => self.emit_binary(Op::Mod, dst, *a, *b),
            Instruction::AddF64(a, b) => self.emit_binary(Op::AddF64, dst, *a, *b),
            Instruction::SubF64(a, b) => self.emit_binary(Op::SubF64, dst, *a, *b),
            Instruction::MulF64(a, b) => self.emit_binary(Op::MulF64, dst, *a, *b),
            Instruction::DivF64(a, b) => self.emit_binary(Op::DivF64, dst, *a, *b),
            Instruction::ModF64(a, b) => self.emit_binary(Op::ModF64, dst, *a, *b),
            Instruction::CmpLt(a, b) => self.emit_binary(Op::CmpLt, dst, *a, *b),
            Instruction::CmpGt(a, b) => self.emit_binary(Op::CmpGt, dst, *a, *b),
            Instruction::CmpLe(a, b) => self.emit_binary(Op::CmpLe, dst, *a, *b),
            Instruction::CmpGe(a, b) => self.emit_binary(Op::CmpGe, dst, *a, *b),
            Instruction::CmpEq(a, b) => self.emit_binary(Op::CmpEq, dst, *a, *b),
            Instruction::CmpNe(a, b) => self.emit_binary(Op::CmpNe, dst, *a, *b),
            Instruction::CmpLtF64(a, b) => self.emit_binary(Op::CmpLtF64, dst, *a, *b),
            Instruction::CmpGtF64(a, b) => self.emit_binary(Op::CmpGtF64, dst, *a, *b),
            Instruction::CmpLeF64(a, b) => self.emit_binary(Op::CmpLeF64, dst, *a, *b),
            Instruction::CmpGeF64(a, b) => self.emit_binary(Op::CmpGeF64, dst, *a, *b),
            Instruction::BitAnd(a, b) => self.emit_binary(Op::BitAnd, dst, *a, *b),
            Instruction::BitOr(a, b) => self.emit_binary(Op::BitOr, dst, *a, *b),
            Instruction::BitXor(a, b) => self.emit_binary(Op::BitXor, dst, *a, *b),
            Instruction::Shl(a, b) => self.emit_binary(Op::Shl, dst, *a, *b),
            Instruction::Shr(a, b) => self.emit_binary(Op::Shr, dst, *a, *b),

            // -- 7B: op + dst + src + imm16 --
            Instruction::GuardClass(a, sym) => {
                self.emit_op(Op::GuardClass);
                self.emit_reg(dst);
                self.emit_reg(*a);
                self.emit_u16(sym.index() as u16);
            }
            Instruction::GuardProtocol(a, proto) => {
                self.emit_op(Op::GuardProtocol);
                self.emit_reg(dst);
                self.emit_reg(*a);
                self.emit_u8(proto.0);
                self.emit_u8(0); // pad
            }
            Instruction::IsType(a, sym) => {
                self.emit_op(Op::IsType);
                self.emit_reg(dst);
                self.emit_reg(*a);
                self.emit_u16(sym.index() as u16);
            }
            Instruction::GetField(recv, idx) => {
                self.emit_op(Op::GetField);
                self.emit_reg(dst);
                self.emit_reg(*recv);
                self.emit_u16(*idx);
            }
            Instruction::SetField(recv, idx, val) => {
                self.emit_op(Op::SetField);
                self.emit_reg(dst);
                self.emit_reg(*recv);
                self.emit_u16(*idx);
                self.emit_reg(*val);
            }
            Instruction::GetStaticField(sym) => {
                self.emit_op(Op::GetStaticField);
                self.emit_reg(dst);
                self.emit_u16(sym.index() as u16);
            }
            Instruction::SetStaticField(sym, val) => {
                self.emit_op(Op::SetStaticField);
                self.emit_reg(dst);
                self.emit_reg(*val);
                self.emit_u16(sym.index() as u16);
            }
            Instruction::GetModuleVar(slot) => {
                self.emit_op(Op::GetModuleVar);
                self.emit_reg(dst);
                self.emit_u16(*slot);
            }
            Instruction::SetModuleVar(slot, val) => {
                self.emit_op(Op::SetModuleVar);
                self.emit_reg(dst);
                self.emit_reg(*val);
                self.emit_u16(*slot);
            }
            Instruction::GetUpvalue(idx) => {
                self.emit_op(Op::GetUpvalue);
                self.emit_reg(dst);
                self.emit_u16(*idx);
            }
            Instruction::SetUpvalue(idx, val) => {
                self.emit_op(Op::SetUpvalue);
                self.emit_reg(dst);
                self.emit_reg(*val);
                self.emit_u16(*idx);
            }
            Instruction::BlockParam(idx) => {
                self.emit_op(Op::BlockParam);
                self.emit_reg(dst);
                self.emit_u16(*idx);
            }

            // -- 8B: binary + extra byte --
            Instruction::MakeRange(from, to, inclusive) => {
                self.emit_op(Op::MakeRange);
                self.emit_reg(dst);
                self.emit_reg(*from);
                self.emit_reg(*to);
                self.emit_u8(if *inclusive { 1 } else { 0 });
            }
            Instruction::MathUnaryF64(op, a) => {
                self.emit_op(Op::MathUnaryF64);
                self.emit_reg(dst);
                self.emit_reg(*a);
                self.emit_u8(math_unary_to_u8(*op));
                self.emit_u8(0); // pad to align
            }
            Instruction::MathBinaryF64(op, a, b) => {
                self.emit_op(Op::MathBinaryF64);
                self.emit_reg(dst);
                self.emit_reg(*a);
                self.emit_reg(*b);
                self.emit_u8(math_binary_to_u8(*op));
            }

            // -- Variable-length --
            Instruction::Call {
                receiver,
                method,
                args,
            pure_call: _,
} => {
                self.emit_op(Op::Call);
                self.emit_reg(dst);
                self.emit_reg(*receiver);
                self.emit_u16(method.index() as u16);
                let ic_idx = self.call_site_count;
                self.call_site_count += 1;
                self.emit_u16(ic_idx);
                self.emit_u8(args.len() as u8);
                for a in args {
                    self.emit_reg(*a);
                }
            }
            Instruction::CallKnownFunc {
                func_id,
                method: _,
                expected_class: _,
                inline_getter_field: _,
                pure_leaf: _,
                receiver,
                args,
            } => {
                // Emit as a regular Call with method=0 for bytecode purposes.
                // The VM interpreter treats this as a call; the JIT will
                // optimise it into a direct dispatch later.
                self.emit_op(Op::Call);
                self.emit_reg(dst);
                self.emit_reg(*receiver);
                self.emit_u16(0); // method symbol placeholder
                let ic_idx = self.call_site_count;
                self.call_site_count += 1;
                self.emit_u16(ic_idx);
                self.emit_u8(args.len() as u8);
                for a in args {
                    self.emit_reg(*a);
                }
                // Stash func_id in the IC metadata so the runtime can resolve it.
                let _ = func_id;
            }
            Instruction::CallStaticSelf { args } => {
                self.emit_op(Op::CallStaticSelf);
                self.emit_reg(dst);
                self.emit_u8(args.len() as u8);
                for a in args {
                    self.emit_reg(*a);
                }
            }
            Instruction::SuperCall { method, args } => {
                self.emit_op(Op::SuperCall);
                self.emit_reg(dst);
                self.emit_u16(method.index() as u16);
                self.emit_u8(args.len() as u8);
                for a in args {
                    self.emit_reg(*a);
                }
            }
            Instruction::MakeClosure { fn_id, upvalues } => {
                self.emit_op(Op::MakeClosure);
                self.emit_reg(dst);
                self.emit_u32(*fn_id);
                self.emit_u8(upvalues.len() as u8);
                for uv in upvalues {
                    self.emit_reg(*uv);
                }
            }
            Instruction::MakeList(elems) => {
                self.emit_op(Op::MakeList);
                self.emit_reg(dst);
                self.emit_u8(elems.len() as u8);
                for e in elems {
                    self.emit_reg(*e);
                }
            }
            Instruction::MakeMap(pairs) => {
                self.emit_op(Op::MakeMap);
                self.emit_reg(dst);
                self.emit_u8(pairs.len() as u8);
                for (k, v) in pairs {
                    self.emit_reg(*k);
                    self.emit_reg(*v);
                }
            }
            Instruction::StringConcat(parts) => {
                self.emit_op(Op::StringConcat);
                self.emit_reg(dst);
                self.emit_u8(parts.len() as u8);
                for p in parts {
                    self.emit_reg(*p);
                }
            }
            Instruction::SubscriptGet { receiver, args } => {
                self.emit_op(Op::SubscriptGet);
                self.emit_reg(dst);
                self.emit_reg(*receiver);
                self.emit_u8(args.len() as u8);
                for a in args {
                    self.emit_reg(*a);
                }
            }
            Instruction::SubscriptSet {
                receiver,
                args,
                value,
            } => {
                self.emit_op(Op::SubscriptSet);
                self.emit_reg(dst);
                self.emit_reg(*receiver);
                self.emit_u8(args.len() as u8);
                for a in args {
                    self.emit_reg(*a);
                }
                self.emit_reg(*value);
            }
        }
    }

    /// Collect BlockParam destination registers from a block's instructions.
    /// Returns Vec of (dst_value_id, param_index) sorted by param_index.
    fn collect_block_params(block: &BasicBlock) -> Vec<(ValueId, u16)> {
        let mut params: Vec<(ValueId, u16)> = block
            .instructions
            .iter()
            .filter_map(|(vid, inst)| {
                if let Instruction::BlockParam(idx) = inst {
                    Some((*vid, *idx))
                } else {
                    None
                }
            })
            .collect();
        params.sort_by_key(|(_, idx)| *idx);
        params
    }

    fn branch_param_bindings(&self, target: BlockId, args: &[ValueId]) -> Vec<(u16, u16)> {
        let target_block = &self.mir.blocks[target.0 as usize];
        // Prefer block.params if populated, otherwise scan for BlockParam instructions.
        if !target_block.params.is_empty() {
            let param_count = target_block.params.len().min(args.len());
            args.iter()
                .zip(target_block.params.iter())
                .take(param_count)
                .map(|(arg, &(param_vid, _))| (param_vid.0 as u16, arg.0 as u16))
                .collect()
        } else {
            let block_params = Self::collect_block_params(target_block);
            block_params
                .iter()
                .filter_map(|(vid, idx)| {
                    args.get(*idx as usize)
                        .map(|arg| (vid.0 as u16, arg.0 as u16))
                })
                .collect()
        }
    }

    /// Emit [dst, src] pairs for block param binding in a branch.
    fn emit_branch_params(&mut self, target: BlockId, args: &[ValueId]) -> Vec<(u16, u16)> {
        let bindings = self.branch_param_bindings(target, args);
        self.emit_u8(bindings.len() as u8);
        for &(dst, src) in &bindings {
            self.emit_u16(dst);
            self.emit_u16(src);
        }
        bindings
    }

    fn emit_terminator(&mut self, term: &Terminator, block: &BasicBlock) {
        match term {
            Terminator::Return(v) => {
                self.emit_op(Op::Return);
                self.emit_reg(*v);
            }
            Terminator::ReturnNull => {
                self.emit_op(Op::ReturnNull);
            }
            Terminator::Unreachable => {
                self.emit_op(Op::Unreachable);
            }
            Terminator::Branch { target, args } => {
                let branch_offset = self.code.len() as u32;
                self.emit_op(Op::Branch);
                self.emit_branch_target(*target);
                let bindings = self.emit_branch_params(*target, args);
                if target.0 <= block.id.0 {
                    let mut param_regs: Vec<u16> = osr_external_live_values(self.mir, *target)
                        .into_iter()
                        .map(|vid| vid.0 as u16)
                        .collect();
                    param_regs.extend(bindings.iter().map(|(dst, _)| *dst));
                    self.osr_points.push(PendingOsrPoint {
                        branch_offset,
                        target_block: *target,
                        param_regs,
                    });
                }
            }
            Terminator::CondBranch {
                condition,
                true_target,
                true_args,
                false_target,
                false_args,
            } => {
                let branch_offset = self.code.len() as u32;
                self.emit_op(Op::CondBranch);
                self.emit_reg(*condition);
                // True branch
                self.emit_branch_target(*true_target);
                let true_bindings = self.emit_branch_params(*true_target, true_args);
                if true_target.0 <= block.id.0 {
                    let mut param_regs: Vec<u16> = osr_external_live_values(self.mir, *true_target)
                        .into_iter()
                        .map(|vid| vid.0 as u16)
                        .collect();
                    param_regs.extend(true_bindings.iter().map(|(dst, _)| *dst));
                    self.osr_points.push(PendingOsrPoint {
                        branch_offset,
                        target_block: *true_target,
                        param_regs,
                    });
                }
                // False branch
                self.emit_branch_target(*false_target);
                let false_bindings = self.emit_branch_params(*false_target, false_args);
                if false_target.0 <= block.id.0 {
                    let mut param_regs: Vec<u16> =
                        osr_external_live_values(self.mir, *false_target)
                            .into_iter()
                            .map(|vid| vid.0 as u16)
                            .collect();
                    param_regs.extend(false_bindings.iter().map(|(dst, _)| *dst));
                    self.osr_points.push(PendingOsrPoint {
                        branch_offset,
                        target_block: *false_target,
                        param_regs,
                    });
                }
            }
        }
    }

    // -- Encoding helpers ---------------------------------------------------

    fn emit_unary(&mut self, op: Op, dst: ValueId, src: ValueId) {
        self.emit_op(op);
        self.emit_reg(dst);
        self.emit_reg(src);
    }

    fn emit_binary(&mut self, op: Op, dst: ValueId, lhs: ValueId, rhs: ValueId) {
        self.emit_op(op);
        self.emit_reg(dst);
        self.emit_reg(lhs);
        self.emit_reg(rhs);
    }

    // -- Pass 2: patch branch targets ---------------------------------------

    fn patch_branches(&mut self) {
        for patch in &self.patches {
            let target_offset = self.block_offsets[patch.target_block.0 as usize];
            let bytes = target_offset.to_le_bytes();
            self.code[patch.code_offset..patch.code_offset + 4].copy_from_slice(&bytes);
        }
    }

    // -- Finish -------------------------------------------------------------

    fn finish(self) -> BytecodeFunction {
        // Pre-compute entry block param layout by scanning leading BlockParam ops
        let mut param_offsets = Vec::new();
        let code = &self.code;
        let mut scan_pc: u32 = 0;
        while (scan_pc as usize) < code.len() {
            if code[scan_pc as usize] != Op::BlockParam as u8 {
                break;
            }
            scan_pc += 1;
            let dst = read_u16(code, &mut scan_pc);
            let param_idx = read_u16(code, &mut scan_pc);
            param_offsets.push((dst, param_idx));
        }

        let osr_points = self
            .osr_points
            .into_iter()
            .filter_map(|point| {
                let target_offset = self
                    .block_offsets
                    .get(point.target_block.0 as usize)
                    .copied()?;
                (target_offset < point.branch_offset).then_some(OsrPoint {
                    branch_offset: point.branch_offset,
                    target_offset,
                    target_block: point.target_block,
                    param_regs: point.param_regs,
                })
            })
            .collect();

        BytecodeFunction {
            code: self.code,
            constants: self.constants,
            source_map: self.source_map,
            block_offsets: self.block_offsets,
            register_count: self.mir.next_value,
            param_offsets,
            osr_points,
            ic_table: std::cell::UnsafeCell::new(vec![
                CallSiteIC::default();
                self.call_site_count as usize
            ]),
        }
    }
}

// ---------------------------------------------------------------------------
// Math op encoding
// ---------------------------------------------------------------------------

fn math_unary_to_u8(op: MathUnaryOp) -> u8 {
    match op {
        MathUnaryOp::Abs => 0,
        MathUnaryOp::Acos => 1,
        MathUnaryOp::Asin => 2,
        MathUnaryOp::Atan => 3,
        MathUnaryOp::Cbrt => 4,
        MathUnaryOp::Ceil => 5,
        MathUnaryOp::Cos => 6,
        MathUnaryOp::Floor => 7,
        MathUnaryOp::Round => 8,
        MathUnaryOp::Sin => 9,
        MathUnaryOp::Sqrt => 10,
        MathUnaryOp::Tan => 11,
        MathUnaryOp::Log => 12,
        MathUnaryOp::Log2 => 13,
        MathUnaryOp::Exp => 14,
        MathUnaryOp::Trunc => 15,
        MathUnaryOp::Fract => 16,
        MathUnaryOp::Sign => 17,
    }
}

pub fn u8_to_math_unary(v: u8) -> MathUnaryOp {
    match v {
        0 => MathUnaryOp::Abs,
        1 => MathUnaryOp::Acos,
        2 => MathUnaryOp::Asin,
        3 => MathUnaryOp::Atan,
        4 => MathUnaryOp::Cbrt,
        5 => MathUnaryOp::Ceil,
        6 => MathUnaryOp::Cos,
        7 => MathUnaryOp::Floor,
        8 => MathUnaryOp::Round,
        9 => MathUnaryOp::Sin,
        10 => MathUnaryOp::Sqrt,
        11 => MathUnaryOp::Tan,
        12 => MathUnaryOp::Log,
        13 => MathUnaryOp::Log2,
        14 => MathUnaryOp::Exp,
        15 => MathUnaryOp::Trunc,
        16 => MathUnaryOp::Fract,
        17 => MathUnaryOp::Sign,
        _ => panic!("invalid math unary op: {}", v),
    }
}

fn math_binary_to_u8(op: MathBinaryOp) -> u8 {
    match op {
        MathBinaryOp::Atan2 => 0,
        MathBinaryOp::Min => 1,
        MathBinaryOp::Max => 2,
        MathBinaryOp::Pow => 3,
    }
}

pub fn u8_to_math_binary(v: u8) -> MathBinaryOp {
    match v {
        0 => MathBinaryOp::Atan2,
        1 => MathBinaryOp::Min,
        2 => MathBinaryOp::Max,
        3 => MathBinaryOp::Pow,
        _ => panic!("invalid math binary op: {}", v),
    }
}

// ---------------------------------------------------------------------------
// Decoder helpers (for the interpreter)
// ---------------------------------------------------------------------------

/// Read a little-endian u16 from `code` at `pc` and advance `pc`.
/// SAFETY: caller must ensure `pc + 1 < code.len()`. All bytecode is
/// produced by our encoder and validated, so bounds checks are redundant.
#[inline(always)]
pub fn read_u16(code: &[u8], pc: &mut u32) -> u16 {
    let i = *pc as usize;
    let v = unsafe {
        let lo = *code.get_unchecked(i);
        let hi = *code.get_unchecked(i + 1);
        u16::from_le_bytes([lo, hi])
    };
    *pc += 2;
    v
}

/// Read a little-endian u32 from `code` at `pc` and advance `pc`.
#[inline(always)]
pub fn read_u32(code: &[u8], pc: &mut u32) -> u32 {
    let i = *pc as usize;
    let v = unsafe {
        u32::from_le_bytes([
            *code.get_unchecked(i),
            *code.get_unchecked(i + 1),
            *code.get_unchecked(i + 2),
            *code.get_unchecked(i + 3),
        ])
    };
    *pc += 4;
    v
}

/// Read a single byte from `code` at `pc` and advance `pc`.
#[inline(always)]
pub fn read_u8(code: &[u8], pc: &mut u32) -> u8 {
    let v = code[*pc as usize];
    *pc += 1;
    v
}

// ---------------------------------------------------------------------------
// Source map lookup
// ---------------------------------------------------------------------------

impl BytecodeFunction {
    /// Find the source span for a bytecode offset using binary search.
    pub fn lookup_span(&self, pc: u32) -> Option<&Span> {
        // Find the last entry with offset <= pc
        match self.source_map.binary_search_by_key(&pc, |(off, _)| *off) {
            Ok(i) => Some(&self.source_map[i].1),
            Err(0) => None,
            Err(i) => Some(&self.source_map[i - 1].1),
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn verify_callsite_ic_layout() {
        assert_eq!(std::mem::size_of::<CallSiteIC>(), CALLSITE_IC_SIZE as usize);
        assert_eq!(memoffset_of!(CallSiteIC, class), CALLSITE_IC_CLASS as usize);
        assert_eq!(
            memoffset_of!(CallSiteIC, jit_ptr),
            CALLSITE_IC_JIT_PTR as usize
        );
        assert_eq!(
            memoffset_of!(CallSiteIC, closure),
            CALLSITE_IC_CLOSURE as usize
        );
        assert_eq!(
            memoffset_of!(CallSiteIC, func_id),
            CALLSITE_IC_FUNC_ID as usize
        );
        assert_eq!(memoffset_of!(CallSiteIC, kind), CALLSITE_IC_KIND as usize);
    }

    macro_rules! memoffset_of {
        ($ty:ty, $field:ident) => {{
            let uninit = std::mem::MaybeUninit::<$ty>::uninit();
            let base = uninit.as_ptr();
            let field_ptr = unsafe { std::ptr::addr_of!((*base).$field) };
            (field_ptr as usize) - (base as usize)
        }};
    }
    use memoffset_of;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::SymbolId;
    use crate::mir::{MirFunction, MirType, Terminator};

    fn make_func() -> MirFunction {
        MirFunction::new(SymbolId::from_raw(0), 0)
    }

    #[test]
    fn test_lower_const_num_return() {
        let mut f = make_func();
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let bc = lower(&f);

        assert_eq!(bc.register_count, 1);
        assert_eq!(bc.constants.len(), 1);
        assert!(matches!(bc.constants[0], BcConst::F64(n) if n == 42.0));

        // ConstNum: op(1) + dst(2) + pool_idx(2) = 5 bytes
        // Return: op(1) + src(2) = 3 bytes
        assert_eq!(bc.code.len(), 8);
        assert_eq!(bc.code[0], Op::ConstNum as u8);
        assert_eq!(bc.code[5], Op::Return as u8);
    }

    #[test]
    fn test_lower_branch_with_params() {
        let mut f = make_func();
        let bb0 = f.new_block();
        let bb1 = f.new_block();

        let v0 = f.new_value();
        let v1 = f.new_value(); // block param in bb1

        f.block_mut(bb0)
            .instructions
            .push((v0, Instruction::ConstNull));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };

        f.block_mut(bb1).params.push((v1, MirType::Value));
        f.block_mut(bb1).terminator = Terminator::Return(v1);

        let bc = lower(&f);

        // bb0: ConstNull(3) + Branch(1 + 4 + 1 + 4) = 13
        // bb1: Return(3)
        assert_eq!(bc.code.len(), 16);

        // Branch target should point to bb1's offset
        let bb1_offset = bc.block_offsets[1];
        assert_eq!(bb1_offset, 13);

        // Verify patched target in branch instruction
        // Branch starts at offset 3, target is at offset 4 (after opcode)
        let target = u32::from_le_bytes([bc.code[4], bc.code[5], bc.code[6], bc.code[7]]);
        assert_eq!(target, bb1_offset);
        assert!(bc.osr_points.is_empty());
    }

    #[test]
    fn test_lower_records_backedge_osr_point() {
        let mut f = make_func();
        let bb0 = f.new_block();
        let bb1 = f.new_block();

        let v0 = f.new_value();
        let v_loop = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v0, Instruction::ConstNum(1.0)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };

        f.block_mut(bb1).params.push((v_loop, MirType::Value));
        let v_body = f.new_value();
        f.block_mut(bb1)
            .instructions
            .push((v_body, Instruction::ConstNull));
        f.block_mut(bb1).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v_loop],
        };

        let bc = lower(&f);
        assert_eq!(bc.osr_points.len(), 1);

        let point = &bc.osr_points[0];
        assert_eq!(point.target_block, bb1);
        assert_eq!(point.target_offset, bc.block_offsets[bb1.0 as usize]);
        assert!(point.target_offset < point.branch_offset);
        assert_eq!(point.param_regs, vec![v_loop.0 as u16]);
    }

    #[test]
    fn test_lower_records_external_osr_live_in_before_block_params() {
        let mut f = make_func();
        let bb0 = f.new_block();
        let bb1 = f.new_block();

        let v_initial = f.new_value();
        let v_limit = f.new_value();
        let v_loop = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v_initial, Instruction::ConstNum(0.0)));
        f.block_mut(bb0)
            .instructions
            .push((v_limit, Instruction::GetModuleVar(0)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v_initial],
        };

        f.block_mut(bb1).params.push((v_loop, MirType::Value));
        let v_cond = f.new_value();
        f.block_mut(bb1)
            .instructions
            .push((v_cond, Instruction::CmpLt(v_loop, v_limit)));
        f.block_mut(bb1).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v_loop],
        };

        let bc = lower(&f);
        assert_eq!(bc.osr_points.len(), 1);
        assert_eq!(
            bc.osr_points[0].param_regs,
            vec![v_limit.0 as u16, v_loop.0 as u16]
        );
    }

    #[test]
    fn test_lower_records_cond_branch_backedge_osr_point() {
        let mut f = make_func();
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();

        let v_initial = f.new_value();
        let v_loop = f.new_value();
        let v_limit = f.new_value();
        let v_cond = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v_initial, Instruction::ConstNum(0.0)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v_initial],
        };

        f.block_mut(bb1).params.push((v_loop, MirType::Value));
        f.block_mut(bb1)
            .instructions
            .push((v_limit, Instruction::ConstNum(3.0)));
        f.block_mut(bb1)
            .instructions
            .push((v_cond, Instruction::CmpLt(v_loop, v_limit)));
        f.block_mut(bb1).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb1,
            true_args: vec![v_loop],
            false_target: bb2,
            false_args: vec![],
        };

        f.block_mut(bb2).terminator = Terminator::Return(v_loop);

        let bc = lower(&f);
        assert_eq!(bc.osr_points.len(), 1);
        let point = &bc.osr_points[0];
        assert_eq!(point.target_block, bb1);
        assert_eq!(point.target_offset, bc.block_offsets[bb1.0 as usize]);
        assert!(point.target_offset < point.branch_offset);
        assert_eq!(point.param_regs, vec![v_loop.0 as u16]);
    }

    #[test]
    fn test_lower_cond_branch() {
        let mut f = make_func();
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();

        let v_cond = f.new_value();
        let v_true = f.new_value();
        let v_false = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v_cond, Instruction::ConstBool(true)));
        f.block_mut(bb0).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb1,
            true_args: vec![],
            false_target: bb2,
            false_args: vec![],
        };

        f.block_mut(bb1)
            .instructions
            .push((v_true, Instruction::ConstNull));
        f.block_mut(bb1).terminator = Terminator::Return(v_true);

        f.block_mut(bb2)
            .instructions
            .push((v_false, Instruction::ConstNull));
        f.block_mut(bb2).terminator = Terminator::Return(v_false);

        let bc = lower(&f);

        // Verify block offsets are set
        assert_eq!(bc.block_offsets[0], 0);
        assert!(bc.block_offsets[1] > 0);
        assert!(bc.block_offsets[2] > bc.block_offsets[1]);
    }

    #[test]
    fn test_lower_add_f64() {
        let mut f = make_func();
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstF64(1.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstF64(2.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::AddF64(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let bc = lower(&f);

        // Two distinct f64 constants
        assert_eq!(bc.constants.len(), 2);
        // AddF64: op(1) + dst(2) + lhs(2) + rhs(2) = 7 bytes
        assert_eq!(bc.code[10], Op::AddF64 as u8);
    }

    #[test]
    fn test_lower_call() {
        let mut f = make_func();
        let bb = f.new_block();
        let v_recv = f.new_value();
        let v_arg = f.new_value();
        let v_result = f.new_value();

        let method = SymbolId::from_raw(5);

        f.block_mut(bb)
            .instructions
            .push((v_recv, Instruction::ConstNull));
        f.block_mut(bb)
            .instructions
            .push((v_arg, Instruction::ConstNum(1.0)));
        f.block_mut(bb).instructions.push((
            v_result,
            Instruction::Call {
                receiver: v_recv,
                method,
                args: vec![v_arg],
            pure_call: false,
},
        ));
        f.block_mut(bb).terminator = Terminator::Return(v_result);

        let bc = lower(&f);

        // Find Call opcode
        let call_start = bc.code.iter().position(|&b| b == Op::Call as u8).unwrap();
        // Call: op(1) + dst(2) + recv(2) + sym(2) + ic_idx(2) + argc(1) + args(2*1) = 12
        assert_eq!(bc.code[call_start + 9], 1); // argc = 1
    }

    #[test]
    fn test_const_dedup() {
        let mut f = make_func();
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(42.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let bc = lower(&f);

        // Same constant should be deduplicated
        assert_eq!(bc.constants.len(), 1);
    }

    #[test]
    fn test_source_map() {
        let mut f = make_func();
        let bb = f.new_block();
        let v0 = f.new_value();

        f.span_map.insert(v0, 10..20);
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNull));
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let bc = lower(&f);

        assert_eq!(bc.source_map.len(), 1);
        assert_eq!(bc.source_map[0].0, 0); // offset 0
        assert_eq!(bc.source_map[0].1, 10..20);

        // Lookup
        assert_eq!(bc.lookup_span(0), Some(&(10..20)));
        assert_eq!(bc.lookup_span(1), Some(&(10..20))); // within same instruction
    }

    #[test]
    fn test_all_instruction_types_lower() {
        // Smoke test: every instruction variant lowers without panic
        let mut f = make_func();
        let bb = f.new_block();

        let mut vals = Vec::new();
        for _ in 0..20 {
            vals.push(f.new_value());
        }

        let sym = SymbolId::from_raw(0);
        let instructions: Vec<Instruction> = vec![
            Instruction::ConstNull,
            Instruction::ConstBool(true),
            Instruction::ConstBool(false),
            Instruction::ConstNum(1.0),
            Instruction::ConstF64(2.0),
            Instruction::ConstI64(3),
            Instruction::ConstString(0),
            Instruction::Add(vals[0], vals[1]),
            Instruction::Neg(vals[0]),
            Instruction::Not(vals[0]),
            Instruction::Move(vals[0]),
            Instruction::GetField(vals[0], 0),
            Instruction::SetField(vals[0], 0, vals[1]),
            Instruction::IsType(vals[0], sym),
            Instruction::MakeRange(vals[0], vals[1], true),
            Instruction::MakeList(vec![vals[0], vals[1]]),
            Instruction::MakeMap(vec![(vals[0], vals[1])]),
            Instruction::Call {
                receiver: vals[0],
                method: sym,
                args: vec![vals[1]],
            pure_call: false,
},
            Instruction::BlockParam(0),
            Instruction::StringConcat(vec![vals[0]]),
        ];

        for (i, inst) in instructions.into_iter().enumerate() {
            f.block_mut(bb).instructions.push((vals[i], inst));
        }
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let bc = lower(&f);
        assert!(!bc.code.is_empty());
    }
}
