/// Platform-agnostic machine IR for WrenLift code generation.
///
/// A RISC-style 3-address instruction set using virtual registers.
/// Designed to lower cleanly to both ARM64 and x86_64 (and WASM).
///
/// Key design choices:
/// - Two register files: GP (integers, pointers, NaN-boxed Values) and FP (unboxed f64)
/// - 3-address: `dst = op lhs, rhs` (ARM64-native; x86_64 lowered via mov+op)
/// - Explicit results from comparisons via `CSet` (no implicit flag registers)
/// - Load/store architecture: memory only via `Ldr`/`Str`
/// - Virtual registers resolved by linear-scan register allocator before emission

pub mod aarch64;
pub mod cfg;
pub mod regalloc;
pub mod runtime_fns;
pub mod wasm;
pub mod x86_64;

use std::fmt;

// ---------------------------------------------------------------------------
// Virtual Registers
// ---------------------------------------------------------------------------

/// Register class — general-purpose, floating-point, or vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// 64-bit general purpose (integers, pointers, NaN-boxed Values).
    Gp,
    /// 64-bit floating point scalar (unboxed f64).
    Fp,
    /// SIMD vector register (128-bit or 256-bit packed f64).
    Vec,
}

/// A virtual register. Allocated by the register allocator to a physical register or spill slot.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg {
    /// Register index (unique within its class).
    pub index: u32,
    /// Which register file this belongs to.
    pub class: RegClass,
}

impl VReg {
    #[inline]
    pub fn gp(index: u32) -> Self {
        Self {
            index,
            class: RegClass::Gp,
        }
    }

    #[inline]
    pub fn fp(index: u32) -> Self {
        Self {
            index,
            class: RegClass::Fp,
        }
    }

    #[inline]
    pub fn vec(index: u32) -> Self {
        Self {
            index,
            class: RegClass::Vec,
        }
    }

    pub fn is_gp(self) -> bool {
        self.class == RegClass::Gp
    }

    pub fn is_fp(self) -> bool {
        self.class == RegClass::Fp
    }

    pub fn is_vec(self) -> bool {
        self.class == RegClass::Vec
    }
}

impl fmt::Debug for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.class {
            RegClass::Gp => write!(f, "r{}", self.index),
            RegClass::Fp => write!(f, "d{}", self.index),
            RegClass::Vec => write!(f, "v{}", self.index),
        }
    }
}

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

/// A branch target label. Resolved to an offset during emission.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub u32);

impl fmt::Debug for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Memory operand
// ---------------------------------------------------------------------------

/// A memory address: `[base + offset]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mem {
    pub base: VReg,
    pub offset: i32,
}

impl Mem {
    pub fn new(base: VReg, offset: i32) -> Self {
        Self { base, offset }
    }
}

impl fmt::Display for Mem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.offset == 0 {
            write!(f, "[{}]", self.base)
        } else if self.offset > 0 {
            write!(f, "[{} + {}]", self.base, self.offset)
        } else {
            write!(f, "[{} - {}]", self.base, -self.offset)
        }
    }
}

// ---------------------------------------------------------------------------
// Vector width
// ---------------------------------------------------------------------------

/// SIMD vector width. Determines lane count for packed f64 operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecWidth {
    /// 128-bit: 2× f64. Universal: NEON, SSE2, WASM SIMD.
    V128,
    /// 256-bit: 4× f64. x86_64 AVX/AVX2.
    V256,
}

impl VecWidth {
    /// Number of f64 lanes.
    pub fn lanes(self) -> usize {
        match self {
            VecWidth::V128 => 2,
            VecWidth::V256 => 4,
        }
    }

    /// Width in bytes.
    pub fn bytes(self) -> usize {
        match self {
            VecWidth::V128 => 16,
            VecWidth::V256 => 32,
        }
    }
}

impl fmt::Display for VecWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VecWidth::V128 => write!(f, "v128"),
            VecWidth::V256 => write!(f, "v256"),
        }
    }
}

// ---------------------------------------------------------------------------
// Condition codes
// ---------------------------------------------------------------------------

/// Condition code for comparisons and conditional branches.
/// Named after ARM64 convention; maps 1:1 to x86_64 condition codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    /// Equal (Z=1) — x86: je/sete
    Eq,
    /// Not equal (Z=0) — x86: jne/setne
    Ne,
    /// Signed less than (N!=V) — x86: jl/setl
    Lt,
    /// Signed less or equal (Z=1 or N!=V) — x86: jle/setle
    Le,
    /// Signed greater than (Z=0 and N=V) — x86: jg/setg
    Gt,
    /// Signed greater or equal (N=V) — x86: jge/setge
    Ge,
    /// Unsigned below (C=0) — x86: jb/setb
    Below,
    /// Unsigned below or equal (C=0 or Z=1) — x86: jbe/setbe
    BelowEq,
    /// Unsigned above (C=1 and Z=0) — x86: ja/seta
    Above,
    /// Unsigned above or equal (C=1) — x86: jae/setae
    AboveEq,
}

impl Cond {
    /// Return the inverted condition.
    pub fn invert(self) -> Self {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Lt => Cond::Ge,
            Cond::Le => Cond::Gt,
            Cond::Gt => Cond::Le,
            Cond::Ge => Cond::Lt,
            Cond::Below => Cond::AboveEq,
            Cond::BelowEq => Cond::Above,
            Cond::Above => Cond::BelowEq,
            Cond::AboveEq => Cond::Below,
        }
    }
}

// ---------------------------------------------------------------------------
// Machine Instruction ADT
// ---------------------------------------------------------------------------

/// Platform-agnostic machine instruction.
///
/// 3-address RISC form. Each variant maps to 1-3 native instructions on
/// both ARM64 and x86_64.
#[derive(Debug, Clone)]
pub enum MachInst {
    // =====================================================================
    // Data Movement
    // =====================================================================

    /// Load a 64-bit immediate into a GP register.
    /// ARM64: up to 4x `movz`/`movk`; x86_64: `mov r64, imm64`
    LoadImm { dst: VReg, bits: u64 },

    /// Load an f64 constant into an FP register.
    /// Emitted as: load constant from pool, or GP transfer.
    LoadFpImm { dst: VReg, value: f64 },

    /// GP → GP register move.
    Mov { dst: VReg, src: VReg },

    /// FP → FP register move.
    FMov { dst: VReg, src: VReg },

    /// Bitwise transfer GP → FP (e.g. NaN-boxed Value → f64 for unboxing).
    /// ARM64: `fmov d0, x0`; x86_64: `movq xmm0, rax`
    BitcastGpToFp { dst: VReg, src: VReg },

    /// Bitwise transfer FP → GP (e.g. f64 → NaN-boxed Value for boxing).
    /// ARM64: `fmov x0, d0`; x86_64: `movq rax, xmm0`
    BitcastFpToGp { dst: VReg, src: VReg },

    // =====================================================================
    // Integer Arithmetic
    // =====================================================================

    /// `dst = lhs + rhs`
    IAdd { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = src + imm` (small immediates, common for stack offsets)
    IAddImm { dst: VReg, src: VReg, imm: i32 },

    /// `dst = lhs - rhs`
    ISub { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = lhs * rhs`
    IMul { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = lhs / rhs` (signed)
    IDiv { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = acc - (lhs * rhs)` — multiply-subtract for remainder.
    /// ARM64: `msub`; x86_64: `imul` + `sub` (or use `idiv` which gives both)
    IMulSub {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
        acc: VReg,
    },

    /// `dst = -src` (integer negate)
    INeg { dst: VReg, src: VReg },

    // =====================================================================
    // Bitwise
    // =====================================================================

    /// `dst = lhs & rhs`
    And { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = src & imm`
    AndImm { dst: VReg, src: VReg, imm: u64 },

    /// `dst = lhs | rhs`
    Or { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = src | imm`
    OrImm { dst: VReg, src: VReg, imm: u64 },

    /// `dst = lhs ^ rhs`
    Xor { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = ~src` (bitwise NOT)
    Not { dst: VReg, src: VReg },

    /// `dst = lhs << rhs` (logical shift left)
    Shl { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = lhs >> rhs` (arithmetic shift right, sign-extending)
    Sar { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = lhs >>> rhs` (logical shift right, zero-extending)
    Shr { dst: VReg, lhs: VReg, rhs: VReg },

    // =====================================================================
    // Floating-Point Arithmetic
    // =====================================================================

    /// `dst = lhs + rhs` (f64)
    FAdd { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = lhs - rhs` (f64)
    FSub { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = lhs * rhs` (f64)
    FMul { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = lhs / rhs` (f64)
    FDiv { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = -src` (f64 negate)
    FNeg { dst: VReg, src: VReg },

    /// `dst = |src|` (f64 absolute value)
    /// ARM64: `fabs`; x86_64: AND with sign-bit mask
    FAbs { dst: VReg, src: VReg },

    /// `dst = sqrt(src)` (f64 square root)
    /// ARM64: `fsqrt`; x86_64: `sqrtsd`
    FSqrt { dst: VReg, src: VReg },

    /// `dst = floor(src)` (round toward −∞)
    /// ARM64: `frintm`; x86_64: `roundsd` imm=0x09 (SSE4.1)
    FFloor { dst: VReg, src: VReg },

    /// `dst = ceil(src)` (round toward +∞)
    /// ARM64: `frintp`; x86_64: `roundsd` imm=0x0A (SSE4.1)
    FCeil { dst: VReg, src: VReg },

    /// `dst = round(src)` (round to nearest, ties to even)
    /// ARM64: `frintn`; x86_64: `roundsd` imm=0x08 (SSE4.1)
    FRound { dst: VReg, src: VReg },

    /// `dst = trunc(src)` (round toward zero)
    /// ARM64: `frintz`; x86_64: `roundsd` imm=0x0B (SSE4.1)
    FTrunc { dst: VReg, src: VReg },

    /// `dst = min(lhs, rhs)` (f64 minimum)
    /// ARM64: `fmin`; x86_64: `minsd`
    FMin { dst: VReg, lhs: VReg, rhs: VReg },

    /// `dst = max(lhs, rhs)` (f64 maximum)
    /// ARM64: `fmax`; x86_64: `maxsd`
    FMax { dst: VReg, lhs: VReg, rhs: VReg },

    // =====================================================================
    // Fused Multiply-Add (scalar)
    // =====================================================================

    /// `dst = (a * b) + c` — single rounding, higher precision than separate mul+add.
    /// ARM64: `fmadd`; x86_64: `vfmadd231sd` (FMA3)
    FMAdd {
        dst: VReg,
        a: VReg,
        b: VReg,
        c: VReg,
    },

    /// `dst = (a * b) - c`
    /// ARM64: `fmsub`; x86_64: `vfmsub231sd`
    FMSub {
        dst: VReg,
        a: VReg,
        b: VReg,
        c: VReg,
    },

    /// `dst = -(a * b) + c` (negated multiply-add)
    /// ARM64: `fnmadd`; x86_64: `vfnmadd231sd`
    FNMAdd {
        dst: VReg,
        a: VReg,
        b: VReg,
        c: VReg,
    },

    /// `dst = -(a * b) - c`
    /// ARM64: `fnmsub`; x86_64: `vfnmsub231sd`
    FNMSub {
        dst: VReg,
        a: VReg,
        b: VReg,
        c: VReg,
    },

    // =====================================================================
    // SIMD / Vector (packed f64)
    // =====================================================================
    //
    // Operates on `Vec` class registers holding `width` packed f64 lanes:
    //   V128 = 2× f64 (NEON, SSE2, WASM SIMD)
    //   V256 = 4× f64 (AVX/AVX2)
    //
    // The lane count is implicit from VecWidth: lanes = width_bytes / 8.

    /// Load packed f64 vector from aligned memory.
    /// ARM64: `ld1 {v0.2d}, [x0]`; x86_64: `vmovapd ymm0, [rax]`
    VLoad {
        dst: VReg,
        mem: Mem,
        width: VecWidth,
    },

    /// Store packed f64 vector to aligned memory.
    VStore {
        src: VReg,
        mem: Mem,
        width: VecWidth,
    },

    /// Packed f64 add: `dst[i] = lhs[i] + rhs[i]`
    VFAdd {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
        width: VecWidth,
    },

    /// Packed f64 subtract.
    VFSub {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
        width: VecWidth,
    },

    /// Packed f64 multiply.
    VFMul {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
        width: VecWidth,
    },

    /// Packed f64 divide.
    VFDiv {
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
        width: VecWidth,
    },

    /// Packed fused multiply-add: `dst[i] = a[i] * b[i] + c[i]`
    /// ARM64: `fmla v0.2d, v1.2d, v2.2d`; x86_64: `vfmadd231pd`
    VFMAdd {
        dst: VReg,
        a: VReg,
        b: VReg,
        c: VReg,
        width: VecWidth,
    },

    /// Broadcast scalar f64 to all lanes: `dst[0..N] = src`
    /// ARM64: `dup v0.2d, d1`; x86_64: `vbroadcastsd`
    VBroadcast {
        dst: VReg,
        src: VReg,
        width: VecWidth,
    },

    /// Extract scalar f64 from a specific lane.
    /// ARM64: `mov d0, v1.d[lane]`; x86_64: depends on lane
    VExtractLane {
        dst: VReg,
        src: VReg,
        lane: u8,
    },

    /// Insert scalar f64 into a specific lane (other lanes unchanged).
    VInsertLane {
        dst: VReg,
        src: VReg,
        lane: u8,
        val: VReg,
    },

    /// Packed f64 negate: `dst[i] = -src[i]`
    VFNeg {
        dst: VReg,
        src: VReg,
        width: VecWidth,
    },

    /// Horizontal add of all lanes into a scalar: `dst = sum(src[0..N])`
    /// Useful for reductions (dot products, array sums).
    VReduceAdd {
        dst: VReg,
        src: VReg,
        width: VecWidth,
    },

    // =====================================================================
    // Conversions
    // =====================================================================

    /// Truncate f64 → i64 (for bitwise ops on Wren numbers).
    /// ARM64: `fcvtzs`; x86_64: `cvttsd2si`
    FCvtToI64 { dst: VReg, src: VReg },

    /// Convert i64 → f64.
    /// ARM64: `scvtf`; x86_64: `cvtsi2sd`
    I64CvtToF { dst: VReg, src: VReg },

    // =====================================================================
    // Comparison
    // =====================================================================

    /// Compare two GP registers. Sets internal flags.
    /// Must be immediately followed by `CSet` or `BCond`.
    ICmp { lhs: VReg, rhs: VReg },

    /// Compare a GP register against a 64-bit immediate.
    ICmpImm { lhs: VReg, imm: u64 },

    /// Compare two FP registers. Sets internal flags.
    FCmp { lhs: VReg, rhs: VReg },

    /// Materialize a condition flag into a GP register (0 or 1).
    /// ARM64: `cset`; x86_64: `setCC` + `movzx`
    CSet { dst: VReg, cond: Cond },

    // =====================================================================
    // Memory
    // =====================================================================

    /// Load 64-bit value from memory into a GP register.
    Ldr { dst: VReg, mem: Mem },

    /// Store 64-bit value from GP register to memory.
    Str { src: VReg, mem: Mem },

    /// Load 64-bit f64 from memory into an FP register.
    FLdr { dst: VReg, mem: Mem },

    /// Store 64-bit f64 from FP register to memory.
    FStr { src: VReg, mem: Mem },

    // =====================================================================
    // Control Flow
    // =====================================================================

    /// Unconditional branch.
    Jmp { target: Label },

    /// Conditional branch on the last comparison's flags.
    JmpIf { cond: Cond, target: Label },

    /// Branch if GP register is zero.
    /// ARM64: `cbz`; x86_64: `test r,r` + `jz`
    JmpZero { src: VReg, target: Label },

    /// Branch if GP register is non-zero.
    JmpNonZero { src: VReg, target: Label },

    /// Test a single bit and branch if zero.
    /// ARM64: `tbz`; x86_64: `bt` + `jnc`
    TestBitJmpZero {
        src: VReg,
        bit: u8,
        target: Label,
    },

    /// Test a single bit and branch if non-zero.
    TestBitJmpNonZero {
        src: VReg,
        bit: u8,
        target: Label,
    },

    // =====================================================================
    // Calls & Returns
    // =====================================================================

    /// Indirect call through a GP register holding a function pointer.
    /// ARM64: `blr`; x86_64: `call *reg`
    CallInd { target: VReg },

    /// Direct call to a named runtime function (resolved at link/patch time).
    /// The caller is responsible for setting up arguments in the correct ABI
    /// registers beforehand.
    CallRuntime {
        name: &'static str,
        /// Argument VRegs — used by regalloc for liveness, not for emission.
        args: Vec<VReg>,
        /// Result VReg (if any).
        ret: Option<VReg>,
    },

    /// Return from function. The return value should already be in the
    /// ABI return register.
    Ret,

    // =====================================================================
    // Stack Frame
    // =====================================================================

    /// Function prologue: save frame pointer, allocate `frame_size` bytes.
    /// ARM64: `stp x29, x30, [sp, #-frame_size]!` + `mov x29, sp`
    /// x86_64: `push rbp` + `mov rbp, rsp` + `sub rsp, frame_size`
    Prologue { frame_size: u32 },

    /// Function epilogue: deallocate stack, restore frame pointer.
    Epilogue { frame_size: u32 },

    /// Push a GP register onto the stack (callee-save spill).
    Push { src: VReg },

    /// Pop from the stack into a GP register (callee-save restore).
    Pop { dst: VReg },

    // =====================================================================
    // Pseudo-instructions
    // =====================================================================

    /// Marks a position in the instruction stream.
    /// Resolved to an offset during emission; emits no code.
    DefLabel(Label),

    /// No operation. May be used for alignment or placeholder.
    Nop,

    /// Trap / unreachable. Causes a fault if reached.
    /// ARM64: `brk #1`; x86_64: `ud2`
    Trap,

    /// Parallel copy — represents the simultaneous binding of block
    /// parameters from branch arguments (SSA phi resolution).
    ///
    /// Each `(dst, src)` pair is logically executed at the same instant,
    /// so `[(r0, r1), (r1, r0)]` is a valid swap. Must be resolved into
    /// sequential moves (with cycle-breaking temps) before emission.
    ///
    /// Call `resolve_parallel_copies()` on the MachFunc to expand these
    /// into safe sequential `Mov`/`FMov` instructions.
    ParallelCopy { copies: Vec<(VReg, VReg)> },
}

impl MachInst {
    /// All virtual registers read by this instruction.
    pub fn uses(&self) -> Vec<VReg> {
        use MachInst::*;
        match self {
            // No uses
            LoadImm { .. }
            | LoadFpImm { .. }
            | Prologue { .. }
            | Epilogue { .. }
            | Jmp { .. }
            | DefLabel(_)
            | Nop
            | Trap
            | Ret => vec![],

            // Single use
            Mov { src, .. }
            | FMov { src, .. }
            | BitcastGpToFp { src, .. }
            | BitcastFpToGp { src, .. }
            | INeg { src, .. }
            | Not { src, .. }
            | FNeg { src, .. }
            | FAbs { src, .. }
            | FSqrt { src, .. }
            | FFloor { src, .. }
            | FCeil { src, .. }
            | FRound { src, .. }
            | FTrunc { src, .. }
            | FCvtToI64 { src, .. }
            | I64CvtToF { src, .. }
            | JmpZero { src, .. }
            | JmpNonZero { src, .. }
            | TestBitJmpZero { src, .. }
            | TestBitJmpNonZero { src, .. }
            | CallInd { target: src, .. }
            | Push { src } => vec![*src],

            Pop { .. } => vec![],

            IAddImm { src, .. } | AndImm { src, .. } | OrImm { src, .. } => vec![*src],

            ICmpImm { lhs, .. } => vec![*lhs],

            // Two uses
            IAdd { lhs, rhs, .. }
            | ISub { lhs, rhs, .. }
            | IMul { lhs, rhs, .. }
            | IDiv { lhs, rhs, .. }
            | And { lhs, rhs, .. }
            | Or { lhs, rhs, .. }
            | Xor { lhs, rhs, .. }
            | Shl { lhs, rhs, .. }
            | Sar { lhs, rhs, .. }
            | Shr { lhs, rhs, .. }
            | FAdd { lhs, rhs, .. }
            | FSub { lhs, rhs, .. }
            | FMul { lhs, rhs, .. }
            | FDiv { lhs, rhs, .. }
            | FMin { lhs, rhs, .. }
            | FMax { lhs, rhs, .. }
            | ICmp { lhs, rhs }
            | FCmp { lhs, rhs }
            | VFAdd { lhs, rhs, .. }
            | VFSub { lhs, rhs, .. }
            | VFMul { lhs, rhs, .. }
            | VFDiv { lhs, rhs, .. } => vec![*lhs, *rhs],

            // Three uses
            IMulSub {
                lhs, rhs, acc, ..
            } => vec![*lhs, *rhs, *acc],
            FMAdd { a, b, c, .. }
            | FMSub { a, b, c, .. }
            | FNMAdd { a, b, c, .. }
            | FNMSub { a, b, c, .. }
            | VFMAdd { a, b, c, .. } => vec![*a, *b, *c],

            // Vector single-source
            VBroadcast { src, .. }
            | VFNeg { src, .. }
            | VReduceAdd { src, .. }
            | VExtractLane { src, .. } => vec![*src],

            // Vector insert: src vec + scalar val
            VInsertLane { src, val, .. } => vec![*src, *val],

            // Memory: base is a use
            Ldr { mem, .. } | FLdr { mem, .. } | VLoad { mem, .. } => vec![mem.base],
            Str { src, mem } => vec![*src, mem.base],
            FStr { src, mem } | VStore { src, mem, .. } => vec![*src, mem.base],

            // Conditional ops use nothing (flags are implicit)
            CSet { .. } | JmpIf { .. } => vec![],

            // Runtime calls
            CallRuntime { args, .. } => args.clone(),

            // Parallel copy: all sources are uses
            ParallelCopy { copies } => copies.iter().map(|(_, src)| *src).collect(),
        }
    }

    /// The virtual register defined (written) by this instruction, if any.
    pub fn def(&self) -> Option<VReg> {
        use MachInst::*;
        match self {
            LoadImm { dst, .. }
            | LoadFpImm { dst, .. }
            | Mov { dst, .. }
            | FMov { dst, .. }
            | BitcastGpToFp { dst, .. }
            | BitcastFpToGp { dst, .. }
            | IAdd { dst, .. }
            | IAddImm { dst, .. }
            | ISub { dst, .. }
            | IMul { dst, .. }
            | IDiv { dst, .. }
            | IMulSub { dst, .. }
            | INeg { dst, .. }
            | And { dst, .. }
            | AndImm { dst, .. }
            | Or { dst, .. }
            | OrImm { dst, .. }
            | Xor { dst, .. }
            | Not { dst, .. }
            | Shl { dst, .. }
            | Sar { dst, .. }
            | Shr { dst, .. }
            | FAdd { dst, .. }
            | FSub { dst, .. }
            | FMul { dst, .. }
            | FDiv { dst, .. }
            | FNeg { dst, .. }
            | FAbs { dst, .. }
            | FSqrt { dst, .. }
            | FFloor { dst, .. }
            | FCeil { dst, .. }
            | FRound { dst, .. }
            | FTrunc { dst, .. }
            | FMin { dst, .. }
            | FMax { dst, .. }
            | FMAdd { dst, .. }
            | FMSub { dst, .. }
            | FNMAdd { dst, .. }
            | FNMSub { dst, .. }
            | VLoad { dst, .. }
            | VFAdd { dst, .. }
            | VFSub { dst, .. }
            | VFMul { dst, .. }
            | VFDiv { dst, .. }
            | VFMAdd { dst, .. }
            | VBroadcast { dst, .. }
            | VExtractLane { dst, .. }
            | VInsertLane { dst, .. }
            | VFNeg { dst, .. }
            | VReduceAdd { dst, .. }
            | FCvtToI64 { dst, .. }
            | I64CvtToF { dst, .. }
            | CSet { dst, .. }
            | Ldr { dst, .. }
            | FLdr { dst, .. }
            | Pop { dst } => Some(*dst),

            CallRuntime { ret, .. } => *ret,

            // ParallelCopy defines multiple — return None; regalloc handles specially.
            ParallelCopy { .. } => None,

            _ => None,
        }
    }

    /// All virtual registers defined by this instruction (plural for ParallelCopy).
    pub fn defs(&self) -> Vec<VReg> {
        if let MachInst::ParallelCopy { copies } = self {
            copies.iter().map(|(dst, _)| *dst).collect()
        } else {
            self.def().into_iter().collect()
        }
    }

    /// Is this a branch or return (terminator)?
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            MachInst::Jmp { .. }
                | MachInst::JmpIf { .. }
                | MachInst::JmpZero { .. }
                | MachInst::JmpNonZero { .. }
                | MachInst::TestBitJmpZero { .. }
                | MachInst::TestBitJmpNonZero { .. }
                | MachInst::Ret
                | MachInst::Trap
        )
    }

    /// Does this instruction have side effects beyond defining its result?
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            MachInst::Str { .. }
                | MachInst::FStr { .. }
                | MachInst::VStore { .. }
                | MachInst::CallInd { .. }
                | MachInst::CallRuntime { .. }
                | MachInst::Push { .. }
                | MachInst::Pop { .. }
                | MachInst::Prologue { .. }
                | MachInst::Epilogue { .. }
                | MachInst::Trap
        ) || self.is_terminator()
    }
}

// ---------------------------------------------------------------------------
// Machine function
// ---------------------------------------------------------------------------

/// A function's worth of machine instructions with virtual registers.
pub struct MachFunc {
    pub name: String,
    pub insts: Vec<MachInst>,
    pub frame_size: u32,
    next_gp: u32,
    next_fp: u32,
    next_vec: u32,
    next_label: u32,
}

impl MachFunc {
    pub fn new(name: String) -> Self {
        Self {
            name,
            insts: Vec::new(),
            frame_size: 0,
            next_gp: 0,
            next_fp: 0,
            next_vec: 0,
            next_label: 0,
        }
    }

    /// Allocate a fresh GP virtual register.
    pub fn new_gp(&mut self) -> VReg {
        let r = VReg::gp(self.next_gp);
        self.next_gp += 1;
        r
    }

    /// Allocate a fresh FP virtual register.
    pub fn new_fp(&mut self) -> VReg {
        let r = VReg::fp(self.next_fp);
        self.next_fp += 1;
        r
    }

    /// Allocate a fresh vector virtual register.
    pub fn new_vec(&mut self) -> VReg {
        let r = VReg::vec(self.next_vec);
        self.next_vec += 1;
        r
    }

    /// Allocate a fresh label.
    pub fn new_label(&mut self) -> Label {
        let l = Label(self.next_label);
        self.next_label += 1;
        l
    }

    pub fn emit(&mut self, inst: MachInst) {
        self.insts.push(inst);
    }

    pub fn num_gp_vregs(&self) -> u32 {
        self.next_gp
    }

    pub fn num_fp_vregs(&self) -> u32 {
        self.next_fp
    }

    pub fn num_vec_vregs(&self) -> u32 {
        self.next_vec
    }

    /// Pretty-print the machine function.
    pub fn display(&self) -> String {
        let mut out = format!("mach_func {}:\n", self.name);
        for inst in &self.insts {
            match inst {
                MachInst::DefLabel(l) => out.push_str(&format!("{}:\n", l)),
                _ => out.push_str(&format!("    {}\n", format_inst(inst))),
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Physical registers (target-specific but shared definitions)
// ---------------------------------------------------------------------------

/// Physical register index for a specific target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysReg {
    pub class: RegClass,
    pub hw_enc: u8,
}

impl PhysReg {
    pub const fn gp(hw_enc: u8) -> Self {
        Self {
            class: RegClass::Gp,
            hw_enc,
        }
    }
    pub const fn fp(hw_enc: u8) -> Self {
        Self {
            class: RegClass::Fp,
            hw_enc,
        }
    }
}

/// ABI calling convention information for a target.
pub struct AbiInfo {
    /// GP registers used for passing integer/pointer arguments.
    pub gp_arg_regs: &'static [PhysReg],
    /// FP registers used for passing f64 arguments.
    pub fp_arg_regs: &'static [PhysReg],
    /// GP register for return value.
    pub gp_ret: PhysReg,
    /// FP register for return value.
    pub fp_ret: PhysReg,
    /// Caller-saved (volatile) GP registers.
    pub gp_caller_saved: &'static [PhysReg],
    /// Callee-saved (non-volatile) GP registers.
    pub gp_callee_saved: &'static [PhysReg],
    /// Frame pointer register.
    pub frame_ptr: PhysReg,
    /// Stack pointer register.
    pub stack_ptr: PhysReg,
    /// Link register (ARM64 has one, x86 uses stack).
    pub link_reg: Option<PhysReg>,
}

// ---------------------------------------------------------------------------
// ARM64 physical registers
// ---------------------------------------------------------------------------

pub mod phys_aarch64 {
    use super::PhysReg;

    pub const X0: PhysReg = PhysReg::gp(0);
    pub const X1: PhysReg = PhysReg::gp(1);
    pub const X2: PhysReg = PhysReg::gp(2);
    pub const X3: PhysReg = PhysReg::gp(3);
    pub const X4: PhysReg = PhysReg::gp(4);
    pub const X5: PhysReg = PhysReg::gp(5);
    pub const X6: PhysReg = PhysReg::gp(6);
    pub const X7: PhysReg = PhysReg::gp(7);
    pub const X8: PhysReg = PhysReg::gp(8);
    pub const X9: PhysReg = PhysReg::gp(9);
    pub const X10: PhysReg = PhysReg::gp(10);
    pub const X11: PhysReg = PhysReg::gp(11);
    pub const X12: PhysReg = PhysReg::gp(12);
    pub const X13: PhysReg = PhysReg::gp(13);
    pub const X14: PhysReg = PhysReg::gp(14);
    pub const X15: PhysReg = PhysReg::gp(15);
    pub const X16: PhysReg = PhysReg::gp(16); // IP0, scratch
    pub const X17: PhysReg = PhysReg::gp(17); // IP1, scratch
    pub const X18: PhysReg = PhysReg::gp(18); // platform register (avoid)
    pub const X19: PhysReg = PhysReg::gp(19);
    pub const X20: PhysReg = PhysReg::gp(20);
    pub const X21: PhysReg = PhysReg::gp(21);
    pub const X22: PhysReg = PhysReg::gp(22);
    pub const X23: PhysReg = PhysReg::gp(23);
    pub const X24: PhysReg = PhysReg::gp(24);
    pub const X25: PhysReg = PhysReg::gp(25);
    pub const X26: PhysReg = PhysReg::gp(26);
    pub const X27: PhysReg = PhysReg::gp(27);
    pub const X28: PhysReg = PhysReg::gp(28);
    pub const X29: PhysReg = PhysReg::gp(29); // FP
    pub const X30: PhysReg = PhysReg::gp(30); // LR

    pub const D0: PhysReg = PhysReg::fp(0);
    pub const D1: PhysReg = PhysReg::fp(1);
    pub const D2: PhysReg = PhysReg::fp(2);
    pub const D3: PhysReg = PhysReg::fp(3);
    pub const D4: PhysReg = PhysReg::fp(4);
    pub const D5: PhysReg = PhysReg::fp(5);
    pub const D6: PhysReg = PhysReg::fp(6);
    pub const D7: PhysReg = PhysReg::fp(7);
    // d8-d15: callee-saved
    // d16-d31: caller-saved

    use super::AbiInfo;

    pub static ABI: AbiInfo = AbiInfo {
        gp_arg_regs: &[X0, X1, X2, X3, X4, X5, X6, X7],
        fp_arg_regs: &[D0, D1, D2, D3, D4, D5, D6, D7],
        gp_ret: X0,
        fp_ret: D0,
        gp_caller_saved: &[X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17],
        gp_callee_saved: &[X19, X20, X21, X22, X23, X24, X25, X26, X27, X28],
        frame_ptr: X29,
        stack_ptr: PhysReg::gp(31), // SP is encoded as 31 in some contexts
        link_reg: Some(X30),
    };
}

// ---------------------------------------------------------------------------
// x86_64 physical registers
// ---------------------------------------------------------------------------

pub mod phys_x86_64 {
    use super::PhysReg;

    // Standard encoding order
    pub const RAX: PhysReg = PhysReg::gp(0);
    pub const RCX: PhysReg = PhysReg::gp(1);
    pub const RDX: PhysReg = PhysReg::gp(2);
    pub const RBX: PhysReg = PhysReg::gp(3);
    pub const RSP: PhysReg = PhysReg::gp(4);
    pub const RBP: PhysReg = PhysReg::gp(5);
    pub const RSI: PhysReg = PhysReg::gp(6);
    pub const RDI: PhysReg = PhysReg::gp(7);
    pub const R8: PhysReg = PhysReg::gp(8);
    pub const R9: PhysReg = PhysReg::gp(9);
    pub const R10: PhysReg = PhysReg::gp(10);
    pub const R11: PhysReg = PhysReg::gp(11);
    pub const R12: PhysReg = PhysReg::gp(12);
    pub const R13: PhysReg = PhysReg::gp(13);
    pub const R14: PhysReg = PhysReg::gp(14);
    pub const R15: PhysReg = PhysReg::gp(15);

    pub const XMM0: PhysReg = PhysReg::fp(0);
    pub const XMM1: PhysReg = PhysReg::fp(1);
    pub const XMM2: PhysReg = PhysReg::fp(2);
    pub const XMM3: PhysReg = PhysReg::fp(3);
    pub const XMM4: PhysReg = PhysReg::fp(4);
    pub const XMM5: PhysReg = PhysReg::fp(5);
    pub const XMM6: PhysReg = PhysReg::fp(6);
    pub const XMM7: PhysReg = PhysReg::fp(7);

    use super::AbiInfo;

    /// System V AMD64 ABI (Linux, macOS).
    pub static ABI: AbiInfo = AbiInfo {
        gp_arg_regs: &[RDI, RSI, RDX, RCX, R8, R9],
        fp_arg_regs: &[XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7],
        gp_ret: RAX,
        fp_ret: XMM0,
        gp_caller_saved: &[RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11],
        gp_callee_saved: &[RBX, R12, R13, R14, R15],
        frame_ptr: RBP,
        stack_ptr: RSP,
        link_reg: None, // x86 uses stack for return address
    };
}

// ---------------------------------------------------------------------------
// Parallel copy resolution
// ---------------------------------------------------------------------------

/// Resolve all `ParallelCopy` pseudo-instructions in a `MachFunc` into
/// sequential `Mov`/`FMov` instructions, correctly handling cycles.
///
/// A parallel copy `[(a, b), (b, a)]` (swap) cannot be lowered to two
/// sequential moves without clobbering. This pass detects cycles and
/// breaks them using a fresh temporary register.
///
/// Algorithm:
/// 1. Build a dependency graph: dst → src.
/// 2. Emit non-cycle copies first (leaves of the DAG).
/// 3. For each cycle, allocate a temp, break the cycle via:
///    `tmp = first_src; emit chain; first_dst = tmp`.
pub fn resolve_parallel_copies(mf: &mut MachFunc) {
    let old_insts = std::mem::take(&mut mf.insts);
    let mut resolved = Vec::with_capacity(old_insts.len());

    for inst in old_insts {
        if let MachInst::ParallelCopy { copies } = inst {
            resolve_one_pcopy(&copies, &mut resolved, mf);
        } else {
            resolved.push(inst);
        }
    }

    mf.insts = resolved;
}

fn resolve_one_pcopy(
    copies: &[(VReg, VReg)],
    out: &mut Vec<MachInst>,
    mf: &mut MachFunc,
) {
    if copies.is_empty() {
        return;
    }

    // Filter out identity copies.
    let copies: Vec<(VReg, VReg)> = copies
        .iter()
        .filter(|(dst, src)| dst != src)
        .copied()
        .collect();

    if copies.is_empty() {
        return;
    }

    // Track which dsts still need to be resolved and what their source is.
    let pending: Vec<(VReg, VReg)> = copies.clone();
    let mut emitted = vec![false; pending.len()];

    // Iteratively emit copies whose dst is not a source of any remaining copy.
    // This handles the DAG (non-cycle) portion.
    let mut progress = true;
    while progress {
        progress = false;
        for i in 0..pending.len() {
            if emitted[i] {
                continue;
            }
            let (dst, _) = pending[i];
            // Check if dst is used as a source by any other pending copy.
            let blocks_other = pending.iter().enumerate().any(|(j, (_, src))| {
                j != i && !emitted[j] && *src == dst
            });
            if !blocks_other {
                let (dst, src) = pending[i];
                emit_mov(out, dst, src);
                emitted[i] = true;
                progress = true;
            }
        }
    }

    // Whatever remains forms cycles. Break each cycle with a temp.
    for i in 0..pending.len() {
        if emitted[i] {
            continue;
        }

        // Walk the cycle starting at i.
        let cycle_start_dst = pending[i].0;
        let cycle_start_src = pending[i].1;

        // Allocate a temp of the same class as the cycle's registers.
        let tmp = match cycle_start_dst.class {
            RegClass::Gp => mf.new_gp(),
            RegClass::Fp => mf.new_fp(),
            RegClass::Vec => mf.new_vec(),
        };

        // Save the first source into temp.
        emit_mov(out, tmp, cycle_start_src);
        emitted[i] = true;

        // Follow the chain: find the copy whose src == cycle_start_dst,
        // emit it, then follow its dst, etc.
        let mut current_dst = cycle_start_dst;
        loop {
            // Find the pending copy whose source is current_dst.
            let next = pending.iter().enumerate().find(|(j, (_, src))| {
                !emitted[*j] && *src == current_dst
            });
            match next {
                Some((j, &(dst, src))) => {
                    emit_mov(out, dst, src);
                    emitted[j] = true;
                    current_dst = dst;
                }
                None => break,
            }
        }

        // Close the cycle: the last dst gets the temp value.
        emit_mov(out, current_dst, tmp);
    }
}

fn emit_mov(out: &mut Vec<MachInst>, dst: VReg, src: VReg) {
    match (dst.class, src.class) {
        (RegClass::Fp, RegClass::Fp) => out.push(MachInst::FMov { dst, src }),
        // Vec-to-vec: use a V128 move (full register copy).
        (RegClass::Vec, RegClass::Vec) => out.push(MachInst::FMov { dst, src }),
        _ => out.push(MachInst::Mov { dst, src }),
    }
}

// ---------------------------------------------------------------------------
// Compilation target
// ---------------------------------------------------------------------------

/// Target architecture for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    X86_64,
    Aarch64,
    Wasm,
}

// ---------------------------------------------------------------------------
// MIR → Machine IR lowering
// ---------------------------------------------------------------------------

use crate::mir::{BlockId, Instruction, MirFunction, Terminator, ValueId};
use std::collections::HashMap;

/// Lower a MIR function to platform-agnostic machine instructions.
pub fn lower_mir(mir: &MirFunction) -> MachFunc {
    let interner = crate::intern::Interner::new();
    lower_mir_with_interner(mir, &interner)
}

/// Lower a MIR function with access to an interner for symbol resolution.
pub fn lower_mir_with_interner(mir: &MirFunction, interner: &crate::intern::Interner) -> MachFunc {
    let mut mf = MachFunc::new(format!("fn_{}", mir.name.index()));
    let mut ctx = LowerCtx::new(&mut mf, mir, interner);
    ctx.lower();
    mf
}

// ---------------------------------------------------------------------------
// Full compilation pipeline
// ---------------------------------------------------------------------------

/// Compiled output from the full pipeline.
pub enum CompiledFunction {
    /// x86_64 machine code bytes.
    X86_64(x86_64::EmittedCode),
    /// aarch64 executable buffer.
    Aarch64(aarch64::CompiledCode),
    /// WebAssembly module bytes.
    Wasm(wasm::WasmModule),
}

impl CompiledFunction {
    /// Make the compiled code executable, returning an owned handle.
    ///
    /// The returned `ExecutableFunction` owns the mmap'd memory.
    /// Function pointers obtained from it are valid for its lifetime.
    /// WASM modules cannot be made directly executable (use a WASM runtime).
    pub fn into_executable(self) -> Result<ExecutableFunction, String> {
        match self {
            CompiledFunction::X86_64(code) => {
                let exec = code.make_executable()?;
                Ok(ExecutableFunction::X86_64(exec))
            }
            CompiledFunction::Aarch64(code) => {
                Ok(ExecutableFunction::Aarch64(code))
            }
            CompiledFunction::Wasm(_) => {
                Err("WASM modules cannot be made directly executable; use a WASM runtime".into())
            }
        }
    }

    /// Get the WASM module bytes, if this is a WASM compilation.
    pub fn as_wasm(&self) -> Option<&wasm::WasmModule> {
        match self {
            CompiledFunction::Wasm(m) => Some(m),
            _ => None,
        }
    }
}

/// Executable function backed by mmap'd memory. Drop to release.
pub enum ExecutableFunction {
    X86_64(x86_64::ExecutableCode),
    Aarch64(aarch64::CompiledCode),
}

impl ExecutableFunction {
    /// Get a callable function pointer.
    ///
    /// # Safety
    /// Caller must ensure `F` matches the compiled function's ABI and that
    /// the target architecture matches the host.
    pub unsafe fn as_fn<F: Copy>(&self) -> F {
        match self {
            ExecutableFunction::X86_64(code) => code.as_fn(),
            ExecutableFunction::Aarch64(code) => code.as_fn(),
        }
    }

    /// Whether this function can be called on the current host.
    pub fn is_native(&self) -> bool {
        match self {
            ExecutableFunction::X86_64(_) => cfg!(target_arch = "x86_64"),
            ExecutableFunction::Aarch64(_) => cfg!(target_arch = "aarch64"),
        }
    }
}

/// Compile a MIR function to native code or WASM for the given target.
///
/// Native pipeline: MIR → lower → regalloc → sentinel fixup → emit.
/// WASM pipeline: MIR → emit_mir (bypasses MachInst entirely).
pub fn compile_function(mir: &MirFunction, target: Target) -> Result<CompiledFunction, String> {
    let interner = crate::intern::Interner::new();
    compile_function_with_interner(mir, target, &interner)
}

/// Compile with access to the VM's interner for symbol resolution.
pub fn compile_function_with_interner(
    mir: &MirFunction,
    target: Target,
    interner: &crate::intern::Interner,
) -> Result<CompiledFunction, String> {
    match target {
        Target::Wasm => {
            // WASM: lower directly from MIR, no MachInst layer needed.
            let module = wasm::emit_mir(mir)?;
            Ok(CompiledFunction::Wasm(module))
        }
        _ => {
            // 1. Lower MIR → MachFunc (virtual registers)
            let mut mach = lower_mir_with_interner(mir, interner);
            // 1b. Link runtime calls: CallRuntime → LoadImm + CallInd
            runtime_fns::link_runtime_calls(&mut mach);
            // Native: register allocation + emit.
            let target_regs = match target {
                Target::X86_64 => regalloc::x86_64_target_regs(),
                Target::Aarch64 => regalloc::aarch64_target_regs(),
                Target::Wasm => unreachable!(),
            };
            regalloc::allocate_registers(&mut mach, &target_regs);
            fixup_sentinels(&mut mach, target);

            match target {
                Target::X86_64 => {
                    let emitted = x86_64::emit(&mach)?;
                    Ok(CompiledFunction::X86_64(emitted))
                }
                Target::Aarch64 => {
                    let compiled = aarch64::emit(&mach)?;
                    Ok(CompiledFunction::Aarch64(compiled))
                }
                Target::Wasm => unreachable!(),
            }
        }
    }
}

/// Replace sentinel VReg indices (frame pointer, spill scratch) with
/// actual hardware register encodings for the target.
fn fixup_sentinels(mach: &mut MachFunc, target: Target) {
    let (fp_enc, gp_scratch, fp_scratch): (u32, u32, u32) = match target {
        Target::X86_64 => (5, 11, 15),     // RBP, R11, XMM15
        Target::Aarch64 => (29, 16, 16),   // X29, X16, D16
        Target::Wasm => return,            // WASM has no physical registers.
    };

    for inst in &mut mach.insts {
        fixup_vreg_sentinels(inst, fp_enc, gp_scratch, fp_scratch);
    }
}

/// Rewrite sentinel indices in a single instruction.
fn fixup_vreg_sentinels(inst: &mut MachInst, fp_enc: u32, gp_scratch: u32, fp_scratch: u32) {
    // Visit every VReg field in the instruction.
    let fix = |v: &mut VReg| {
        if v.index == u32::MAX && v.class == RegClass::Gp {
            v.index = fp_enc; // frame pointer
        } else if v.index == u32::MAX - 1 {
            v.index = match v.class {
                RegClass::Gp => gp_scratch,
                RegClass::Fp | RegClass::Vec => fp_scratch,
            };
        }
    };

    // Apply to all register fields in the instruction.
    // We use the mutable visitor pattern since MachInst fields are public.
    match inst {
        MachInst::LoadImm { dst, .. } | MachInst::LoadFpImm { dst, .. } => fix(dst),
        MachInst::Mov { dst, src } | MachInst::FMov { dst, src }
        | MachInst::BitcastGpToFp { dst, src } | MachInst::BitcastFpToGp { dst, src }
        | MachInst::INeg { dst, src } | MachInst::Not { dst, src }
        | MachInst::FNeg { dst, src }
        | MachInst::FAbs { dst, src } | MachInst::FSqrt { dst, src }
        | MachInst::FFloor { dst, src } | MachInst::FCeil { dst, src }
        | MachInst::FRound { dst, src } | MachInst::FTrunc { dst, src }
        | MachInst::FCvtToI64 { dst, src } | MachInst::I64CvtToF { dst, src } => {
            fix(dst); fix(src);
        }
        MachInst::IAddImm { dst, src, .. } | MachInst::AndImm { dst, src, .. }
        | MachInst::OrImm { dst, src, .. } => {
            fix(dst); fix(src);
        }
        MachInst::IAdd { dst, lhs, rhs } | MachInst::ISub { dst, lhs, rhs }
        | MachInst::IMul { dst, lhs, rhs } | MachInst::IDiv { dst, lhs, rhs }
        | MachInst::And { dst, lhs, rhs } | MachInst::Or { dst, lhs, rhs }
        | MachInst::Xor { dst, lhs, rhs }
        | MachInst::Shl { dst, lhs, rhs } | MachInst::Sar { dst, lhs, rhs }
        | MachInst::Shr { dst, lhs, rhs }
        | MachInst::FAdd { dst, lhs, rhs } | MachInst::FSub { dst, lhs, rhs }
        | MachInst::FMul { dst, lhs, rhs } | MachInst::FDiv { dst, lhs, rhs }
        | MachInst::FMin { dst, lhs, rhs } | MachInst::FMax { dst, lhs, rhs } => {
            fix(dst); fix(lhs); fix(rhs);
        }
        MachInst::IMulSub { dst, lhs, rhs, acc } => {
            fix(dst); fix(lhs); fix(rhs); fix(acc);
        }
        MachInst::FMAdd { dst, a, b, c } | MachInst::FMSub { dst, a, b, c }
        | MachInst::FNMAdd { dst, a, b, c } | MachInst::FNMSub { dst, a, b, c } => {
            fix(dst); fix(a); fix(b); fix(c);
        }
        MachInst::ICmp { lhs, rhs } | MachInst::FCmp { lhs, rhs } => {
            fix(lhs); fix(rhs);
        }
        MachInst::ICmpImm { lhs, .. } => fix(lhs),
        MachInst::CSet { dst, .. } => fix(dst),
        MachInst::Ldr { dst, mem } | MachInst::FLdr { dst, mem } => {
            fix(dst); fix(&mut mem.base);
        }
        MachInst::Str { src, mem } | MachInst::FStr { src, mem } => {
            fix(src); fix(&mut mem.base);
        }
        MachInst::VLoad { dst, mem, .. } => { fix(dst); fix(&mut mem.base); }
        MachInst::VStore { src, mem, .. } => { fix(src); fix(&mut mem.base); }
        MachInst::VFAdd { dst, lhs, rhs, .. } | MachInst::VFSub { dst, lhs, rhs, .. }
        | MachInst::VFMul { dst, lhs, rhs, .. } | MachInst::VFDiv { dst, lhs, rhs, .. } => {
            fix(dst); fix(lhs); fix(rhs);
        }
        MachInst::VFMAdd { dst, a, b, c, .. } => { fix(dst); fix(a); fix(b); fix(c); }
        MachInst::VBroadcast { dst, src, .. } | MachInst::VFNeg { dst, src, .. }
        | MachInst::VReduceAdd { dst, src, .. } | MachInst::VExtractLane { dst, src, .. } => {
            fix(dst); fix(src);
        }
        MachInst::VInsertLane { dst, src, val, .. } => { fix(dst); fix(src); fix(val); }
        MachInst::JmpZero { src, .. } | MachInst::JmpNonZero { src, .. }
        | MachInst::TestBitJmpZero { src, .. } | MachInst::TestBitJmpNonZero { src, .. } => {
            fix(src);
        }
        MachInst::CallInd { target } => fix(target),
        MachInst::Push { src } => fix(src),
        MachInst::Pop { dst } => fix(dst),
        MachInst::CallRuntime { args, ret, .. } => {
            for a in args.iter_mut() { fix(a); }
            if let Some(r) = ret { fix(r); }
        }
        MachInst::ParallelCopy { copies } => {
            for (d, s) in copies.iter_mut() { fix(d); fix(s); }
        }
        MachInst::Prologue { .. } | MachInst::Epilogue { .. }
        | MachInst::Jmp { .. } | MachInst::JmpIf { .. }
        | MachInst::DefLabel(_) | MachInst::Nop | MachInst::Trap | MachInst::Ret => {}
    }
}

struct LowerCtx<'a> {
    mf: &'a mut MachFunc,
    mir: &'a MirFunction,
    /// MIR ValueId → VReg mapping.
    val_map: HashMap<ValueId, VReg>,
    /// MIR BlockId → Label mapping.
    block_labels: HashMap<BlockId, Label>,
    /// Interner for resolving symbol names (e.g. IsType class names).
    interner: &'a crate::intern::Interner,
}

impl<'a> LowerCtx<'a> {
    fn new(mf: &'a mut MachFunc, mir: &'a MirFunction, interner: &'a crate::intern::Interner) -> Self {
        // Pre-allocate labels for each block.
        let mut block_labels = HashMap::new();
        for block in &mir.blocks {
            let label = mf.new_label();
            block_labels.insert(block.id, label);
        }
        Self {
            mf,
            mir,
            val_map: HashMap::new(),
            block_labels,
            interner,
        }
    }

    /// Get or create the VReg for a MIR value.
    fn vreg_for(&mut self, val: ValueId) -> VReg {
        if let Some(&r) = self.val_map.get(&val) {
            return r;
        }
        // Default to GP — callers can override for FP values.
        let r = self.mf.new_gp();
        self.val_map.insert(val, r);
        r
    }

    /// Get or create an FP VReg for a MIR value known to be f64.
    fn fp_vreg_for(&mut self, val: ValueId) -> VReg {
        if let Some(&r) = self.val_map.get(&val) {
            return r;
        }
        let r = self.mf.new_fp();
        self.val_map.insert(val, r);
        r
    }

    fn label_for(&self, block: BlockId) -> Label {
        self.block_labels[&block]
    }

    fn lower(&mut self) {
        self.mf.emit(MachInst::Prologue { frame_size: 0 });

        for block_idx in 0..self.mir.blocks.len() {
            let block = &self.mir.blocks[block_idx];
            let label = self.block_labels[&block.id];
            self.mf.emit(MachInst::DefLabel(label));

            // Block parameters are pre-assigned vregs by predecessors.
            for (val, _ty) in &block.params {
                self.vreg_for(*val);
            }

            let insts: Vec<_> = block.instructions.clone();
            for (val, inst) in &insts {
                self.lower_inst(*val, inst);
            }

            let term = block.terminator.clone();
            self.lower_terminator(&term);
        }
    }

    fn lower_inst(&mut self, dst_val: ValueId, inst: &Instruction) {
        match inst {
            // -- Constants (boxed = NaN-boxed in GP register) --
            Instruction::ConstNum(n) => {
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::LoadImm {
                    dst,
                    bits: n.to_bits(),
                });
            }
            Instruction::ConstBool(b) => {
                let dst = self.vreg_for(dst_val);
                let bits = if *b { 0x7FFC_0000_0000_0002 } else { 0x7FFC_0000_0000_0001 };
                self.mf.emit(MachInst::LoadImm { dst, bits });
            }
            Instruction::ConstNull => {
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::LoadImm {
                    dst,
                    bits: 0x7FFC_0000_0000_0000,
                });
            }
            Instruction::ConstString(idx) => {
                // Load address from string table (placeholder: index as immediate)
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::LoadImm {
                    dst,
                    bits: *idx as u64,
                });
            }

            // -- Unboxed constants --
            Instruction::ConstF64(n) => {
                let dst = self.fp_vreg_for(dst_val);
                self.mf.emit(MachInst::LoadFpImm { dst, value: *n });
            }
            Instruction::ConstI64(n) => {
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::LoadImm {
                    dst,
                    bits: *n as u64,
                });
            }

            // -- Unboxed f64 arithmetic (the big win) --
            Instruction::AddF64(a, b) => self.emit_fp_binop(dst_val, *a, *b, FpBinOp::Add),
            Instruction::SubF64(a, b) => self.emit_fp_binop(dst_val, *a, *b, FpBinOp::Sub),
            Instruction::MulF64(a, b) => self.emit_fp_binop(dst_val, *a, *b, FpBinOp::Mul),
            Instruction::DivF64(a, b) => self.emit_fp_binop(dst_val, *a, *b, FpBinOp::Div),
            Instruction::ModF64(a, b) => {
                // f64 remainder: no single instruction on either arch.
                // Lower to: a - trunc(a/b) * b
                let la = self.fp_vreg_for(*a);
                let lb = self.fp_vreg_for(*b);
                let dst = self.fp_vreg_for(dst_val);
                let quot = self.mf.new_fp();
                let trunc_gp = self.mf.new_gp();
                let trunc_fp = self.mf.new_fp();
                let prod = self.mf.new_fp();
                self.mf.emit(MachInst::FDiv {
                    dst: quot,
                    lhs: la,
                    rhs: lb,
                });
                self.mf.emit(MachInst::FCvtToI64 {
                    dst: trunc_gp,
                    src: quot,
                });
                self.mf.emit(MachInst::I64CvtToF {
                    dst: trunc_fp,
                    src: trunc_gp,
                });
                self.mf.emit(MachInst::FMul {
                    dst: prod,
                    lhs: trunc_fp,
                    rhs: lb,
                });
                self.mf.emit(MachInst::FSub {
                    dst,
                    lhs: la,
                    rhs: prod,
                });
            }
            Instruction::NegF64(a) => {
                let la = self.fp_vreg_for(*a);
                let dst = self.fp_vreg_for(dst_val);
                self.mf.emit(MachInst::FNeg { dst, src: la });
            }

            // -- Unboxed f64 comparisons --
            Instruction::CmpLtF64(a, b) => self.emit_fp_cmp(dst_val, *a, *b, Cond::Lt),
            Instruction::CmpGtF64(a, b) => self.emit_fp_cmp(dst_val, *a, *b, Cond::Gt),
            Instruction::CmpLeF64(a, b) => self.emit_fp_cmp(dst_val, *a, *b, Cond::Le),
            Instruction::CmpGeF64(a, b) => self.emit_fp_cmp(dst_val, *a, *b, Cond::Ge),

            // -- Boxing / Unboxing --
            Instruction::Unbox(a) => {
                let src = self.vreg_for(*a);
                let dst = self.fp_vreg_for(dst_val);
                self.mf.emit(MachInst::BitcastGpToFp { dst, src });
            }
            Instruction::Box(a) => {
                let src = self.fp_vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::BitcastFpToGp { dst, src });
            }

            // -- Move --
            Instruction::Move(a) => {
                let src = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::Mov { dst, src });
            }

            // -- Guards --
            Instruction::GuardNum(a) => {
                // A NaN-boxed number is any f64 that is NOT one of our tagged
                // special values. Check: bits & QNAN_MASK != QNAN_MASK, or
                // more precisely: (bits + 0x0004_0000_0000_0000) >> 63 == 0
                // For now, emit a runtime call as placeholder.
                let src = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::Mov { dst, src });
                // TODO: inline guard check + branch to deopt
            }
            Instruction::GuardBool(a) => {
                let src = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::Mov { dst, src });
            }

            // -- Boxed arithmetic → runtime calls --
            Instruction::Add(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_num_add"),
            Instruction::Sub(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_num_sub"),
            Instruction::Mul(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_num_mul"),
            Instruction::Div(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_num_div"),
            Instruction::Mod(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_num_mod"),
            Instruction::Neg(a) => {
                let la = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_num_neg",
                    args: vec![la],
                    ret: Some(dst),
                });
            }

            // -- Boxed comparisons → runtime calls --
            Instruction::CmpLt(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_cmp_lt"),
            Instruction::CmpGt(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_cmp_gt"),
            Instruction::CmpLe(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_cmp_le"),
            Instruction::CmpGe(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_cmp_ge"),
            Instruction::CmpEq(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_cmp_eq"),
            Instruction::CmpNe(a, b) => self.emit_boxed_binop(dst_val, *a, *b, "wren_cmp_ne"),

            // -- Logical --
            Instruction::Not(a) => {
                let la = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_not",
                    args: vec![la],
                    ret: Some(dst),
                });
            }

            // -- Bitwise (truncate to i32, operate, convert back) --
            Instruction::BitAnd(a, b) => self.emit_bitwise(dst_val, *a, *b, BitwiseOp::And),
            Instruction::BitOr(a, b) => self.emit_bitwise(dst_val, *a, *b, BitwiseOp::Or),
            Instruction::BitXor(a, b) => self.emit_bitwise(dst_val, *a, *b, BitwiseOp::Xor),
            Instruction::Shl(a, b) => self.emit_bitwise(dst_val, *a, *b, BitwiseOp::Shl),
            Instruction::Shr(a, b) => self.emit_bitwise(dst_val, *a, *b, BitwiseOp::Shr),
            Instruction::BitNot(a) => {
                // Inline: unbox → truncate → NOT → convert back → rebox
                let la = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                let fa = self.mf.new_fp();
                let ia = self.mf.new_gp();
                let result_i = self.mf.new_gp();
                let result_f = self.mf.new_fp();
                self.mf.emit(MachInst::BitcastGpToFp { dst: fa, src: la });
                self.mf.emit(MachInst::FCvtToI64 { dst: ia, src: fa });
                self.mf.emit(MachInst::Not { dst: result_i, src: ia });
                self.mf.emit(MachInst::I64CvtToF { dst: result_f, src: result_i });
                self.mf.emit(MachInst::BitcastFpToGp { dst, src: result_f });
            }

            // -- Object operations: inline GEP field access --
            Instruction::GetField(recv, idx) => {
                use crate::runtime::object_layout::*;
                let recv_reg = self.vreg_for(*recv);
                let dst = self.vreg_for(dst_val);

                // 1. Extract object pointer: AND with PTR_MASK (48-bit)
                let obj_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::AndImm {
                    dst: obj_ptr,
                    src: recv_reg,
                    imm: 0x0000_FFFF_FFFF_FFFF, // PTR_MASK
                });

                // 2. Load fields pointer: obj_ptr + INSTANCE_FIELDS
                let fields_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::Ldr {
                    dst: fields_ptr,
                    mem: Mem::new(obj_ptr, INSTANCE_FIELDS),
                });

                // 3. Load field value: fields_ptr + idx * VALUE_SIZE
                let field_offset = (*idx as i32) * VALUE_SIZE;
                self.mf.emit(MachInst::Ldr {
                    dst,
                    mem: Mem::new(fields_ptr, field_offset),
                });
            }
            Instruction::SetField(recv, idx, val) => {
                use crate::runtime::object_layout::*;
                let recv_reg = self.vreg_for(*recv);
                let v = self.vreg_for(*val);
                let dst = self.vreg_for(dst_val);

                // 1. Extract object pointer
                let obj_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::AndImm {
                    dst: obj_ptr,
                    src: recv_reg,
                    imm: 0x0000_FFFF_FFFF_FFFF,
                });

                // 2. Load fields pointer
                let fields_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::Ldr {
                    dst: fields_ptr,
                    mem: Mem::new(obj_ptr, INSTANCE_FIELDS),
                });

                // 3. Store field value
                let field_offset = (*idx as i32) * VALUE_SIZE;
                self.mf.emit(MachInst::Str {
                    src: v,
                    mem: Mem::new(fields_ptr, field_offset),
                });

                // SetField result is the stored value
                self.mf.emit(MachInst::Mov { dst, src: v });
            }
            Instruction::GetModuleVar(idx) => {
                let dst = self.vreg_for(dst_val);
                let idx_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: idx_reg,
                    bits: *idx as u64,
                });
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_get_module_var",
                    args: vec![idx_reg],
                    ret: Some(dst),
                });
            }
            Instruction::SetModuleVar(idx, val) => {
                let v = self.vreg_for(*val);
                let dst = self.vreg_for(dst_val);
                let idx_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: idx_reg,
                    bits: *idx as u64,
                });
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_set_module_var",
                    args: vec![idx_reg, v],
                    ret: Some(dst),
                });
            }

            // -- Calls --
            Instruction::Call {
                receiver,
                method,
                args,
            } => {
                let r = self.vreg_for(*receiver);
                let dst = self.vreg_for(dst_val);
                let mut call_args = vec![r];
                let method_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: method_reg,
                    bits: method.index() as u64,
                });
                call_args.push(method_reg);
                for a in args {
                    call_args.push(self.vreg_for(*a));
                }
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_call",
                    args: call_args,
                    ret: Some(dst),
                });
            }
            Instruction::SuperCall { method, args } => {
                let dst = self.vreg_for(dst_val);
                let method_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: method_reg,
                    bits: method.index() as u64,
                });
                let mut call_args = vec![method_reg];
                for a in args {
                    call_args.push(self.vreg_for(*a));
                }
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_super_call",
                    args: call_args,
                    ret: Some(dst),
                });
            }

            // -- Collections & misc → runtime calls --
            Instruction::MakeClosure { fn_id, upvalues } => {
                let dst = self.vreg_for(dst_val);
                let fn_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: fn_reg,
                    bits: *fn_id as u64,
                });
                let mut call_args = vec![fn_reg];
                for uv in upvalues {
                    call_args.push(self.vreg_for(*uv));
                }
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_make_closure",
                    args: call_args,
                    ret: Some(dst),
                });
            }
            Instruction::GetUpvalue(idx) => {
                let dst = self.vreg_for(dst_val);
                let idx_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: idx_reg,
                    bits: *idx as u64,
                });
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_get_upvalue",
                    args: vec![idx_reg],
                    ret: Some(dst),
                });
            }
            Instruction::SetUpvalue(idx, val) => {
                let v = self.vreg_for(*val);
                let dst = self.vreg_for(dst_val);
                let idx_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: idx_reg,
                    bits: *idx as u64,
                });
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_set_upvalue",
                    args: vec![idx_reg, v],
                    ret: Some(dst),
                });
            }
            Instruction::MakeList(elems) => {
                let dst = self.vreg_for(dst_val);
                let args: Vec<VReg> = elems.iter().map(|e| self.vreg_for(*e)).collect();
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_make_list",
                    args,
                    ret: Some(dst),
                });
            }
            Instruction::MakeMap(pairs) => {
                let dst = self.vreg_for(dst_val);
                let mut args = Vec::new();
                for (k, v) in pairs {
                    args.push(self.vreg_for(*k));
                    args.push(self.vreg_for(*v));
                }
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_make_map",
                    args,
                    ret: Some(dst),
                });
            }
            Instruction::MakeRange(from, to, inclusive) => {
                let f = self.vreg_for(*from);
                let t = self.vreg_for(*to);
                let dst = self.vreg_for(dst_val);
                let incl_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: incl_reg,
                    bits: *inclusive as u64,
                });
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_make_range",
                    args: vec![f, t, incl_reg],
                    ret: Some(dst),
                });
            }
            Instruction::StringConcat(parts) => {
                let dst = self.vreg_for(dst_val);
                let args: Vec<VReg> = parts.iter().map(|p| self.vreg_for(*p)).collect();
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_string_concat",
                    args,
                    ret: Some(dst),
                });
            }
            Instruction::ToString(a) => {
                let la = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_to_string",
                    args: vec![la],
                    ret: Some(dst),
                });
            }
            // -- IsType: inline tag checks for primitives, class ptr for objects --
            Instruction::IsType(a, sym) => {
                use crate::runtime::object_layout::*;
                let la = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);

                // NaN-box constants
                const QNAN: u64 = 0x7FFC_0000_0000_0000;
                const TAG_OBJ: u64 = (1u64 << 63) | QNAN;
                const TAG_NULL: u64 = QNAN;
                const TAG_FALSE: u64 = QNAN | 1;
                const TAG_TRUE: u64 = QNAN | 2;

                // Resolve the type name from the interner
                let type_name = self.interner.resolve(*sym);
                match type_name {
                    "Num" => {
                        // is Num: (val & QNAN) != QNAN
                        // Equivalently: the QNAN bits aren't all set → it's a number
                        let masked = self.mf.new_gp();
                        self.mf.emit(MachInst::AndImm { dst: masked, src: la, imm: QNAN });
                        let qnan_reg = self.mf.new_gp();
                        self.mf.emit(MachInst::LoadImm { dst: qnan_reg, bits: QNAN });
                        self.mf.emit(MachInst::ICmp { lhs: masked, rhs: qnan_reg });
                        self.mf.emit(MachInst::CSet { dst, cond: Cond::Ne });
                    }
                    "Bool" => {
                        // is Bool: val == TAG_TRUE || val == TAG_FALSE
                        let is_true = self.mf.new_gp();
                        let is_false = self.mf.new_gp();
                        let t_reg = self.mf.new_gp();
                        let f_reg = self.mf.new_gp();
                        self.mf.emit(MachInst::LoadImm { dst: t_reg, bits: TAG_TRUE });
                        self.mf.emit(MachInst::ICmp { lhs: la, rhs: t_reg });
                        self.mf.emit(MachInst::CSet { dst: is_true, cond: Cond::Eq });
                        self.mf.emit(MachInst::LoadImm { dst: f_reg, bits: TAG_FALSE });
                        self.mf.emit(MachInst::ICmp { lhs: la, rhs: f_reg });
                        self.mf.emit(MachInst::CSet { dst: is_false, cond: Cond::Eq });
                        self.mf.emit(MachInst::Or { dst, lhs: is_true, rhs: is_false });
                    }
                    "Null" => {
                        // is Null: val == TAG_NULL
                        let null_reg = self.mf.new_gp();
                        self.mf.emit(MachInst::LoadImm { dst: null_reg, bits: TAG_NULL });
                        self.mf.emit(MachInst::ICmp { lhs: la, rhs: null_reg });
                        self.mf.emit(MachInst::CSet { dst, cond: Cond::Eq });
                    }
                    _ => {
                        // Object type: extract obj ptr → load class ptr → compare
                        // First check if it's even an object
                        let is_obj_label = self.mf.new_label();
                        let done_label = self.mf.new_label();

                        let tag_masked = self.mf.new_gp();
                        let tag_obj_reg = self.mf.new_gp();
                        self.mf.emit(MachInst::AndImm { dst: tag_masked, src: la, imm: TAG_OBJ });
                        self.mf.emit(MachInst::LoadImm { dst: tag_obj_reg, bits: TAG_OBJ });
                        self.mf.emit(MachInst::ICmp { lhs: tag_masked, rhs: tag_obj_reg });
                        self.mf.emit(MachInst::JmpIf { cond: Cond::Eq, target: is_obj_label });
                        // Not an object → false
                        self.mf.emit(MachInst::LoadImm { dst, bits: 0 });
                        self.mf.emit(MachInst::Jmp { target: done_label });

                        // Is an object → load class name sym and compare
                        self.mf.emit(MachInst::DefLabel(is_obj_label));
                        let obj_ptr = self.mf.new_gp();
                        self.mf.emit(MachInst::AndImm {
                            dst: obj_ptr, src: la, imm: 0x0000_FFFF_FFFF_FFFF,
                        });
                        // Load class pointer from header
                        let class_ptr = self.mf.new_gp();
                        self.mf.emit(MachInst::Ldr { dst: class_ptr, mem: Mem::new(obj_ptr, HEADER_CLASS) });
                        // Load class name sym from ObjClass (name field is after header)
                        // For now, fall back to runtime for class hierarchy comparison
                        let sym_reg = self.mf.new_gp();
                        self.mf.emit(MachInst::LoadImm { dst: sym_reg, bits: sym.index() as u64 });
                        self.mf.emit(MachInst::CallRuntime {
                            name: "wren_is_type",
                            args: vec![la, sym_reg],
                            ret: Some(dst),
                        });

                        self.mf.emit(MachInst::DefLabel(done_label));
                    }
                }
            }
            Instruction::GuardClass(a, sym) => {
                let la = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                let sym_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: sym_reg,
                    bits: sym.index() as u64,
                });
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_guard_class",
                    args: vec![la, sym_reg],
                    ret: Some(dst),
                });
            }
            Instruction::GuardProtocol(a, proto_id) => {
                // Protocol guard: checks the class's protocol bitset at runtime.
                // For now, emit as a runtime call; the devirt pass eliminates
                // most of these before codegen.
                let la = self.vreg_for(*a);
                let dst = self.vreg_for(dst_val);
                let pid_reg = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm {
                    dst: pid_reg,
                    bits: proto_id.0 as u64,
                });
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_guard_protocol",
                    args: vec![la, pid_reg],
                    ret: Some(dst),
                });
            }
            // -- Subscript: inline GEP for single-index list access --
            Instruction::SubscriptGet { receiver, args } if args.len() == 1 => {
                use crate::runtime::object_layout::*;
                let recv_reg = self.vreg_for(*receiver);
                let idx_reg = self.vreg_for(args[0]);
                let dst = self.vreg_for(dst_val);

                // 1. Extract list pointer from NaN-boxed receiver
                let list_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::AndImm {
                    dst: list_ptr, src: recv_reg, imm: 0x0000_FFFF_FFFF_FFFF,
                });

                // 2. Convert NaN-boxed index to i64
                let idx_fp = self.mf.new_fp();
                let idx_int = self.mf.new_gp();
                self.mf.emit(MachInst::BitcastGpToFp { dst: idx_fp, src: idx_reg });
                self.mf.emit(MachInst::FCvtToI64 { dst: idx_int, src: idx_fp });

                // 3. Bounds check: load count, compare, trap if out of bounds
                let count_reg = self.mf.new_gp();
                self.mf.emit(MachInst::Ldr { dst: count_reg, mem: Mem::new(list_ptr, LIST_COUNT) });
                self.mf.emit(MachInst::ICmp { lhs: idx_int, rhs: count_reg });
                let ok_label = self.mf.new_label();
                self.mf.emit(MachInst::JmpIf { cond: Cond::Below, target: ok_label });
                self.mf.emit(MachInst::Trap);
                self.mf.emit(MachInst::DefLabel(ok_label));

                // 4. Load element: elements_ptr + idx * 8
                let elements_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::Ldr { dst: elements_ptr, mem: Mem::new(list_ptr, LIST_ELEMENTS) });
                let shift_amt = self.mf.new_gp();
                let byte_offset = self.mf.new_gp();
                let elem_addr = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm { dst: shift_amt, bits: 3 });
                self.mf.emit(MachInst::Shl { dst: byte_offset, lhs: idx_int, rhs: shift_amt });
                self.mf.emit(MachInst::IAdd { dst: elem_addr, lhs: elements_ptr, rhs: byte_offset });
                self.mf.emit(MachInst::Ldr { dst, mem: Mem::new(elem_addr, 0) });
            }
            Instruction::SubscriptSet { receiver, args, value } if args.len() == 1 => {
                use crate::runtime::object_layout::*;
                let recv_reg = self.vreg_for(*receiver);
                let idx_reg = self.vreg_for(args[0]);
                let val_reg = self.vreg_for(*value);
                let dst = self.vreg_for(dst_val);

                // 1. Extract list pointer
                let list_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::AndImm {
                    dst: list_ptr, src: recv_reg, imm: 0x0000_FFFF_FFFF_FFFF,
                });

                // 2. Convert NaN-boxed index to i64
                let idx_fp = self.mf.new_fp();
                let idx_int = self.mf.new_gp();
                self.mf.emit(MachInst::BitcastGpToFp { dst: idx_fp, src: idx_reg });
                self.mf.emit(MachInst::FCvtToI64 { dst: idx_int, src: idx_fp });

                // 3. Bounds check
                let count_reg = self.mf.new_gp();
                self.mf.emit(MachInst::Ldr { dst: count_reg, mem: Mem::new(list_ptr, LIST_COUNT) });
                self.mf.emit(MachInst::ICmp { lhs: idx_int, rhs: count_reg });
                let ok_label = self.mf.new_label();
                self.mf.emit(MachInst::JmpIf { cond: Cond::Below, target: ok_label });
                self.mf.emit(MachInst::Trap);
                self.mf.emit(MachInst::DefLabel(ok_label));

                // 4. Store element: elements_ptr + idx * 8
                let elements_ptr = self.mf.new_gp();
                self.mf.emit(MachInst::Ldr { dst: elements_ptr, mem: Mem::new(list_ptr, LIST_ELEMENTS) });
                let shift_amt = self.mf.new_gp();
                let byte_offset = self.mf.new_gp();
                let elem_addr = self.mf.new_gp();
                self.mf.emit(MachInst::LoadImm { dst: shift_amt, bits: 3 });
                self.mf.emit(MachInst::Shl { dst: byte_offset, lhs: idx_int, rhs: shift_amt });
                self.mf.emit(MachInst::IAdd { dst: elem_addr, lhs: elements_ptr, rhs: byte_offset });
                self.mf.emit(MachInst::Str { src: val_reg, mem: Mem::new(elem_addr, 0) });
                self.mf.emit(MachInst::Mov { dst, src: val_reg });
            }
            // Multi-index subscript: fall back to runtime call
            Instruction::SubscriptGet { receiver, args } => {
                let r = self.vreg_for(*receiver);
                let dst = self.vreg_for(dst_val);
                let mut call_args = vec![r];
                for a in args {
                    call_args.push(self.vreg_for(*a));
                }
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_subscript_get",
                    args: call_args,
                    ret: Some(dst),
                });
            }
            Instruction::SubscriptSet {
                receiver,
                args,
                value,
            } => {
                let r = self.vreg_for(*receiver);
                let v = self.vreg_for(*value);
                let dst = self.vreg_for(dst_val);
                let mut call_args = vec![r];
                for a in args {
                    call_args.push(self.vreg_for(*a));
                }
                call_args.push(v);
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_subscript_set",
                    args: call_args,
                    ret: Some(dst),
                });
            }

            // -- Math intrinsics (unboxed f64) --
            Instruction::MathUnaryF64(op, a) => {
                use crate::mir::MathUnaryOp;
                let la = self.fp_vreg_for(*a);
                let dst = self.fp_vreg_for(dst_val);
                match op {
                    // Hardware-native single-instruction ops
                    MathUnaryOp::Abs => self.mf.emit(MachInst::FAbs { dst, src: la }),
                    MathUnaryOp::Sqrt => self.mf.emit(MachInst::FSqrt { dst, src: la }),
                    MathUnaryOp::Floor => self.mf.emit(MachInst::FFloor { dst, src: la }),
                    MathUnaryOp::Ceil => self.mf.emit(MachInst::FCeil { dst, src: la }),
                    MathUnaryOp::Round => self.mf.emit(MachInst::FRound { dst, src: la }),
                    MathUnaryOp::Trunc => self.mf.emit(MachInst::FTrunc { dst, src: la }),
                    // Transcendentals — fall back to libm via CallRuntime
                    _ => {
                        self.mf.emit(MachInst::CallRuntime {
                            name: match op {
                                MathUnaryOp::Acos => "acos",
                                MathUnaryOp::Asin => "asin",
                                MathUnaryOp::Atan => "atan",
                                MathUnaryOp::Cbrt => "cbrt",
                                MathUnaryOp::Cos => "cos",
                                MathUnaryOp::Sin => "sin",
                                MathUnaryOp::Tan => "tan",
                                MathUnaryOp::Log => "log",
                                MathUnaryOp::Log2 => "log2",
                                MathUnaryOp::Exp => "exp",
                                MathUnaryOp::Fract => "wren_fract",
                                MathUnaryOp::Sign => "wren_sign",
                                // Already handled above
                                MathUnaryOp::Abs | MathUnaryOp::Sqrt | MathUnaryOp::Floor
                                | MathUnaryOp::Ceil | MathUnaryOp::Round | MathUnaryOp::Trunc => {
                                    unreachable!()
                                }
                            },
                            args: vec![la],
                            ret: Some(dst),
                        });
                    }
                }
            }
            Instruction::MathBinaryF64(op, a, b) => {
                use crate::mir::MathBinaryOp;
                let la = self.fp_vreg_for(*a);
                let lb = self.fp_vreg_for(*b);
                let dst = self.fp_vreg_for(dst_val);
                match op {
                    // Hardware-native single-instruction ops
                    MathBinaryOp::Min => self.mf.emit(MachInst::FMin { dst, lhs: la, rhs: lb }),
                    MathBinaryOp::Max => self.mf.emit(MachInst::FMax { dst, lhs: la, rhs: lb }),
                    // Transcendentals — fall back to libm via CallRuntime
                    _ => {
                        self.mf.emit(MachInst::CallRuntime {
                            name: match op {
                                MathBinaryOp::Atan2 => "atan2",
                                MathBinaryOp::Pow => "pow",
                                MathBinaryOp::Min | MathBinaryOp::Max => unreachable!(),
                            },
                            args: vec![la, lb],
                            ret: Some(dst),
                        });
                    }
                }
            }

            // Block parameters are handled at block entry.
            Instruction::BlockParam(_) => {
                self.vreg_for(dst_val);
            }
        }
    }

    fn lower_terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Return(v) => {
                let src = self.vreg_for(*v);
                // Move return value into GP r0 (ABI return register).
                let ret_reg = VReg::gp(0);
                if src.is_fp() {
                    // FP value → GP: bitcast (e.g. unboxed f64 returned as NaN-boxed bits).
                    self.mf.emit(MachInst::BitcastFpToGp {
                        dst: ret_reg,
                        src,
                    });
                } else {
                    self.mf.emit(MachInst::Mov {
                        dst: ret_reg,
                        src,
                    });
                }
                self.mf
                    .emit(MachInst::Epilogue { frame_size: 0 });
                self.mf.emit(MachInst::Ret);
            }
            Terminator::ReturnNull => {
                let ret_reg = VReg::gp(0);
                self.mf.emit(MachInst::LoadImm {
                    dst: ret_reg,
                    bits: 0x7FFC_0000_0000_0000,
                });
                self.mf
                    .emit(MachInst::Epilogue { frame_size: 0 });
                self.mf.emit(MachInst::Ret);
            }
            Terminator::Branch { target, args } => {
                // Bind branch args to target block params.
                self.emit_block_args(*target, args);
                let label = self.label_for(*target);
                self.mf.emit(MachInst::Jmp { target: label });
            }
            Terminator::CondBranch {
                condition,
                true_target,
                true_args,
                false_target,
                false_args,
            } => {
                let cond = self.vreg_for(*condition);

                let false_label = self.label_for(*false_target);
                let true_label = self.label_for(*true_target);

                // Evaluate truthiness.
                let truthy_reg = self.mf.new_gp();
                self.mf.emit(MachInst::CallRuntime {
                    name: "wren_is_truthy",
                    args: vec![cond],
                    ret: Some(truthy_reg),
                });

                // Both branches may pass different args to different blocks.
                // We need a landing pad for each side that does its own
                // parallel copy before jumping to the real target.

                let has_true_args = !true_args.is_empty();
                let has_false_args = !false_args.is_empty();

                if !has_true_args && !has_false_args {
                    // No block params on either side — simple conditional.
                    self.mf.emit(MachInst::JmpZero {
                        src: truthy_reg,
                        target: false_label,
                    });
                    self.mf.emit(MachInst::Jmp { target: true_label });
                } else {
                    // Create landing pads for phi resolution.
                    let true_pad = self.mf.new_label();
                    let false_pad = self.mf.new_label();

                    self.mf.emit(MachInst::JmpZero {
                        src: truthy_reg,
                        target: false_pad,
                    });

                    // True landing pad (fall-through).
                    self.mf.emit(MachInst::DefLabel(true_pad));
                    self.emit_block_args(*true_target, true_args);
                    self.mf.emit(MachInst::Jmp { target: true_label });

                    // False landing pad.
                    self.mf.emit(MachInst::DefLabel(false_pad));
                    self.emit_block_args(*false_target, false_args);
                    self.mf.emit(MachInst::Jmp { target: false_label });
                }
            }
            Terminator::Unreachable => {
                self.mf.emit(MachInst::Trap);
            }
        }
    }

    /// Emit a ParallelCopy to bind branch arguments to target block parameters.
    fn emit_block_args(&mut self, target: BlockId, args: &[ValueId]) {
        if args.is_empty() {
            return;
        }
        let target_idx = target.0 as usize;
        if target_idx >= self.mir.blocks.len() {
            return;
        }
        let params: Vec<ValueId> = self.mir.blocks[target_idx]
            .params
            .iter()
            .map(|(v, _)| *v)
            .collect();
        let copies: Vec<(VReg, VReg)> = params
            .iter()
            .zip(args.iter())
            .map(|(param, arg)| {
                let dst = self.vreg_for(*param);
                let src = self.vreg_for(*arg);
                (dst, src)
            })
            .collect();
        if !copies.is_empty() {
            self.mf.emit(MachInst::ParallelCopy { copies });
        }
    }

    fn emit_fp_binop(&mut self, dst_val: ValueId, a: ValueId, b: ValueId, op: FpBinOp) {
        let la = self.fp_vreg_for(a);
        let lb = self.fp_vreg_for(b);
        let dst = self.fp_vreg_for(dst_val);
        self.mf.emit(match op {
            FpBinOp::Add => MachInst::FAdd {
                dst,
                lhs: la,
                rhs: lb,
            },
            FpBinOp::Sub => MachInst::FSub {
                dst,
                lhs: la,
                rhs: lb,
            },
            FpBinOp::Mul => MachInst::FMul {
                dst,
                lhs: la,
                rhs: lb,
            },
            FpBinOp::Div => MachInst::FDiv {
                dst,
                lhs: la,
                rhs: lb,
            },
        });
    }

    fn emit_fp_cmp(&mut self, dst_val: ValueId, a: ValueId, b: ValueId, cond: Cond) {
        let la = self.fp_vreg_for(a);
        let lb = self.fp_vreg_for(b);
        let dst = self.vreg_for(dst_val);
        self.mf.emit(MachInst::FCmp { lhs: la, rhs: lb });
        self.mf.emit(MachInst::CSet { dst, cond });
    }

    fn emit_boxed_binop(
        &mut self,
        dst_val: ValueId,
        a: ValueId,
        b: ValueId,
        runtime_fn: &'static str,
    ) {
        let la = self.vreg_for(a);
        let lb = self.vreg_for(b);
        let dst = self.vreg_for(dst_val);
        self.mf.emit(MachInst::CallRuntime {
            name: runtime_fn,
            args: vec![la, lb],
            ret: Some(dst),
        });
    }

    /// Inline bitwise: unbox both to GP (truncate f64→i32), integer op, rebox.
    /// Wren bitwise semantics: truncate operands to i32, operate, convert back to f64.
    fn emit_bitwise(&mut self, dst_val: ValueId, a: ValueId, b: ValueId, op: BitwiseOp) {
        let la = self.vreg_for(a);
        let lb = self.vreg_for(b);
        let dst = self.vreg_for(dst_val);

        // Unbox: Value(GP) → f64(FP) → i64(GP) (truncated to i32 semantics)
        let fa = self.mf.new_fp();
        let fb = self.mf.new_fp();
        let ia = self.mf.new_gp();
        let ib = self.mf.new_gp();
        self.mf.emit(MachInst::BitcastGpToFp { dst: fa, src: la });
        self.mf.emit(MachInst::BitcastGpToFp { dst: fb, src: lb });
        self.mf.emit(MachInst::FCvtToI64 { dst: ia, src: fa });
        self.mf.emit(MachInst::FCvtToI64 { dst: ib, src: fb });

        // Integer operation.
        let result_i = self.mf.new_gp();
        self.mf.emit(match op {
            BitwiseOp::And => MachInst::And { dst: result_i, lhs: ia, rhs: ib },
            BitwiseOp::Or => MachInst::Or { dst: result_i, lhs: ia, rhs: ib },
            BitwiseOp::Xor => MachInst::Xor { dst: result_i, lhs: ia, rhs: ib },
            BitwiseOp::Shl => MachInst::Shl { dst: result_i, lhs: ia, rhs: ib },
            BitwiseOp::Shr => MachInst::Sar { dst: result_i, lhs: ia, rhs: ib },
        });

        // Rebox: i64(GP) → f64(FP) → Value(GP)
        let result_f = self.mf.new_fp();
        self.mf.emit(MachInst::I64CvtToF { dst: result_f, src: result_i });
        self.mf.emit(MachInst::BitcastFpToGp { dst, src: result_f });
    }
}

enum FpBinOp {
    Add,
    Sub,
    Mul,
    Div,
}

enum BitwiseOp {
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

// ---------------------------------------------------------------------------
// Instruction formatting
// ---------------------------------------------------------------------------

fn format_inst(inst: &MachInst) -> String {
    use MachInst::*;
    match inst {
        LoadImm { dst, bits } => format!("{} = load_imm 0x{:016x}", dst, bits),
        LoadFpImm { dst, value } => format!("{} = load_fp_imm {}", dst, value),
        Mov { dst, src } => format!("{} = mov {}", dst, src),
        FMov { dst, src } => format!("{} = fmov {}", dst, src),
        BitcastGpToFp { dst, src } => format!("{} = bitcast_gp_to_fp {}", dst, src),
        BitcastFpToGp { dst, src } => format!("{} = bitcast_fp_to_gp {}", dst, src),

        IAdd { dst, lhs, rhs } => format!("{} = iadd {}, {}", dst, lhs, rhs),
        IAddImm { dst, src, imm } => format!("{} = iadd_imm {}, {}", dst, src, imm),
        ISub { dst, lhs, rhs } => format!("{} = isub {}, {}", dst, lhs, rhs),
        IMul { dst, lhs, rhs } => format!("{} = imul {}, {}", dst, lhs, rhs),
        IDiv { dst, lhs, rhs } => format!("{} = idiv {}, {}", dst, lhs, rhs),
        IMulSub { dst, lhs, rhs, acc } => format!("{} = imul_sub {}, {}, {}", dst, lhs, rhs, acc),
        INeg { dst, src } => format!("{} = ineg {}", dst, src),

        And { dst, lhs, rhs } => format!("{} = and {}, {}", dst, lhs, rhs),
        AndImm { dst, src, imm } => format!("{} = and_imm {}, 0x{:x}", dst, src, imm),
        Or { dst, lhs, rhs } => format!("{} = or {}, {}", dst, lhs, rhs),
        OrImm { dst, src, imm } => format!("{} = or_imm {}, 0x{:x}", dst, src, imm),
        Xor { dst, lhs, rhs } => format!("{} = xor {}, {}", dst, lhs, rhs),
        Not { dst, src } => format!("{} = not {}", dst, src),
        Shl { dst, lhs, rhs } => format!("{} = shl {}, {}", dst, lhs, rhs),
        Sar { dst, lhs, rhs } => format!("{} = sar {}, {}", dst, lhs, rhs),
        Shr { dst, lhs, rhs } => format!("{} = shr {}, {}", dst, lhs, rhs),

        FAdd { dst, lhs, rhs } => format!("{} = fadd {}, {}", dst, lhs, rhs),
        FSub { dst, lhs, rhs } => format!("{} = fsub {}, {}", dst, lhs, rhs),
        FMul { dst, lhs, rhs } => format!("{} = fmul {}, {}", dst, lhs, rhs),
        FDiv { dst, lhs, rhs } => format!("{} = fdiv {}, {}", dst, lhs, rhs),
        FNeg { dst, src } => format!("{} = fneg {}", dst, src),
        FAbs { dst, src } => format!("{} = fabs {}", dst, src),
        FSqrt { dst, src } => format!("{} = fsqrt {}", dst, src),
        FFloor { dst, src } => format!("{} = ffloor {}", dst, src),
        FCeil { dst, src } => format!("{} = fceil {}", dst, src),
        FRound { dst, src } => format!("{} = fround {}", dst, src),
        FTrunc { dst, src } => format!("{} = ftrunc {}", dst, src),
        FMin { dst, lhs, rhs } => format!("{} = fmin {}, {}", dst, lhs, rhs),
        FMax { dst, lhs, rhs } => format!("{} = fmax {}, {}", dst, lhs, rhs),

        FMAdd { dst, a, b, c } => format!("{} = fmadd {}, {}, {}", dst, a, b, c),
        FMSub { dst, a, b, c } => format!("{} = fmsub {}, {}, {}", dst, a, b, c),
        FNMAdd { dst, a, b, c } => format!("{} = fnmadd {}, {}, {}", dst, a, b, c),
        FNMSub { dst, a, b, c } => format!("{} = fnmsub {}, {}, {}", dst, a, b, c),

        VLoad { dst, mem, width } => format!("{} = vload.{} {}", dst, width, mem),
        VStore { src, mem, width } => format!("vstore.{} {}, {}", width, src, mem),
        VFAdd { dst, lhs, rhs, width } => format!("{} = vfadd.{} {}, {}", dst, width, lhs, rhs),
        VFSub { dst, lhs, rhs, width } => format!("{} = vfsub.{} {}, {}", dst, width, lhs, rhs),
        VFMul { dst, lhs, rhs, width } => format!("{} = vfmul.{} {}, {}", dst, width, lhs, rhs),
        VFDiv { dst, lhs, rhs, width } => format!("{} = vfdiv.{} {}, {}", dst, width, lhs, rhs),
        VFMAdd { dst, a, b, c, width } => format!("{} = vfmadd.{} {}, {}, {}", dst, width, a, b, c),
        VBroadcast { dst, src, width } => format!("{} = vbroadcast.{} {}", dst, width, src),
        VExtractLane { dst, src, lane } => format!("{} = vextract {}, #{}", dst, src, lane),
        VInsertLane { dst, src, lane, val } => format!("{} = vinsert {}, #{}, {}", dst, src, lane, val),
        VFNeg { dst, src, width } => format!("{} = vfneg.{} {}", dst, width, src),
        VReduceAdd { dst, src, width } => format!("{} = vreduce_add.{} {}", dst, width, src),

        FCvtToI64 { dst, src } => format!("{} = fcvt_to_i64 {}", dst, src),
        I64CvtToF { dst, src } => format!("{} = i64_cvt_to_f {}", dst, src),

        ICmp { lhs, rhs } => format!("icmp {}, {}", lhs, rhs),
        ICmpImm { lhs, imm } => format!("icmp_imm {}, 0x{:x}", lhs, imm),
        FCmp { lhs, rhs } => format!("fcmp {}, {}", lhs, rhs),
        CSet { dst, cond } => format!("{} = cset {:?}", dst, cond),

        Ldr { dst, mem } => format!("{} = ldr {}", dst, mem),
        Str { src, mem } => format!("str {}, {}", src, mem),
        FLdr { dst, mem } => format!("{} = fldr {}", dst, mem),
        FStr { src, mem } => format!("fstr {}, {}", src, mem),

        Jmp { target } => format!("jmp {}", target),
        JmpIf { cond, target } => format!("jmp_if {:?} {}", cond, target),
        JmpZero { src, target } => format!("jmp_zero {}, {}", src, target),
        JmpNonZero { src, target } => format!("jmp_nz {}, {}", src, target),
        TestBitJmpZero { src, bit, target } => {
            format!("tbz {}, #{}, {}", src, bit, target)
        }
        TestBitJmpNonZero { src, bit, target } => {
            format!("tbnz {}, #{}, {}", src, bit, target)
        }

        CallInd { target } => format!("call_ind {}", target),
        CallRuntime { name, args, ret } => {
            let args_str: Vec<String> = args.iter().map(|r| format!("{}", r)).collect();
            match ret {
                Some(r) => format!("{} = call_runtime {}({})", r, name, args_str.join(", ")),
                None => format!("call_runtime {}({})", name, args_str.join(", ")),
            }
        }
        Ret => "ret".to_string(),

        Prologue { frame_size } => format!("prologue frame_size={}", frame_size),
        Epilogue { frame_size } => format!("epilogue frame_size={}", frame_size),
        Push { src } => format!("push {}", src),
        Pop { dst } => format!("pop {}", dst),

        DefLabel(l) => format!("{}:", l),
        Nop => "nop".to_string(),
        Trap => "trap".to_string(),
        ParallelCopy { copies } => {
            let pairs: Vec<String> = copies
                .iter()
                .map(|(dst, src)| format!("{} <- {}", dst, src))
                .collect();
            format!("pcopy [{}]", pairs.join(", "))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::{Instruction, MirFunction, Terminator};

    fn make_mir(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    // -- ADT construction tests --

    #[test]
    fn test_vreg_display() {
        assert_eq!(format!("{}", VReg::gp(0)), "r0");
        assert_eq!(format!("{}", VReg::gp(15)), "r15");
        assert_eq!(format!("{}", VReg::fp(0)), "d0");
        assert_eq!(format!("{}", VReg::fp(7)), "d7");
    }

    #[test]
    fn test_vreg_class() {
        assert!(VReg::gp(0).is_gp());
        assert!(!VReg::gp(0).is_fp());
        assert!(VReg::fp(0).is_fp());
        assert!(!VReg::fp(0).is_gp());
    }

    #[test]
    fn test_label_display() {
        assert_eq!(format!("{}", Label(0)), "L0");
        assert_eq!(format!("{}", Label(42)), "L42");
    }

    #[test]
    fn test_mem_display() {
        assert_eq!(format!("{}", Mem::new(VReg::gp(0), 0)), "[r0]");
        assert_eq!(format!("{}", Mem::new(VReg::gp(1), 16)), "[r1 + 16]");
        assert_eq!(format!("{}", Mem::new(VReg::gp(2), -8)), "[r2 - 8]");
    }

    #[test]
    fn test_cond_invert() {
        assert_eq!(Cond::Eq.invert(), Cond::Ne);
        assert_eq!(Cond::Ne.invert(), Cond::Eq);
        assert_eq!(Cond::Lt.invert(), Cond::Ge);
        assert_eq!(Cond::Ge.invert(), Cond::Lt);
        assert_eq!(Cond::Below.invert(), Cond::AboveEq);
        // Double invert = identity
        assert_eq!(Cond::Lt.invert().invert(), Cond::Lt);
    }

    #[test]
    fn test_mach_func_alloc() {
        let mut mf = MachFunc::new("test".to_string());
        let r0 = mf.new_gp();
        let r1 = mf.new_gp();
        let d0 = mf.new_fp();
        let l0 = mf.new_label();
        assert_eq!(r0, VReg::gp(0));
        assert_eq!(r1, VReg::gp(1));
        assert_eq!(d0, VReg::fp(0));
        assert_eq!(l0, Label(0));
        assert_eq!(mf.num_gp_vregs(), 2);
        assert_eq!(mf.num_fp_vregs(), 1);
    }

    #[test]
    fn test_inst_uses_and_defs() {
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let r2 = VReg::gp(2);

        let inst = MachInst::IAdd {
            dst: r2,
            lhs: r0,
            rhs: r1,
        };
        assert_eq!(inst.def(), Some(r2));
        assert_eq!(inst.uses(), vec![r0, r1]);

        let load = MachInst::LoadImm {
            dst: r0,
            bits: 42,
        };
        assert_eq!(load.def(), Some(r0));
        assert!(load.uses().is_empty());

        let jmp = MachInst::Jmp {
            target: Label(0),
        };
        assert_eq!(jmp.def(), None);
        assert!(jmp.uses().is_empty());
        assert!(jmp.is_terminator());
    }

    #[test]
    fn test_inst_side_effects() {
        assert!(!MachInst::IAdd {
            dst: VReg::gp(0),
            lhs: VReg::gp(1),
            rhs: VReg::gp(2),
        }
        .has_side_effects());

        assert!(MachInst::Str {
            src: VReg::gp(0),
            mem: Mem::new(VReg::gp(1), 0),
        }
        .has_side_effects());

        assert!(MachInst::CallRuntime {
            name: "foo",
            args: vec![],
            ret: None,
        }
        .has_side_effects());

        assert!(MachInst::Ret.has_side_effects());
        assert!(MachInst::Trap.has_side_effects());
    }

    // -- Formatting tests --

    #[test]
    fn test_format_arithmetic() {
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let r2 = VReg::gp(2);
        let inst = MachInst::IAdd {
            dst: r2,
            lhs: r0,
            rhs: r1,
        };
        assert_eq!(format_inst(&inst), "r2 = iadd r0, r1");
    }

    #[test]
    fn test_format_fp() {
        let d0 = VReg::fp(0);
        let d1 = VReg::fp(1);
        let d2 = VReg::fp(2);
        let inst = MachInst::FAdd {
            dst: d2,
            lhs: d0,
            rhs: d1,
        };
        assert_eq!(format_inst(&inst), "d2 = fadd d0, d1");
    }

    #[test]
    fn test_format_memory() {
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let inst = MachInst::Ldr {
            dst: r0,
            mem: Mem::new(r1, 16),
        };
        assert_eq!(format_inst(&inst), "r0 = ldr [r1 + 16]");
    }

    #[test]
    fn test_format_branch() {
        let inst = MachInst::JmpIf {
            cond: Cond::Lt,
            target: Label(3),
        };
        assert_eq!(format_inst(&inst), "jmp_if Lt L3");
    }

    // -- MIR lowering tests --

    #[test]
    fn test_lower_const_num() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let mf = lower_mir(&f);
        let output = mf.display();
        assert!(output.contains("load_imm"));
        assert!(output.contains("ret"));
    }

    #[test]
    fn test_lower_fadd() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(1.0)));
            b.instructions.push((v1, Instruction::ConstF64(2.0)));
            b.instructions.push((v2, Instruction::AddF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let mf = lower_mir(&f);
        let output = mf.display();
        // Unboxed f64 add should produce a single fadd instruction.
        assert!(output.contains("fadd"));
        assert!(output.contains("load_fp_imm"));
    }

    #[test]
    fn test_lower_boxed_add_uses_runtime() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::ConstNum(2.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let mf = lower_mir(&f);
        let output = mf.display();
        // Boxed add should call the runtime.
        assert!(output.contains("wren_num_add"));
    }

    #[test]
    fn test_lower_unbox_box_roundtrip() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(3.14)));
            b.instructions.push((v1, Instruction::Unbox(v0)));
            b.instructions.push((v2, Instruction::Box(v1)));
            b.terminator = Terminator::Return(v2);
        }

        let mf = lower_mir(&f);
        let output = mf.display();
        assert!(output.contains("bitcast_gp_to_fp"));
        assert!(output.contains("bitcast_fp_to_gp"));
    }

    #[test]
    fn test_lower_fcmp() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let bb_t = f.new_block();
        let bb_f = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(1.0)));
            b.instructions.push((v1, Instruction::ConstF64(2.0)));
            b.instructions.push((v2, Instruction::CmpLtF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }
        f.block_mut(bb_t).terminator = Terminator::ReturnNull;
        f.block_mut(bb_f).terminator = Terminator::ReturnNull;

        let mf = lower_mir(&f);
        let output = mf.display();
        assert!(output.contains("fcmp"));
        assert!(output.contains("cset"));
    }

    #[test]
    fn test_lower_branch() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let v0 = f.new_value();
        {
            let b = f.block_mut(bb0);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![],
            };
        }
        f.block_mut(bb1).terminator = Terminator::Return(v0);

        let mf = lower_mir(&f);
        let output = mf.display();
        assert!(output.contains("jmp L"));
    }

    #[test]
    fn test_lower_unreachable() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::Unreachable;

        let mf = lower_mir(&f);
        let output = mf.display();
        assert!(output.contains("trap"));
    }

    #[test]
    fn test_lower_fmod_expansion() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(7.0)));
            b.instructions.push((v1, Instruction::ConstF64(3.0)));
            b.instructions.push((v2, Instruction::ModF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let mf = lower_mir(&f);
        let output = mf.display();
        // fmod expands to: fdiv, fcvt_to_i64, i64_cvt_to_f, fmul, fsub
        assert!(output.contains("fdiv"));
        assert!(output.contains("fcvt_to_i64"));
        assert!(output.contains("i64_cvt_to_f"));
        assert!(output.contains("fmul"));
        assert!(output.contains("fsub"));
    }

    // -- Physical register tests --

    #[test]
    fn test_aarch64_abi() {
        let abi = &phys_aarch64::ABI;
        assert_eq!(abi.gp_arg_regs.len(), 8);
        assert_eq!(abi.gp_ret, phys_aarch64::X0);
        assert_eq!(abi.frame_ptr, phys_aarch64::X29);
        assert!(abi.link_reg.is_some());
    }

    #[test]
    fn test_x86_64_abi() {
        let abi = &phys_x86_64::ABI;
        assert_eq!(abi.gp_arg_regs.len(), 6); // System V: rdi, rsi, rdx, rcx, r8, r9
        assert_eq!(abi.gp_ret, phys_x86_64::RAX);
        assert_eq!(abi.frame_ptr, phys_x86_64::RBP);
        assert!(abi.link_reg.is_none());
    }

    // -- Display test for full function --

    #[test]
    fn test_mach_func_display() {
        let mut mf = MachFunc::new("example".to_string());
        let r0 = mf.new_gp();
        let r1 = mf.new_gp();
        let d0 = mf.new_fp();
        let d1 = mf.new_fp();
        let d2 = mf.new_fp();
        let l0 = mf.new_label();

        mf.emit(MachInst::Prologue { frame_size: 16 });
        mf.emit(MachInst::DefLabel(l0));
        mf.emit(MachInst::LoadImm {
            dst: r0,
            bits: 42u64.to_be(),
        });
        mf.emit(MachInst::BitcastGpToFp { dst: d0, src: r0 });
        mf.emit(MachInst::LoadFpImm { dst: d1, value: 2.0 });
        mf.emit(MachInst::FAdd {
            dst: d2,
            lhs: d0,
            rhs: d1,
        });
        mf.emit(MachInst::BitcastFpToGp { dst: r1, src: d2 });
        mf.emit(MachInst::Epilogue { frame_size: 16 });
        mf.emit(MachInst::Ret);

        let output = mf.display();
        assert!(output.contains("mach_func example:"));
        assert!(output.contains("prologue"));
        assert!(output.contains("fadd"));
        assert!(output.contains("ret"));
    }

    // -- Parallel copy tests --

    #[test]
    fn test_pcopy_format() {
        let inst = MachInst::ParallelCopy {
            copies: vec![
                (VReg::gp(0), VReg::gp(1)),
                (VReg::gp(1), VReg::gp(0)),
            ],
        };
        assert_eq!(format_inst(&inst), "pcopy [r0 <- r1, r1 <- r0]");
    }

    #[test]
    fn test_pcopy_uses_and_defs() {
        let inst = MachInst::ParallelCopy {
            copies: vec![
                (VReg::gp(2), VReg::gp(0)),
                (VReg::gp(3), VReg::gp(1)),
            ],
        };
        assert_eq!(inst.uses(), vec![VReg::gp(0), VReg::gp(1)]);
        assert_eq!(inst.def(), None); // singular def is None for pcopy
        assert_eq!(inst.defs(), vec![VReg::gp(2), VReg::gp(3)]);
    }

    #[test]
    fn test_resolve_simple_no_conflict() {
        // r2 <- r0, r3 <- r1: no overlap, should produce two Movs.
        let mut mf = MachFunc::new("test".to_string());
        mf.emit(MachInst::ParallelCopy {
            copies: vec![
                (VReg::gp(2), VReg::gp(0)),
                (VReg::gp(3), VReg::gp(1)),
            ],
        });
        resolve_parallel_copies(&mut mf);

        assert_eq!(mf.insts.len(), 2);
        // Both should be Mov
        for inst in &mf.insts {
            assert!(matches!(inst, MachInst::Mov { .. }));
        }
    }

    #[test]
    fn test_resolve_identity_copies_eliminated() {
        // r0 <- r0: should produce nothing.
        let mut mf = MachFunc::new("test".to_string());
        mf.emit(MachInst::ParallelCopy {
            copies: vec![(VReg::gp(0), VReg::gp(0))],
        });
        resolve_parallel_copies(&mut mf);
        assert!(mf.insts.is_empty());
    }

    #[test]
    fn test_resolve_swap_cycle() {
        // r0 <- r1, r1 <- r0: a swap. Needs a temp.
        let mut mf = MachFunc::new("test".to_string());
        let _r0 = mf.new_gp(); // index 0
        let _r1 = mf.new_gp(); // index 1
        mf.emit(MachInst::ParallelCopy {
            copies: vec![
                (VReg::gp(0), VReg::gp(1)),
                (VReg::gp(1), VReg::gp(0)),
            ],
        });
        resolve_parallel_copies(&mut mf);

        // Should produce 3 Movs: tmp <- r1, r0(or r1) <- ..., last <- tmp
        assert_eq!(mf.insts.len(), 3);
        for inst in &mf.insts {
            assert!(matches!(inst, MachInst::Mov { .. }));
        }

        // Verify the temp is a new register (index >= 2).
        let uses_temp = mf.insts.iter().any(|inst| {
            if let MachInst::Mov { dst, src } = inst {
                dst.index >= 2 || src.index >= 2
            } else {
                false
            }
        });
        assert!(uses_temp, "swap should use a temp register");
    }

    #[test]
    fn test_resolve_three_way_rotation() {
        // r0 <- r1, r1 <- r2, r2 <- r0: a 3-cycle.
        let mut mf = MachFunc::new("test".to_string());
        for _ in 0..3 {
            mf.new_gp();
        }
        mf.emit(MachInst::ParallelCopy {
            copies: vec![
                (VReg::gp(0), VReg::gp(1)),
                (VReg::gp(1), VReg::gp(2)),
                (VReg::gp(2), VReg::gp(0)),
            ],
        });
        resolve_parallel_copies(&mut mf);

        // 3-cycle needs 4 moves (tmp <- src; chain of 2; close with tmp).
        assert_eq!(mf.insts.len(), 4);
    }

    #[test]
    fn test_resolve_chain_no_cycle() {
        // r2 <- r1, r1 <- r0: a chain, not a cycle. r1 is both dst and src
        // but there's no cycle (r0 is not a dst). Requires correct ordering.
        let mut mf = MachFunc::new("test".to_string());
        for _ in 0..3 {
            mf.new_gp();
        }
        mf.emit(MachInst::ParallelCopy {
            copies: vec![
                (VReg::gp(2), VReg::gp(1)),
                (VReg::gp(1), VReg::gp(0)),
            ],
        });
        resolve_parallel_copies(&mut mf);

        // Should be 2 moves, no temp needed.
        assert_eq!(mf.insts.len(), 2);
        // r2 <- r1 must come before r1 <- r0 to avoid clobbering.
        if let (MachInst::Mov { dst: d1, src: s1 }, MachInst::Mov { dst: d2, src: s2 }) =
            (&mf.insts[0], &mf.insts[1])
        {
            assert_eq!(*d1, VReg::gp(2));
            assert_eq!(*s1, VReg::gp(1));
            assert_eq!(*d2, VReg::gp(1));
            assert_eq!(*s2, VReg::gp(0));
        } else {
            panic!("expected two Mov instructions");
        }
    }

    #[test]
    fn test_resolve_fp_registers() {
        // FP swap: d0 <- d1, d1 <- d0.
        let mut mf = MachFunc::new("test".to_string());
        mf.new_fp();
        mf.new_fp();
        mf.emit(MachInst::ParallelCopy {
            copies: vec![
                (VReg::fp(0), VReg::fp(1)),
                (VReg::fp(1), VReg::fp(0)),
            ],
        });
        resolve_parallel_copies(&mut mf);

        assert_eq!(mf.insts.len(), 3);
        // All should be FMov for FP registers.
        for inst in &mf.insts {
            assert!(matches!(inst, MachInst::FMov { .. }));
        }
    }

    #[test]
    fn test_resolve_preserves_other_insts() {
        // Non-pcopy instructions should pass through unchanged.
        let mut mf = MachFunc::new("test".to_string());
        mf.emit(MachInst::LoadImm {
            dst: VReg::gp(0),
            bits: 42,
        });
        mf.emit(MachInst::ParallelCopy {
            copies: vec![(VReg::gp(1), VReg::gp(0))],
        });
        mf.emit(MachInst::Ret);
        resolve_parallel_copies(&mut mf);

        assert_eq!(mf.insts.len(), 3); // LoadImm, Mov, Ret
        assert!(matches!(mf.insts[0], MachInst::LoadImm { .. }));
        assert!(matches!(mf.insts[1], MachInst::Mov { .. }));
        assert!(matches!(mf.insts[2], MachInst::Ret));
    }

    // -- MIR lowering with block params --

    #[test]
    fn test_lower_branch_with_block_params() {
        use crate::mir::MirType;

        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();

        // bb0: v0 = const 42; branch bb1(v0)
        f.block_mut(bb0)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };

        // bb1(v1: val): return v1
        f.block_mut(bb1).params.push((v1, MirType::Value));
        f.block_mut(bb1).terminator = Terminator::Return(v1);

        let mf = lower_mir(&f);
        let output = mf.display();
        // Should contain a parallel copy for the block param binding.
        assert!(output.contains("pcopy"));
    }

    #[test]
    fn test_lower_cond_branch_with_block_params() {
        use crate::mir::MirType;

        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb0 = f.new_block();
        let bb_t = f.new_block();
        let bb_f = f.new_block();
        let v_cond = f.new_value();
        let v_a = f.new_value();
        let v_b = f.new_value();
        let v_pt = f.new_value();
        let v_pf = f.new_value();

        // bb0: cond=true, a=10, b=20; condbranch cond, bb_t(a), bb_f(b)
        {
            let b = f.block_mut(bb0);
            b.instructions
                .push((v_cond, Instruction::ConstBool(true)));
            b.instructions
                .push((v_a, Instruction::ConstNum(10.0)));
            b.instructions
                .push((v_b, Instruction::ConstNum(20.0)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb_t,
                true_args: vec![v_a],
                false_target: bb_f,
                false_args: vec![v_b],
            };
        }

        f.block_mut(bb_t).params.push((v_pt, MirType::Value));
        f.block_mut(bb_t).terminator = Terminator::Return(v_pt);

        f.block_mut(bb_f).params.push((v_pf, MirType::Value));
        f.block_mut(bb_f).terminator = Terminator::Return(v_pf);

        let mf = lower_mir(&f);
        let output = mf.display();
        // Should have two separate pcopy instructions for true/false paths.
        let pcopy_count = output.matches("pcopy").count();
        assert_eq!(pcopy_count, 2, "expected 2 pcopy for true/false paths, got:\n{}", output);
    }

    // -- FMA tests --

    #[test]
    fn test_fma_format_and_uses() {
        let d0 = VReg::fp(0);
        let d1 = VReg::fp(1);
        let d2 = VReg::fp(2);
        let d3 = VReg::fp(3);

        let fmadd = MachInst::FMAdd { dst: d3, a: d0, b: d1, c: d2 };
        assert_eq!(format_inst(&fmadd), "d3 = fmadd d0, d1, d2");
        assert_eq!(fmadd.uses(), vec![d0, d1, d2]);
        assert_eq!(fmadd.def(), Some(d3));

        let fmsub = MachInst::FMSub { dst: d3, a: d0, b: d1, c: d2 };
        assert_eq!(format_inst(&fmsub), "d3 = fmsub d0, d1, d2");

        let fnmadd = MachInst::FNMAdd { dst: d3, a: d0, b: d1, c: d2 };
        assert_eq!(format_inst(&fnmadd), "d3 = fnmadd d0, d1, d2");

        let fnmsub = MachInst::FNMSub { dst: d3, a: d0, b: d1, c: d2 };
        assert_eq!(format_inst(&fnmsub), "d3 = fnmsub d0, d1, d2");
    }

    // -- Vector / SIMD tests --

    #[test]
    fn test_vec_width() {
        assert_eq!(VecWidth::V128.lanes(), 2);
        assert_eq!(VecWidth::V256.lanes(), 4);
        assert_eq!(VecWidth::V128.bytes(), 16);
        assert_eq!(VecWidth::V256.bytes(), 32);
        assert_eq!(format!("{}", VecWidth::V128), "v128");
        assert_eq!(format!("{}", VecWidth::V256), "v256");
    }

    #[test]
    fn test_vreg_vec_class() {
        let v0 = VReg::vec(0);
        assert!(v0.is_vec());
        assert!(!v0.is_gp());
        assert!(!v0.is_fp());
        assert_eq!(format!("{}", v0), "v0");
    }

    #[test]
    fn test_mach_func_vec_alloc() {
        let mut mf = MachFunc::new("test".to_string());
        let v0 = mf.new_vec();
        let v1 = mf.new_vec();
        assert_eq!(v0, VReg::vec(0));
        assert_eq!(v1, VReg::vec(1));
        assert_eq!(mf.num_vec_vregs(), 2);
    }

    #[test]
    fn test_vec_arithmetic_format() {
        let v0 = VReg::vec(0);
        let v1 = VReg::vec(1);
        let v2 = VReg::vec(2);

        let vfadd = MachInst::VFAdd { dst: v2, lhs: v0, rhs: v1, width: VecWidth::V128 };
        assert_eq!(format_inst(&vfadd), "v2 = vfadd.v128 v0, v1");
        assert_eq!(vfadd.uses(), vec![v0, v1]);
        assert_eq!(vfadd.def(), Some(v2));

        let vfmul = MachInst::VFMul { dst: v2, lhs: v0, rhs: v1, width: VecWidth::V256 };
        assert_eq!(format_inst(&vfmul), "v2 = vfmul.v256 v0, v1");
    }

    #[test]
    fn test_vec_fma_format() {
        let v0 = VReg::vec(0);
        let v1 = VReg::vec(1);
        let v2 = VReg::vec(2);
        let v3 = VReg::vec(3);

        let vfmadd = MachInst::VFMAdd { dst: v3, a: v0, b: v1, c: v2, width: VecWidth::V128 };
        assert_eq!(format_inst(&vfmadd), "v3 = vfmadd.v128 v0, v1, v2");
        assert_eq!(vfmadd.uses(), vec![v0, v1, v2]);
        assert_eq!(vfmadd.def(), Some(v3));
    }

    #[test]
    fn test_vec_broadcast_extract() {
        let d0 = VReg::fp(0);
        let v0 = VReg::vec(0);
        let d1 = VReg::fp(1);

        let bcast = MachInst::VBroadcast { dst: v0, src: d0, width: VecWidth::V128 };
        assert_eq!(format_inst(&bcast), "v0 = vbroadcast.v128 d0");
        assert_eq!(bcast.uses(), vec![d0]);
        assert_eq!(bcast.def(), Some(v0));

        let extract = MachInst::VExtractLane { dst: d1, src: v0, lane: 1 };
        assert_eq!(format_inst(&extract), "d1 = vextract v0, #1");
        assert_eq!(extract.uses(), vec![v0]);
        assert_eq!(extract.def(), Some(d1));
    }

    #[test]
    fn test_vec_load_store() {
        let r0 = VReg::gp(0);
        let v0 = VReg::vec(0);

        let vload = MachInst::VLoad { dst: v0, mem: Mem::new(r0, 0), width: VecWidth::V128 };
        assert_eq!(format_inst(&vload), "v0 = vload.v128 [r0]");
        assert_eq!(vload.uses(), vec![r0]);
        assert_eq!(vload.def(), Some(v0));
        assert!(!vload.has_side_effects());

        let vstore = MachInst::VStore { src: v0, mem: Mem::new(r0, 32), width: VecWidth::V256 };
        assert_eq!(format_inst(&vstore), "vstore.v256 v0, [r0 + 32]");
        assert!(vstore.has_side_effects());
    }

    #[test]
    fn test_vec_reduce_and_neg() {
        let d0 = VReg::fp(0);
        let v0 = VReg::vec(0);
        let v1 = VReg::vec(1);

        let reduce = MachInst::VReduceAdd { dst: d0, src: v0, width: VecWidth::V128 };
        assert_eq!(format_inst(&reduce), "d0 = vreduce_add.v128 v0");
        assert_eq!(reduce.uses(), vec![v0]);
        assert_eq!(reduce.def(), Some(d0));

        let vneg = MachInst::VFNeg { dst: v1, src: v0, width: VecWidth::V128 };
        assert_eq!(format_inst(&vneg), "v1 = vfneg.v128 v0");
    }

    #[test]
    fn test_vec_insert_lane() {
        let v0 = VReg::vec(0);
        let v1 = VReg::vec(1);
        let d0 = VReg::fp(0);

        let insert = MachInst::VInsertLane { dst: v1, src: v0, lane: 0, val: d0 };
        assert_eq!(format_inst(&insert), "v1 = vinsert v0, #0, d0");
        assert_eq!(insert.uses(), vec![v0, d0]);
        assert_eq!(insert.def(), Some(v1));
    }

    /// Simulates a vectorized dot product: sum += a[i] * b[i]
    #[test]
    fn test_vec_dot_product_sequence() {
        let mut mf = MachFunc::new("dot".to_string());
        let base_a = mf.new_gp();
        let base_b = mf.new_gp();
        let va = mf.new_vec();
        let vb = mf.new_vec();
        let acc = mf.new_vec();
        let result_scalar = mf.new_fp();

        // acc = 0 (broadcast 0.0)
        let zero = mf.new_fp();
        mf.emit(MachInst::LoadFpImm { dst: zero, value: 0.0 });
        mf.emit(MachInst::VBroadcast { dst: acc, src: zero, width: VecWidth::V128 });

        // Load 2 f64s from a[] and b[]
        mf.emit(MachInst::VLoad { dst: va, mem: Mem::new(base_a, 0), width: VecWidth::V128 });
        mf.emit(MachInst::VLoad { dst: vb, mem: Mem::new(base_b, 0), width: VecWidth::V128 });

        // acc = va * vb + acc (fused multiply-add)
        mf.emit(MachInst::VFMAdd { dst: acc, a: va, b: vb, c: acc, width: VecWidth::V128 });

        // Horizontal reduce to scalar
        mf.emit(MachInst::VReduceAdd { dst: result_scalar, src: acc, width: VecWidth::V128 });

        mf.emit(MachInst::Ret);

        let output = mf.display();
        assert!(output.contains("vbroadcast.v128"));
        assert!(output.contains("vload.v128"));
        assert!(output.contains("vfmadd.v128"));
        assert!(output.contains("vreduce_add.v128"));
    }

    // -- Full pipeline tests --

    #[test]
    fn test_compile_function_x86_64() {
        // Use unboxed f64 ops (no runtime calls needed).
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(10.0)));
            b.instructions.push((v1, Instruction::ConstF64(32.0)));
            b.instructions.push((v2, Instruction::AddF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let result = compile_function(&f, Target::X86_64);
        assert!(result.is_ok(), "compile_function failed: {:?}", result.err());
        if let Ok(CompiledFunction::X86_64(code)) = result {
            assert!(code.len() > 0, "emitted code should not be empty");
        } else {
            panic!("expected X86_64 variant");
        }
    }

    #[test]
    fn test_compile_function_f64_ops_x86_64() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(3.0)));
            b.instructions.push((v1, Instruction::ConstF64(4.0)));
            b.instructions.push((v2, Instruction::AddF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let result = compile_function(&f, Target::X86_64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_function_with_branch_x86_64() {
        use crate::mir::MirType;

        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();

        f.block_mut(bb0).instructions.push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };
        f.block_mut(bb1).params.push((v1, MirType::Value));
        f.block_mut(bb1).terminator = Terminator::Return(v1);

        let result = compile_function(&f, Target::X86_64);
        assert!(result.is_ok(), "branch pipeline failed: {:?}", result.err());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_compile_function_aarch64() {
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(5.0)));
            b.instructions.push((v1, Instruction::ConstF64(7.0)));
            b.instructions.push((v2, Instruction::AddF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let result = compile_function(&f, Target::Aarch64);
        assert!(result.is_ok(), "aarch64 pipeline failed: {:?}", result.err());
    }

    // -- JIT execution tests (native arch only) --

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_jit_execute_add_f64() {
        // MIR: return 3.0 + 4.0 = 7.0 (as NaN-boxed bits in GP return reg)
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(3.0)));
            b.instructions.push((v1, Instruction::ConstF64(4.0)));
            b.instructions.push((v2, Instruction::AddF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let compiled = compile_function(&f, Target::Aarch64).unwrap();
        let exec = compiled.into_executable().unwrap();
        assert!(exec.is_native());

        unsafe {
            let func: extern "C" fn() -> u64 = exec.as_fn();
            let result_bits = func();
            let result = f64::from_bits(result_bits);
            assert_eq!(result, 7.0, "JIT: 3.0 + 4.0 should be 7.0, got {}", result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_jit_execute_mul_sub() {
        // MIR: (10.0 * 3.0) - 5.0 = 25.0
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        let v4 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(10.0)));
            b.instructions.push((v1, Instruction::ConstF64(3.0)));
            b.instructions.push((v2, Instruction::MulF64(v0, v1)));
            b.instructions.push((v3, Instruction::ConstF64(5.0)));
            b.instructions.push((v4, Instruction::SubF64(v2, v3)));
            b.terminator = Terminator::Return(v4);
        }

        let compiled = compile_function(&f, Target::Aarch64).unwrap();
        let exec = compiled.into_executable().unwrap();

        unsafe {
            let func: extern "C" fn() -> u64 = exec.as_fn();
            let result = f64::from_bits(func());
            assert_eq!(result, 25.0, "JIT: (10*3)-5 should be 25, got {}", result);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_jit_execute_const_num_boxed() {
        // MIR: return ConstNum(42.0) — returns NaN-boxed Value bits
        use crate::runtime::value::Value;

        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb).instructions.push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let compiled = compile_function(&f, Target::Aarch64).unwrap();
        let exec = compiled.into_executable().unwrap();

        unsafe {
            let func: extern "C" fn() -> u64 = exec.as_fn();
            let bits = func();
            let val = Value::from_bits(bits);
            assert_eq!(val.as_num(), Some(42.0), "JIT: ConstNum(42) should return 42.0");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_jit_execute_return_null() {
        use crate::runtime::value::Value;

        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let compiled = compile_function(&f, Target::Aarch64).unwrap();
        let exec = compiled.into_executable().unwrap();

        unsafe {
            let func: extern "C" fn() -> u64 = exec.as_fn();
            let bits = func();
            let val = Value::from_bits(bits);
            assert!(val.is_null(), "JIT: ReturnNull should return null, got bits 0x{:016x}", bits);
        }
    }

    #[test]
    fn test_into_executable_x86_64() {
        // Verify x86_64 code can be made executable (mmap succeeds).
        let mut interner = Interner::new();
        let mut f = make_mir(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb).instructions.push((v0, Instruction::ConstF64(1.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let compiled = compile_function(&f, Target::X86_64).unwrap();
        let exec = compiled.into_executable();
        assert!(exec.is_ok(), "mmap should succeed: {:?}", exec.err());

        let exec = exec.unwrap();
        // On aarch64 host, x86_64 code is not native.
        #[cfg(target_arch = "aarch64")]
        assert!(!exec.is_native());
        #[cfg(target_arch = "x86_64")]
        assert!(exec.is_native());
    }

    #[test]
    fn test_fixup_sentinels_x86_64() {
        let mut mf = MachFunc::new("test".to_string());
        // Simulate a spill load using sentinel registers.
        mf.emit(MachInst::Ldr {
            dst: VReg::gp(u32::MAX - 1),    // GP scratch sentinel
            mem: Mem::new(VReg::gp(u32::MAX), -8), // frame ptr sentinel
        });

        fixup_sentinels(&mut mf, Target::X86_64);

        if let MachInst::Ldr { dst, mem } = &mf.insts[0] {
            assert_eq!(dst.index, 11, "GP scratch should be R11 on x86_64");
            assert_eq!(mem.base.index, 5, "frame ptr should be RBP on x86_64");
        } else {
            panic!("expected Ldr");
        }
    }

    #[test]
    fn test_fixup_sentinels_aarch64() {
        let mut mf = MachFunc::new("test".to_string());
        mf.emit(MachInst::FLdr {
            dst: VReg::fp(u32::MAX - 1),    // FP scratch sentinel
            mem: Mem::new(VReg::gp(u32::MAX), -16), // frame ptr sentinel
        });

        fixup_sentinels(&mut mf, Target::Aarch64);

        if let MachInst::FLdr { dst, mem } = &mf.insts[0] {
            assert_eq!(dst.index, 16, "FP scratch should be D16 on aarch64");
            assert_eq!(mem.base.index, 29, "frame ptr should be X29 on aarch64");
        } else {
            panic!("expected FLdr");
        }
    }

    // -- Inline GEP tests --

    #[test]
    fn test_subscript_get_inline_gep() {
        let mut interner = crate::intern::Interner::new();
        let name = interner.intern("test_subscript_get");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v_list = f.new_value();
        let v_idx = f.new_value();
        let v_result = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v_list, Instruction::BlockParam(0)));
            b.instructions.push((v_idx, Instruction::BlockParam(1)));
            b.instructions.push((v_result, Instruction::SubscriptGet {
                receiver: v_list,
                args: vec![v_idx],
            }));
            b.terminator = Terminator::Return(v_result);
        }
        let mf = lower_mir(&f);
        let output = mf.display();
        println!("Generated code:\n{}", output);
        // Should use inline GEP (and_imm, ldr, shl, iadd) not CallRuntime
        assert!(!output.contains("call_runtime"), "SubscriptGet should use inline GEP, got:\n{}", output);
        assert!(output.contains("and_imm"), "should have ptr extraction: {}", output);
        assert!(output.contains("fcvt_to_i64"), "should convert index to int: {}", output);
        assert!(output.contains("shl"), "should scale index by 8: {}", output);
        assert!(output.contains("trap"), "should have bounds check trap: {}", output);
    }

    #[test]
    fn test_is_type_num_inline() {
        let mut interner = crate::intern::Interner::new();
        let name = interner.intern("test_is_num");
        let num_sym = interner.intern("Num");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v_val = f.new_value();
        let v_result = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v_val, Instruction::BlockParam(0)));
            b.instructions.push((v_result, Instruction::IsType(v_val, num_sym)));
            b.terminator = Terminator::Return(v_result);
        }
        let mf = lower_mir_with_interner(&f, &interner);
        let output = mf.display();
        // Should use inline tag check (and_imm + icmp + cset) not CallRuntime
        assert!(!output.contains("call_runtime"), "IsType(Num) should be inline, got:\n{}", output);
        assert!(output.contains("and_imm"), "should mask QNAN bits: {}", output);
        assert!(output.contains("cset"), "should materialize condition: {}", output);
    }

    #[test]
    fn test_is_type_null_inline() {
        let mut interner = crate::intern::Interner::new();
        let name = interner.intern("test_is_null");
        let null_sym = interner.intern("Null");
        let mut f = MirFunction::new(name, 0);
        let bb = f.new_block();
        let v_val = f.new_value();
        let v_result = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v_val, Instruction::BlockParam(0)));
            b.instructions.push((v_result, Instruction::IsType(v_val, null_sym)));
            b.terminator = Terminator::Return(v_result);
        }
        let mf = lower_mir_with_interner(&f, &interner);
        let output = mf.display();
        assert!(!output.contains("call_runtime"), "IsType(Null) should be inline, got:\n{}", output);
        assert!(output.contains("cset"), "should materialize eq condition: {}", output);
    }
}
