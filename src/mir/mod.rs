/// Mid-level SSA intermediate representation for WrenLift.
///
/// The MIR uses block parameters (not phi nodes) following Cranelift/MLIR
/// style. Each `MirFunction` is a control-flow graph of `BasicBlock`s,
/// each containing a linear sequence of `Instruction`s and a `Terminator`.
///
/// SSA values are referenced by `ValueId`. Each instruction produces at most
/// one value. Block parameters receive values from predecessor branches.
pub mod builder;
pub mod bytecode;
pub mod interp;
pub mod opt;
pub mod ssa;
// Threaded dispatch piggybacks on the host JIT runtime helpers
// (`wren_call_N` for indirect method calls). Wasm builds run BC-
// only and never see threaded code, so the whole module is host-
// gated; the BC interpreter's `threaded_*` references are gated
// to `host` too (see vm_interp.rs / engine.rs).
#[cfg(feature = "host")]
pub mod threaded;

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::intern::SymbolId;

// ---------------------------------------------------------------------------
// IDs (thin newtypes for type safety)
// ---------------------------------------------------------------------------

/// A reference to an SSA value.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ValueId(pub u32);

impl fmt::Debug for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A reference to a basic block.
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct BlockId(pub u32);

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Types (for typed MIR values)
// ---------------------------------------------------------------------------

/// MIR-level type for SSA values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MirType {
    /// A NaN-boxed Wren value (64-bit).
    Value,
    /// An unboxed f64 (for optimized numeric paths).
    F64,
    /// An unboxed boolean.
    Bool,
    /// An unboxed integer (for loop counters, indices).
    I64,
    /// No value (for instructions that don't produce a result).
    Void,
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

/// Unary math operation on an unboxed f64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MathUnaryOp {
    Abs,
    Acos,
    Asin,
    Atan,
    Cbrt,
    Ceil,
    Cos,
    Floor,
    Round,
    Sin,
    Sqrt,
    Tan,
    Log,
    Log2,
    Exp,
    Trunc,
    Fract,
    Sign,
}

impl MathUnaryOp {
    /// Apply this operation to an f64 value.
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Abs => x.abs(),
            Self::Acos => x.acos(),
            Self::Asin => x.asin(),
            Self::Atan => x.atan(),
            Self::Cbrt => x.cbrt(),
            Self::Ceil => x.ceil(),
            Self::Cos => x.cos(),
            Self::Floor => x.floor(),
            Self::Round => x.round(),
            Self::Sin => x.sin(),
            Self::Sqrt => x.sqrt(),
            Self::Tan => x.tan(),
            Self::Log => x.ln(),
            Self::Log2 => x.log2(),
            Self::Exp => x.exp(),
            Self::Trunc => x.trunc(),
            Self::Fract => x.fract(),
            Self::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Pretty-print name for this operation.
    pub fn name(self) -> &'static str {
        match self {
            Self::Abs => "fabs",
            Self::Acos => "facos",
            Self::Asin => "fasin",
            Self::Atan => "fatan",
            Self::Cbrt => "fcbrt",
            Self::Ceil => "fceil",
            Self::Cos => "fcos",
            Self::Floor => "ffloor",
            Self::Round => "fround",
            Self::Sin => "fsin",
            Self::Sqrt => "fsqrt",
            Self::Tan => "ftan",
            Self::Log => "flog",
            Self::Log2 => "flog2",
            Self::Exp => "fexp",
            Self::Trunc => "ftrunc",
            Self::Fract => "ffract",
            Self::Sign => "fsign",
        }
    }
}

/// Binary math operation on two unboxed f64 values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MathBinaryOp {
    Atan2,
    Min,
    Max,
    Pow,
}

impl MathBinaryOp {
    /// Apply this operation to two f64 values.
    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Atan2 => a.atan2(b),
            Self::Min => a.min(b),
            Self::Max => a.max(b),
            Self::Pow => a.powf(b),
        }
    }

    /// Pretty-print name for this operation.
    pub fn name(self) -> &'static str {
        match self {
            Self::Atan2 => "fatan2",
            Self::Min => "fmin",
            Self::Max => "fmax",
            Self::Pow => "fpow",
        }
    }
}

/// A single MIR instruction. Each produces at most one `ValueId`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Instruction {
    // -- Constants ----------------------------------------------------------
    /// Load an f64 constant as a boxed Value.
    ConstNum(f64),
    /// Load a boolean constant as a boxed Value.
    ConstBool(bool),
    /// Load null.
    ConstNull,
    /// Load a string constant (index into string table).
    ConstString(u32),

    // -- Unboxed constants (for optimized paths) ----------------------------
    /// Load an unboxed f64.
    ConstF64(f64),
    /// Load an unboxed i64.
    ConstI64(i64),

    // -- Boxed arithmetic (calls runtime helpers) ---------------------------
    /// Boxed add: Value + Value → Value (type-checked at runtime).
    Add(ValueId, ValueId),
    /// Boxed sub.
    Sub(ValueId, ValueId),
    /// Boxed mul.
    Mul(ValueId, ValueId),
    /// Boxed div.
    Div(ValueId, ValueId),
    /// Boxed mod.
    Mod(ValueId, ValueId),
    /// Boxed negate.
    Neg(ValueId),

    // -- Math intrinsics (unboxed f64, inlined from Num methods) ------------
    /// Unary math intrinsic on unboxed f64 (abs, sin, sqrt, etc.).
    MathUnaryF64(MathUnaryOp, ValueId),
    /// Binary math intrinsic on unboxed f64 (atan2, min, max, pow).
    MathBinaryF64(MathBinaryOp, ValueId, ValueId),

    // -- Unboxed f64 arithmetic (the big optimization win) ------------------
    /// Unboxed f64 add.
    AddF64(ValueId, ValueId),
    /// Unboxed f64 sub.
    SubF64(ValueId, ValueId),
    /// Unboxed f64 mul.
    MulF64(ValueId, ValueId),
    /// Unboxed f64 div.
    DivF64(ValueId, ValueId),
    /// Unboxed f64 mod.
    ModF64(ValueId, ValueId),
    /// Unboxed f64 negate.
    NegF64(ValueId),

    // -- Comparison ---------------------------------------------------------
    /// Compare less-than (boxed).
    CmpLt(ValueId, ValueId),
    /// Compare greater-than (boxed).
    CmpGt(ValueId, ValueId),
    /// Compare less-or-equal (boxed).
    CmpLe(ValueId, ValueId),
    /// Compare greater-or-equal (boxed).
    CmpGe(ValueId, ValueId),
    /// Compare equal (boxed).
    CmpEq(ValueId, ValueId),
    /// Compare not-equal (boxed).
    CmpNe(ValueId, ValueId),

    // -- Unboxed comparison → Bool ------------------------------------------
    CmpLtF64(ValueId, ValueId),
    CmpGtF64(ValueId, ValueId),
    CmpLeF64(ValueId, ValueId),
    CmpGeF64(ValueId, ValueId),

    // -- Logical / bitwise --------------------------------------------------
    /// Logical not (any value → Bool).
    Not(ValueId),
    /// Bitwise AND (boxed).
    BitAnd(ValueId, ValueId),
    /// Bitwise OR.
    BitOr(ValueId, ValueId),
    /// Bitwise XOR.
    BitXor(ValueId, ValueId),
    /// Bitwise NOT.
    BitNot(ValueId),
    /// Shift left.
    Shl(ValueId, ValueId),
    /// Shift right.
    Shr(ValueId, ValueId),

    // -- Type guards (for speculative optimization) -------------------------
    /// Assert that a value is a Num. Branches to deopt if not.
    GuardNum(ValueId),
    /// Assert that a value is a Bool.
    GuardBool(ValueId),
    /// Assert that a value is an instance of a specific class.
    GuardClass(ValueId, SymbolId),
    /// Assert that a value conforms to a protocol (protocol-based devirtualization guard).
    GuardProtocol(ValueId, crate::sema::protocol::ProtocolId),

    // -- Boxing / unboxing --------------------------------------------------
    /// Unbox a Value → f64 (assumes GuardNum passed).
    Unbox(ValueId),
    /// Box an f64 → Value.
    Box(ValueId),

    // -- Object operations --------------------------------------------------
    /// Read an instance field by index.
    GetField(ValueId, u16),
    /// Write an instance field by index.
    SetField(ValueId, u16, ValueId),
    /// Read a static field (__name) from the defining class.
    GetStaticField(SymbolId),
    /// Write a static field (__name) on the defining class.
    SetStaticField(SymbolId, ValueId),
    /// Read a module variable by index.
    GetModuleVar(u16),
    /// Write a module variable by index.
    SetModuleVar(u16, ValueId),

    // -- Calls --------------------------------------------------------------
    /// Method call: receiver, method symbol, args → result.
    Call {
        receiver: ValueId,
        method: SymbolId,
        args: Vec<ValueId>,
        /// MIR-builder annotation: this call dispatches to a method
        /// known not to mutate observable heap state (Num/String
        /// arithmetic + comparisons, Math, etc.). Used by CSE to keep
        /// its memory-read cache valid across the call instead of
        /// flushing on every method dispatch. Defaults to false; the
        /// MIR builder sets it at lowering time when the method
        /// symbol matches a built-in pure operator.
        pure_call: bool,
    },
    /// Direct call to a known function by FuncId. Emitted by speculative
    /// devirtualization when the receiver class is known from IC data.
    /// `expected_class` is the class pointer (as usize) observed at
    /// compile time; the JIT emits an inline class check and falls
    /// back to wren_call_N on polymorphic miss.
    CallKnownFunc {
        func_id: u32,
        method: SymbolId,
        expected_class: usize,
        /// If Some(field_idx), Cranelift emits an inlined field load
        /// (class check + get_field) instead of a function call.
        /// Valid only for trivial getters of the form `{ _field }`.
        inline_getter_field: Option<u16>,
        /// If true, the callee has no internal method calls — Cranelift
        /// can emit a pure `call_indirect` to `jit_code[func_id]` without
        /// any context setup or FFI round-trip.
        pure_leaf: bool,
        receiver: ValueId,
        args: Vec<ValueId>,
    },
    /// Recursive call to the current static method on its defining class.
    CallStaticSelf {
        args: Vec<ValueId>,
    },
    /// Super call: method symbol, args → result.
    SuperCall {
        method: SymbolId,
        args: Vec<ValueId>,
    },

    // -- Closures -----------------------------------------------------------
    /// Create a closure from a function ID and captured upvalues.
    MakeClosure {
        fn_id: u32,
        upvalues: Vec<ValueId>,
    },
    /// Read an upvalue.
    GetUpvalue(u16),
    /// Write an upvalue.
    SetUpvalue(u16, ValueId),

    // -- Collections --------------------------------------------------------
    /// Create a new list from elements.
    MakeList(Vec<ValueId>),
    /// Create a new map from key-value pairs.
    MakeMap(Vec<(ValueId, ValueId)>),
    /// Create a range (from, to, is_inclusive).
    MakeRange(ValueId, ValueId, bool),

    // -- String interpolation -----------------------------------------------
    /// Concatenate string parts.
    StringConcat(Vec<ValueId>),
    /// Convert value to string (for interpolation).
    ToString(ValueId),

    // -- Type test ----------------------------------------------------------
    /// `value is ClassName` → Bool.
    IsType(ValueId, SymbolId),

    // -- Subscript ----------------------------------------------------------
    /// Subscript get: `receiver[args]`.
    SubscriptGet {
        receiver: ValueId,
        args: Vec<ValueId>,
    },
    /// Subscript set: `receiver[args] = value`.
    SubscriptSet {
        receiver: ValueId,
        args: Vec<ValueId>,
        value: ValueId,
    },

    // -- Misc ---------------------------------------------------------------
    /// Move/copy a value (used during SSA construction).
    Move(ValueId),
    /// A block parameter (receives value from predecessors).
    BlockParam(u16),
    // -- Speculation guards (appended last: serialised bundles number
    // variants by position) -------------------------------------------------
    /// Raw Bool: the value is an object whose class is this pointer.
    /// Planted by the call inliner; JIT-only.
    ClassIs(ValueId, usize),
    /// Raw Bool: the value is exactly this object. JIT-only.
    ObjectIs(ValueId, usize),
    /// Raw Bool: the value is a closure of this `ObjFn` pointer. JIT-only.
    ClosureFnIs(ValueId, usize),

    // -- Integer arithmetic on values proven integral (JIT-only) ------------
    AddI64(ValueId, ValueId),
    SubI64(ValueId, ValueId),
    MulI64(ValueId, ValueId),
    /// Truncated remainder, as fmod on integers; the divisor is a nonzero constant.
    RemI64(ValueId, ValueId),
    BandI64(ValueId, ValueId),
    NegI64(ValueId),
    /// Raw Bool comparisons.
    CmpLtI64(ValueId, ValueId),
    CmpGtI64(ValueId, ValueId),
    CmpLeI64(ValueId, ValueId),
    CmpGeI64(ValueId, ValueId),
    /// Exact conversion of a proven-integral value back to f64.
    I64ToF64(ValueId),
}

impl Instruction {
    /// Does this instruction have side effects?
    /// Pure instructions can be eliminated by DCE if unused.
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Instruction::SetField(..)
                | Instruction::SetModuleVar(..)
                | Instruction::GuardNum(..)
                | Instruction::GuardBool(..)
                | Instruction::GuardClass(..)
                | Instruction::GuardProtocol(..)
                | Instruction::Call { .. }
                | Instruction::CallKnownFunc { .. }
                | Instruction::CallStaticSelf { .. }
                | Instruction::SuperCall { .. }
                | Instruction::SetUpvalue(..)
                | Instruction::SubscriptSet { .. }
                // Allocation instructions create new mutable objects — they must
                // NOT be hoisted out of loops by LICM even when their arguments
                // are loop-invariant, because each execution must produce a
                // distinct object.
                | Instruction::MakeList(..)
                | Instruction::MakeMap(..)
                | Instruction::MakeRange { .. }
                | Instruction::MakeClosure { .. }
                | Instruction::StringConcat(..)
                | Instruction::ToString(..)
                | Instruction::SetStaticField(..)
        )
    }

    /// Get the values this instruction reads.
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Instruction::ConstNum(_)
            | Instruction::ConstBool(_)
            | Instruction::ConstNull
            | Instruction::ConstString(_)
            | Instruction::ConstF64(_)
            | Instruction::ConstI64(_)
            | Instruction::GetModuleVar(_)
            | Instruction::GetUpvalue(_)
            | Instruction::BlockParam(_) => vec![],

            Instruction::Add(a, b)
            | Instruction::Sub(a, b)
            | Instruction::Mul(a, b)
            | Instruction::Div(a, b)
            | Instruction::Mod(a, b)
            | Instruction::AddF64(a, b)
            | Instruction::SubF64(a, b)
            | Instruction::MulF64(a, b)
            | Instruction::DivF64(a, b)
            | Instruction::ModF64(a, b)
            | Instruction::CmpLt(a, b)
            | Instruction::CmpGt(a, b)
            | Instruction::CmpLe(a, b)
            | Instruction::CmpGe(a, b)
            | Instruction::CmpEq(a, b)
            | Instruction::CmpNe(a, b)
            | Instruction::CmpLtF64(a, b)
            | Instruction::CmpGtF64(a, b)
            | Instruction::CmpLeF64(a, b)
            | Instruction::CmpGeF64(a, b)
            | Instruction::BitAnd(a, b)
            | Instruction::BitOr(a, b)
            | Instruction::BitXor(a, b)
            | Instruction::Shl(a, b)
            | Instruction::Shr(a, b)
            | Instruction::MathBinaryF64(_, a, b) => vec![*a, *b],

            Instruction::Neg(a)
            | Instruction::NegF64(a)
            | Instruction::Not(a)
            | Instruction::BitNot(a)
            | Instruction::GuardNum(a)
            | Instruction::GuardBool(a)
            | Instruction::Unbox(a)
            | Instruction::Box(a)
            | Instruction::Move(a)
            | Instruction::ToString(a)
            | Instruction::MathUnaryF64(_, a) => vec![*a],

            Instruction::GuardClass(a, _)
            | Instruction::GuardProtocol(a, _)
            | Instruction::IsType(a, _)
            | Instruction::ClassIs(a, _)
            | Instruction::ObjectIs(a, _)
            | Instruction::ClosureFnIs(a, _)
            | Instruction::NegI64(a)
            | Instruction::I64ToF64(a) => vec![*a],
            Instruction::AddI64(a, b)
            | Instruction::SubI64(a, b)
            | Instruction::MulI64(a, b)
            | Instruction::RemI64(a, b)
            | Instruction::BandI64(a, b)
            | Instruction::CmpLtI64(a, b)
            | Instruction::CmpGtI64(a, b)
            | Instruction::CmpLeI64(a, b)
            | Instruction::CmpGeI64(a, b) => vec![*a, *b],

            Instruction::GetField(recv, _) => vec![*recv],
            Instruction::SetField(recv, _, val) => vec![*recv, *val],
            Instruction::GetStaticField(_) => vec![],
            Instruction::SetStaticField(_, val) => vec![*val],
            Instruction::SetModuleVar(_, val) => vec![*val],
            Instruction::SetUpvalue(_, val) => vec![*val],

            Instruction::Call { receiver, args, .. }
            | Instruction::CallKnownFunc { receiver, args, .. } => {
                let mut ops = vec![*receiver];
                ops.extend(args);
                ops
            }
            Instruction::CallStaticSelf { args } => args.clone(),
            Instruction::SuperCall { args, .. } => args.clone(),

            Instruction::MakeClosure { upvalues, .. } => upvalues.clone(),
            Instruction::MakeList(elems) => elems.clone(),
            Instruction::MakeMap(pairs) => pairs.iter().flat_map(|(k, v)| [*k, *v]).collect(),
            Instruction::MakeRange(from, to, _) => vec![*from, *to],
            Instruction::StringConcat(parts) => parts.clone(),
            Instruction::SubscriptGet { receiver, args } => {
                let mut ops = vec![*receiver];
                ops.extend(args);
                ops
            }
            Instruction::SubscriptSet {
                receiver,
                args,
                value,
            } => {
                let mut ops = vec![*receiver];
                ops.extend(args);
                ops.push(*value);
                ops
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Terminator
// ---------------------------------------------------------------------------

/// How a basic block ends.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Terminator {
    /// Return a value from the function.
    Return(ValueId),
    /// Return null (implicit return).
    ReturnNull,
    /// Unconditional jump.
    Branch { target: BlockId, args: Vec<ValueId> },
    /// Conditional branch.
    CondBranch {
        condition: ValueId,
        true_target: BlockId,
        true_args: Vec<ValueId>,
        false_target: BlockId,
        false_args: Vec<ValueId>,
    },
    /// Unreachable (after a deopt guard fails, etc.).
    Unreachable,
}

impl Terminator {
    /// Get all successor block IDs.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Return(_) | Terminator::ReturnNull | Terminator::Unreachable => vec![],
            Terminator::Branch { target, .. } => vec![*target],
            Terminator::CondBranch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
        }
    }

    /// Get all values used by this terminator.
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Terminator::Return(v) => vec![*v],
            Terminator::ReturnNull | Terminator::Unreachable => vec![],
            Terminator::Branch { args, .. } => args.clone(),
            Terminator::CondBranch {
                condition,
                true_args,
                false_args,
                ..
            } => {
                let mut ops = vec![*condition];
                ops.extend(true_args);
                ops.extend(false_args);
                ops
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Basic block
// ---------------------------------------------------------------------------

/// A basic block in the MIR CFG.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BasicBlock {
    /// Block identifier.
    pub id: BlockId,
    /// Block parameters (like phi nodes, but explicit).
    pub params: Vec<(ValueId, MirType)>,
    /// Instructions in this block.
    pub instructions: Vec<(ValueId, Instruction)>,
    /// How this block ends.
    pub terminator: Terminator,
    /// Predecessor block IDs (populated during CFG construction).
    pub predecessors: Vec<BlockId>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            params: Vec::new(),
            instructions: Vec::new(),
            terminator: Terminator::Unreachable,
            predecessors: Vec::new(),
        }
    }

    /// All values defined in this block (params + instruction results).
    pub fn defined_values(&self) -> Vec<ValueId> {
        let mut vals: Vec<ValueId> = self.params.iter().map(|(v, _)| *v).collect();
        vals.extend(self.instructions.iter().map(|(v, _)| *v));
        vals
    }

    /// All values used in this block (instruction operands + terminator operands).
    pub fn used_values(&self) -> Vec<ValueId> {
        let mut vals = Vec::new();
        for (_, inst) in &self.instructions {
            vals.extend(inst.operands());
        }
        vals.extend(self.terminator.operands());
        vals
    }
}

// ---------------------------------------------------------------------------
// MIR Function
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Module-level MIR output (includes classes)
// ---------------------------------------------------------------------------

/// The complete MIR output for a module: top-level code + class definitions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModuleMir {
    pub top_level: MirFunction,
    pub classes: Vec<ClassMir>,
    /// Closure / nested function bodies referenced by MakeClosure instructions.
    pub closures: Vec<MirFunction>,
}

/// MIR for a user-defined class.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassMir {
    pub name: SymbolId,
    pub superclass: Option<SymbolId>,
    pub methods: Vec<MethodMir>,
    pub num_fields: u16,
    /// Protocols this class conforms to (computed during compilation).
    pub protocols: crate::sema::protocol::ProtocolSet,
    /// Runtime-visible attributes (`#key`, `#!` ones are stripped).
    #[serde(default)]
    pub attributes: Vec<AttrEntry>,
    /// Value of `#!native = "..."` on a foreign class — the native
    /// library name (or path) whose symbols back this class's methods.
    /// Only populated when the class is `foreign`.
    #[serde(default)]
    pub native_library: Option<String>,
    /// One entry per `foreign` method declared inside this class. These
    /// have no Wren body; at class install time the runtime looks up
    /// each `symbol` inside `native_library` (falling back to
    /// `bind_foreign_method_fn`) and binds a `WrenForeignMethodFn`
    /// trampoline.
    #[serde(default)]
    pub foreign_methods: Vec<ForeignMethodMir>,
}

/// MIR for a single method within a class.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MethodMir {
    /// Wren method signature (e.g. `"foo(_)"`, `"bar"`, `"[_]"`).
    pub signature: String,
    pub is_static: bool,
    pub is_constructor: bool,
    pub mir: MirFunction,
    /// Runtime-visible attributes attached to the method declaration.
    #[serde(default)]
    pub attributes: Vec<AttrEntry>,
}

/// Stub record for a `foreign` method. Carries only what the
/// runtime needs to resolve it at install time — no MIR body exists
/// because the implementation lives in an external shared library
/// (or the host's `bind_foreign_method_fn` callback).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForeignMethodMir {
    /// Wren method signature (e.g. "open(_)"). Used both for method-
    /// table binding and as the default symbol lookup key when `symbol`
    /// is `None`.
    pub signature: String,
    pub is_static: bool,
    /// Value of `#!symbol = "..."` if present. When absent the runtime
    /// loader falls back to the method's base name.
    #[serde(default)]
    pub symbol: Option<String>,
}

/// Flattened attribute record stored on MIR classes and methods. Each
/// entry represents one `key [= value]` pair; group entries share a
/// `group` name and flags store `value = None`. Compile-time attributes
/// (`#!`) are stripped before an `AttrEntry` is ever constructed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AttrEntry {
    pub group: Option<String>,
    pub key: String,
    pub value: Option<AttrValue>,
}

/// Literal payload of an attribute. Mirrors the parse-side
/// `ast::AttributeLiteral`, but owns its data so it can survive
/// `ModuleBlob` round-trips.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AttrValue {
    Num(f64),
    Str(String),
    Bool(bool),
    Null,
    Ident(String),
}

// ---------------------------------------------------------------------------
// MirFunction
// ---------------------------------------------------------------------------

/// A compiled function in MIR form.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MirFunction {
    /// Function name (for debugging).
    pub name: SymbolId,
    /// Number of parameters.
    pub arity: u8,
    /// Basic blocks (entry block is always `blocks[0]`).
    pub blocks: Vec<BasicBlock>,
    /// String constant table.
    pub strings: Vec<String>,
    /// Next available ValueId.
    pub next_value: u32,
    /// Next available BlockId.
    pub next_block: u32,
    /// Source span map: ValueId → source byte range (for runtime error reporting).
    pub span_map: std::collections::HashMap<ValueId, crate::ast::Span>,
    /// Block parameters the type specialiser assumed to be Num on the
    /// strength of their in-function edges. An OSR entry feeding one
    /// of these from the interpreter must check the value first.
    /// Compile-time only; never part of a serialised bundle.
    #[serde(skip)]
    pub speculated_num_params: Vec<ValueId>,
    /// Block parameters introduced by scalar replacement, mapped to the
    /// object parameter they split and the field they carry. Lets an
    /// OSR entry rebuild them from the object the interpreter holds.
    /// Compile-time only.
    #[serde(skip)]
    pub scalar_param_sources: std::collections::HashMap<ValueId, (ValueId, u16)>,
    /// Loop headers the interpreter never transfers into: headers of
    /// generic loop copies the inliner made, and headers whose loop had
    /// no inline-cache data at compile time (the interpreter keeps
    /// running those to warm them up). No OSR entry is compiled for them.
    /// Compile-time only.
    #[serde(skip)]
    pub osr_excluded: std::collections::HashSet<BlockId>,
}

impl MirFunction {
    pub fn new(name: SymbolId, arity: u8) -> Self {
        Self {
            name,
            arity,
            blocks: Vec::new(),
            strings: Vec::new(),
            next_value: 0,
            next_block: 0,
            span_map: std::collections::HashMap::new(),
            speculated_num_params: Vec::new(),
            scalar_param_sources: std::collections::HashMap::new(),
            osr_excluded: std::collections::HashSet::new(),
        }
    }

    /// Allocate a new ValueId.
    pub fn new_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    /// Allocate a new BasicBlock and return its ID.
    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.next_block);
        self.next_block += 1;
        self.blocks.push(BasicBlock::new(id));
        id
    }

    /// Get a block by its ID.
    pub fn block(&self, id: BlockId) -> &BasicBlock {
        &self.blocks[id.0 as usize]
    }

    /// Get a mutable block by its ID.
    pub fn block_mut(&mut self, id: BlockId) -> &mut BasicBlock {
        &mut self.blocks[id.0 as usize]
    }

    /// Add a string constant and return its index.
    pub fn add_string(&mut self, s: String) -> u32 {
        let idx = self.strings.len() as u32;
        self.strings.push(s);
        idx
    }

    /// The entry block (always block 0).
    pub fn entry_block(&self) -> BlockId {
        BlockId(0)
    }

    /// True if every non-`CallStaticSelf` instruction in this function is
    /// non-side-effecting, treating recursive self-calls as an unknown
    /// that resolves to "pure" when the rest of the body is pure.
    ///
    /// Used by CSE to decide whether `CallStaticSelf` invalidates the
    /// memory-read cache: a recursive numeric helper (`fact`, `fib`,
    /// etc.) made entirely of arithmetic + builtin pure calls + a
    /// recursive tail can keep the cache live across the recursion.
    ///
    /// Sound because `CallStaticSelf` always dispatches to *this*
    /// function — the only way the call could be impure is if the
    /// function itself has another impure instruction, in which case
    /// this check returns false.
    ///
    /// Allocation instructions (`MakeList`, `MakeMap`, etc.) count as
    /// side-effecting because each call must produce a distinct
    /// identity; ditto `StringConcat` / `ToString`. A "purely
    /// arithmetic" function with one of those is not eligible.
    pub fn is_pure_self_recursive(&self) -> bool {
        for block in &self.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::CallStaticSelf { .. } => {}
                    Instruction::Call { pure_call, .. } if *pure_call => {}
                    other if other.has_side_effects() => return false,
                    _ => {}
                }
            }
        }
        true
    }

    /// Dump a human-readable text representation of this function.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("fn {} (arity={}):\n", self.name, self.arity));
        if !self.strings.is_empty() {
            out.push_str("  strings:");
            for (i, s) in self.strings.iter().enumerate() {
                out.push_str(&format!(" [{}]={:?}", i, s));
            }
            out.push('\n');
        }
        for block in &self.blocks {
            // Block header with params
            out.push_str(&format!("  {}(", block.id));
            for (i, (val, ty)) in block.params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{}: {:?}", val, ty));
            }
            out.push_str("):\n");
            // Instructions
            for (val, inst) in &block.instructions {
                out.push_str(&format!("    {} = {:?}\n", val, inst));
            }
            // Terminator
            out.push_str(&format!("    -> {:?}\n", block.terminator));
        }
        out
    }

    /// Remap all SymbolId references in this function using a mapping function.
    ///
    /// This is needed when merging a parse interner into the VM interner,
    /// since the same string may have different SymbolId indices in each.
    pub fn remap_symbols<F>(&mut self, remap: F)
    where
        F: Fn(SymbolId) -> SymbolId,
    {
        // Remap function name
        self.name = remap(self.name);

        // Remap all instructions in all blocks
        for block in &mut self.blocks {
            for (_dst, inst) in &mut block.instructions {
                match inst {
                    Instruction::ConstString(idx) => {
                        let old = SymbolId::from_raw(*idx);
                        *idx = remap(old).index();
                    }
                    Instruction::Call { method, .. } => {
                        *method = remap(*method);
                    }
                    Instruction::CallStaticSelf { .. } => {}
                    Instruction::SuperCall { method, .. } => {
                        *method = remap(*method);
                    }
                    Instruction::GuardClass(_, cls) => {
                        *cls = remap(*cls);
                    }
                    Instruction::IsType(_, ty) => {
                        *ty = remap(*ty);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Populate predecessor lists from terminator edges.
    /// Drop empty, unreachable blocks at the end of the block list, such
    /// as the block the builder opens after an explicit `return`. Only
    /// trailing blocks go, so no block id changes.
    pub fn trim_dead_tail(&mut self) {
        loop {
            let n = self.blocks.len();
            if n <= 1 {
                return;
            }
            let last = &self.blocks[n - 1];
            let dead = last.instructions.is_empty()
                && last.params.is_empty()
                && matches!(last.terminator, Terminator::Unreachable);
            if !dead {
                return;
            }
            let id = last.id;
            let referenced = self
                .blocks
                .iter()
                .any(|b| b.terminator.successors().contains(&id));
            if referenced {
                return;
            }
            self.blocks.pop();
            self.next_block = self.blocks.len() as u32;
        }
    }

    pub fn compute_predecessors(&mut self) {
        // Clear existing.
        for block in &mut self.blocks {
            block.predecessors.clear();
        }
        // Build predecessor lists.
        let edges: Vec<(BlockId, Vec<BlockId>)> = self
            .blocks
            .iter()
            .map(|b| (b.id, b.terminator.successors()))
            .collect();
        for (src, succs) in edges {
            for succ in succs {
                self.blocks[succ.0 as usize].predecessors.push(src);
            }
        }
    }

    /// Pretty-print the MIR function.
    /// Pretty-print the MIR function in a CLIF-inspired text format.
    ///
    /// Output looks like:
    /// ```text
    /// function %test(i64) -> val {
    /// bb0(v0: val):
    ///     v1 = const.num 42.0
    ///     v2 = add v0, v1
    ///     return v2
    /// }
    /// ```
    pub fn pretty_print(&self, interner: &crate::intern::Interner) -> String {
        let mut out = format!(
            "function %{}({}) {{\n",
            interner.resolve(self.name),
            self.arity
        );

        for block in &self.blocks {
            // Block header with params
            out.push_str(&format!("{}(", block.id));
            for (i, (val, ty)) in block.params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("{}: {}", val, fmt_type(ty)));
            }
            out.push_str("):\n");

            // Predecessor comment
            if !block.predecessors.is_empty() {
                out.push_str("    ; preds:");
                for pred in &block.predecessors {
                    out.push_str(&format!(" {}", pred));
                }
                out.push('\n');
            }

            // Instructions
            for (val, inst) in &block.instructions {
                out.push_str(&format!(
                    "    {} = {}\n",
                    val,
                    fmt_instruction(inst, interner)
                ));
            }

            // Terminator
            out.push_str(&format!("    {}\n", fmt_terminator(&block.terminator)));
        }

        out.push_str("}\n");
        out
    }
}

/// Return true for MIR values the OSR entry can recreate exactly.
pub fn is_osr_rematerializable(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::ConstNum(_)
            | Instruction::ConstBool(_)
            | Instruction::ConstNull
            | Instruction::ConstF64(_)
            | Instruction::ConstI64(_)
    )
}

/// Constants defined outside `target`'s reachable region that can be rebuilt
/// in a native OSR entry instead of being passed from the interpreter.
pub fn osr_rematerializable_defs(
    func: &MirFunction,
    target: BlockId,
) -> HashMap<ValueId, Instruction> {
    let reachable = osr_reachable_blocks(func, target);
    let mut defs = HashMap::new();
    for (idx, block) in func.blocks.iter().enumerate() {
        if reachable.contains(&idx) {
            continue;
        }
        for &(vid, ref inst) in &block.instructions {
            if is_osr_rematerializable(inst) {
                defs.insert(vid, inst.clone());
            }
        }
    }
    defs
}

/// Values used by a loop/header region but defined outside it, excluding
/// constants that can be rematerialized. The order is deterministic and is
/// part of the bytecode-to-native OSR ABI.
/// Live-in value sets per block, one bit per value id.
pub struct LiveSets {
    words: usize,
    bits: Vec<u64>,
}

impl LiveSets {
    /// Whether `v` is live on entry to block `b`.
    pub fn contains(&self, b: usize, v: ValueId) -> bool {
        let i = v.0 as usize;
        i / 64 < self.words && self.bits[b * self.words + i / 64] >> (i % 64) & 1 == 1
    }

    /// The values live on entry to block `b`, ascending.
    pub fn iter(&self, b: usize) -> impl Iterator<Item = ValueId> + '_ {
        let row = &self.bits[b * self.words..(b + 1) * self.words];
        row.iter().enumerate().flat_map(|(w, &word)| {
            let mut bits = word;
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let tz = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(ValueId((w * 64 + tz) as u32))
            })
        })
    }
}

/// Classic backward liveness: the values live on entry to each block.
pub fn live_in_sets(func: &MirFunction) -> LiveSets {
    let n = func.blocks.len();
    let max_value = func
        .blocks
        .iter()
        .flat_map(|b| {
            b.params
                .iter()
                .map(|(v, _)| v.0)
                .chain(b.instructions.iter().map(|(v, _)| v.0))
                .chain(b.used_values().into_iter().map(|v| v.0))
        })
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(0)
        .max(func.next_value as usize);
    let words = max_value.div_ceil(64).max(1);
    let mut gens = vec![0u64; n * words];
    let mut defs = vec![0u64; n * words];
    let set = |bits: &mut [u64], v: ValueId| {
        let i = v.0 as usize;
        bits[i / 64] |= 1 << (i % 64);
    };
    let has = |bits: &[u64], v: ValueId| {
        let i = v.0 as usize;
        bits[i / 64] >> (i % 64) & 1 == 1
    };
    for (b, block) in func.blocks.iter().enumerate() {
        let d = &mut defs[b * words..(b + 1) * words];
        let mut g = vec![0u64; words];
        for &(param, _) in &block.params {
            set(d, param);
        }
        for &(dst, ref inst) in &block.instructions {
            for op in inst.operands() {
                if !has(d, op) {
                    set(&mut g, op);
                }
            }
            set(d, dst);
        }
        for op in block.terminator.operands() {
            if !has(d, op) {
                set(&mut g, op);
            }
        }
        gens[b * words..(b + 1) * words].copy_from_slice(&g);
    }
    let mut live = gens.clone();
    let mut out = vec![0u64; words];
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            out.iter_mut().for_each(|w| *w = 0);
            for succ in func.blocks[b].terminator.successors() {
                let s = succ.0 as usize;
                if s < n {
                    for w in 0..words {
                        out[w] |= live[s * words + w];
                    }
                }
            }
            for (w, out_w) in out.iter().enumerate() {
                let idx = b * words + w;
                let next = gens[idx] | (out_w & !defs[idx]);
                if next != live[idx] {
                    live[idx] = next;
                    changed = true;
                }
            }
        }
    }
    LiveSets { words, bits: live }
}

pub fn osr_external_live_values(func: &MirFunction, target: BlockId) -> Vec<ValueId> {
    // A value is live-in at the target if some path from the target
    // uses it before redefining it. That includes values defined inside
    // the loop region on an earlier trip around an enclosing loop (a
    // range built in the outer body and consumed by the inner header),
    // which a reachability-only rule misclassifies as internal.
    let live_in = live_in_sets(func);
    let Some(target_block) = func.blocks.get(target.0 as usize) else {
        return Vec::new();
    };
    let rematerializable = osr_rematerializable_defs(func, target);
    let params: HashSet<ValueId> = target_block.params.iter().map(|&(p, _)| p).collect();
    live_in
        .iter(target.0 as usize)
        .filter(|v| !params.contains(v) && !rematerializable.contains_key(v))
        .collect()
}

/// Set of block indices reachable via successor edges from `start`.
/// Guards against out-of-range successor targets so malformed MIR
/// can't index out of bounds. Used by OSR entry analysis — layout
/// validity check and `osr_external_live_values` MUST agree on
/// reachability or val_map / external_args get out of sync and
/// lowering blows up.
pub fn osr_reachable_blocks(func: &MirFunction, start: BlockId) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    fn dfs(idx: usize, func: &MirFunction, reachable: &mut HashSet<usize>) {
        if idx >= func.blocks.len() || !reachable.insert(idx) {
            return;
        }
        for succ in func.blocks[idx].terminator.successors() {
            dfs(succ.0 as usize, func, reachable);
        }
    }
    dfs(start.0 as usize, func, &mut reachable);
    reachable
}

// ---------------------------------------------------------------------------
// CLIF-style formatting helpers
// ---------------------------------------------------------------------------

fn fmt_type(ty: &MirType) -> &'static str {
    match ty {
        MirType::Value => "val",
        MirType::F64 => "f64",
        MirType::Bool => "bool",
        MirType::I64 => "i64",
        MirType::Void => "void",
    }
}

fn fmt_val_list(vals: &[ValueId]) -> String {
    vals.iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join(", ")
}

fn fmt_instruction(inst: &Instruction, interner: &crate::intern::Interner) -> String {
    match inst {
        Instruction::ConstNum(n) => format!("const.num {}", n),
        Instruction::ConstBool(b) => format!("const.bool {}", b),
        Instruction::ConstNull => "const.null".to_string(),
        Instruction::ConstString(idx) => format!("const.str @{}", idx),
        Instruction::ConstF64(n) => format!("const.f64 {}", n),
        Instruction::ConstI64(n) => format!("const.i64 {}", n),

        Instruction::Add(a, b) => format!("add {}, {}", a, b),
        Instruction::Sub(a, b) => format!("sub {}, {}", a, b),
        Instruction::Mul(a, b) => format!("mul {}, {}", a, b),
        Instruction::Div(a, b) => format!("div {}, {}", a, b),
        Instruction::Mod(a, b) => format!("mod {}, {}", a, b),
        Instruction::Neg(a) => format!("neg {}", a),

        Instruction::AddF64(a, b) => format!("fadd {}, {}", a, b),
        Instruction::SubF64(a, b) => format!("fsub {}, {}", a, b),
        Instruction::MulF64(a, b) => format!("fmul {}, {}", a, b),
        Instruction::DivF64(a, b) => format!("fdiv {}, {}", a, b),
        Instruction::ModF64(a, b) => format!("fmod {}, {}", a, b),
        Instruction::NegF64(a) => format!("fneg {}", a),

        Instruction::MathUnaryF64(op, a) => format!("{} {}", op.name(), a),
        Instruction::MathBinaryF64(op, a, b) => format!("{} {}, {}", op.name(), a, b),

        Instruction::CmpLt(a, b) => format!("icmp.lt {}, {}", a, b),
        Instruction::CmpGt(a, b) => format!("icmp.gt {}, {}", a, b),
        Instruction::CmpLe(a, b) => format!("icmp.le {}, {}", a, b),
        Instruction::CmpGe(a, b) => format!("icmp.ge {}, {}", a, b),
        Instruction::CmpEq(a, b) => format!("icmp.eq {}, {}", a, b),
        Instruction::CmpNe(a, b) => format!("icmp.ne {}, {}", a, b),

        Instruction::CmpLtF64(a, b) => format!("fcmp.lt {}, {}", a, b),
        Instruction::CmpGtF64(a, b) => format!("fcmp.gt {}, {}", a, b),
        Instruction::CmpLeF64(a, b) => format!("fcmp.le {}, {}", a, b),
        Instruction::CmpGeF64(a, b) => format!("fcmp.ge {}, {}", a, b),

        Instruction::Not(a) => format!("not {}", a),
        Instruction::BitAnd(a, b) => format!("band {}, {}", a, b),
        Instruction::BitOr(a, b) => format!("bor {}, {}", a, b),
        Instruction::BitXor(a, b) => format!("bxor {}, {}", a, b),
        Instruction::BitNot(a) => format!("bnot {}", a),
        Instruction::Shl(a, b) => format!("ishl {}, {}", a, b),
        Instruction::Shr(a, b) => format!("sshr {}, {}", a, b),

        Instruction::GuardNum(a) => format!("guard.num {}", a),
        Instruction::GuardBool(a) => format!("guard.bool {}", a),
        Instruction::GuardClass(a, sym) => {
            format!("guard.class {}, %{}", a, interner.resolve(*sym))
        }
        Instruction::GuardProtocol(a, pid) => {
            let name = crate::sema::protocol::BUILTIN_PROTOCOLS
                .get(pid.0 as usize)
                .map(|p| p.name)
                .unwrap_or("?");
            format!("guard.protocol {}, {}", a, name)
        }

        Instruction::Unbox(a) => format!("unbox {}", a),
        Instruction::Box(a) => format!("box {}", a),

        Instruction::GetField(recv, idx) => format!("get_field {}, #{}", recv, idx),
        Instruction::SetField(recv, idx, val) => {
            format!("set_field {}, #{}, {}", recv, idx, val)
        }
        Instruction::GetStaticField(sym) => format!("get_static_field :{}", sym.index()),
        Instruction::SetStaticField(sym, val) => {
            format!("set_static_field :{}, {}", sym.index(), val)
        }
        Instruction::GetModuleVar(idx) => format!("get_module_var @{}", idx),
        Instruction::SetModuleVar(idx, val) => format!("set_module_var @{}, {}", idx, val),

        Instruction::Call {
            receiver,
            method,
            args,
            pure_call,
        } => format!(
            "call{} {}.%{}({})",
            if *pure_call { "[pure]" } else { "" },
            receiver,
            interner.resolve(*method),
            fmt_val_list(args)
        ),
        Instruction::CallStaticSelf { args } => {
            format!("call_static_self({})", fmt_val_list(args))
        }
        Instruction::CallKnownFunc {
            func_id,
            method: _,
            expected_class: _,
            inline_getter_field: _,
            pure_leaf: _,
            receiver,
            args,
        } => format!(
            "call_known FuncId({}) {}({})",
            func_id,
            receiver,
            fmt_val_list(args)
        ),
        Instruction::SuperCall { method, args } => {
            format!(
                "super_call %{}({})",
                interner.resolve(*method),
                fmt_val_list(args)
            )
        }

        Instruction::MakeClosure { fn_id, upvalues } => {
            format!("make_closure fn#{}, [{}]", fn_id, fmt_val_list(upvalues))
        }
        Instruction::GetUpvalue(idx) => format!("get_upvalue #{}", idx),
        Instruction::SetUpvalue(idx, val) => format!("set_upvalue #{}, {}", idx, val),

        Instruction::MakeList(elems) => format!("make_list [{}]", fmt_val_list(elems)),
        Instruction::MakeMap(pairs) => {
            let entries: Vec<String> = pairs.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
            format!("make_map {{{}}}", entries.join(", "))
        }
        Instruction::MakeRange(from, to, inclusive) => {
            let op = if *inclusive { ".." } else { "..." };
            format!("make_range {}{}{}", from, op, to)
        }

        Instruction::StringConcat(parts) => format!("str_concat [{}]", fmt_val_list(parts)),
        Instruction::ToString(a) => format!("to_string {}", a),
        Instruction::ClassIs(a, class) => format!("class_is {}, {:#x}", a, class),
        Instruction::ObjectIs(a, obj) => format!("object_is {}, {:#x}", a, obj),
        Instruction::AddI64(a, b) => format!("iadd {}, {}", a, b),
        Instruction::SubI64(a, b) => format!("isub {}, {}", a, b),
        Instruction::MulI64(a, b) => format!("imul {}, {}", a, b),
        Instruction::RemI64(a, b) => format!("irem {}, {}", a, b),
        Instruction::BandI64(a, b) => format!("iand {}, {}", a, b),
        Instruction::NegI64(a) => format!("ineg {}", a),
        Instruction::CmpLtI64(a, b) => format!("icmp_i64.lt {}, {}", a, b),
        Instruction::CmpGtI64(a, b) => format!("icmp_i64.gt {}, {}", a, b),
        Instruction::CmpLeI64(a, b) => format!("icmp_i64.le {}, {}", a, b),
        Instruction::CmpGeI64(a, b) => format!("icmp_i64.ge {}, {}", a, b),
        Instruction::I64ToF64(a) => format!("i64_to_f64 {}", a),
        Instruction::ClosureFnIs(a, f) => format!("closure_fn_is {}, {:#x}", a, f),

        Instruction::IsType(a, sym) => {
            format!("is_type {}, %{}", a, interner.resolve(*sym))
        }

        Instruction::SubscriptGet { receiver, args } => {
            format!("subscript_get {}[{}]", receiver, fmt_val_list(args))
        }
        Instruction::SubscriptSet {
            receiver,
            args,
            value,
        } => {
            format!(
                "subscript_set {}[{}] = {}",
                receiver,
                fmt_val_list(args),
                value
            )
        }

        Instruction::Move(a) => format!("move {}", a),
        Instruction::BlockParam(idx) => format!("block_param #{}", idx),
    }
}

fn fmt_terminator(term: &Terminator) -> String {
    match term {
        Terminator::Return(v) => format!("return {}", v),
        Terminator::ReturnNull => "return".to_string(),
        Terminator::Branch { target, args } => {
            if args.is_empty() {
                format!("jump {}", target)
            } else {
                format!("jump {}({})", target, fmt_val_list(args))
            }
        }
        Terminator::CondBranch {
            condition,
            true_target,
            true_args,
            false_target,
            false_args,
        } => {
            let true_part = if true_args.is_empty() {
                format!("{}", true_target)
            } else {
                format!("{}({})", true_target, fmt_val_list(true_args))
            };
            let false_part = if false_args.is_empty() {
                format!("{}", false_target)
            } else {
                format!("{}({})", false_target, fmt_val_list(false_args))
            };
            format!("brif {}, {}, {}", condition, true_part, false_part)
        }
        Terminator::Unreachable => "unreachable".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;

    fn test_fn(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_value_id_display() {
        assert_eq!(format!("{}", ValueId(0)), "v0");
        assert_eq!(format!("{}", ValueId(42)), "v42");
    }

    #[test]
    fn test_block_id_display() {
        assert_eq!(format!("{}", BlockId(0)), "bb0");
        assert_eq!(format!("{}", BlockId(3)), "bb3");
    }

    #[test]
    fn test_new_value_ids_sequential() {
        let mut interner = Interner::new();
        let mut f = test_fn(&mut interner);
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        assert_eq!(v0, ValueId(0));
        assert_eq!(v1, ValueId(1));
        assert_eq!(v2, ValueId(2));
    }

    #[test]
    fn test_new_blocks_sequential() {
        let mut interner = Interner::new();
        let mut f = test_fn(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        assert_eq!(bb0, BlockId(0));
        assert_eq!(bb1, BlockId(1));
        assert_eq!(f.blocks.len(), 2);
    }

    #[test]
    fn test_entry_block() {
        let mut interner = Interner::new();
        let mut f = test_fn(&mut interner);
        f.new_block();
        assert_eq!(f.entry_block(), BlockId(0));
    }

    #[test]
    fn test_string_table() {
        let mut interner = Interner::new();
        let mut f = test_fn(&mut interner);
        let idx0 = f.add_string("hello".to_string());
        let idx1 = f.add_string("world".to_string());
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(f.strings[0], "hello");
        assert_eq!(f.strings[1], "world");
    }

    #[test]
    fn test_instruction_side_effects() {
        assert!(!Instruction::ConstNum(1.0).has_side_effects());
        assert!(!Instruction::Add(ValueId(0), ValueId(1)).has_side_effects());
        assert!(!Instruction::AddF64(ValueId(0), ValueId(1)).has_side_effects());
        assert!(Instruction::Call {
            receiver: ValueId(0),
            method: SymbolId::from_raw(0),
            args: vec![],
            pure_call: false,
        }
        .has_side_effects());
        assert!(Instruction::SetField(ValueId(0), 0, ValueId(1)).has_side_effects());
        assert!(Instruction::SetModuleVar(0, ValueId(0)).has_side_effects());
    }

    #[test]
    fn test_instruction_operands() {
        assert!(Instruction::ConstNum(1.0).operands().is_empty());
        assert_eq!(
            Instruction::Add(ValueId(0), ValueId(1)).operands(),
            vec![ValueId(0), ValueId(1)]
        );
        assert_eq!(Instruction::Neg(ValueId(2)).operands(), vec![ValueId(2)]);
        assert_eq!(
            Instruction::Call {
                receiver: ValueId(0),
                method: SymbolId::from_raw(1),
                args: vec![ValueId(2), ValueId(3)],
                pure_call: false,
            }
            .operands(),
            vec![ValueId(0), ValueId(2), ValueId(3)]
        );
    }

    #[test]
    fn test_terminator_successors() {
        assert!(Terminator::ReturnNull.successors().is_empty());
        assert_eq!(
            Terminator::Branch {
                target: BlockId(1),
                args: vec![]
            }
            .successors(),
            vec![BlockId(1)]
        );
        assert_eq!(
            Terminator::CondBranch {
                condition: ValueId(0),
                true_target: BlockId(1),
                true_args: vec![],
                false_target: BlockId(2),
                false_args: vec![],
            }
            .successors(),
            vec![BlockId(1), BlockId(2)]
        );
    }

    #[test]
    fn test_compute_predecessors() {
        let mut interner = Interner::new();
        let mut f = test_fn(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();

        f.block_mut(bb0).terminator = Terminator::CondBranch {
            condition: ValueId(0),
            true_target: bb1,
            true_args: vec![],
            false_target: bb2,
            false_args: vec![],
        };
        f.block_mut(bb1).terminator = Terminator::Branch {
            target: bb2,
            args: vec![],
        };
        f.block_mut(bb2).terminator = Terminator::ReturnNull;

        f.compute_predecessors();

        assert!(f.block(bb0).predecessors.is_empty());
        assert_eq!(f.block(bb1).predecessors, vec![bb0]);
        assert_eq!(f.block(bb2).predecessors, vec![bb0, bb1]);
    }

    #[test]
    fn test_block_defined_and_used_values() {
        let mut block = BasicBlock::new(BlockId(0));
        block.params.push((ValueId(0), MirType::Value));
        block
            .instructions
            .push((ValueId(1), Instruction::ConstNum(42.0)));
        block
            .instructions
            .push((ValueId(2), Instruction::Add(ValueId(0), ValueId(1))));
        block.terminator = Terminator::Return(ValueId(2));

        let defined = block.defined_values();
        assert_eq!(defined, vec![ValueId(0), ValueId(1), ValueId(2)]);

        let used = block.used_values();
        assert!(used.contains(&ValueId(0)));
        assert!(used.contains(&ValueId(1)));
        assert!(used.contains(&ValueId(2)));
    }

    #[test]
    fn test_pretty_print() {
        let mut interner = Interner::new();
        let mut f = test_fn(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();

        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(1.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(2.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Add(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let output = f.pretty_print(&interner);
        assert!(output.contains("function %test(0)"));
        assert!(output.contains("bb0"));
        assert!(output.contains("const.num 1"));
        assert!(output.contains("add v0, v1"));
        assert!(output.contains("return v2"));
    }

    #[test]
    fn test_mir_type_variants() {
        assert_eq!(MirType::Value, MirType::Value);
        assert_ne!(MirType::Value, MirType::F64);
        assert_ne!(MirType::F64, MirType::Bool);
    }

    #[test]
    fn test_block_params() {
        let mut interner = Interner::new();
        let mut f = test_fn(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();

        // bb0 branches to bb1 with a value.
        let v0 = f.new_value();
        f.block_mut(bb0)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));

        let v1 = f.new_value();
        f.block_mut(bb1).params.push((v1, MirType::Value));

        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };
        f.block_mut(bb1).terminator = Terminator::Return(v1);

        f.compute_predecessors();
        assert_eq!(f.block(bb1).predecessors, vec![bb0]);
        assert_eq!(f.block(bb1).params.len(), 1);
    }
}

/// The representation type of every value: boxed `Value`, raw `F64`,
/// `I64`, or `Bool` for comparison results; `Void` for ids without a
/// definition.
pub fn infer_value_types(mir: &MirFunction) -> Vec<MirType> {
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
                | Instruction::IsType(..)
                | Instruction::ClassIs(..)
                | Instruction::ObjectIs(..)
                | Instruction::ClosureFnIs(..)
                | Instruction::CmpLtI64(..)
                | Instruction::CmpGtI64(..)
                | Instruction::CmpLeI64(..)
                | Instruction::CmpGeI64(..) => MirType::Bool,
                Instruction::AddI64(..)
                | Instruction::SubI64(..)
                | Instruction::MulI64(..)
                | Instruction::RemI64(..)
                | Instruction::BandI64(..)
                | Instruction::NegI64(_) => MirType::I64,
                Instruction::I64ToF64(_) => MirType::F64,
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
