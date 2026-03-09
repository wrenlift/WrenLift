/// Mid-level SSA intermediate representation for WrenLift.
///
/// The MIR uses block parameters (not phi nodes) following Cranelift/MLIR
/// style. Each `MirFunction` is a control-flow graph of `BasicBlock`s,
/// each containing a linear sequence of `Instruction`s and a `Terminator`.
///
/// SSA values are referenced by `ValueId`. Each instruction produces at most
/// one value. Block parameters receive values from predecessor branches.

pub mod builder;
pub mod interp;
pub mod opt;
pub mod ssa;

use std::fmt;

use crate::intern::SymbolId;

// ---------------------------------------------------------------------------
// IDs (thin newtypes for type safety)
// ---------------------------------------------------------------------------

/// A reference to an SSA value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
                if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone)]
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

    /// Subscript get: receiver[args].
    SubscriptGet {
        receiver: ValueId,
        args: Vec<ValueId>,
    },
    /// Subscript set: receiver[args] = value.
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
}

impl Instruction {
    /// Does this instruction have side effects?
    /// Pure instructions can be eliminated by DCE if unused.
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            Instruction::SetField(..)
                | Instruction::SetModuleVar(..)
                | Instruction::Call { .. }
                | Instruction::SuperCall { .. }
                | Instruction::SetUpvalue(..)
                | Instruction::SubscriptSet { .. }
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

            Instruction::GuardClass(a, _) | Instruction::IsType(a, _) => vec![*a],

            Instruction::GetField(recv, _) => vec![*recv],
            Instruction::SetField(recv, _, val) => vec![*recv, *val],
            Instruction::SetModuleVar(_, val) => vec![*val],
            Instruction::SetUpvalue(_, val) => vec![*val],

            Instruction::Call { receiver, args, .. } => {
                let mut ops = vec![*receiver];
                ops.extend(args);
                ops
            }
            Instruction::SuperCall { args, .. } => args.clone(),

            Instruction::MakeClosure { upvalues, .. } => upvalues.clone(),
            Instruction::MakeList(elems) => elems.clone(),
            Instruction::MakeMap(pairs) => {
                pairs.iter().flat_map(|(k, v)| [*k, *v]).collect()
            }
            Instruction::MakeRange(from, to, _) => vec![*from, *to],
            Instruction::StringConcat(parts) => parts.clone(),
            Instruction::SubscriptGet { receiver, args } => {
                let mut ops = vec![*receiver];
                ops.extend(args);
                ops
            }
            Instruction::SubscriptSet { receiver, args, value } => {
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
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return a value from the function.
    Return(ValueId),
    /// Return null (implicit return).
    ReturnNull,
    /// Unconditional jump.
    Branch {
        target: BlockId,
        args: Vec<ValueId>,
    },
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct ModuleMir {
    pub top_level: MirFunction,
    pub classes: Vec<ClassMir>,
    /// Closure / nested function bodies referenced by MakeClosure instructions.
    pub closures: Vec<MirFunction>,
}

/// MIR for a user-defined class.
#[derive(Debug, Clone)]
pub struct ClassMir {
    pub name: SymbolId,
    pub superclass: Option<SymbolId>,
    pub methods: Vec<MethodMir>,
    pub num_fields: u16,
}

/// MIR for a single method within a class.
#[derive(Debug, Clone)]
pub struct MethodMir {
    /// Wren method signature (e.g. "foo(_)", "bar", "[_]").
    pub signature: String,
    pub is_static: bool,
    pub is_constructor: bool,
    pub mir: MirFunction,
}

// ---------------------------------------------------------------------------
// MirFunction
// ---------------------------------------------------------------------------

/// A compiled function in MIR form.
#[derive(Debug, Clone)]
pub struct MirFunction {
    /// Function name (for debugging).
    pub name: SymbolId,
    /// Number of parameters.
    pub arity: u8,
    /// Basic blocks (entry block is always blocks[0]).
    pub blocks: Vec<BasicBlock>,
    /// String constant table.
    pub strings: Vec<String>,
    /// Next available ValueId.
    pub next_value: u32,
    /// Next available BlockId.
    pub next_block: u32,
    /// Source span map: ValueId → source byte range (for runtime error reporting).
    pub span_map: std::collections::HashMap<ValueId, crate::ast::Span>,
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
            out.push_str(&format!(
                "    {}\n",
                fmt_terminator(&block.terminator)
            ));
        }

        out.push_str("}\n");
        out
    }
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

        Instruction::Unbox(a) => format!("unbox {}", a),
        Instruction::Box(a) => format!("box {}", a),

        Instruction::GetField(recv, idx) => format!("get_field {}, #{}", recv, idx),
        Instruction::SetField(recv, idx, val) => {
            format!("set_field {}, #{}, {}", recv, idx, val)
        }
        Instruction::GetModuleVar(idx) => format!("get_module_var @{}", idx),
        Instruction::SetModuleVar(idx, val) => format!("set_module_var @{}, {}", idx, val),

        Instruction::Call {
            receiver,
            method,
            args,
        } => format!(
            "call {}.%{}({})",
            receiver,
            interner.resolve(*method),
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
            let entries: Vec<String> = pairs
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect();
            format!("make_map {{{}}}", entries.join(", "))
        }
        Instruction::MakeRange(from, to, inclusive) => {
            let op = if *inclusive { ".." } else { "..." };
            format!("make_range {}{}{}", from, op, to)
        }

        Instruction::StringConcat(parts) => format!("str_concat [{}]", fmt_val_list(parts)),
        Instruction::ToString(a) => format!("to_string {}", a),

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
        assert_eq!(
            Instruction::Neg(ValueId(2)).operands(),
            vec![ValueId(2)]
        );
        assert_eq!(
            Instruction::Call {
                receiver: ValueId(0),
                method: SymbolId::from_raw(1),
                args: vec![ValueId(2), ValueId(3)],
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
