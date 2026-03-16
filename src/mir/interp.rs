/// MIR interpreter for testing optimization passes.
///
/// Executes a `MirFunction` by walking basic blocks, evaluating instructions,
/// and following control flow. Used to verify that optimization passes preserve
/// semantics: `eval(f) == eval(optimize(f))`.
///
/// Supports: constants, arithmetic (boxed + unboxed), comparisons, logical/bitwise,
/// guards, box/unbox, module vars, moves, block params, control flow.
///
/// Returns `InterpError::Unsupported` for runtime-dependent ops (calls, closures,
/// field access, collections) that require a full VM.
use std::collections::HashMap;

use crate::runtime::value::Value;

use super::{BlockId, Instruction, MirFunction, Terminator, ValueId};

// ---------------------------------------------------------------------------
// Interpreter value
// ---------------------------------------------------------------------------

/// A value held by the interpreter. Tracks whether it's boxed or unboxed
/// so we can correctly handle Box/Unbox instructions.
#[derive(Debug, Clone, Copy)]
pub enum InterpValue {
    /// A NaN-boxed Wren value.
    Boxed(Value),
    /// An unboxed f64 (from ConstF64, unboxed arithmetic, Unbox).
    F64(f64),
    /// An unboxed i64 (from ConstI64).
    I64(i64),
    /// An unboxed bool (from unboxed comparisons).
    Bool(bool),
}

impl InterpValue {
    /// Extract the boxed Value, or error if unboxed.
    pub fn as_boxed(self) -> Result<Value, InterpError> {
        match self {
            InterpValue::Boxed(v) => Ok(v),
            other => Err(InterpError::TypeMismatch(format!(
                "expected boxed Value, got {:?}",
                other
            ))),
        }
    }

    /// Extract as f64. Works for unboxed F64, I64, and boxed nums.
    pub fn as_f64(self) -> Result<f64, InterpError> {
        match self {
            InterpValue::F64(n) => Ok(n),
            InterpValue::I64(n) => Ok(n as f64),
            InterpValue::Boxed(v) => v
                .as_num()
                .ok_or_else(|| InterpError::TypeMismatch(format!("expected num, got {:?}", v))),
            other => Err(InterpError::TypeMismatch(format!(
                "expected f64, got {:?}",
                other
            ))),
        }
    }

    /// Convert to a NaN-boxed Value.
    pub fn to_value(self) -> Value {
        match self {
            InterpValue::Boxed(v) => v,
            InterpValue::F64(n) => Value::num(n),
            InterpValue::I64(n) => Value::num(n as f64),
            InterpValue::Bool(b) => Value::bool(b),
        }
    }

    /// Extract as i64.
    pub fn as_i64(self) -> Result<i64, InterpError> {
        match self {
            InterpValue::I64(n) => Ok(n),
            other => Err(InterpError::TypeMismatch(format!(
                "expected i64, got {:?}",
                other
            ))),
        }
    }

    /// Extract as bool (unboxed).
    pub fn as_bool_val(self) -> Result<bool, InterpError> {
        match self {
            InterpValue::Bool(b) => Ok(b),
            other => Err(InterpError::TypeMismatch(format!(
                "expected bool, got {:?}",
                other
            ))),
        }
    }

    /// Check truthiness for control flow (CondBranch).
    /// Boxed: Wren falsy = null or false. Unboxed bool: direct. Others: truthy.
    pub fn is_truthy(self) -> bool {
        match self {
            InterpValue::Boxed(v) => !v.is_falsy(),
            InterpValue::Bool(b) => b,
            InterpValue::F64(_) | InterpValue::I64(_) => true,
        }
    }
}

impl PartialEq for InterpValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (InterpValue::Boxed(a), InterpValue::Boxed(b)) => a.equals(*b),
            (InterpValue::F64(a), InterpValue::F64(b)) => a == b,
            (InterpValue::I64(a), InterpValue::I64(b)) => a == b,
            (InterpValue::Bool(a), InterpValue::Bool(b)) => a == b,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum InterpError {
    /// Instruction not supported by the interpreter.
    Unsupported(String),
    /// Type mismatch during execution.
    TypeMismatch(String),
    /// SSA value not found.
    UndefinedValue(ValueId),
    /// Block not found.
    UndefinedBlock(BlockId),
    /// Guard check failed (deoptimization would happen).
    GuardFailed(String),
    /// Exceeded step limit.
    StepLimitExceeded,
    /// Division by zero.
    DivisionByZero,
    /// Hit unreachable terminator.
    Unreachable,
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpError::Unsupported(msg) => write!(f, "unsupported: {}", msg),
            InterpError::TypeMismatch(msg) => write!(f, "type mismatch: {}", msg),
            InterpError::UndefinedValue(v) => write!(f, "undefined value: {}", v),
            InterpError::UndefinedBlock(b) => write!(f, "undefined block: {}", b),
            InterpError::GuardFailed(msg) => write!(f, "guard failed: {}", msg),
            InterpError::StepLimitExceeded => write!(f, "step limit exceeded"),
            InterpError::DivisionByZero => write!(f, "division by zero"),
            InterpError::Unreachable => write!(f, "hit unreachable"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared pure instruction evaluator
// ---------------------------------------------------------------------------

/// Evaluate a pure MIR instruction using the provided SSA value map.
///
/// Handles all computational instructions (arithmetic, comparisons, guards,
/// box/unbox, bitwise, constants, moves). Returns `Err(InterpError::Unsupported)`
/// for instructions that require VM/runtime access (calls, collections, module
/// vars, strings, fields, closures).
///
/// Both the standalone MIR interpreter and the VM interpreter delegate here
/// to avoid duplicating evaluation logic.
pub fn eval_pure_instruction(
    inst: &Instruction,
    values: &[InterpValue],
) -> Result<InterpValue, InterpError> {
    let get = |id: ValueId| -> Result<InterpValue, InterpError> {
        values
            .get(id.0 as usize)
            .copied()
            .ok_or(InterpError::UndefinedValue(id))
    };

    match inst {
        // -- Constants --
        Instruction::ConstNum(n) => Ok(InterpValue::Boxed(Value::num(*n))),
        Instruction::ConstBool(b) => Ok(InterpValue::Boxed(Value::bool(*b))),
        Instruction::ConstNull => Ok(InterpValue::Boxed(Value::null())),
        Instruction::ConstString(_) => {
            // Strings are opaque in pure evaluation (no allocator).
            // The VM interpreter handles ConstString before calling this.
            Ok(InterpValue::Boxed(Value::null()))
        }
        Instruction::ConstF64(n) => Ok(InterpValue::F64(*n)),
        Instruction::ConstI64(n) => Ok(InterpValue::I64(*n)),

        // -- Boxed arithmetic --
        Instruction::Add(a, b) => boxed_binop(&get, *a, *b, |x, y| x + y),
        Instruction::Sub(a, b) => boxed_binop(&get, *a, *b, |x, y| x - y),
        Instruction::Mul(a, b) => boxed_binop(&get, *a, *b, |x, y| x * y),
        Instruction::Div(a, b) => boxed_binop(&get, *a, *b, |x, y| x / y),
        Instruction::Mod(a, b) => boxed_binop(&get, *a, *b, |x, y| x % y),
        Instruction::Neg(a) => {
            let n = get_boxed_num(&get, *a)?;
            Ok(InterpValue::Boxed(Value::num(-n)))
        }

        // -- Unboxed f64 arithmetic --
        Instruction::AddF64(a, b) => f64_binop(&get, *a, *b, |x, y| x + y),
        Instruction::SubF64(a, b) => f64_binop(&get, *a, *b, |x, y| x - y),
        Instruction::MulF64(a, b) => f64_binop(&get, *a, *b, |x, y| x * y),
        Instruction::DivF64(a, b) => f64_binop(&get, *a, *b, |x, y| x / y),
        Instruction::ModF64(a, b) => f64_binop(&get, *a, *b, |x, y| x % y),
        Instruction::NegF64(a) => Ok(InterpValue::F64(-get(*a)?.as_f64()?)),

        // -- Math intrinsics --
        Instruction::MathUnaryF64(op, a) => Ok(InterpValue::F64(op.apply(get(*a)?.as_f64()?))),
        Instruction::MathBinaryF64(op, a, b) => Ok(InterpValue::F64(
            op.apply(get(*a)?.as_f64()?, get(*b)?.as_f64()?),
        )),

        // -- Boxed comparisons --
        Instruction::CmpLt(a, b) => boxed_cmp(&get, *a, *b, |x, y| x < y),
        Instruction::CmpGt(a, b) => boxed_cmp(&get, *a, *b, |x, y| x > y),
        Instruction::CmpLe(a, b) => boxed_cmp(&get, *a, *b, |x, y| x <= y),
        Instruction::CmpGe(a, b) => boxed_cmp(&get, *a, *b, |x, y| x >= y),
        Instruction::CmpEq(a, b) => {
            let lhs = get(*a)?.to_value();
            let rhs = get(*b)?.to_value();
            // For non-instance objects (nums, bools, null, strings), use value equality.
            // For instances, return error so vm_interp can dispatch user-defined ==(_).
            if lhs.is_object()
                && !lhs.is_string_object()
                && !rhs.is_null()
                && !rhs.is_bool()
                && !rhs.is_num()
            {
                Err(InterpError::TypeMismatch("CmpEq on objects".into()))
            } else {
                Ok(InterpValue::Boxed(Value::bool(lhs == rhs)))
            }
        }
        Instruction::CmpNe(a, b) => {
            let lhs = get(*a)?.to_value();
            let rhs = get(*b)?.to_value();
            if lhs.is_object()
                && !lhs.is_string_object()
                && !rhs.is_null()
                && !rhs.is_bool()
                && !rhs.is_num()
            {
                Err(InterpError::TypeMismatch("CmpNe on objects".into()))
            } else {
                Ok(InterpValue::Boxed(Value::bool(lhs != rhs)))
            }
        }

        // -- Unboxed f64 comparisons --
        Instruction::CmpLtF64(a, b) => f64_cmp(&get, *a, *b, |x, y| x < y),
        Instruction::CmpGtF64(a, b) => f64_cmp(&get, *a, *b, |x, y| x > y),
        Instruction::CmpLeF64(a, b) => f64_cmp(&get, *a, *b, |x, y| x <= y),
        Instruction::CmpGeF64(a, b) => f64_cmp(&get, *a, *b, |x, y| x >= y),

        // -- Logical --
        Instruction::Not(a) => Ok(InterpValue::Boxed(Value::bool(!get(*a)?.is_truthy()))),

        // -- Bitwise --
        Instruction::BitAnd(a, b) => bitwise_binop(&get, *a, *b, |x, y| x & y),
        Instruction::BitOr(a, b) => bitwise_binop(&get, *a, *b, |x, y| x | y),
        Instruction::BitXor(a, b) => bitwise_binop(&get, *a, *b, |x, y| x ^ y),
        Instruction::Shl(a, b) => bitwise_binop(&get, *a, *b, |x, y| x << (y & 31)),
        Instruction::Shr(a, b) => bitwise_binop(&get, *a, *b, |x, y| x >> (y & 31)),
        Instruction::BitNot(a) => {
            let n = get_boxed_num(&get, *a)? as i32;
            Ok(InterpValue::Boxed(Value::num((!n) as f64)))
        }

        // -- Guards (lenient: accepts both boxed and unboxed forms) --
        Instruction::GuardNum(a) => {
            let v = get(*a)?;
            match v {
                InterpValue::Boxed(val) if val.is_num() => Ok(v),
                InterpValue::F64(_) | InterpValue::I64(_) => Ok(v),
                _ => Err(InterpError::GuardFailed("GuardNum: not a number".into())),
            }
        }
        Instruction::GuardBool(a) => {
            let v = get(*a)?;
            match v {
                InterpValue::Boxed(val) if val.is_bool() => Ok(v),
                InterpValue::Bool(_) => Ok(v),
                _ => Err(InterpError::GuardFailed("GuardBool: not a bool".into())),
            }
        }

        // -- Box / Unbox (handles already-unboxed values gracefully) --
        Instruction::Unbox(a) => match get(*a)? {
            InterpValue::Boxed(val) => match val.as_num() {
                Some(n) => Ok(InterpValue::F64(n)),
                None => Err(InterpError::TypeMismatch("unbox: not a number".into())),
            },
            InterpValue::F64(n) => Ok(InterpValue::F64(n)),
            InterpValue::I64(n) => Ok(InterpValue::F64(n as f64)),
            InterpValue::Bool(_) => Err(InterpError::TypeMismatch("unbox: bool".into())),
        },
        Instruction::Box(a) => Ok(InterpValue::Boxed(get(*a)?.to_value())),

        // -- Move --
        Instruction::Move(a) => get(*a),

        // -- BlockParam (pre-bound by bind_block_params, no-op here) --
        Instruction::BlockParam(_) => Ok(InterpValue::Boxed(Value::null())),

        // -- VM/runtime-specific (unsupported in pure evaluation) --
        Instruction::GuardClass(_, _) => Err(InterpError::Unsupported("GuardClass".into())),
        Instruction::GuardProtocol(_, _) => Err(InterpError::Unsupported("GuardProtocol".into())),
        Instruction::GetModuleVar(_) => Err(InterpError::Unsupported("GetModuleVar".into())),
        Instruction::SetModuleVar(_, _) => Err(InterpError::Unsupported("SetModuleVar".into())),
        Instruction::Call { .. } => Err(InterpError::Unsupported("Call".into())),
        Instruction::SuperCall { .. } => Err(InterpError::Unsupported("SuperCall".into())),
        Instruction::GetField(_, _) => Err(InterpError::Unsupported("GetField".into())),
        Instruction::SetField(_, _, _) => Err(InterpError::Unsupported("SetField".into())),
        Instruction::GetUpvalue(_) => Err(InterpError::Unsupported("GetUpvalue".into())),
        Instruction::SetUpvalue(_, _) => Err(InterpError::Unsupported("SetUpvalue".into())),
        Instruction::MakeClosure { .. } => Err(InterpError::Unsupported("MakeClosure".into())),
        Instruction::MakeList(_) => Err(InterpError::Unsupported("MakeList".into())),
        Instruction::MakeMap(_) => Err(InterpError::Unsupported("MakeMap".into())),
        Instruction::MakeRange(_, _, _) => Err(InterpError::Unsupported("MakeRange".into())),
        Instruction::StringConcat(_) => Err(InterpError::Unsupported("StringConcat".into())),
        Instruction::ToString(_) => Err(InterpError::Unsupported("ToString".into())),
        Instruction::IsType(_, _) => Err(InterpError::Unsupported("IsType".into())),
        Instruction::SubscriptGet { .. } => Err(InterpError::Unsupported("SubscriptGet".into())),
        Instruction::SubscriptSet { .. } => Err(InterpError::Unsupported("SubscriptSet".into())),
        Instruction::GetStaticField(_) => Err(InterpError::Unsupported("GetStaticField".into())),
        Instruction::SetStaticField(_, _) => Err(InterpError::Unsupported("SetStaticField".into())),
        Instruction::CallStaticSelf { .. } => {
            Err(InterpError::Unsupported("CallStaticSelf".into()))
        }
    }
}

// -- Shared helpers for eval_pure_instruction --

fn get_boxed_num(
    get: &impl Fn(ValueId) -> Result<InterpValue, InterpError>,
    id: ValueId,
) -> Result<f64, InterpError> {
    get(id)?.as_f64()
}

fn boxed_binop(
    get: &impl Fn(ValueId) -> Result<InterpValue, InterpError>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> f64,
) -> Result<InterpValue, InterpError> {
    Ok(InterpValue::Boxed(Value::num(op(
        get_boxed_num(get, a)?,
        get_boxed_num(get, b)?,
    ))))
}

fn f64_binop(
    get: &impl Fn(ValueId) -> Result<InterpValue, InterpError>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> f64,
) -> Result<InterpValue, InterpError> {
    Ok(InterpValue::F64(op(get(a)?.as_f64()?, get(b)?.as_f64()?)))
}

fn boxed_cmp(
    get: &impl Fn(ValueId) -> Result<InterpValue, InterpError>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> bool,
) -> Result<InterpValue, InterpError> {
    Ok(InterpValue::Boxed(Value::bool(op(
        get_boxed_num(get, a)?,
        get_boxed_num(get, b)?,
    ))))
}

fn f64_cmp(
    get: &impl Fn(ValueId) -> Result<InterpValue, InterpError>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> bool,
) -> Result<InterpValue, InterpError> {
    Ok(InterpValue::Bool(op(get(a)?.as_f64()?, get(b)?.as_f64()?)))
}

fn bitwise_binop(
    get: &impl Fn(ValueId) -> Result<InterpValue, InterpError>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(i32, i32) -> i32,
) -> Result<InterpValue, InterpError> {
    let lhs = get_boxed_num(get, a)? as i32;
    let rhs = get_boxed_num(get, b)? as i32;
    Ok(InterpValue::Boxed(Value::num(op(lhs, rhs) as f64)))
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

const MAX_STEPS: usize = 100_000;

/// Interpreter state.
pub struct Interp<'a> {
    func: &'a MirFunction,
    values: Vec<InterpValue>,
    module_vars: HashMap<u16, InterpValue>,
    steps: usize,
}

impl<'a> Interp<'a> {
    pub fn new(func: &'a MirFunction) -> Self {
        Self {
            func,
            values: vec![InterpValue::Boxed(Value::UNDEFINED); func.next_value as usize],
            module_vars: HashMap::new(),
            steps: 0,
        }
    }

    /// Run the function to completion, returning the result value.
    pub fn run(&mut self) -> Result<InterpValue, InterpError> {
        let mut current_block = self.func.entry_block();

        loop {
            let block = self
                .func
                .blocks
                .get(current_block.0 as usize)
                .ok_or(InterpError::UndefinedBlock(current_block))?;

            // Execute all instructions in the block.
            for (result_id, inst) in &block.instructions {
                self.step()?;
                let val = self.eval_instruction(inst)?;
                let idx = result_id.0 as usize;
                if idx >= self.values.len() {
                    self.values
                        .resize(idx + 1, InterpValue::Boxed(Value::UNDEFINED));
                }
                self.values[idx] = val;
            }

            self.step()?;

            // Execute terminator.
            match &block.terminator {
                Terminator::Return(v) => {
                    return self.get(*v);
                }
                Terminator::ReturnNull => {
                    return Ok(InterpValue::Boxed(Value::null()));
                }
                Terminator::Branch { target, args } => {
                    self.bind_block_params(*target, args)?;
                    current_block = *target;
                }
                Terminator::CondBranch {
                    condition,
                    true_target,
                    true_args,
                    false_target,
                    false_args,
                } => {
                    let cond = self.get(*condition)?;
                    if cond.is_truthy() {
                        self.bind_block_params(*true_target, true_args)?;
                        current_block = *true_target;
                    } else {
                        self.bind_block_params(*false_target, false_args)?;
                        current_block = *false_target;
                    }
                }
                Terminator::Unreachable => {
                    return Err(InterpError::Unreachable);
                }
            }
        }
    }

    fn step(&mut self) -> Result<(), InterpError> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            Err(InterpError::StepLimitExceeded)
        } else {
            Ok(())
        }
    }

    fn get(&self, id: ValueId) -> Result<InterpValue, InterpError> {
        self.values
            .get(id.0 as usize)
            .copied()
            .ok_or(InterpError::UndefinedValue(id))
    }

    /// Bind branch arguments to the target block's parameters.
    fn bind_block_params(&mut self, target: BlockId, args: &[ValueId]) -> Result<(), InterpError> {
        let block = self
            .func
            .blocks
            .get(target.0 as usize)
            .ok_or(InterpError::UndefinedBlock(target))?;
        let params: Vec<ValueId> = block.params.iter().map(|(v, _)| *v).collect();
        for (param_id, arg_id) in params.iter().zip(args.iter()) {
            let val = self.get(*arg_id)?;
            let idx = param_id.0 as usize;
            if idx >= self.values.len() {
                self.values
                    .resize(idx + 1, InterpValue::Boxed(Value::UNDEFINED));
            }
            self.values[idx] = val;
        }
        Ok(())
    }

    fn eval_instruction(&mut self, inst: &Instruction) -> Result<InterpValue, InterpError> {
        // Module vars use Interp-local storage, handle them here.
        // Everything else delegates to the shared evaluator.
        match inst {
            Instruction::GetModuleVar(idx) => self
                .module_vars
                .get(idx)
                .copied()
                .ok_or(InterpError::UndefinedValue(ValueId(*idx as u32))),
            Instruction::SetModuleVar(idx, val) => {
                let v = self.get(*val)?;
                self.module_vars.insert(*idx, v);
                Ok(InterpValue::Boxed(Value::null()))
            }
            _ => eval_pure_instruction(inst, &self.values),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Evaluate a MIR function and return the result.
pub fn eval(func: &MirFunction) -> Result<InterpValue, InterpError> {
    let mut interp = Interp::new(func);
    interp.run()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::{Interner, SymbolId};
    use crate::mir::{MirType, Terminator};

    /// Helper: create a MirFunction with a given name.
    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    // -- Constants --

    #[test]
    fn test_const_num() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    #[test]
    fn test_const_bool() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstBool(true)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::bool(true)));
    }

    #[test]
    fn test_const_null() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNull));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::null()));
    }

    #[test]
    fn test_const_f64() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstF64(1.234)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::F64(1.234));
    }

    #[test]
    fn test_const_i64() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstI64(99)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::I64(99));
    }

    #[test]
    fn test_return_null_terminator() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::null()));
    }

    // -- Boxed arithmetic --

    #[test]
    fn test_boxed_add() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(10.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(32.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Add(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    #[test]
    fn test_boxed_sub() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(50.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(8.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Sub(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    #[test]
    fn test_boxed_mul() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(6.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(7.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Mul(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    #[test]
    fn test_boxed_div() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(84.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(2.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Div(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    #[test]
    fn test_boxed_mod() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(47.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(5.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Mod(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(2.0)));
    }

    #[test]
    fn test_boxed_neg() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::Neg(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(-42.0)));
    }

    // -- Unboxed f64 arithmetic --

    #[test]
    fn test_f64_add() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstF64(1.5)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstF64(2.5)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::AddF64(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::F64(4.0));
    }

    #[test]
    fn test_f64_sub_mul_div_mod_neg() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v_sub = f.new_value();
        let v_mul = f.new_value();
        let v_div = f.new_value();
        let v_mod = f.new_value();
        let v_neg = f.new_value();

        let b = f.block_mut(bb);
        b.instructions.push((v0, Instruction::ConstF64(10.0)));
        b.instructions.push((v1, Instruction::ConstF64(3.0)));
        b.instructions.push((v_sub, Instruction::SubF64(v0, v1)));
        b.instructions.push((v_mul, Instruction::MulF64(v0, v1)));
        b.instructions.push((v_div, Instruction::DivF64(v0, v1)));
        b.instructions.push((v_mod, Instruction::ModF64(v0, v1)));
        b.instructions.push((v_neg, Instruction::NegF64(v0)));
        b.terminator = Terminator::Return(v_sub);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::F64(7.0));
    }

    // -- Comparisons --

    #[test]
    fn test_boxed_cmp_lt() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
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
            .push((v2, Instruction::CmpLt(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::bool(true)));
    }

    #[test]
    fn test_boxed_cmp_eq() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(5.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(5.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::CmpEq(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::bool(true)));
    }

    #[test]
    fn test_boxed_cmp_ne() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(5.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(3.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::CmpNe(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::bool(true)));
    }

    #[test]
    fn test_f64_cmp_lt() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
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
            .push((v2, Instruction::CmpLtF64(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Bool(true));
    }

    // -- Logical --

    #[test]
    fn test_not_true() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstBool(true)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::Not(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::bool(false)));
    }

    #[test]
    fn test_not_null() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNull));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::Not(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        // null is falsy → !null = true
        assert_eq!(result, InterpValue::Boxed(Value::bool(true)));
    }

    // -- Bitwise --

    #[test]
    fn test_bitwise_and() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(0xFF.into())));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(0x0F.into())));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::BitAnd(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(0x0F as f64)));
    }

    #[test]
    fn test_bitwise_not() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(0.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::BitNot(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(-1.0)));
    }

    #[test]
    fn test_shl() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(1.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::ConstNum(4.0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Shl(v0, v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(16.0)));
    }

    // -- Guards --

    #[test]
    fn test_guard_num_pass() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::GuardNum(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    #[test]
    fn test_guard_num_fail() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNull));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::GuardNum(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f);
        assert!(matches!(result, Err(InterpError::GuardFailed(_))));
    }

    #[test]
    fn test_guard_bool_pass() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstBool(false)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::GuardBool(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::bool(false)));
    }

    // -- Box / Unbox --

    #[test]
    fn test_unbox_then_box() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(1.234)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::Unbox(v0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::Box(v1)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(1.234)));
    }

    // -- Move --

    #[test]
    fn test_move() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(99.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::Move(v0)));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(99.0)));
    }

    // -- Module vars --

    #[test]
    fn test_module_var_set_get() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb)
            .instructions
            .push((v1, Instruction::SetModuleVar(0, v0)));
        f.block_mut(bb)
            .instructions
            .push((v2, Instruction::GetModuleVar(0)));
        f.block_mut(bb).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    // -- Control flow: Branch --

    #[test]
    fn test_unconditional_branch() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let v0 = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v0, Instruction::ConstNum(42.0)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0],
        };

        let v1 = f.new_value();
        f.block_mut(bb1).params.push((v1, MirType::Value));
        f.block_mut(bb1).terminator = Terminator::Return(v1);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(42.0)));
    }

    // -- Control flow: CondBranch --

    #[test]
    fn test_cond_branch_true() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb_true = f.new_block();
        let bb_false = f.new_block();

        let v_cond = f.new_value();
        let v_yes = f.new_value();
        let v_no = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v_cond, Instruction::ConstBool(true)));
        f.block_mut(bb0).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb_true,
            true_args: vec![],
            false_target: bb_false,
            false_args: vec![],
        };

        f.block_mut(bb_true)
            .instructions
            .push((v_yes, Instruction::ConstNum(1.0)));
        f.block_mut(bb_true).terminator = Terminator::Return(v_yes);

        f.block_mut(bb_false)
            .instructions
            .push((v_no, Instruction::ConstNum(0.0)));
        f.block_mut(bb_false).terminator = Terminator::Return(v_no);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(1.0)));
    }

    #[test]
    fn test_cond_branch_false() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb_true = f.new_block();
        let bb_false = f.new_block();

        let v_cond = f.new_value();
        let v_yes = f.new_value();
        let v_no = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v_cond, Instruction::ConstBool(false)));
        f.block_mut(bb0).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb_true,
            true_args: vec![],
            false_target: bb_false,
            false_args: vec![],
        };

        f.block_mut(bb_true)
            .instructions
            .push((v_yes, Instruction::ConstNum(1.0)));
        f.block_mut(bb_true).terminator = Terminator::Return(v_yes);

        f.block_mut(bb_false)
            .instructions
            .push((v_no, Instruction::ConstNum(0.0)));
        f.block_mut(bb_false).terminator = Terminator::Return(v_no);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(0.0)));
    }

    #[test]
    fn test_cond_branch_null_is_falsy() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb_true = f.new_block();
        let bb_false = f.new_block();

        let v_cond = f.new_value();
        let v_yes = f.new_value();
        let v_no = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v_cond, Instruction::ConstNull));
        f.block_mut(bb0).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb_true,
            true_args: vec![],
            false_target: bb_false,
            false_args: vec![],
        };

        f.block_mut(bb_true)
            .instructions
            .push((v_yes, Instruction::ConstNum(1.0)));
        f.block_mut(bb_true).terminator = Terminator::Return(v_yes);

        f.block_mut(bb_false)
            .instructions
            .push((v_no, Instruction::ConstNum(0.0)));
        f.block_mut(bb_false).terminator = Terminator::Return(v_no);

        let result = eval(&f).unwrap();
        // null is falsy
        assert_eq!(result, InterpValue::Boxed(Value::num(0.0)));
    }

    // -- Loop with block params (e.g. sum 1..5) --

    #[test]
    fn test_loop_sum() {
        // Compute 1+2+3+4+5 = 15 using a loop with block params.
        //
        // bb0:
        //   v0 = const.num 0     ; sum
        //   v1 = const.num 1     ; i
        //   jump bb1(v0, v1)
        //
        // bb1(v2: val, v3: val): ; sum, i
        //   v4 = const.num 6
        //   v5 = icmp.lt v3, v4  ; i < 6
        //   brif v5, bb2, bb3
        //
        // bb2:                   ; loop body
        //   v6 = add v2, v3     ; sum + i
        //   v7 = const.num 1
        //   v8 = add v3, v7     ; i + 1
        //   jump bb1(v6, v8)
        //
        // bb3:                   ; exit
        //   return v2

        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();

        // bb0
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb0)
            .instructions
            .push((v0, Instruction::ConstNum(0.0)));
        f.block_mut(bb0)
            .instructions
            .push((v1, Instruction::ConstNum(1.0)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v0, v1],
        };

        // bb1 — header
        let v2 = f.new_value();
        let v3 = f.new_value();
        let v4 = f.new_value();
        let v5 = f.new_value();
        f.block_mut(bb1).params.push((v2, MirType::Value));
        f.block_mut(bb1).params.push((v3, MirType::Value));
        f.block_mut(bb1)
            .instructions
            .push((v4, Instruction::ConstNum(6.0)));
        f.block_mut(bb1)
            .instructions
            .push((v5, Instruction::CmpLt(v3, v4)));
        f.block_mut(bb1).terminator = Terminator::CondBranch {
            condition: v5,
            true_target: bb2,
            true_args: vec![],
            false_target: bb3,
            false_args: vec![],
        };

        // bb2 — body
        let v6 = f.new_value();
        let v7 = f.new_value();
        let v8 = f.new_value();
        f.block_mut(bb2)
            .instructions
            .push((v6, Instruction::Add(v2, v3)));
        f.block_mut(bb2)
            .instructions
            .push((v7, Instruction::ConstNum(1.0)));
        f.block_mut(bb2)
            .instructions
            .push((v8, Instruction::Add(v3, v7)));
        f.block_mut(bb2).terminator = Terminator::Branch {
            target: bb1,
            args: vec![v6, v8],
        };

        // bb3 — exit
        f.block_mut(bb3).terminator = Terminator::Return(v2);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(15.0)));
    }

    // -- Chain of expressions --

    #[test]
    fn test_complex_expression() {
        // (2 + 3) * 4 - 1 = 19
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value(); // 2
        let v1 = f.new_value(); // 3
        let v2 = f.new_value(); // 2+3=5
        let v3 = f.new_value(); // 4
        let v4 = f.new_value(); // 5*4=20
        let v5 = f.new_value(); // 1
        let v6 = f.new_value(); // 20-1=19

        let b = f.block_mut(bb);
        b.instructions.push((v0, Instruction::ConstNum(2.0)));
        b.instructions.push((v1, Instruction::ConstNum(3.0)));
        b.instructions.push((v2, Instruction::Add(v0, v1)));
        b.instructions.push((v3, Instruction::ConstNum(4.0)));
        b.instructions.push((v4, Instruction::Mul(v2, v3)));
        b.instructions.push((v5, Instruction::ConstNum(1.0)));
        b.instructions.push((v6, Instruction::Sub(v4, v5)));
        b.terminator = Terminator::Return(v6);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(19.0)));
    }

    // -- Unbox → f64 arithmetic → Box roundtrip --

    #[test]
    fn test_unbox_compute_rebox() {
        // Unbox two nums, add as f64, rebox.
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value(); // boxed 10
        let v1 = f.new_value(); // boxed 20
        let v2 = f.new_value(); // unboxed 10.0
        let v3 = f.new_value(); // unboxed 20.0
        let v4 = f.new_value(); // f64 add = 30.0
        let v5 = f.new_value(); // rebox = Value(30.0)

        let b = f.block_mut(bb);
        b.instructions.push((v0, Instruction::ConstNum(10.0)));
        b.instructions.push((v1, Instruction::ConstNum(20.0)));
        b.instructions.push((v2, Instruction::Unbox(v0)));
        b.instructions.push((v3, Instruction::Unbox(v1)));
        b.instructions.push((v4, Instruction::AddF64(v2, v3)));
        b.instructions.push((v5, Instruction::Box(v4)));
        b.terminator = Terminator::Return(v5);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(30.0)));
    }

    // -- Unreachable --

    #[test]
    fn test_unreachable() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::Unreachable;

        let result = eval(&f);
        assert!(matches!(result, Err(InterpError::Unreachable)));
    }

    // -- Step limit --

    #[test]
    fn test_step_limit() {
        // Infinite loop: bb0 → bb0.
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::Branch {
            target: bb,
            args: vec![],
        };

        let result = eval(&f);
        assert!(matches!(result, Err(InterpError::StepLimitExceeded)));
    }

    // -- Unsupported ops --

    #[test]
    fn test_unsupported_call() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNull));
        f.block_mut(bb).instructions.push((
            v1,
            Instruction::Call {
                receiver: v0,
                method: SymbolId::from_raw(0),
                args: vec![],
            },
        ));
        f.block_mut(bb).terminator = Terminator::Return(v1);

        let result = eval(&f);
        assert!(matches!(result, Err(InterpError::Unsupported(_))));
    }

    // -- Multi-block with block param args --

    #[test]
    fn test_cond_branch_with_args() {
        // bb0: cond = true, branch → bb_merge with different values
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb_merge = f.new_block();

        let v_cond = f.new_value();
        let v_yes = f.new_value();
        let v_no = f.new_value();
        let v_result = f.new_value();

        f.block_mut(bb0)
            .instructions
            .push((v_cond, Instruction::ConstBool(true)));
        f.block_mut(bb0)
            .instructions
            .push((v_yes, Instruction::ConstNum(100.0)));
        f.block_mut(bb0)
            .instructions
            .push((v_no, Instruction::ConstNum(200.0)));
        f.block_mut(bb0).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb_merge,
            true_args: vec![v_yes],
            false_target: bb_merge,
            false_args: vec![v_no],
        };

        f.block_mut(bb_merge)
            .params
            .push((v_result, MirType::Value));
        f.block_mut(bb_merge).terminator = Terminator::Return(v_result);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(100.0)));
    }

    // -- Fibonacci (recursive-like via loop) --

    #[test]
    fn test_fibonacci_loop() {
        // Compute fib(10) = 55 using a loop.
        //
        // bb0:
        //   v0 = const.num 0      ; a
        //   v1 = const.num 1      ; b
        //   v2 = const.num 10     ; n
        //   v3 = const.num 0      ; i
        //   jump bb1(v0, v1, v3)
        //
        // bb1(v4, v5, v6):        ; a, b, i
        //   v7 = icmp.lt v6, v2   ; i < n
        //   brif v7, bb2, bb3
        //
        // bb2:
        //   v8 = add v4, v5       ; a + b
        //   v9 = const.num 1
        //   v10 = add v6, v9      ; i + 1
        //   jump bb1(v5, v8, v10) ; a=b, b=a+b, i=i+1
        //
        // bb3:
        //   return v4

        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();

        // bb0
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        {
            let b = f.block_mut(bb0);
            b.instructions.push((v0, Instruction::ConstNum(0.0)));
            b.instructions.push((v1, Instruction::ConstNum(1.0)));
            b.instructions.push((v2, Instruction::ConstNum(10.0)));
            b.instructions.push((v3, Instruction::ConstNum(0.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v0, v1, v3],
            };
        }

        // bb1
        let v4 = f.new_value();
        let v5 = f.new_value();
        let v6 = f.new_value();
        let v7 = f.new_value();
        {
            let b = f.block_mut(bb1);
            b.params.push((v4, MirType::Value));
            b.params.push((v5, MirType::Value));
            b.params.push((v6, MirType::Value));
            b.instructions.push((v7, Instruction::CmpLt(v6, v2)));
            b.terminator = Terminator::CondBranch {
                condition: v7,
                true_target: bb2,
                true_args: vec![],
                false_target: bb3,
                false_args: vec![],
            };
        }

        // bb2
        let v8 = f.new_value();
        let v9 = f.new_value();
        let v10 = f.new_value();
        {
            let b = f.block_mut(bb2);
            b.instructions.push((v8, Instruction::Add(v4, v5)));
            b.instructions.push((v9, Instruction::ConstNum(1.0)));
            b.instructions.push((v10, Instruction::Add(v6, v9)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v5, v8, v10],
            };
        }

        // bb3
        f.block_mut(bb3).terminator = Terminator::Return(v4);

        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::num(55.0)));
    }
}
