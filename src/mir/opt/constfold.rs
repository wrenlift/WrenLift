/// Constant folding and propagation.
///
/// Evaluates constant expressions at compile time:
/// - Arithmetic: `ConstNum(2) + ConstNum(3)` → `ConstNum(5)`
/// - Comparisons: `ConstNum(1) < ConstNum(2)` → `ConstBool(true)`
/// - Branch elimination: `CondBranch` on constant → unconditional `Branch`
/// - Propagation through `Move`, `Box`, `Unbox`

use std::collections::HashMap;

use super::MirPass;
use crate::mir::{Instruction, MirFunction, Terminator, ValueId};
use crate::runtime::value::Value;

// ---------------------------------------------------------------------------
// Constant value representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum ConstVal {
    Num(f64),
    Bool(bool),
    Null,
    F64(f64),
    I64(i64),
}

impl ConstVal {
    fn is_truthy(&self) -> bool {
        match self {
            ConstVal::Null => false,
            ConstVal::Bool(b) => *b,
            ConstVal::Num(_) | ConstVal::F64(_) | ConstVal::I64(_) => true,
        }
    }
}

fn const_to_instruction(cv: &ConstVal) -> Instruction {
    match cv {
        ConstVal::Num(n) => Instruction::ConstNum(*n),
        ConstVal::Bool(b) => Instruction::ConstBool(*b),
        ConstVal::Null => Instruction::ConstNull,
        ConstVal::F64(n) => Instruction::ConstF64(*n),
        ConstVal::I64(n) => Instruction::ConstI64(*n),
    }
}

fn const_to_value(cv: &ConstVal) -> Value {
    match cv {
        ConstVal::Num(n) => Value::num(*n),
        ConstVal::Bool(b) => Value::bool(*b),
        ConstVal::Null => Value::null(),
        _ => Value::null(),
    }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

pub struct ConstFold;

impl MirPass for ConstFold {
    fn name(&self) -> &str {
        "constfold"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut constants: HashMap<ValueId, ConstVal> = HashMap::new();
        let mut changed = false;

        for block_idx in 0..func.blocks.len() {
            for inst_idx in 0..func.blocks[block_idx].instructions.len() {
                let (val_id, ref inst) = func.blocks[block_idx].instructions[inst_idx];

                if let Some(cv) = extract_constant(inst) {
                    constants.insert(val_id, cv);
                } else if let Some(cv) = try_fold(inst, &constants) {
                    func.blocks[block_idx].instructions[inst_idx] =
                        (val_id, const_to_instruction(&cv));
                    constants.insert(val_id, cv);
                    changed = true;
                } else if let Some(cv) = try_propagate(inst, &constants) {
                    if should_replace(inst, &cv) {
                        func.blocks[block_idx].instructions[inst_idx] =
                            (val_id, const_to_instruction(&cv));
                        changed = true;
                    }
                    constants.insert(val_id, cv);
                }
            }

            if let Some(new_term) =
                try_fold_terminator(&func.blocks[block_idx].terminator, &constants)
            {
                func.blocks[block_idx].terminator = new_term;
                changed = true;
            }
        }

        changed
    }
}

// ---------------------------------------------------------------------------
// Constant extraction
// ---------------------------------------------------------------------------

fn extract_constant(inst: &Instruction) -> Option<ConstVal> {
    match inst {
        Instruction::ConstNum(n) => Some(ConstVal::Num(*n)),
        Instruction::ConstBool(b) => Some(ConstVal::Bool(*b)),
        Instruction::ConstNull => Some(ConstVal::Null),
        Instruction::ConstF64(n) => Some(ConstVal::F64(*n)),
        Instruction::ConstI64(n) => Some(ConstVal::I64(*n)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Folding
// ---------------------------------------------------------------------------

fn try_fold(inst: &Instruction, constants: &HashMap<ValueId, ConstVal>) -> Option<ConstVal> {
    match inst {
        // Boxed arithmetic
        Instruction::Add(a, b) => num_binop(constants, *a, *b, |x, y| x + y),
        Instruction::Sub(a, b) => num_binop(constants, *a, *b, |x, y| x - y),
        Instruction::Mul(a, b) => num_binop(constants, *a, *b, |x, y| x * y),
        Instruction::Div(a, b) => num_binop(constants, *a, *b, |x, y| x / y),
        Instruction::Mod(a, b) => num_binop(constants, *a, *b, |x, y| x % y),
        Instruction::Neg(a) => match constants.get(a) {
            Some(ConstVal::Num(n)) => Some(ConstVal::Num(-n)),
            _ => None,
        },

        // Unboxed f64 arithmetic
        Instruction::AddF64(a, b) => f64_binop(constants, *a, *b, |x, y| x + y),
        Instruction::SubF64(a, b) => f64_binop(constants, *a, *b, |x, y| x - y),
        Instruction::MulF64(a, b) => f64_binop(constants, *a, *b, |x, y| x * y),
        Instruction::DivF64(a, b) => f64_binop(constants, *a, *b, |x, y| x / y),
        Instruction::ModF64(a, b) => f64_binop(constants, *a, *b, |x, y| x % y),
        Instruction::NegF64(a) => match constants.get(a) {
            Some(ConstVal::F64(n)) => Some(ConstVal::F64(-n)),
            _ => None,
        },

        // Boxed comparisons
        Instruction::CmpLt(a, b) => num_cmp(constants, *a, *b, |x, y| x < y),
        Instruction::CmpGt(a, b) => num_cmp(constants, *a, *b, |x, y| x > y),
        Instruction::CmpLe(a, b) => num_cmp(constants, *a, *b, |x, y| x <= y),
        Instruction::CmpGe(a, b) => num_cmp(constants, *a, *b, |x, y| x >= y),
        Instruction::CmpEq(a, b) => match (constants.get(a), constants.get(b)) {
            (Some(ConstVal::Num(x)), Some(ConstVal::Num(y))) => Some(ConstVal::Bool(x == y)),
            (Some(ConstVal::Bool(x)), Some(ConstVal::Bool(y))) => Some(ConstVal::Bool(x == y)),
            (Some(ConstVal::Null), Some(ConstVal::Null)) => Some(ConstVal::Bool(true)),
            (Some(va), Some(vb))
                if !matches!(
                    (va, vb),
                    (ConstVal::F64(_) | ConstVal::I64(_), _)
                        | (_, ConstVal::F64(_) | ConstVal::I64(_))
                ) =>
            {
                Some(ConstVal::Bool(const_to_value(va).equals(const_to_value(vb))))
            }
            _ => None,
        },
        Instruction::CmpNe(a, b) => match (constants.get(a), constants.get(b)) {
            (Some(ConstVal::Num(x)), Some(ConstVal::Num(y))) => Some(ConstVal::Bool(x != y)),
            (Some(ConstVal::Bool(x)), Some(ConstVal::Bool(y))) => Some(ConstVal::Bool(x != y)),
            (Some(ConstVal::Null), Some(ConstVal::Null)) => Some(ConstVal::Bool(false)),
            (Some(va), Some(vb))
                if !matches!(
                    (va, vb),
                    (ConstVal::F64(_) | ConstVal::I64(_), _)
                        | (_, ConstVal::F64(_) | ConstVal::I64(_))
                ) =>
            {
                Some(ConstVal::Bool(
                    !const_to_value(va).equals(const_to_value(vb)),
                ))
            }
            _ => None,
        },

        // Logical
        Instruction::Not(a) => constants.get(a).map(|cv| ConstVal::Bool(!cv.is_truthy())),

        // Bitwise
        Instruction::BitAnd(a, b) => int_binop(constants, *a, *b, |x, y| x & y),
        Instruction::BitOr(a, b) => int_binop(constants, *a, *b, |x, y| x | y),
        Instruction::BitXor(a, b) => int_binop(constants, *a, *b, |x, y| x ^ y),
        Instruction::BitNot(a) => match constants.get(a) {
            Some(ConstVal::Num(n)) => Some(ConstVal::Num((!(*n as i32)) as f64)),
            _ => None,
        },
        Instruction::Shl(a, b) => int_binop(constants, *a, *b, |x, y| x << (y & 31)),
        Instruction::Shr(a, b) => int_binop(constants, *a, *b, |x, y| x >> (y & 31)),

        // Math intrinsics
        Instruction::MathUnaryF64(op, a) => match constants.get(a) {
            Some(ConstVal::F64(n)) => Some(ConstVal::F64(op.apply(*n))),
            _ => None,
        },
        Instruction::MathBinaryF64(op, a, b) => f64_binop(constants, *a, *b, |x, y| op.apply(x, y)),

        _ => None,
    }
}

fn try_propagate(inst: &Instruction, constants: &HashMap<ValueId, ConstVal>) -> Option<ConstVal> {
    match inst {
        Instruction::Move(a) => constants.get(a).copied(),
        Instruction::Unbox(a) => match constants.get(a) {
            Some(ConstVal::Num(n)) => Some(ConstVal::F64(*n)),
            _ => None,
        },
        Instruction::Box(a) => match constants.get(a) {
            Some(ConstVal::F64(n)) => Some(ConstVal::Num(*n)),
            _ => None,
        },
        // Unboxed comparisons: track Bool for branch folding, don't replace instruction.
        Instruction::CmpLtF64(a, b) => f64_cmp(constants, *a, *b, |x, y| x < y),
        Instruction::CmpGtF64(a, b) => f64_cmp(constants, *a, *b, |x, y| x > y),
        Instruction::CmpLeF64(a, b) => f64_cmp(constants, *a, *b, |x, y| x <= y),
        Instruction::CmpGeF64(a, b) => f64_cmp(constants, *a, *b, |x, y| x >= y),
        _ => None,
    }
}

fn should_replace(inst: &Instruction, cv: &ConstVal) -> bool {
    match (inst, cv) {
        (Instruction::Move(_), _) => true,
        (Instruction::Unbox(_), ConstVal::F64(_)) => true,
        (Instruction::Box(_), ConstVal::Num(_)) => true,
        (Instruction::CmpLtF64(..), _)
        | (Instruction::CmpGtF64(..), _)
        | (Instruction::CmpLeF64(..), _)
        | (Instruction::CmpGeF64(..), _) => false,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Terminator folding
// ---------------------------------------------------------------------------

fn try_fold_terminator(
    term: &Terminator,
    constants: &HashMap<ValueId, ConstVal>,
) -> Option<Terminator> {
    if let Terminator::CondBranch {
        condition,
        true_target,
        true_args,
        false_target,
        false_args,
    } = term
    {
        if let Some(cv) = constants.get(condition) {
            return if cv.is_truthy() {
                Some(Terminator::Branch {
                    target: *true_target,
                    args: true_args.clone(),
                })
            } else {
                Some(Terminator::Branch {
                    target: *false_target,
                    args: false_args.clone(),
                })
            };
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn num_binop(
    constants: &HashMap<ValueId, ConstVal>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> f64,
) -> Option<ConstVal> {
    match (constants.get(&a), constants.get(&b)) {
        (Some(ConstVal::Num(x)), Some(ConstVal::Num(y))) => Some(ConstVal::Num(op(*x, *y))),
        _ => None,
    }
}

fn f64_binop(
    constants: &HashMap<ValueId, ConstVal>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> f64,
) -> Option<ConstVal> {
    match (constants.get(&a), constants.get(&b)) {
        (Some(ConstVal::F64(x)), Some(ConstVal::F64(y))) => Some(ConstVal::F64(op(*x, *y))),
        _ => None,
    }
}

fn num_cmp(
    constants: &HashMap<ValueId, ConstVal>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> bool,
) -> Option<ConstVal> {
    match (constants.get(&a), constants.get(&b)) {
        (Some(ConstVal::Num(x)), Some(ConstVal::Num(y))) => Some(ConstVal::Bool(op(*x, *y))),
        _ => None,
    }
}

fn f64_cmp(
    constants: &HashMap<ValueId, ConstVal>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(f64, f64) -> bool,
) -> Option<ConstVal> {
    match (constants.get(&a), constants.get(&b)) {
        (Some(ConstVal::F64(x)), Some(ConstVal::F64(y))) => Some(ConstVal::Bool(op(*x, *y))),
        _ => None,
    }
}

fn int_binop(
    constants: &HashMap<ValueId, ConstVal>,
    a: ValueId,
    b: ValueId,
    op: impl Fn(i32, i32) -> i32,
) -> Option<ConstVal> {
    match (constants.get(&a), constants.get(&b)) {
        (Some(ConstVal::Num(x)), Some(ConstVal::Num(y))) => {
            Some(ConstVal::Num(op(*x as i32, *y as i32) as f64))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::interp::{eval, InterpValue};
    use crate::mir::Terminator;
    use crate::runtime::value::Value;

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_fold_arithmetic() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(2.0)));
            b.instructions.push((v1, Instruction::ConstNum(3.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let before = eval(&f).unwrap();
        assert!(ConstFold.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        let (_, ref inst) = f.block(bb).instructions[2];
        assert!(matches!(inst, Instruction::ConstNum(n) if *n == 5.0));
    }

    #[test]
    fn test_fold_nested() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        let v4 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(2.0)));
            b.instructions.push((v1, Instruction::ConstNum(3.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.instructions.push((v3, Instruction::ConstNum(4.0)));
            b.instructions.push((v4, Instruction::Mul(v2, v3)));
            b.terminator = Terminator::Return(v4);
        }

        let before = eval(&f).unwrap();
        assert!(ConstFold.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(20.0)));
    }

    #[test]
    fn test_fold_comparison() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::ConstNum(2.0)));
            b.instructions.push((v2, Instruction::CmpLt(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        assert!(ConstFold.run(&mut f));
        let result = eval(&f).unwrap();
        assert_eq!(result, InterpValue::Boxed(Value::bool(true)));
    }

    #[test]
    fn test_fold_branch_elimination() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let v_cond = f.new_value();
        let v_yes = f.new_value();
        let v_no = f.new_value();

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
            .push((v_yes, Instruction::ConstNum(1.0)));
        f.block_mut(bb1).terminator = Terminator::Return(v_yes);
        f.block_mut(bb2)
            .instructions
            .push((v_no, Instruction::ConstNum(0.0)));
        f.block_mut(bb2).terminator = Terminator::Return(v_no);

        let before = eval(&f).unwrap();
        assert!(ConstFold.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert!(matches!(f.block(bb0).terminator, Terminator::Branch { .. }));
    }

    #[test]
    fn test_fold_unboxed_f64() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstF64(2.5)));
            b.instructions.push((v1, Instruction::ConstF64(3.5)));
            b.instructions.push((v2, Instruction::AddF64(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let before = eval(&f).unwrap();
        assert!(ConstFold.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        let (_, ref inst) = f.block(bb).instructions[2];
        assert!(matches!(inst, Instruction::ConstF64(n) if *n == 6.0));
    }

    #[test]
    fn test_fold_box_unbox() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(42.0)));
            b.instructions.push((v1, Instruction::Unbox(v0)));
            b.instructions.push((v2, Instruction::Box(v1)));
            b.terminator = Terminator::Return(v2);
        }

        let before = eval(&f).unwrap();
        assert!(ConstFold.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        let (_, ref inst1) = f.block(bb).instructions[1];
        assert!(matches!(inst1, Instruction::ConstF64(n) if *n == 42.0));
        let (_, ref inst2) = f.block(bb).instructions[2];
        assert!(matches!(inst2, Instruction::ConstNum(n) if *n == 42.0));
    }

    #[test]
    fn test_no_fold_non_constant() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::GetModuleVar(0)));
            b.instructions.push((v1, Instruction::ConstNum(1.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        assert!(!ConstFold.run(&mut f));
        let (_, ref inst) = f.block(bb).instructions[2];
        assert!(matches!(inst, Instruction::Add(..)));
    }
}
