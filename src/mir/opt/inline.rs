/// Type specialization (devirtualization).
///
/// When operands of boxed arithmetic are known to be Num (via ConstNum or
/// GuardNum), replaces the boxed operation with Unbox → unboxed op → Box.
/// This eliminates runtime type checks and enables further f64 optimizations.

use std::collections::HashSet;

use super::MirPass;
use crate::mir::{Instruction, MirFunction, ValueId};

pub struct TypeSpecialize;

impl MirPass for TypeSpecialize {
    fn name(&self) -> &str {
        "type-specialize"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut known_nums: HashSet<ValueId> = HashSet::new();
        let mut changed = false;

        for block_idx in 0..func.blocks.len() {
            let old_instructions =
                std::mem::take(&mut func.blocks[block_idx].instructions);
            let mut new_instructions = Vec::new();

            for (val_id, inst) in &old_instructions {
                match inst {
                    Instruction::ConstNum(_) | Instruction::GuardNum(_) | Instruction::Box(_) => {
                        known_nums.insert(*val_id);
                        new_instructions.push((*val_id, inst.clone()));
                    }

                    Instruction::Add(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Add);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Sub(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Sub);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Mul(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Mul);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Div(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Div);
                        known_nums.insert(*val_id);
                        changed = true;
                    }
                    Instruction::Mod(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_binop(func, &mut new_instructions, *val_id, *a, *b, BinOp::Mod);
                        known_nums.insert(*val_id);
                        changed = true;
                    }

                    Instruction::Neg(a) if known_nums.contains(a) => {
                        let ua = func.new_value();
                        let result_f = func.new_value();
                        new_instructions.push((ua, Instruction::Unbox(*a)));
                        new_instructions.push((result_f, Instruction::NegF64(ua)));
                        new_instructions.push((*val_id, Instruction::Box(result_f)));
                        known_nums.insert(*val_id);
                        changed = true;
                    }

                    Instruction::CmpLt(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Lt);
                        changed = true;
                    }
                    Instruction::CmpGt(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Gt);
                        changed = true;
                    }
                    Instruction::CmpLe(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Le);
                        changed = true;
                    }
                    Instruction::CmpGe(a, b)
                        if known_nums.contains(a) && known_nums.contains(b) =>
                    {
                        expand_cmp(func, &mut new_instructions, *val_id, *a, *b, CmpOp::Ge);
                        changed = true;
                    }

                    _ => {
                        new_instructions.push((*val_id, inst.clone()));
                    }
                }
            }

            func.blocks[block_idx].instructions = new_instructions;
        }

        changed
    }
}

enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

enum CmpOp {
    Lt,
    Gt,
    Le,
    Ge,
}

fn expand_binop(
    func: &mut MirFunction,
    out: &mut Vec<(ValueId, Instruction)>,
    result_id: ValueId,
    a: ValueId,
    b: ValueId,
    op: BinOp,
) {
    let ua = func.new_value();
    let ub = func.new_value();
    let result_f = func.new_value();

    out.push((ua, Instruction::Unbox(a)));
    out.push((ub, Instruction::Unbox(b)));
    out.push((
        result_f,
        match op {
            BinOp::Add => Instruction::AddF64(ua, ub),
            BinOp::Sub => Instruction::SubF64(ua, ub),
            BinOp::Mul => Instruction::MulF64(ua, ub),
            BinOp::Div => Instruction::DivF64(ua, ub),
            BinOp::Mod => Instruction::ModF64(ua, ub),
        },
    ));
    out.push((result_id, Instruction::Box(result_f)));
}

fn expand_cmp(
    func: &mut MirFunction,
    out: &mut Vec<(ValueId, Instruction)>,
    result_id: ValueId,
    a: ValueId,
    b: ValueId,
    op: CmpOp,
) {
    let ua = func.new_value();
    let ub = func.new_value();

    out.push((ua, Instruction::Unbox(a)));
    out.push((ub, Instruction::Unbox(b)));
    out.push((
        result_id,
        match op {
            CmpOp::Lt => Instruction::CmpLtF64(ua, ub),
            CmpOp::Gt => Instruction::CmpGtF64(ua, ub),
            CmpOp::Le => Instruction::CmpLeF64(ua, ub),
            CmpOp::Ge => Instruction::CmpGeF64(ua, ub),
        },
    ));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::interp::{eval, InterpValue};
    use crate::mir::{Instruction, Terminator};
    use crate::runtime::value::Value;

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_specialize_add() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::ConstNum(20.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let before = eval(&f).unwrap();
        assert!(TypeSpecialize.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(30.0)));
        let last = f.block(bb).instructions.iter().find(|(v, _)| *v == v2);
        assert!(matches!(last, Some((_, Instruction::Box(_)))));
    }

    #[test]
    fn test_non_num_not_specialized() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::GetModuleVar(0)));
            b.instructions.push((v1, Instruction::ConstNum(10.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        assert!(!TypeSpecialize.run(&mut f));
        let (_, ref inst) = f.block(bb).instructions[2];
        assert!(matches!(inst, Instruction::Add(..)));
    }

    #[test]
    fn test_specialize_neg() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(42.0)));
            b.instructions.push((v1, Instruction::Neg(v0)));
            b.terminator = Terminator::Return(v1);
        }

        let before = eval(&f).unwrap();
        assert!(TypeSpecialize.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(-42.0)));
    }

    #[test]
    fn test_specialize_guard_num() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::GetModuleVar(0)));
            b.instructions.push((v1, Instruction::GuardNum(v0)));
            b.instructions.push((v2, Instruction::ConstNum(5.0)));
            b.instructions.push((v3, Instruction::Add(v1, v2)));
            b.terminator = Terminator::Return(v3);
        }

        assert!(TypeSpecialize.run(&mut f));
        let last = f.block(bb).instructions.iter().find(|(v, _)| *v == v3);
        assert!(matches!(last, Some((_, Instruction::Box(_)))));
    }

    #[test]
    fn test_specialize_comparison() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let bb_t = f.new_block();
        let bb_f = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v_yes = f.new_value();
        let v_no = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::ConstNum(2.0)));
            b.instructions.push((v2, Instruction::CmpLt(v0, v1)));
            b.terminator = Terminator::CondBranch {
                condition: v2,
                true_target: bb_t,
                true_args: vec![],
                false_target: bb_f,
                false_args: vec![],
            };
        }
        f.block_mut(bb_t)
            .instructions
            .push((v_yes, Instruction::ConstNum(1.0)));
        f.block_mut(bb_t).terminator = Terminator::Return(v_yes);
        f.block_mut(bb_f)
            .instructions
            .push((v_no, Instruction::ConstNum(0.0)));
        f.block_mut(bb_f).terminator = Terminator::Return(v_no);

        let before = eval(&f).unwrap();
        assert!(TypeSpecialize.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(1.0)));
    }
}
