/// Dead code elimination.
///
/// - Removes instructions whose results are never used and have no side effects.
/// - Clears unreachable blocks (no path from entry).

use std::collections::HashSet;

use super::MirPass;
use crate::mir::{BlockId, MirFunction, Terminator, ValueId};

pub struct Dce;

impl MirPass for Dce {
    fn name(&self) -> &str {
        "dce"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut changed = false;
        if remove_unreachable_blocks(func) {
            changed = true;
        }
        if remove_dead_instructions(func) {
            changed = true;
        }
        changed
    }
}

fn remove_unreachable_blocks(func: &mut MirFunction) -> bool {
    if func.blocks.is_empty() {
        return false;
    }

    let mut reachable = HashSet::new();
    let mut worklist = vec![func.entry_block()];
    while let Some(bid) = worklist.pop() {
        if !reachable.insert(bid) {
            continue;
        }
        for succ in func.block(bid).terminator.successors() {
            if !reachable.contains(&succ) {
                worklist.push(succ);
            }
        }
    }

    let mut changed = false;
    for idx in 0..func.blocks.len() {
        let bid = BlockId(idx as u32);
        if !reachable.contains(&bid) {
            let block = &mut func.blocks[idx];
            if !block.instructions.is_empty()
                || !matches!(block.terminator, Terminator::Unreachable)
            {
                block.instructions.clear();
                block.params.clear();
                block.terminator = Terminator::Unreachable;
                changed = true;
            }
        }
    }
    changed
}

fn remove_dead_instructions(func: &mut MirFunction) -> bool {
    let used = compute_used_values(func);
    let mut changed = false;
    for block in &mut func.blocks {
        let before = block.instructions.len();
        block
            .instructions
            .retain(|(vid, inst)| used.contains(vid) || inst.has_side_effects());
        if block.instructions.len() != before {
            changed = true;
        }
    }
    changed
}

fn compute_used_values(func: &MirFunction) -> HashSet<ValueId> {
    let mut used = HashSet::new();

    // Seed from terminators.
    for block in &func.blocks {
        for v in block.terminator.operands() {
            used.insert(v);
        }
    }

    // Seed from side-effecting instructions.
    for block in &func.blocks {
        for (_, inst) in &block.instructions {
            if inst.has_side_effects() {
                for op in inst.operands() {
                    used.insert(op);
                }
            }
        }
    }

    // Fixpoint: mark operands of used instructions.
    let mut progress = true;
    while progress {
        progress = false;
        for block in &func.blocks {
            for (vid, inst) in &block.instructions {
                if used.contains(vid) {
                    for op in inst.operands() {
                        if used.insert(op) {
                            progress = true;
                        }
                    }
                }
            }
        }
    }

    used
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
    fn test_remove_unused_values() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(42.0)));
            b.instructions.push((v1, Instruction::ConstNum(99.0)));
            b.terminator = Terminator::Return(v0);
        }

        let before = eval(&f).unwrap();
        assert!(Dce.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(f.block(bb).instructions.len(), 1);
    }

    #[test]
    fn test_preserve_side_effects() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(42.0)));
            b.instructions
                .push((v1, Instruction::SetModuleVar(0, v0)));
            b.instructions.push((v2, Instruction::ConstNum(99.0)));
            b.terminator = Terminator::Return(v2);
        }

        assert!(!Dce.run(&mut f));
        assert_eq!(f.block(bb).instructions.len(), 3);
    }

    #[test]
    fn test_remove_chain_of_dead_code() {
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
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::ConstNum(2.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.instructions.push((v3, Instruction::Mul(v2, v2)));
            b.instructions.push((v4, Instruction::ConstNum(99.0)));
            b.terminator = Terminator::Return(v4);
        }

        let before = eval(&f).unwrap();
        assert!(Dce.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(f.block(bb).instructions.len(), 1);
    }

    #[test]
    fn test_remove_unreachable_blocks() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();

        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb2,
            args: vec![],
        };
        f.block_mut(bb1)
            .instructions
            .push((v0, Instruction::ConstNum(99.0)));
        f.block_mut(bb1).terminator = Terminator::Return(v0);
        f.block_mut(bb2)
            .instructions
            .push((v1, Instruction::ConstNum(42.0)));
        f.block_mut(bb2).terminator = Terminator::Return(v1);

        let before = eval(&f).unwrap();
        assert!(Dce.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(42.0)));
        assert!(f.block(bb1).instructions.is_empty());
        assert!(matches!(
            f.block(bb1).terminator,
            Terminator::Unreachable
        ));
    }

    #[test]
    fn test_dce_preserves_used_chain() {
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

        assert!(!Dce.run(&mut f));
        assert_eq!(f.block(bb).instructions.len(), 3);
    }
}
