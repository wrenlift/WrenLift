//! Drop a guard that the same guard on the same value has passed on
//! every path to it.
//!
//! An object's class never changes and a Num stays a Num, so a passed
//! `GuardClassAt` or Num guard holds for the rest of the body. Values
//! are compared through copies. A block entered by a back edge starts
//! with nothing proven: an OSR entry arrives there from the interpreter,
//! which ran none of the guards before it. Every other block is entered
//! only from blocks before it in reverse postorder, so one pass in that
//! order sees each block's predecessors first. A dropped guard's result
//! is its operand.

use std::collections::{HashMap, HashSet};

use super::licm::compute_rpo;
use super::{MirPass, replace_uses_in_func};
use crate::mir::{Instruction, MirFunction, ValueId};

pub struct RedundantGuards;

/// What a guard proves about its value's root.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Fact {
    Class(ValueId, usize),
    Num(ValueId),
}

impl MirPass for RedundantGuards {
    fn name(&self) -> &str {
        "redundant-guards"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        let copy_of: HashMap<ValueId, ValueId> = func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|(v, i)| match i {
                Instruction::Move(a) => Some((*v, *a)),
                _ => None,
            })
            .collect();
        let root = |mut v: ValueId| {
            while let Some(a) = copy_of.get(&v) {
                v = *a;
            }
            v
        };
        let fact = |inst: &Instruction| match inst {
            Instruction::GuardClassAt { value, class, .. } => {
                Some(Fact::Class(root(*value), *class))
            }
            Instruction::GuardNumAt { value, .. } | Instruction::GuardNum(value) => {
                Some(Fact::Num(root(*value)))
            }
            _ => None,
        };

        func.compute_predecessors();
        let rpo = compute_rpo(func);
        let mut order = vec![usize::MAX; func.blocks.len()];
        for (i, b) in rpo.iter().enumerate() {
            order[b.0 as usize] = i;
        }
        let mut facts_out: Vec<Option<HashSet<Fact>>> = vec![None; func.blocks.len()];
        let mut dropped: HashMap<ValueId, ValueId> = HashMap::new();
        for bid in &rpo {
            let bi = bid.0 as usize;
            let block = &func.blocks[bi];
            let back_edge = block
                .predecessors
                .iter()
                .any(|p| order.get(p.0 as usize).is_none_or(|&o| o >= order[bi]));
            let mut facts: HashSet<Fact> = HashSet::new();
            if !back_edge {
                let mut preds = block
                    .predecessors
                    .iter()
                    .filter_map(|p| facts_out[p.0 as usize].as_ref());
                if let Some(first) = preds.next() {
                    facts = first.clone();
                    for other in preds {
                        facts.retain(|f| other.contains(f));
                    }
                }
            }
            for (v, inst) in &block.instructions {
                let Some(f) = fact(inst) else {
                    continue;
                };
                if !facts.insert(f) {
                    match inst {
                        Instruction::GuardClassAt { value, .. }
                        | Instruction::GuardNumAt { value, .. } => {
                            dropped.insert(*v, *value);
                        }
                        _ => {}
                    }
                }
            }
            facts_out[bi] = Some(facts);
        }
        if dropped.is_empty() {
            return false;
        }
        for block in &mut func.blocks {
            block.instructions.retain(|(v, _)| !dropped.contains_key(v));
        }
        replace_uses_in_func(func, &dropped);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::{BlockId, Terminator};

    fn class_guard(value: ValueId, class: usize) -> Instruction {
        Instruction::GuardClassAt {
            value,
            class,
            pc: 0,
            live: vec![],
        }
    }

    #[test]
    fn a_guard_every_path_passed_is_dropped() {
        let mut interner = Interner::new();
        let mut f = MirFunction::new(interner.intern("test"), 1);
        let entry = f.new_block();
        let then = f.new_block();
        let obj = f.new_value();
        let first = f.new_value();
        let copy = f.new_value();
        let second = f.new_value();
        let other = f.new_value();
        let field = f.new_value();
        f.block_mut(entry).instructions = vec![
            (obj, Instruction::BlockParam(0)),
            (first, class_guard(obj, 0x1000)),
        ];
        f.block_mut(entry).terminator = Terminator::Branch {
            target: then,
            args: vec![],
        };
        f.block_mut(then).instructions = vec![
            (copy, Instruction::Move(obj)),
            (second, class_guard(copy, 0x1000)),
            (other, class_guard(copy, 0x2000)),
            (field, Instruction::GetField(second, 0)),
        ];
        f.block_mut(then).terminator = Terminator::Return(field);
        assert!(RedundantGuards.run(&mut f));
        let insts = &f.block(then).instructions;
        assert!(!insts.iter().any(|(v, _)| *v == second));
        assert!(insts.iter().any(|(v, _)| *v == other));
        assert!(
            insts
                .iter()
                .any(|(v, i)| *v == field && matches!(i, Instruction::GetField(r, 0) if *r == copy))
        );
    }

    /// A loop an OSR entry can enter, and the block after it, keep their
    /// guards though the one before the loop dominates both.
    #[test]
    fn a_loop_header_and_what_follows_it_start_unproven() {
        let mut interner = Interner::new();
        let mut f = MirFunction::new(interner.intern("test"), 1);
        let entry = f.new_block();
        let header = f.new_block();
        let body = f.new_block();
        let exit = f.new_block();
        let obj = f.new_value();
        let flag = f.new_value();
        let before = f.new_value();
        let in_loop = f.new_value();
        let after = f.new_value();
        f.block_mut(entry).instructions = vec![
            (obj, Instruction::BlockParam(0)),
            (flag, Instruction::ConstBool(true)),
            (before, class_guard(obj, 0x1000)),
        ];
        f.block_mut(entry).terminator = Terminator::Branch {
            target: header,
            args: vec![],
        };
        f.block_mut(header).terminator = Terminator::CondBranch {
            condition: flag,
            true_target: body,
            true_args: vec![],
            false_target: exit,
            false_args: vec![],
        };
        f.block_mut(body).instructions = vec![(in_loop, class_guard(obj, 0x1000))];
        f.block_mut(body).terminator = Terminator::Branch {
            target: BlockId(header.0),
            args: vec![],
        };
        f.block_mut(exit).instructions = vec![(after, class_guard(obj, 0x1000))];
        f.block_mut(exit).terminator = Terminator::ReturnNull;
        assert!(!RedundantGuards.run(&mut f));
    }
}
