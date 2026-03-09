/// Scalar replacement of aggregates.
///
/// Replaces non-escaping `MakeList` allocations with individual scalar values
/// when all `SubscriptGet` accesses use constant indices. This eliminates the
/// heap allocation entirely — the list elements live as SSA values.
use std::collections::HashMap;

use super::MirPass;
use crate::mir::{Instruction, MirFunction, ValueId};

pub struct Sra;

impl MirPass for Sra {
    fn name(&self) -> &str {
        "sra"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        // Step 1: Escape analysis.
        let non_escaping = super::escape::analyze(func);
        if non_escaping.is_empty() {
            return false;
        }

        // Step 2: Collect MakeList elements and constants.
        let mut list_elements: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
        let mut const_nums: HashMap<ValueId, f64> = HashMap::new();

        for block in &func.blocks {
            for (val_id, inst) in &block.instructions {
                if let Instruction::MakeList(elems) = inst {
                    if non_escaping.contains(val_id) {
                        list_elements.insert(*val_id, elems.clone());
                    }
                }
                match inst {
                    Instruction::ConstNum(n) => {
                        const_nums.insert(*val_id, *n);
                    }
                    Instruction::ConstI64(n) => {
                        const_nums.insert(*val_id, *n as f64);
                    }
                    _ => {}
                }
            }
        }

        if list_elements.is_empty() {
            return false;
        }

        // Step 3: Block lists with non-constant index accesses.
        let mut blocked = Vec::new();
        for block in &func.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::SubscriptGet { receiver, args }
                    | Instruction::SubscriptSet { receiver, args, .. }
                        if list_elements.contains_key(receiver)
                            && !args.is_empty()
                            && !const_nums.contains_key(&args[0]) =>
                    {
                        blocked.push(*receiver);
                    }
                    _ => {}
                }
            }
        }
        for b in &blocked {
            list_elements.remove(b);
        }
        if list_elements.is_empty() {
            return false;
        }

        // Step 4: Replace SubscriptGet with Move of the element.
        let mut changed = false;
        let current_elements = list_elements.clone();

        for block_idx in 0..func.blocks.len() {
            for inst_idx in 0..func.blocks[block_idx].instructions.len() {
                let (val_id, ref inst) = func.blocks[block_idx].instructions[inst_idx];
                if let Instruction::SubscriptGet { receiver, args } = inst {
                    if let Some(elems) = current_elements.get(receiver) {
                        if !args.is_empty() {
                            if let Some(&idx_f) = const_nums.get(&args[0]) {
                                let idx = idx_f as usize;
                                if idx < elems.len() {
                                    func.blocks[block_idx].instructions[inst_idx] =
                                        (val_id, Instruction::Move(elems[idx]));
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        changed
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

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_basic_list_to_scalars() {
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
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::ConstNum(20.0)));
            b.instructions
                .push((v2, Instruction::MakeList(vec![v0, v1])));
            b.instructions.push((v3, Instruction::ConstNum(0.0)));
            b.instructions.push((
                v4,
                Instruction::SubscriptGet {
                    receiver: v2,
                    args: vec![v3],
                },
            ));
            b.terminator = Terminator::Return(v4);
        }

        assert!(Sra.run(&mut f));
        let (_, ref inst) = f.block(bb).instructions[4];
        assert!(matches!(inst, Instruction::Move(v) if *v == v0));
    }

    #[test]
    fn test_dynamic_index_blocks_sra() {
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
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::ConstNum(20.0)));
            b.instructions
                .push((v2, Instruction::MakeList(vec![v0, v1])));
            b.instructions.push((v3, Instruction::GetModuleVar(0)));
            b.instructions.push((
                v4,
                Instruction::SubscriptGet {
                    receiver: v2,
                    args: vec![v3],
                },
            ));
            b.terminator = Terminator::Return(v4);
        }

        assert!(!Sra.run(&mut f));
    }

    #[test]
    fn test_escaping_list_not_replaced() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::MakeList(vec![v0])));
            b.terminator = Terminator::Return(v1);
        }

        assert!(!Sra.run(&mut f));
    }

    #[test]
    fn test_second_element_access() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        let v4 = f.new_value();
        let v5 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(10.0)));
            b.instructions.push((v1, Instruction::ConstNum(20.0)));
            b.instructions.push((v2, Instruction::ConstNum(30.0)));
            b.instructions
                .push((v3, Instruction::MakeList(vec![v0, v1, v2])));
            b.instructions.push((v4, Instruction::ConstNum(1.0)));
            b.instructions.push((
                v5,
                Instruction::SubscriptGet {
                    receiver: v3,
                    args: vec![v4],
                },
            ));
            b.terminator = Terminator::Return(v5);
        }

        assert!(Sra.run(&mut f));
        let (_, ref inst) = f.block(bb).instructions[5];
        assert!(matches!(inst, Instruction::Move(v) if *v == v1));
    }

    #[test]
    fn test_no_allocations() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(42.0)));
            b.terminator = Terminator::Return(v0);
        }

        assert!(!Sra.run(&mut f));
    }
}
