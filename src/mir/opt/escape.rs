/// Escape analysis for heap allocations.
///
/// Determines which `MakeList`/`MakeMap`/`MakeRange` values do NOT escape
/// the function, enabling scalar replacement (SRA) to eliminate them.
///
/// A value escapes if it is returned, stored to a field/variable/upvalue,
/// passed as a call argument, or captured by a closure.
use std::collections::HashSet;

use crate::mir::{Instruction, MirFunction, Terminator, ValueId};

/// Returns the set of allocation ValueIds that do NOT escape.
pub fn analyze(func: &MirFunction) -> HashSet<ValueId> {
    // Collect all allocation values.
    let mut allocations: HashSet<ValueId> = HashSet::new();
    for block in &func.blocks {
        for (val_id, inst) in &block.instructions {
            if matches!(
                inst,
                Instruction::MakeList(_) | Instruction::MakeMap(_) | Instruction::MakeRange(..)
            ) {
                allocations.insert(*val_id);
            }
        }
    }

    if allocations.is_empty() {
        return HashSet::new();
    }

    // Find escaping allocations.
    let mut escaping: HashSet<ValueId> = HashSet::new();

    for block in &func.blocks {
        for (_, inst) in &block.instructions {
            check_inst_escapes(inst, &allocations, &mut escaping);
        }

        // Check terminator.
        match &block.terminator {
            Terminator::Return(v) if allocations.contains(v) => {
                escaping.insert(*v);
            }
            Terminator::Branch { args, .. } => {
                for arg in args {
                    if allocations.contains(arg) {
                        escaping.insert(*arg);
                    }
                }
            }
            Terminator::CondBranch {
                condition,
                true_args,
                false_args,
                ..
            } => {
                if allocations.contains(condition) {
                    escaping.insert(*condition);
                }
                for arg in true_args.iter().chain(false_args.iter()) {
                    if allocations.contains(arg) {
                        escaping.insert(*arg);
                    }
                }
            }
            _ => {}
        }
    }

    allocations.difference(&escaping).copied().collect()
}

fn check_inst_escapes(
    inst: &Instruction,
    allocations: &HashSet<ValueId>,
    escaping: &mut HashSet<ValueId>,
) {
    match inst {
        // SubscriptGet: receiver doesn't escape, but args might.
        Instruction::SubscriptGet { args, .. } => {
            for arg in args {
                if allocations.contains(arg) {
                    escaping.insert(*arg);
                }
            }
        }
        // SubscriptSet: receiver doesn't escape, but value and args do.
        Instruction::SubscriptSet { args, value, .. } => {
            if allocations.contains(value) {
                escaping.insert(*value);
            }
            for arg in args {
                if allocations.contains(arg) {
                    escaping.insert(*arg);
                }
            }
        }
        Instruction::SetField(_, _, val)
        | Instruction::SetModuleVar(_, val)
        | Instruction::SetUpvalue(_, val)
            if allocations.contains(val) =>
        {
            escaping.insert(*val);
        }
        Instruction::Call { receiver, args, .. } => {
            if allocations.contains(receiver) {
                escaping.insert(*receiver);
            }
            for arg in args {
                if allocations.contains(arg) {
                    escaping.insert(*arg);
                }
            }
        }
        Instruction::SuperCall { args, .. } => {
            for arg in args {
                if allocations.contains(arg) {
                    escaping.insert(*arg);
                }
            }
        }
        Instruction::MakeClosure { upvalues, .. } => {
            for uv in upvalues {
                if allocations.contains(uv) {
                    escaping.insert(*uv);
                }
            }
        }
        Instruction::MakeList(elems) => {
            for elem in elems {
                if allocations.contains(elem) {
                    escaping.insert(*elem);
                }
            }
        }
        Instruction::MakeMap(pairs) => {
            for (k, v) in pairs {
                if allocations.contains(k) {
                    escaping.insert(*k);
                }
                if allocations.contains(v) {
                    escaping.insert(*v);
                }
            }
        }
        Instruction::MakeRange(from, to, _) => {
            if allocations.contains(from) {
                escaping.insert(*from);
            }
            if allocations.contains(to) {
                escaping.insert(*to);
            }
        }
        Instruction::StringConcat(parts) => {
            for part in parts {
                if allocations.contains(part) {
                    escaping.insert(*part);
                }
            }
        }
        Instruction::ToString(v) if allocations.contains(v) => {
            escaping.insert(*v);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::{Interner, SymbolId};
    use crate::mir::{Instruction, MirFunction, Terminator};

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_local_only_does_not_escape() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(0.0)));
            b.instructions.push((v1, Instruction::ConstNum(1.0)));
            b.instructions
                .push((v2, Instruction::MakeList(vec![v0, v1])));
            b.instructions.push((
                v3,
                Instruction::SubscriptGet {
                    receiver: v2,
                    args: vec![v0],
                },
            ));
            b.terminator = Terminator::Return(v3);
        }

        let non_escaping = analyze(&f);
        assert!(non_escaping.contains(&v2));
    }

    #[test]
    fn test_returned_escapes() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::MakeList(vec![v0])));
            b.terminator = Terminator::Return(v1);
        }

        let non_escaping = analyze(&f);
        assert!(!non_escaping.contains(&v1));
    }

    #[test]
    fn test_field_stored_escapes() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v_recv = f.new_value();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v_recv, Instruction::GetModuleVar(0)));
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::MakeList(vec![v0])));
            b.instructions
                .push((v2, Instruction::SetField(v_recv, 0, v1)));
            b.terminator = Terminator::ReturnNull;
        }

        let non_escaping = analyze(&f);
        assert!(!non_escaping.contains(&v1));
    }

    #[test]
    fn test_call_arg_escapes() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v_recv = f.new_value();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v_recv, Instruction::GetModuleVar(0)));
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::MakeList(vec![v0])));
            b.instructions.push((
                v2,
                Instruction::Call {
                    receiver: v_recv,
                    method: SymbolId::from_raw(0),
                    args: vec![v1],
                pure_call: false,
},
            ));
            b.terminator = Terminator::ReturnNull;
        }

        let non_escaping = analyze(&f);
        assert!(!non_escaping.contains(&v1));
    }

    #[test]
    fn test_closure_capture_escapes() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(1.0)));
            b.instructions.push((v1, Instruction::MakeList(vec![v0])));
            b.instructions.push((
                v2,
                Instruction::MakeClosure {
                    fn_id: 0,
                    upvalues: vec![v1],
                },
            ));
            b.terminator = Terminator::ReturnNull;
        }

        let non_escaping = analyze(&f);
        assert!(!non_escaping.contains(&v1));
    }
}
