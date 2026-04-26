/// Common subexpression elimination via value numbering.
///
/// For each pure instruction, if an identical computation was already performed,
/// replace all uses of the new result with the earlier one. The redundant
/// instruction becomes dead and is cleaned up by DCE.
use std::collections::HashMap;

use super::{replace_uses_in_func, MirPass};
use crate::mir::{Instruction, MirFunction, ValueId};

pub struct Cse;

impl MirPass for Cse {
    fn name(&self) -> &str {
        "cse"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let mut replacements: HashMap<ValueId, ValueId> = HashMap::new();

        // CSE within each block only — merging across blocks requires
        // dominance analysis to avoid referencing values from non-dominating
        // blocks (e.g. merging constants between then/else branches).
        for block in &func.blocks {
            // Two caches:
            //  - `pure`: instructions whose result never depends on heap state
            //    (Add, ConstNum, Move, etc.). Free to merge across the whole
            //    block; calls / stores can't change their answer.
            //  - `memory`: reads that DO depend on heap state (SubscriptGet,
            //    method dispatches, etc.). A side-effecting instruction in
            //    between two such reads can have mutated the receiver, so we
            //    drop this cache whenever one is encountered. This is the
            //    minimal memory-dependency tracking the pass needs to stay
            //    sound around boxed-upvalue closures (where outer's read of
            //    the box can't be merged across the closure call that
            //    mutates it).
            let mut pure: HashMap<CseKey, ValueId> = HashMap::new();
            let mut memory: HashMap<CseKey, ValueId> = HashMap::new();
            for (val_id, inst) in &block.instructions {
                // Side-effecting instructions invalidate every cached
                // memory-dependent read. Pure cache survives.
                if inst.has_side_effects() {
                    if inst_may_write_memory(inst) {
                        memory.clear();
                    }
                    continue;
                }
                if let Some(key) = make_key(inst, &replacements) {
                    let cache = if inst_reads_memory(inst) {
                        &mut memory
                    } else {
                        &mut pure
                    };
                    if let Some(&existing) = cache.get(&key) {
                        replacements.insert(*val_id, existing);
                    } else {
                        cache.insert(key, *val_id);
                    }
                }
            }
        }

        if replacements.is_empty() {
            return false;
        }

        replace_uses_in_func(func, &replacements);
        true
    }
}

// ---------------------------------------------------------------------------
// CSE key
// ---------------------------------------------------------------------------

#[derive(Hash, Eq, PartialEq, Clone)]
struct CseKey(Vec<u64>);

fn resolve(v: ValueId, replacements: &HashMap<ValueId, ValueId>) -> ValueId {
    let mut current = v;
    let mut depth = 0;
    while let Some(&next) = replacements.get(&current) {
        if next == current || depth > 100 {
            break;
        }
        current = next;
        depth += 1;
    }
    current
}

/// True for instructions whose result depends on mutable heap state, so
/// they should be dropped from the CSE cache whenever a side-effecting
/// instruction (call, store, etc.) intervenes within the block.
fn inst_reads_memory(inst: &Instruction) -> bool {
    matches!(inst, Instruction::SubscriptGet { .. })
}

/// True for instructions that can mutate observable heap state — anything
/// else cached as a memory read may now return a stale value. Calls
/// flagged `pure_call` (see `Instruction::Call::pure_call`) are exempt:
/// the MIR builder marks them based on a known-pure-methods list (Num
/// arithmetic, comparisons, Math intrinsics) so a `subscript_get`
/// before a `+` and another after the `+` can still merge.
///
/// This is the seed of the per-function effect-summary system in the
/// Phase 6 roadmap. Real call-graph propagation would let CSE keep
/// the cache across user-defined leaf methods too; for now we lean on
/// builtin-method names.
fn inst_may_write_memory(inst: &Instruction) -> bool {
    match inst {
        Instruction::Call { pure_call, .. } => !*pure_call,
        Instruction::CallKnownFunc { .. }
        | Instruction::CallStaticSelf { .. }
        | Instruction::SuperCall { .. }
        | Instruction::SetField(..)
        | Instruction::SetUpvalue(..)
        | Instruction::SubscriptSet { .. }
        | Instruction::SetStaticField(..)
        | Instruction::SetModuleVar(..) => true,
        _ => false,
    }
}

fn make_key(inst: &Instruction, replacements: &HashMap<ValueId, ValueId>) -> Option<CseKey> {
    // Don't CSE block params, mutable reads that need observed identity,
    // or allocations (each must produce a distinct object).
    if matches!(
        inst,
        Instruction::BlockParam(_)
            | Instruction::GetModuleVar(_)
            | Instruction::GetUpvalue(_)
            | Instruction::GetField(..)
            | Instruction::MakeClosure { .. }
            | Instruction::MakeList(..)
            | Instruction::MakeMap(..)
    ) {
        return None;
    }

    let mut key = Vec::new();
    key.push(inst_discriminant(inst) as u64);

    // Extra constant data.
    match inst {
        Instruction::ConstNum(n) | Instruction::ConstF64(n) => key.push(n.to_bits()),
        Instruction::ConstBool(b) => key.push(*b as u64),
        Instruction::ConstI64(n) => key.push(*n as u64),
        Instruction::ConstString(idx) => key.push(*idx as u64),
        Instruction::GetField(_, idx) => key.push(*idx as u64),
        Instruction::GuardClass(_, sym) | Instruction::IsType(_, sym) => {
            key.push(sym.index() as u64);
        }
        Instruction::MakeRange(_, _, incl) => key.push(*incl as u64),
        Instruction::MathUnaryF64(op, _) => key.push(*op as u64),
        Instruction::MathBinaryF64(op, _, _) => key.push(*op as u64),
        _ => {}
    }

    // Resolved operands.
    for op in inst.operands() {
        key.push(resolve(op, replacements).0 as u64);
    }

    Some(CseKey(key))
}

fn inst_discriminant(inst: &Instruction) -> u32 {
    use Instruction::*;
    match inst {
        ConstNum(_) => 0,
        ConstBool(_) => 1,
        ConstNull => 2,
        ConstString(_) => 3,
        ConstF64(_) => 4,
        ConstI64(_) => 5,
        Add(..) => 6,
        Sub(..) => 7,
        Mul(..) => 8,
        Div(..) => 9,
        Mod(..) => 10,
        Neg(..) => 11,
        AddF64(..) => 12,
        SubF64(..) => 13,
        MulF64(..) => 14,
        DivF64(..) => 15,
        ModF64(..) => 16,
        NegF64(..) => 17,
        CmpLt(..) => 18,
        CmpGt(..) => 19,
        CmpLe(..) => 20,
        CmpGe(..) => 21,
        CmpEq(..) => 22,
        CmpNe(..) => 23,
        CmpLtF64(..) => 24,
        CmpGtF64(..) => 25,
        CmpLeF64(..) => 26,
        CmpGeF64(..) => 27,
        Not(..) => 28,
        BitAnd(..) => 29,
        BitOr(..) => 30,
        BitXor(..) => 31,
        BitNot(..) => 32,
        Shl(..) => 33,
        Shr(..) => 34,
        GuardNum(..) => 35,
        GuardBool(..) => 36,
        GuardClass(..) => 37,
        Unbox(..) => 38,
        Box(..) => 39,
        GetField(..) => 40,
        SetField(..) => 41,
        GetModuleVar(..) => 42,
        SetModuleVar(..) => 43,
        Call { .. } => 44,
        CallStaticSelf { .. } => 45,
        SuperCall { .. } => 46,
        MakeClosure { .. } => 47,
        GetUpvalue(..) => 48,
        SetUpvalue(..) => 49,
        MakeList(..) => 50,
        MakeMap(..) => 51,
        MakeRange(..) => 52,
        StringConcat(..) => 53,
        ToString(..) => 54,
        IsType(..) => 55,
        SubscriptGet { .. } => 56,
        SubscriptSet { .. } => 57,
        Move(..) => 58,
        BlockParam(..) => 59,
        MathUnaryF64(..) => 60,
        MathBinaryF64(..) => 61,
        GuardProtocol(..) => 62,
        GetStaticField(..) => 63,
        SetStaticField(..) => 64,
        CallKnownFunc { .. } => 65,
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
    use crate::mir::{Instruction, Terminator};
    use crate::runtime::value::Value;

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    #[test]
    fn test_same_ops_merged() {
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
            b.instructions.push((v0, Instruction::ConstNum(5.0)));
            b.instructions.push((v1, Instruction::ConstNum(3.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.instructions.push((v3, Instruction::Add(v0, v1)));
            b.instructions.push((v4, Instruction::Add(v2, v3)));
            b.terminator = Terminator::Return(v4);
        }

        let before = eval(&f).unwrap();
        assert!(Cse.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        // v4 should use v2 twice
        let (_, ref inst) = f.block(bb).instructions[4];
        assert!(matches!(inst, Instruction::Add(a, b) if *a == v2 && *b == v2));
    }

    #[test]
    fn test_different_ops_kept() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(5.0)));
            b.instructions.push((v1, Instruction::ConstNum(3.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.instructions.push((v3, Instruction::Sub(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        assert!(!Cse.run(&mut f));
    }

    #[test]
    fn test_calls_not_merged() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let method = crate::intern::SymbolId::from_raw(0);
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNull));
            b.instructions.push((
                v1,
                Instruction::Call {
                    receiver: v0,
                    method,
                    args: vec![],
                pure_call: false,
},
            ));
            b.instructions.push((
                v2,
                Instruction::Call {
                    receiver: v0,
                    method,
                    args: vec![],
                pure_call: false,
},
            ));
            b.terminator = Terminator::Return(v1);
        }

        assert!(!Cse.run(&mut f));
    }

    #[test]
    fn test_constants_deduped() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(42.0)));
            b.instructions.push((v1, Instruction::ConstNum(42.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.terminator = Terminator::Return(v2);
        }

        let before = eval(&f).unwrap();
        assert!(Cse.run(&mut f));
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        let (_, ref inst) = f.block(bb).instructions[2];
        assert!(matches!(inst, Instruction::Add(a, b) if *a == v0 && *b == v0));
    }

    #[test]
    fn test_cse_with_interpreter() {
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
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.instructions.push((v3, Instruction::Add(v0, v1)));
            b.instructions.push((v4, Instruction::Mul(v2, v3)));
            b.terminator = Terminator::Return(v4);
        }

        let before = eval(&f).unwrap();
        Cse.run(&mut f);
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(after, InterpValue::Boxed(Value::num(900.0)));
    }
}
