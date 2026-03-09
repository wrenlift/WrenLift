/// Optimization pass infrastructure and utilities.
///
/// Each pass implements `MirPass`. The pass manager runs passes in sequence,
/// optionally iterating to fixpoint.

pub mod constfold;
pub mod cse;
pub mod dce;
pub mod devirt;
pub mod escape;
pub mod inline;
pub mod licm;
pub mod sra;

use std::collections::HashMap;

use super::{Instruction, MirFunction, Terminator, ValueId};

// ---------------------------------------------------------------------------
// Pass trait
// ---------------------------------------------------------------------------

/// An optimization pass over a single MIR function.
pub trait MirPass {
    /// Human-readable pass name.
    fn name(&self) -> &str;
    /// Run the pass. Returns `true` if the function was modified.
    fn run(&self, func: &mut MirFunction) -> bool;
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Run a sequence of passes once. Returns `true` if any pass modified the function.
pub fn run_pipeline(func: &mut MirFunction, passes: &[&dyn MirPass]) -> bool {
    let mut changed = false;
    for pass in passes {
        if pass.run(func) {
            changed = true;
        }
    }
    changed
}

/// Run passes repeatedly until no pass reports changes or `max_iters` is reached.
/// Returns the number of iterations executed.
pub fn run_to_fixpoint(
    func: &mut MirFunction,
    passes: &[&dyn MirPass],
    max_iters: usize,
) -> usize {
    let mut iters = 0;
    loop {
        iters += 1;
        if !run_pipeline(func, passes) || iters >= max_iters {
            break;
        }
    }
    iters
}

// ---------------------------------------------------------------------------
// Replace uses utility
// ---------------------------------------------------------------------------

/// Replace all uses of values according to the replacement map.
/// Follows chains: if v1→v2 and v2→v3, v1 resolves to v3.
pub fn replace_uses_in_func(func: &mut MirFunction, map: &HashMap<ValueId, ValueId>) {
    if map.is_empty() {
        return;
    }
    for block in &mut func.blocks {
        for (_, inst) in &mut block.instructions {
            replace_in_inst(inst, map);
        }
        replace_in_term(&mut block.terminator, map);
    }
}

fn resolve(v: ValueId, map: &HashMap<ValueId, ValueId>) -> ValueId {
    let mut current = v;
    let mut depth = 0;
    while let Some(&next) = map.get(&current) {
        if next == current || depth > 100 {
            break;
        }
        current = next;
        depth += 1;
    }
    current
}

fn replace_in_inst(inst: &mut Instruction, map: &HashMap<ValueId, ValueId>) {
    use Instruction::*;
    match inst {
        // No operands
        ConstNum(_) | ConstBool(_) | ConstNull | ConstString(_) | ConstF64(_) | ConstI64(_)
        | GetModuleVar(_) | GetUpvalue(_) | BlockParam(_) => {}

        // One operand
        Neg(a) | NegF64(a) | Not(a) | BitNot(a) | GuardNum(a) | GuardBool(a) | Unbox(a)
        | Box(a) | Move(a) | ToString(a) | MathUnaryF64(_, a) => {
            *a = resolve(*a, map);
        }
        GuardClass(a, _) | GuardProtocol(a, _) | IsType(a, _) => {
            *a = resolve(*a, map);
        }
        GetField(recv, _) => {
            *recv = resolve(*recv, map);
        }

        // Two operands
        Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Mod(a, b) | AddF64(a, b)
        | SubF64(a, b) | MulF64(a, b) | DivF64(a, b) | ModF64(a, b) | CmpLt(a, b)
        | CmpGt(a, b) | CmpLe(a, b) | CmpGe(a, b) | CmpEq(a, b) | CmpNe(a, b)
        | CmpLtF64(a, b) | CmpGtF64(a, b) | CmpLeF64(a, b) | CmpGeF64(a, b) | BitAnd(a, b)
        | BitOr(a, b) | BitXor(a, b) | Shl(a, b) | Shr(a, b) | MathBinaryF64(_, a, b) => {
            *a = resolve(*a, map);
            *b = resolve(*b, map);
        }

        // Special multi-operand
        SetField(recv, _, val) => {
            *recv = resolve(*recv, map);
            *val = resolve(*val, map);
        }
        SetModuleVar(_, val) | SetUpvalue(_, val) => {
            *val = resolve(*val, map);
        }
        Call {
            receiver, args, ..
        } => {
            *receiver = resolve(*receiver, map);
            for arg in args.iter_mut() {
                *arg = resolve(*arg, map);
            }
        }
        SuperCall { args, .. } => {
            for arg in args.iter_mut() {
                *arg = resolve(*arg, map);
            }
        }
        MakeClosure { upvalues, .. } => {
            for uv in upvalues.iter_mut() {
                *uv = resolve(*uv, map);
            }
        }
        MakeList(elems) => {
            for elem in elems.iter_mut() {
                *elem = resolve(*elem, map);
            }
        }
        MakeMap(pairs) => {
            for (k, v) in pairs.iter_mut() {
                *k = resolve(*k, map);
                *v = resolve(*v, map);
            }
        }
        MakeRange(from, to, _) => {
            *from = resolve(*from, map);
            *to = resolve(*to, map);
        }
        StringConcat(parts) => {
            for part in parts.iter_mut() {
                *part = resolve(*part, map);
            }
        }
        SubscriptGet { receiver, args } => {
            *receiver = resolve(*receiver, map);
            for arg in args.iter_mut() {
                *arg = resolve(*arg, map);
            }
        }
        SubscriptSet {
            receiver,
            args,
            value,
        } => {
            *receiver = resolve(*receiver, map);
            for arg in args.iter_mut() {
                *arg = resolve(*arg, map);
            }
            *value = resolve(*value, map);
        }
    }
}

fn replace_in_term(term: &mut Terminator, map: &HashMap<ValueId, ValueId>) {
    match term {
        Terminator::Return(v) => *v = resolve(*v, map),
        Terminator::ReturnNull | Terminator::Unreachable => {}
        Terminator::Branch { args, .. } => {
            for arg in args.iter_mut() {
                *arg = resolve(*arg, map);
            }
        }
        Terminator::CondBranch {
            condition,
            true_args,
            false_args,
            ..
        } => {
            *condition = resolve(*condition, map);
            for arg in true_args.iter_mut() {
                *arg = resolve(*arg, map);
            }
            for arg in false_args.iter_mut() {
                *arg = resolve(*arg, map);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::interp::eval;
    use crate::mir::{Instruction, Terminator};

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    struct NoopPass;
    impl MirPass for NoopPass {
        fn name(&self) -> &str {
            "noop"
        }
        fn run(&self, _func: &mut MirFunction) -> bool {
            false
        }
    }

    #[test]
    fn test_pipeline_runs_all_passes() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v0, Instruction::ConstNum(1.0)));
        f.block_mut(bb).terminator = Terminator::Return(v0);

        let changed = run_pipeline(&mut f, &[&NoopPass, &NoopPass]);
        assert!(!changed);
    }

    #[test]
    fn test_fixpoint_converges() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::ReturnNull;

        let iters = run_to_fixpoint(&mut f, &[&NoopPass], 10);
        assert_eq!(iters, 1);
    }

    #[test]
    fn test_constfold_then_dce_pipeline() {
        // v0=2, v1=3, v2=add(v0,v1), v3=5, return v3
        // After constfold+DCE: only v3=5 remains
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v0 = f.new_value();
        let v1 = f.new_value();
        let v2 = f.new_value();
        let v3 = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((v0, Instruction::ConstNum(2.0)));
            b.instructions.push((v1, Instruction::ConstNum(3.0)));
            b.instructions.push((v2, Instruction::Add(v0, v1)));
            b.instructions.push((v3, Instruction::ConstNum(5.0)));
            b.terminator = Terminator::Return(v3);
        }

        let before = eval(&f).unwrap();
        let cf = constfold::ConstFold;
        let dce = dce::Dce;
        run_pipeline(&mut f, &[&cf, &dce]);
        let after = eval(&f).unwrap();
        assert_eq!(before, after);
        assert_eq!(f.block(bb).instructions.len(), 1);
    }
}
