//! Carry provably-Num block parameters as raw f64.
//!
//! The type specialiser leaves every value boxed and wraps each f64
//! operation in an unbox/box pair, so a loop accumulator crosses the
//! integer/float register file twice per operation. Where every edge
//! feeds a parameter from a `Box`, a Num constant or another such
//! parameter, the parameter becomes f64: the edges pass the unboxed
//! value, unboxes of the parameter collapse to the parameter itself, and
//! any remaining boxed use reads one `Box` planted at the top of the
//! parameter's block. Runs once, after the rest of the JIT pipeline.

use std::collections::{HashMap, HashSet};

use super::{MirPass, replace_uses_in_func};
use crate::mir::{BlockId, Instruction, MirFunction, MirType, Terminator, ValueId};

pub struct UnboxParams;

impl MirPass for UnboxParams {
    fn name(&self) -> &str {
        "unbox-params"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let defs: HashMap<ValueId, Instruction> = func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter().map(|(v, i)| (*v, i.clone())))
            .collect();

        // Optimistic fixed point over the non-entry parameters.
        let mut chosen: HashSet<ValueId> = func
            .blocks
            .iter()
            .skip(1)
            .flat_map(|b| {
                b.params
                    .iter()
                    .filter(|(_, t)| *t == MirType::Value)
                    .map(|(p, _)| *p)
            })
            .collect();
        loop {
            let mut dropped = false;
            for block in &func.blocks {
                for (target, args) in edges(&block.terminator) {
                    let params = &func.blocks[target.0 as usize].params;
                    for (i, arg) in args.iter().enumerate() {
                        let Some(&(p, _)) = params.get(i) else {
                            continue;
                        };
                        if chosen.contains(&p) && f64_source(*arg, &defs, &chosen).is_none() {
                            chosen.remove(&p);
                            dropped = true;
                        }
                    }
                }
            }
            if !dropped {
                break;
            }
        }

        // Retype the parameters and rewrite the edges.
        for block in &mut func.blocks {
            for (p, t) in &mut block.params {
                if chosen.contains(p) {
                    *t = MirType::F64;
                }
            }
        }
        for bi in 0..func.blocks.len() {
            let mut consts: Vec<(ValueId, Instruction)> = Vec::new();
            let mut cache: HashMap<u64, ValueId> = HashMap::new();
            let mut unboxes: HashMap<ValueId, ValueId> = HashMap::new();
            let targets: Vec<(BlockId, Vec<ValueId>)> = edges(&func.blocks[bi].terminator)
                .into_iter()
                .map(|(t, a)| (t, a.to_vec()))
                .collect();
            let mut new_args: Vec<Vec<ValueId>> = Vec::new();
            for (target, args) in &targets {
                let params: Vec<ValueId> = func.blocks[target.0 as usize]
                    .params
                    .iter()
                    .map(|(p, _)| *p)
                    .collect();
                let mut out = args.clone();
                for (i, arg) in args.iter().enumerate() {
                    let Some(p) = params.get(i) else { continue };
                    if !chosen.contains(p) {
                        continue;
                    }
                    out[i] = match f64_source(*arg, &defs, &chosen).expect("checked above") {
                        Source::Value(v) => v,
                        Source::Guarded(g) => *unboxes.entry(g).or_insert_with(|| {
                            let id = ValueId(func.next_value);
                            func.next_value += 1;
                            consts.push((id, Instruction::Unbox(g)));
                            id
                        }),
                        Source::Const(c) => *cache.entry(c.to_bits()).or_insert_with(|| {
                            let id = ValueId(func.next_value);
                            func.next_value += 1;
                            consts.push((id, Instruction::ConstF64(c)));
                            id
                        }),
                    };
                }
                new_args.push(out);
            }
            let block = &mut func.blocks[bi];
            block.instructions.extend(consts);
            match &mut block.terminator {
                Terminator::Branch { args, .. } => *args = new_args.remove(0),
                Terminator::CondBranch {
                    true_args,
                    false_args,
                    ..
                } => {
                    *true_args = new_args.remove(0);
                    *false_args = new_args.remove(0);
                }
                _ => {}
            }
        }

        // Unboxes and moves of a chosen parameter are the parameter, and
        // an unbox of a box is the boxed f64.
        let mut alias: HashMap<ValueId, ValueId> = HashMap::new();
        loop {
            let mut grew = false;
            for block in &func.blocks {
                for (v, inst) in &block.instructions {
                    if alias.contains_key(v) {
                        continue;
                    }
                    if let Instruction::Unbox(a) | Instruction::Move(a) = inst {
                        let root = alias.get(a).copied().unwrap_or(*a);
                        if chosen.contains(&root) {
                            alias.insert(*v, root);
                            grew = true;
                            continue;
                        }
                    }
                    if let Instruction::Unbox(a) = inst {
                        let mut a = alias.get(a).copied().unwrap_or(*a);
                        while let Some(Instruction::Move(s)) = defs.get(&a) {
                            a = alias.get(s).copied().unwrap_or(*s);
                        }
                        if let Some(Instruction::Box(f)) = defs.get(&a) {
                            alias.insert(*v, alias.get(f).copied().unwrap_or(*f));
                            grew = true;
                        }
                    }
                }
            }
            if !grew {
                break;
            }
        }
        for block in &mut func.blocks {
            block.instructions.retain(|(v, _)| !alias.contains_key(v));
        }
        replace_uses_in_func(func, &alias);

        // Every other use reads a box planted next to the use, so a box
        // only needed on a rare path costs nothing on the others. An
        // edge into an f64 parameter passes the f64 whether this pass
        // retyped the parameter or an earlier one did.
        let chosen_params: Vec<Vec<bool>> = func
            .blocks
            .iter()
            .map(|b| b.params.iter().map(|(_, t)| *t == MirType::F64).collect())
            .collect();
        for bi in 0..func.blocks.len() {
            let mut boxed: HashMap<ValueId, ValueId> = HashMap::new();
            let mut out: Vec<(ValueId, Instruction)> = Vec::new();
            let old = std::mem::take(&mut func.blocks[bi].instructions);
            let mut next_value = func.next_value;
            let mut box_of = |p: ValueId, out: &mut Vec<(ValueId, Instruction)>| -> ValueId {
                *boxed.entry(p).or_insert_with(|| {
                    let b = ValueId(next_value);
                    next_value += 1;
                    out.push((b, Instruction::Box(p)));
                    b
                })
            };
            for (v, mut inst) in old {
                if !consumes_f64(&inst) {
                    let needs: Vec<ValueId> = inst
                        .operands()
                        .into_iter()
                        .filter(|o| chosen.contains(o))
                        .collect();
                    if !needs.is_empty() {
                        let map: HashMap<ValueId, ValueId> =
                            needs.iter().map(|p| (*p, box_of(*p, &mut out))).collect();
                        super::replace_in_inst(&mut inst, &map);
                    }
                }
                out.push((v, inst));
            }
            let mut term =
                std::mem::replace(&mut func.blocks[bi].terminator, Terminator::Unreachable);
            let mut term_needs: Vec<ValueId> = Vec::new();
            match &term {
                Terminator::Return(v) if chosen.contains(v) => term_needs.push(*v),
                Terminator::CondBranch { condition, .. } if chosen.contains(condition) => {
                    term_needs.push(*condition)
                }
                _ => {}
            }
            for (target, args) in edges(&term) {
                for (i, a) in args.iter().enumerate() {
                    let boxed_param = !chosen_params[target.0 as usize]
                        .get(i)
                        .copied()
                        .unwrap_or(false);
                    if boxed_param && chosen.contains(a) {
                        term_needs.push(*a);
                    }
                }
            }
            if !term_needs.is_empty() {
                let map: HashMap<ValueId, ValueId> = term_needs
                    .iter()
                    .map(|p| (*p, box_of(*p, &mut out)))
                    .collect();
                let targets: Vec<&[bool]> = edges(&term)
                    .into_iter()
                    .map(|(t, _)| chosen_params[t.0 as usize].as_slice())
                    .collect();
                rewrite_term_boxed(&mut term, &map, &targets);
            }
            func.next_value = next_value;
            func.blocks[bi].instructions = out;
            func.blocks[bi].terminator = term;
        }
        true
    }
}

enum Source {
    Value(ValueId),
    Const(f64),
    /// A boxed value a guard has proven a Num; the edge unboxes it.
    Guarded(ValueId),
}

/// The f64 behind a boxed edge argument, if it has one.
fn f64_source(
    v: ValueId,
    defs: &HashMap<ValueId, Instruction>,
    chosen: &HashSet<ValueId>,
) -> Option<Source> {
    let mut cur = v;
    for _ in 0..64 {
        if chosen.contains(&cur) {
            return Some(Source::Value(cur));
        }
        match defs.get(&cur)? {
            Instruction::Box(f) => return Some(Source::Value(*f)),
            Instruction::ConstNum(c) => return Some(Source::Const(*c)),
            Instruction::GuardNumAt { .. } | Instruction::GuardNum(_) => {
                return Some(Source::Guarded(cur));
            }
            Instruction::Move(a) => cur = *a,
            _ => return None,
        }
    }
    None
}

fn edges(term: &Terminator) -> Vec<(BlockId, &[ValueId])> {
    match term {
        Terminator::Branch { target, args } => vec![(*target, args.as_slice())],
        Terminator::CondBranch {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => vec![
            (*true_target, true_args.as_slice()),
            (*false_target, false_args.as_slice()),
        ],
        _ => Vec::new(),
    }
}

fn consumes_f64(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::AddF64(..)
            | Instruction::SubF64(..)
            | Instruction::MulF64(..)
            | Instruction::DivF64(..)
            | Instruction::ModF64(..)
            | Instruction::NegF64(_)
            | Instruction::CmpLtF64(..)
            | Instruction::CmpGtF64(..)
            | Instruction::CmpLeF64(..)
            | Instruction::CmpGeF64(..)
            | Instruction::MathUnaryF64(..)
            | Instruction::MathBinaryF64(..)
            | Instruction::Box(_)
    )
}

/// Box the arguments of edges into parameters that stayed boxed, and a
/// returned parameter.
fn rewrite_term_boxed(
    term: &mut Terminator,
    boxed: &HashMap<ValueId, ValueId>,
    chosen_targets: &[&[bool]],
) {
    let fix = |args: &mut Vec<ValueId>, chosen: &[bool]| {
        for (i, a) in args.iter_mut().enumerate() {
            if !chosen.get(i).copied().unwrap_or(false)
                && let Some(b) = boxed.get(a)
            {
                *a = *b;
            }
        }
    };
    match term {
        Terminator::Return(v) => {
            if let Some(b) = boxed.get(v) {
                *v = *b;
            }
        }
        Terminator::Branch { args, .. } => fix(args, chosen_targets[0]),
        Terminator::CondBranch {
            condition,
            true_args,
            false_args,
            ..
        } => {
            if let Some(b) = boxed.get(condition) {
                *condition = *b;
            }
            fix(true_args, chosen_targets[0]);
            fix(false_args, chosen_targets[1]);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;

    #[test]
    fn an_unbox_of_a_copied_box_is_the_boxed_f64() {
        let mut interner = Interner::new();
        let mut f = MirFunction::new(interner.intern("test"), 0);
        let bb = f.new_block();
        let x = f.new_value();
        let boxed = f.new_value();
        let copy = f.new_value();
        let unboxed = f.new_value();
        let sum = f.new_value();
        let out = f.new_value();
        {
            let b = f.block_mut(bb);
            b.instructions.push((x, Instruction::ConstF64(2.0)));
            b.instructions.push((boxed, Instruction::Box(x)));
            b.instructions.push((copy, Instruction::Move(boxed)));
            b.instructions.push((unboxed, Instruction::Unbox(copy)));
            b.instructions.push((sum, Instruction::AddF64(unboxed, x)));
            b.instructions.push((out, Instruction::Box(sum)));
            b.terminator = Terminator::Return(out);
        }
        UnboxParams.run(&mut f);
        let insts = &f.block(bb).instructions;
        assert!(!insts.iter().any(|(v, _)| *v == unboxed));
        assert!(
            insts
                .iter()
                .any(|(v, i)| *v == sum && matches!(i, Instruction::AddF64(a, _) if *a == x))
        );
    }
}
