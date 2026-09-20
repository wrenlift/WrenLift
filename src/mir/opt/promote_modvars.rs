//! Module variables a loop writes are carried around it as values.
//!
//! A module-level loop keeps its state in module variables, which the
//! loop otherwise loads and stores through the module's cell on every
//! trip. When the loop runs no code of the program's own — no call,
//! and no operator on a receiver not known to be a Num — nothing but
//! the loop can observe the variable while it runs, so each written
//! slot becomes a header parameter: the preheader reads it once, every
//! read in the loop is the current value, every write defines the
//! next, and every exit stores it back. A deopt inside the loop stores
//! it back too, through a live entry that names the slot, and a loop
//! entry from the interpreter reads the slot for the parameter.
//!
//! A slot an operator is applied to is guarded to be a Num where the
//! preheader reads it, when the engine has said where the loop's entry
//! is in the bytecode; the guard resumes the interpreter there, and
//! the loop's numbers are then known to the passes that follow.
//!
//! The reads of a slot the loop never writes leave the loop later
//! (loop-invariant code motion).

use std::collections::{HashMap, HashSet};

use super::MirPass;
use super::licm::{compute_dominators, compute_rpo, detect_loops, merge_loops_by_header};
use crate::mir::{
    BlockId, DEOPT_MODVAR_REG, DeoptReg, DeoptSource, Instruction, MirFunction, MirType,
    Terminator, ValueId, known_num_values,
};

pub struct PromoteModuleVars;

impl MirPass for PromoteModuleVars {
    fn name(&self) -> &str {
        "promote-module-vars"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        func.compute_predecessors();
        let rpo = compute_rpo(func);
        let idom = compute_dominators(func, &rpo);
        let mut loops = merge_loops_by_header(&detect_loops(func, &idom));
        // Outer loops first: a slot promoted around an outer loop is a
        // value inside its inner loops already.
        loops.sort_by_key(|l| std::cmp::Reverse(l.body.len()));
        let nums = known_num_values(func);
        let mut changed = false;
        for lp in loops {
            // A promotion adds blocks and parameters; the loops it
            // leaves are the same blocks, so the shape is recomputed
            // per loop.
            func.compute_predecessors();
            let rpo = compute_rpo(func);
            let body: HashSet<BlockId> = lp.body.iter().copied().collect();
            let guardable = func.loop_entries.contains_key(&lp.header);
            if let Some(plan) = plan(func, &body, &nums, guardable)
                && promote(func, lp.header, &body, &plan, &rpo)
            {
                changed = true;
            }
        }
        changed
    }
}

/// What a loop's promotion carries: the slots it writes, and those of
/// them the preheader guards to be Nums.
struct Plan {
    slots: Vec<u16>,
    guarded: Vec<u16>,
}

/// The plan for the loop, when it can carry its written slots: it
/// runs no code of the program's own — counting a guarded slot's read
/// as a Num — and transfers to no other body from inside.
fn plan(
    func: &MirFunction,
    body: &HashSet<BlockId>,
    nums: &HashSet<ValueId>,
    guardable: bool,
) -> Option<Plan> {
    let mut slots = Vec::new();
    let mut reads: HashMap<ValueId, u16> = HashMap::new();
    for &b in body {
        for (v, inst) in &func.block(b).instructions {
            match inst {
                Instruction::SetModuleVar(slot, _) if !slots.contains(slot) => slots.push(*slot),
                Instruction::GetModuleVar(slot) => {
                    reads.insert(*v, *slot);
                }
                Instruction::ColdLoopExit { .. } => return None,
                _ => {}
            }
        }
    }
    if slots.is_empty() {
        return None;
    }
    let is_read = |v: ValueId| reads.get(&v).is_some_and(|s| slots.contains(s));
    let mut guarded = Vec::new();
    for &b in body {
        for (_, inst) in &func.block(b).instructions {
            if !inst.may_run_code(&|v| nums.contains(&v)) {
                continue;
            }
            if !guardable || inst.may_run_code(&|v| nums.contains(&v) || is_read(v)) {
                return None;
            }
            for v in inst.operands() {
                if let Some(&slot) = reads.get(&v)
                    && slots.contains(&slot)
                    && !guarded.contains(&slot)
                {
                    guarded.push(slot);
                }
            }
        }
    }
    Some(Plan { slots, guarded })
}

/// Carry the plan's slots around the loop at `header`. Returns false
/// when the loop's shape is not one this handles: every predecessor of
/// a body block other than the header must be in the body, the header
/// must have exactly one predecessor outside it (its preheader), and a
/// guard's resume state must be expressible before the loop.
fn promote(
    func: &mut MirFunction,
    header: BlockId,
    body: &HashSet<BlockId>,
    plan: &Plan,
    rpo: &[BlockId],
) -> bool {
    let slots = &plan.slots;
    let outside: Vec<BlockId> = func
        .block(header)
        .predecessors
        .iter()
        .copied()
        .filter(|p| !body.contains(p))
        .collect();
    let [pre] = outside[..] else {
        return false;
    };
    let Terminator::Branch {
        target,
        args: pre_args,
    } = &func.block(pre).terminator
    else {
        return false;
    };
    if *target != header {
        return false;
    }
    let pre_args = pre_args.clone();
    for &b in body {
        if b != header && func.block(b).predecessors.iter().any(|p| !body.contains(p)) {
            return false;
        }
    }
    // A guard before the loop resumes the interpreter at the header
    // with the registers live there: a header parameter's is the
    // value the preheader passes for it; anything else must be
    // defined before the loop.
    let guard_at = if plan.guarded.is_empty() {
        None
    } else {
        let (pc, live) = func.loop_entries.get(&header).cloned().unwrap();
        let params: Vec<ValueId> = func.block(header).params.iter().map(|(v, _)| *v).collect();
        let defined_in_body: HashSet<ValueId> = body
            .iter()
            .flat_map(|b| {
                let blk = func.block(*b);
                blk.params
                    .iter()
                    .map(|(v, _)| *v)
                    .chain(blk.instructions.iter().map(|(v, _)| *v))
            })
            .collect();
        let mut mapped = Vec::with_capacity(live.len());
        for r in live {
            let DeoptSource::Value(v) = r.source else {
                return false;
            };
            let source = if let Some(i) = params.iter().position(|p| *p == v) {
                match pre_args.get(i) {
                    Some(a) => *a,
                    None => return false,
                }
            } else if defined_in_body.contains(&v) {
                return false;
            } else {
                v
            };
            mapped.push(DeoptReg {
                reg: r.reg,
                source: DeoptSource::Value(source),
            });
        }
        Some((pc, mapped))
    };
    let order: Vec<BlockId> = rpo.iter().copied().filter(|b| body.contains(b)).collect();

    // The preheader reads each slot; the header carries it.
    let mut header_param: HashMap<u16, ValueId> = HashMap::new();
    for &slot in slots {
        let mut read = func.new_value();
        func.block_mut(pre)
            .instructions
            .push((read, Instruction::GetModuleVar(slot)));
        if let Some((pc, live)) = &guard_at
            && plan.guarded.contains(&slot)
        {
            let guarded = func.new_value();
            func.block_mut(pre).instructions.push((
                guarded,
                Instruction::GuardNumAt {
                    value: read,
                    pc: *pc,
                    live: live.clone(),
                    call_pc: *pc,
                    call_live: Vec::new(),
                },
            ));
            read = guarded;
        }
        if let Terminator::Branch { args, .. } = &mut func.block_mut(pre).terminator {
            args.push(read);
        }
        let param = func.new_value();
        func.block_mut(header).params.push((param, MirType::Value));
        func.promoted_modvar_params.insert(param, slot);
        header_param.insert(slot, param);
    }

    // The value of each slot on entry to each body block: the header's
    // parameter, a block's own parameter where paths merge, else what
    // its one predecessor leaves.
    let mut exit: HashMap<BlockId, HashMap<u16, ValueId>> = HashMap::new();
    let mut merged: HashSet<BlockId> = HashSet::new();
    for &b in &order {
        let mut cur: HashMap<u16, ValueId> = if b == header {
            header_param.clone()
        } else if func.block(b).predecessors.len() == 1 {
            let p = func.block(b).predecessors[0];
            exit[&p].clone()
        } else {
            merged.insert(b);
            let mut params = HashMap::new();
            for &slot in slots {
                let v = func.new_value();
                func.block_mut(b).params.push((v, MirType::Value));
                // A loop entered from the interpreter here (an inner
                // loop's header) reads the slot for it as well.
                func.promoted_modvar_params.insert(v, slot);
                params.insert(slot, v);
            }
            params
        };
        let block = func.block_mut(b);
        for (_, inst) in &mut block.instructions {
            match inst {
                Instruction::GetModuleVar(slot) if header_param.contains_key(slot) => {
                    *inst = Instruction::Move(cur[slot]);
                }
                Instruction::SetModuleVar(slot, v) if header_param.contains_key(slot) => {
                    let v = *v;
                    cur.insert(*slot, v);
                    *inst = Instruction::Move(v);
                }
                Instruction::GuardClassAt { live, .. } | Instruction::SlowPathExit { live, .. } => {
                    for &slot in slots {
                        live.push(DeoptReg {
                            reg: DEOPT_MODVAR_REG | slot as u32,
                            source: DeoptSource::Value(cur[&slot]),
                        });
                    }
                }
                Instruction::GuardNumAt {
                    live, call_live, ..
                } => {
                    for &slot in slots {
                        let r = DeoptReg {
                            reg: DEOPT_MODVAR_REG | slot as u32,
                            source: DeoptSource::Value(cur[&slot]),
                        };
                        live.push(r.clone());
                        call_live.push(r);
                    }
                }
                _ => {}
            }
        }
        exit.insert(b, cur);
    }

    // Edges into the header and into merged blocks pass the values
    // along; edges out of the loop store them back on the way, each
    // through a block of its own.
    enum Edge {
        Jump,
        True,
        False,
    }
    type Leaving = (BlockId, Edge, Vec<(u16, ValueId)>);
    let mut leaving: Vec<Leaving> = Vec::new();
    for &b in &order {
        let out: Vec<(u16, ValueId)> = slots.iter().map(|s| (*s, exit[&b][s])).collect();
        let outs: Vec<ValueId> = out.iter().map(|(_, v)| *v).collect();
        match &mut func.block_mut(b).terminator {
            Terminator::Branch { target, args } => {
                if *target == header || merged.contains(target) {
                    args.extend(outs.iter().copied());
                } else if !body.contains(target) {
                    leaving.push((b, Edge::Jump, out));
                }
            }
            Terminator::CondBranch {
                true_target,
                true_args,
                false_target,
                false_args,
                ..
            } => {
                for (target, args, edge) in [
                    (true_target, true_args, Edge::True),
                    (false_target, false_args, Edge::False),
                ] {
                    if *target == header || merged.contains(target) {
                        args.extend(outs.iter().copied());
                    } else if !body.contains(target) {
                        leaving.push((b, edge, out.clone()));
                    }
                }
            }
            Terminator::Return(_) | Terminator::ReturnNull => {
                for (slot, v) in &out {
                    let d = func.new_value();
                    func.block_mut(b)
                        .instructions
                        .push((d, Instruction::SetModuleVar(*slot, *v)));
                }
            }
            Terminator::Unreachable => {}
        }
    }
    for (from, edge, out) in leaving {
        let e = func.new_block();
        for (slot, v) in out {
            let d = func.new_value();
            func.block_mut(e)
                .instructions
                .push((d, Instruction::SetModuleVar(slot, v)));
        }
        let (target, args) = match (&mut func.block_mut(from).terminator, &edge) {
            (Terminator::Branch { target, args }, Edge::Jump)
            | (
                Terminator::CondBranch {
                    true_target: target,
                    true_args: args,
                    ..
                },
                Edge::True,
            )
            | (
                Terminator::CondBranch {
                    false_target: target,
                    false_args: args,
                    ..
                },
                Edge::False,
            ) => {
                let t = *target;
                *target = e;
                (t, std::mem::take(args))
            }
            _ => unreachable!(),
        };
        func.block_mut(e).terminator = Terminator::Branch { target, args };
    }
    func.compute_predecessors();
    true
}
