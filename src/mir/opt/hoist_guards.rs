//! Hoist a class guard on a loop-invariant receiver out of the loop.
//!
//! An accessor inlined in place leaves `GuardClassAt` on its receiver at
//! the site, inside the loop the site runs in. When the receiver is
//! defined outside the loop its class is the same on every iteration,
//! so one check before the loop is enough — but a guard resumes the
//! interpreter at a bytecode offset with the registers live there, and
//! that pair is only known at points the compile clone already marked:
//! a `SlowPathExit`, `GuardNumAt` or `GuardClassAt` that dominates the
//! loop and comes after the receiver's definition. The hoisted guard is
//! planted right after such an anchor with the anchor's offset and
//! registers, which is a valid resume point since nothing runs between
//! the two; the guards in the loop become copies of the receiver. The
//! receiver is followed through copies and through loop parameters
//! every back edge passes unchanged to the value the loop was entered
//! with.

use std::collections::{HashMap, HashSet};

use super::MirPass;
use super::licm::{
    compute_dominators, compute_rpo, detect_loops, dominates, merge_loops_by_header,
};
use crate::mir::{BlockId, DeoptReg, Instruction, MirFunction, ValueId};

pub struct HoistGuards;

impl MirPass for HoistGuards {
    fn name(&self) -> &str {
        "hoist-guards"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        if func.blocks.is_empty() {
            return false;
        }
        func.compute_predecessors();
        let rpo = compute_rpo(func);
        let idom = compute_dominators(func, &rpo);
        let loops = merge_loops_by_header(&detect_loops(func, &idom));
        if loops.is_empty() {
            return false;
        }
        let world = World::new(func, &loops);
        // (block, position after which the guard goes, guard)
        let mut plants: Vec<(BlockId, usize, Instruction)> = Vec::new();
        let mut hoisted: HashMap<(ValueId, usize), BlockId> = HashMap::new();
        for lp in &loops {
            let body: HashSet<BlockId> = lp.body.iter().copied().collect();
            for &bid in &lp.body {
                for (_, inst) in &func.block(bid).instructions {
                    let Instruction::GuardClassAt { value, class, .. } = inst else {
                        continue;
                    };
                    let Some(outer) = world.entered_with(func, *value) else {
                        continue;
                    };
                    let Some(&def) = world.def_block.get(&outer) else {
                        continue;
                    };
                    if body.contains(&def) || hoisted.contains_key(&(outer, *class)) {
                        continue;
                    }
                    let Some((ablock, at, pc, live)) = anchor(func, &idom, lp.header, outer, def)
                    else {
                        continue;
                    };
                    plants.push((
                        ablock,
                        at,
                        Instruction::GuardClassAt {
                            value: outer,
                            class: *class,
                            pc,
                            live,
                        },
                    ));
                    hoisted.insert((outer, *class), ablock);
                }
            }
        }
        if plants.is_empty() {
            return false;
        }
        // The guards a hoisted one covers become copies: every guard on
        // a value entered with the same one, in a block the hoisted
        // guard dominates.
        for bi in 0..func.blocks.len() {
            let bid = func.blocks[bi].id;
            for k in 0..func.blocks[bi].instructions.len() {
                let Instruction::GuardClassAt { value, class, .. } =
                    &func.blocks[bi].instructions[k].1
                else {
                    continue;
                };
                let (value, class) = (*value, *class);
                let Some(outer) = world.entered_with(func, value) else {
                    continue;
                };
                let Some(&pb) = hoisted.get(&(outer, class)) else {
                    continue;
                };
                if pb != bid && dominates(&idom, pb.0 as usize, bid.0 as usize) {
                    func.blocks[bi].instructions[k].1 = Instruction::Move(value);
                }
            }
        }
        // Later positions first so earlier ones stay valid.
        plants.sort_by_key(|p| std::cmp::Reverse((p.0, p.1)));
        for (bid, at, guard) in plants {
            let v = func.new_value();
            func.block_mut(bid).instructions.insert(at + 1, (v, guard));
        }
        true
    }
}

/// Every value mapped to the value it stands for outside the loops it
/// is carried around, for a backend's class facts; a value that
/// resolves to nothing maps to itself.
pub fn value_roots(func: &mut MirFunction) -> HashMap<ValueId, ValueId> {
    func.compute_predecessors();
    let rpo = compute_rpo(func);
    let idom = compute_dominators(func, &rpo);
    let loops = merge_loops_by_header(&detect_loops(func, &idom));
    let world = World::new(func, &loops);
    world
        .def_block
        .keys()
        .map(|v| (*v, world.entered_with(func, *v).unwrap_or(*v)))
        .collect()
}

/// Where values are defined and which loop each header parameter is
/// carried around unchanged.
struct World {
    def_block: HashMap<ValueId, BlockId>,
    /// Header parameter → the value the loop was entered with, for a
    /// parameter every back edge passes itself.
    carried: HashMap<ValueId, ValueId>,
    /// Parameter of a block with one predecessor → the argument.
    passed: HashMap<ValueId, ValueId>,
    moves: HashMap<ValueId, ValueId>,
}

impl World {
    fn new(func: &MirFunction, loops: &[super::licm::Loop]) -> Self {
        let def_block: HashMap<ValueId, BlockId> = func
            .blocks
            .iter()
            .flat_map(|b| {
                b.params
                    .iter()
                    .map(move |(p, _)| (*p, b.id))
                    .chain(b.instructions.iter().map(move |(v, _)| (*v, b.id)))
            })
            .collect();
        let moves: HashMap<ValueId, ValueId> = func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|(v, i)| match i {
                Instruction::Move(s) => Some((*v, *s)),
                _ => None,
            })
            .collect();
        let mut carried = HashMap::new();
        let mut passed = HashMap::new();
        let headers: HashMap<BlockId, HashSet<BlockId>> = loops
            .iter()
            .map(|l| (l.header, l.body.iter().copied().collect()))
            .collect();
        for block in &func.blocks {
            let preds = &block.predecessors;
            for (i, (p, _)) in block.params.iter().enumerate() {
                let args: Vec<(BlockId, ValueId)> = preds
                    .iter()
                    .flat_map(|pred| {
                        edge_args(func, *pred, block.id)
                            .into_iter()
                            .filter_map(move |args| args.get(i).map(|a| (*pred, *a)))
                    })
                    .collect();
                if let Some(body) = headers.get(&block.id) {
                    let mut entry = None;
                    let mut ok = true;
                    for (pred, a) in &args {
                        if body.contains(pred) {
                            if a != p {
                                ok = false;
                            }
                        } else if entry.is_none_or(|e| e == *a) {
                            entry = Some(*a);
                        } else {
                            ok = false;
                        }
                    }
                    if ok && let Some(e) = entry {
                        carried.insert(*p, e);
                    }
                } else if preds.len() == 1 && args.len() == 1 {
                    passed.insert(*p, args[0].1);
                }
            }
        }
        Self {
            def_block,
            carried,
            passed,
            moves,
        }
    }

    /// The value `v` stands for outside every loop it is carried
    /// around: copies, pass-through parameters and unchanged loop
    /// parameters are followed to a defined value.
    fn entered_with(&self, func: &MirFunction, v: ValueId) -> Option<ValueId> {
        let mut v = v;
        for _ in 0..64 {
            if let Some(s) = self.moves.get(&v) {
                v = *s;
            } else if let Some(e) = self.carried.get(&v) {
                v = *e;
            } else if let Some(a) = self.passed.get(&v) {
                v = *a;
            } else {
                let b = self.def_block.get(&v)?;
                let is_param = func.block(*b).params.iter().any(|(p, _)| *p == v);
                return if is_param { None } else { Some(v) };
            }
        }
        None
    }
}

/// The arguments `pred` passes to `target`, one list per edge.
fn edge_args(func: &MirFunction, pred: BlockId, target: BlockId) -> Vec<&[ValueId]> {
    use crate::mir::Terminator;
    match &func.block(pred).terminator {
        Terminator::Branch { target: t, args } if *t == target => vec![args.as_slice()],
        Terminator::CondBranch {
            true_target,
            true_args,
            false_target,
            false_args,
            ..
        } => {
            let mut out = Vec::new();
            if *true_target == target {
                out.push(true_args.as_slice());
            }
            if *false_target == target {
                out.push(false_args.as_slice());
            }
            out
        }
        _ => Vec::new(),
    }
}

/// The nearest resume point that dominates `header`, lies outside the
/// loop, and follows the definition of `value`: its block, the index of
/// the anchor instruction, and the offset and registers it resumes with.
fn anchor(
    func: &MirFunction,
    idom: &[usize],
    header: BlockId,
    value: ValueId,
    def: BlockId,
) -> Option<(BlockId, usize, u32, Vec<DeoptReg>)> {
    // Walk the dominator chain up from the header's immediate
    // dominator; the header itself is in the loop.
    let mut b = idom.get(header.0 as usize).copied()?;
    if b == usize::MAX {
        return None;
    }
    loop {
        let block = func.block(BlockId(b as u32));
        let def_at = if block.id == def {
            block.instructions.iter().position(|(v, _)| *v == value)
        } else {
            None
        };
        for (k, (_, inst)) in block.instructions.iter().enumerate().rev() {
            if def_at.is_some_and(|d| k <= d) {
                break;
            }
            let point = match inst {
                // A slow-path exit resumes at the instruction before it,
                // which then runs again: only after one without effects.
                Instruction::SlowPathExit { pc, live }
                    if k > 0 && !block.instructions[k - 1].1.has_side_effects() =>
                {
                    Some((*pc, live))
                }
                Instruction::GuardNumAt { pc, live, .. } => Some((*pc, live)),
                Instruction::GuardClassAt { pc, live, .. } => Some((*pc, live)),
                _ => None,
            };
            if let Some((pc, live)) = point {
                return Some((block.id, k, pc, live.clone()));
            }
        }
        if block.id == def {
            return None;
        }
        let up = idom.get(b).copied().unwrap_or(usize::MAX);
        if up == usize::MAX || up == b {
            return None;
        }
        b = up;
    }
}
