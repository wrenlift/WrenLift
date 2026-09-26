/// Range loop specialization pass.
///
/// Transforms `for (i in from...to)` from generic iterate/iteratorValue calls
/// into a direct counted loop with CmpLtF64 + AddF64 — eliminating two runtime
/// calls per iteration.
///
/// Pattern matched (exclusive range, ascending):
/// ```text
///   v_range = MakeRange(from, to, false)
///   v_init  = Call { receiver: v_range, method: iterate(_), args: [null] }
///   br cond_bb [v_init]
///
///   cond_bb(iter_param):
///     is_false = Not(iter_param)
///     cond_br is_false, exit_bb, body_bb
///
///   body_bb:
///     v_elem = Call { receiver: v_range, method: iteratorValue(_), args: [iter_param] }
///     ... user code uses v_elem ...
///     v_next = Call { receiver: v_range, method: iterate(_), args: [iter_param] }
///     br cond_bb [v_next, ...]
/// ```
///
/// Transformed to:
/// ```text
///   br cond_bb [from]
///
///   cond_bb(i):
///     in_range = CmpLtF64(i, to)
///     cond_br in_range, body_bb, exit_bb   // targets swapped
///
///   body_bb:
///     ... user code uses i directly (iteratorValue was identity) ...
///     one = ConstNum(1.0)
///     next_i = AddF64(i, one)
///     br cond_bb [next_i, ...]
/// ```
///
/// A range counts down when `from > to`. Unless its bounds are constants
/// that ascend, the counted loop's exit asks: a descending range goes on
/// in a copy that counts down from where the count stopped.
use super::{MirPass, remap_inst, remap_term, replace_uses_in_func};
use crate::intern::Interner;
use crate::mir::{BlockId, Instruction, MirFunction, Terminator, ValueId};
use std::collections::{HashMap, HashSet};

pub struct RangeLoop<'a> {
    pub interner: &'a Interner,
    /// Whether a range the loop may count down, because its bounds are
    /// known only at run time or descend, is handled: a counting-down copy
    /// takes over where it descends. Without it the loop counts up.
    pub copy_descending: bool,
}

impl<'a> MirPass for RangeLoop<'a> {
    fn name(&self) -> &str {
        "range_loop"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        let iterate_sym = match self.interner.lookup("iterate(_)") {
            Some(s) => s,
            None => return false,
        };
        let iter_value_sym = match self.interner.lookup("iteratorValue(_)") {
            Some(s) => s,
            None => return false,
        };

        // Collect MakeRange instructions → (range_vid, from_vid, to_vid)
        let mut range_infos: Vec<(ValueId, ValueId, ValueId)> = Vec::new();
        for block in &func.blocks {
            for &(vid, ref inst) in &block.instructions {
                if let Instruction::MakeRange(from, to, inclusive) = inst
                    && !inclusive
                {
                    range_infos.push((vid, *from, *to));
                }
            }
        }

        let mut changed = false;
        for (range_vid, from_vid, to_vid) in range_infos {
            if self.try_optimize(
                func,
                range_vid,
                from_vid,
                to_vid,
                iterate_sym,
                iter_value_sym,
            ) {
                changed = true;
            }
        }
        changed
    }
}

impl<'a> RangeLoop<'a> {
    fn try_optimize(
        &self,
        func: &mut MirFunction,
        range_vid: ValueId,
        from_vid: ValueId,
        to_vid: ValueId,
        iterate_sym: crate::intern::SymbolId,
        iter_value_sym: crate::intern::SymbolId,
    ) -> bool {
        // Find initial iterate call: Call { receiver: range_vid, method: iterate(_), args: [null] }
        let mut init_call: Option<(usize, usize, ValueId)> = None; // (block_idx, inst_idx, result_vid)
        let mut body_iterate: Option<(usize, usize, ValueId, ValueId)> = None; // (block_idx, inst_idx, result_vid, iter_arg)
        let mut body_iter_value: Option<(usize, usize, ValueId, ValueId)> = None;
        // A loop with `continue` iterates the range from more than one
        // place; only the single-latch shape is rewritten.
        let mut iterate_sites = 0usize;
        let mut iter_value_sites = 0usize;

        for (bi, block) in func.blocks.iter().enumerate() {
            for (ii, &(vid, ref inst)) in block.instructions.iter().enumerate() {
                if let Instruction::Call {
                    receiver,
                    method,
                    args,
                    pure_call: _,
                } = inst
                {
                    if *receiver != range_vid {
                        continue;
                    }
                    if method.index() == iterate_sym.index() && args.len() == 1 {
                        // Check if arg is ConstNull (initial call)
                        let arg = args[0];
                        let is_null_arg = func.blocks.iter().any(|b| {
                            b.instructions
                                .iter()
                                .any(|&(v, ref i)| v == arg && matches!(i, Instruction::ConstNull))
                        });
                        if is_null_arg {
                            init_call = Some((bi, ii, vid));
                        } else {
                            body_iterate = Some((bi, ii, vid, arg));
                            iterate_sites += 1;
                        }
                    } else if method.index() == iter_value_sym.index() && args.len() == 1 {
                        body_iter_value = Some((bi, ii, vid, args[0]));
                        iter_value_sites += 1;
                    }
                }
            }
        }

        let (init_bi, init_ii, init_vid) = match init_call {
            Some(v) => v,
            None => return false,
        };
        let (body_bi, body_iter_ii, body_iter_vid, _iter_arg) = match body_iterate {
            Some(v) => v,
            None => return false,
        };
        let (body_val_bi, body_val_ii, iter_val_vid, iter_val_arg) = match body_iter_value {
            Some(v) => v,
            None => return false,
        };
        if iterate_sites != 1 || iter_value_sites != 1 {
            return false;
        }

        // Find cond_bb: the branch target from init_call's block
        let init_block = &func.blocks[init_bi];
        let cond_bid = match &init_block.terminator {
            Terminator::Branch { target, args } => {
                if args.first() != Some(&init_vid) {
                    return false;
                }
                *target
            }
            _ => return false,
        };

        // A preheader between the initial iterate and the header forwards
        // the iterator as its first parameter; look through it.
        let mut cond_bid = cond_bid;
        for _ in 0..8 {
            let Some(b) = func.blocks.iter().find(|b| b.id == cond_bid) else {
                return false;
            };
            match &b.terminator {
                Terminator::Branch { target, args }
                    if !b.params.is_empty() && args.first() == Some(&b.params[0].0) =>
                {
                    cond_bid = *target;
                }
                _ => break,
            }
        }
        let cond_bi = match func.blocks.iter().position(|b| b.id == cond_bid) {
            Some(v) => v,
            None => return false,
        };
        let cond_block = &func.blocks[cond_bi];

        // cond_bb should have iter_param as first block param
        if cond_block.params.is_empty() {
            return false;
        }
        let iter_param = cond_block.params[0].0;

        // Find Not(iter_param) in cond_bb
        let not_pos = match cond_block
            .instructions
            .iter()
            .position(|(_, inst)| matches!(inst, Instruction::Not(v) if *v == iter_param))
        {
            Some(v) => v,
            None => return false,
        };
        let not_vid = cond_block.instructions[not_pos].0;

        // Verify CondBranch { condition: not_vid, true: exit, false: body }
        let (exit_bid, exit_args, body_bid) = match &cond_block.terminator {
            Terminator::CondBranch {
                condition,
                true_target,
                true_args,
                false_target,
                ..
            } if *condition == not_vid => (*true_target, true_args.clone(), *false_target),
            _ => return false,
        };

        // The body's entry holds iteratorValue; the latch, which may be
        // a later block when the body branches, feeds the next iterator
        // straight back to the header.
        if body_bid != func.blocks[body_val_bi].id {
            return false;
        }
        match &func.blocks[body_bi].terminator {
            Terminator::Branch { target, args }
                if *target == cond_bid && args.first() == Some(&body_iter_vid) => {}
            _ => return false,
        }

        // Verify iter_val_arg == iter_param (iteratorValue receives the iterator)
        if iter_val_arg != iter_param {
            return false;
        }
        let shape = Counted {
            cond_bi,
            not_pos,
            not_vid,
            iter_param,
            exit_bid,
            exit_args,
            body_bid,
            body_val_bi,
            body_val_ii,
            iter_val_vid,
            body_bi,
            body_iter_ii,
            to_vid,
        };

        // Without a copy the loop counts up whatever its bounds.
        if !self.copy_descending {
            func.blocks[init_bi].instructions[init_ii].1 = Instruction::Move(from_vid);
            rewrite_counted(func, shape, Direction::Up);
            return true;
        }
        let konst = |v: ValueId| {
            func.blocks
                .iter()
                .flat_map(|b| b.instructions.iter())
                .find_map(|(id, i)| match i {
                    Instruction::ConstNum(n) if *id == v => Some(*n),
                    _ => None,
                })
        };
        match (konst(from_vid), konst(to_vid)) {
            (Some(a), Some(b)) if a <= b => {
                func.blocks[init_bi].instructions[init_ii].1 = Instruction::Move(from_vid);
                rewrite_counted(func, shape, Direction::Up);
                return true;
            }
            (Some(_), Some(_)) => return false,
            _ => {}
        }

        // Bounds known only at run time: the loop counts up in place, so
        // an entry from the interpreter still lands in it, and where the
        // count stops a descending range goes on in a copy that counts
        // down from the value reached. Both copies compute the same
        // values, so the types the specialiser proves hold after either.
        let latch = func.blocks[body_bi].id;
        let region = loop_region(func, cond_bid, latch);
        if escapes(func, &region) {
            return false;
        }
        let header_params: Vec<ValueId> = func.blocks[cond_bi].params.iter().map(|p| p.0).collect();
        let (bmap, vmap) = clone_region(func, &region);
        let m = |v: ValueId| vmap.get(&v).copied().unwrap_or(v);
        let pos = |id: BlockId| bmap[&id].0 as usize;
        let down = Counted {
            cond_bi: pos(func.blocks[shape.cond_bi].id),
            not_vid: m(shape.not_vid),
            iter_param: m(shape.iter_param),
            exit_args: shape.exit_args.iter().map(|v| m(*v)).collect(),
            body_bid: bmap[&shape.body_bid],
            body_val_bi: pos(func.blocks[shape.body_val_bi].id),
            iter_val_vid: m(shape.iter_val_vid),
            body_bi: pos(func.blocks[shape.body_bi].id),
            ..shape.clone()
        };
        let down_header = bmap[&cond_bid];
        func.blocks[init_bi].instructions[init_ii].1 = Instruction::Move(from_vid);
        let exit_args = shape.exit_args.clone();
        rewrite_counted(func, shape, Direction::Up);
        rewrite_counted(func, down, Direction::Down);
        let stop = func.new_block();
        if let Terminator::CondBranch {
            false_target,
            false_args,
            ..
        } = &mut func.blocks[cond_bi].terminator
        {
            *false_target = stop;
            false_args.clear();
        }
        let descending = func.new_value();
        let stop_block = func.block_mut(stop);
        stop_block
            .instructions
            .push((descending, Instruction::CmpGt(from_vid, to_vid)));
        stop_block.terminator = Terminator::CondBranch {
            condition: descending,
            true_target: down_header,
            true_args: header_params,
            false_target: exit_bid,
            false_args: exit_args,
        };
        func.compute_predecessors();
        true
    }
}

/// Which way a counted loop steps.
#[derive(Clone, Copy, PartialEq)]
enum Direction {
    Up,
    Down,
}

/// A matched range loop, by position, for the counted rewrite.
#[derive(Clone)]
struct Counted {
    cond_bi: usize,
    not_pos: usize,
    not_vid: ValueId,
    iter_param: ValueId,
    exit_bid: BlockId,
    exit_args: Vec<ValueId>,
    body_bid: BlockId,
    body_val_bi: usize,
    body_val_ii: usize,
    iter_val_vid: ValueId,
    body_bi: usize,
    body_iter_ii: usize,
    to_vid: ValueId,
}

/// Rewrite the loop `c` describes to count from its header's first
/// parameter to `to`, up or down, whatever the header was entered with.
fn rewrite_counted(func: &mut MirFunction, c: Counted, dir: Direction) {
    let Counted {
        cond_bi,
        not_pos,
        not_vid,
        iter_param,
        exit_bid,
        exit_args,
        body_bid,
        body_val_bi,
        body_val_ii,
        iter_val_vid,
        body_bi,
        body_iter_ii,
        to_vid,
    } = c;

    // 2. In cond_bb: replace Not(iter_param) with CmpLt(iter_param, to_vid)
    //    Use boxed CmpLt (not CmpLtF64) because iter_param is NaN-boxed.
    //    TypeSpecialize will convert to CmpLtF64 on the Optimized tier.
    func.blocks[cond_bi].instructions[not_pos].1 = match dir {
        Direction::Up => Instruction::CmpLt(iter_param, to_vid),
        Direction::Down => Instruction::CmpGt(iter_param, to_vid),
    };
    // The interpreter's own iterator holds `false` once the range is
    // exhausted; an OSR entry at this header must decline that value
    // rather than compare it.
    if !func.speculated_num_params.contains(&iter_param) {
        func.speculated_num_params.push(iter_param);
    }

    // Swap CondBranch targets (CmpLtF64 is true when we should CONTINUE, not exit)
    func.blocks[cond_bi].terminator = Terminator::CondBranch {
        condition: not_vid,
        true_target: body_bid,
        true_args: vec![],
        false_target: exit_bid,
        false_args: exit_args,
    };

    // 3. Replace iteratorValue Call with Move(iter_param)
    //    (Range.iteratorValue is identity — returns the iterator value as-is)
    func.blocks[body_val_bi].instructions[body_val_ii].1 = Instruction::Move(iter_param);

    // 4. Replace iterate Call with AddF64(iter_param, 1.0)
    //    Insert ConstNum(1.0) before the iterate call position
    let one_vid = func.new_value();
    let one_inst = (one_vid, Instruction::ConstNum(1.0));
    // Insert before body_iter_ii (which may have shifted if body_val_ii < body_iter_ii)
    // No shift needed since we only replaced (not inserted) the iteratorValue call above.
    let adjusted_iter_ii = body_iter_ii;
    func.blocks[body_bi]
        .instructions
        .insert(adjusted_iter_ii, one_inst);
    // The old iterate call is now at adjusted_iter_ii + 1
    // Use boxed Add (not AddF64) because iter_param is NaN-boxed.
    func.blocks[body_bi].instructions[adjusted_iter_ii + 1].1 = match dir {
        Direction::Up => Instruction::Add(iter_param, one_vid),
        Direction::Down => Instruction::Sub(iter_param, one_vid),
    };

    // 5. Replace all uses of iter_val_vid with iter_param
    //    (the iteratorValue result is now just iter_param)
    let mut replacements = HashMap::new();
    replacements.insert(iter_val_vid, iter_param);
    replace_uses_in_func(func, &replacements);
}

/// The blocks of the loop with this header and latch.
fn loop_region(func: &MirFunction, header: BlockId, latch: BlockId) -> Vec<BlockId> {
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for b in &func.blocks {
        for s in b.terminator.successors() {
            preds.entry(s).or_default().push(b.id);
        }
    }
    let mut seen: HashSet<BlockId> = HashSet::from([header]);
    let mut work = vec![latch];
    while let Some(b) = work.pop() {
        if seen.insert(b) {
            work.extend(preds.get(&b).into_iter().flatten().copied());
        }
    }
    let mut region: Vec<BlockId> = seen.into_iter().collect();
    region.sort_by_key(|b| b.0);
    region
}

/// Whether a value the region defines is read outside it other than
/// through an edge's arguments, which a clone could not share.
fn escapes(func: &MirFunction, region: &[BlockId]) -> bool {
    let inside: HashSet<BlockId> = region.iter().copied().collect();
    let defined: HashSet<ValueId> = region
        .iter()
        .flat_map(|b| {
            let blk = func.block(*b);
            blk.params
                .iter()
                .map(|p| p.0)
                .chain(blk.instructions.iter().map(|i| i.0))
                .collect::<Vec<_>>()
        })
        .collect();
    func.blocks
        .iter()
        .filter(|b| !inside.contains(&b.id))
        .any(|b| {
            b.instructions
                .iter()
                .any(|(_, i)| i.operands().iter().any(|v| defined.contains(v)))
                || b.terminator.operands().iter().any(|v| defined.contains(v))
        })
}

/// Copy the region's blocks under fresh block and value ids; edges that
/// leave it keep their targets.
fn clone_region(
    func: &mut MirFunction,
    region: &[BlockId],
) -> (HashMap<BlockId, BlockId>, HashMap<ValueId, ValueId>) {
    let mut bmap = HashMap::new();
    let mut vmap = HashMap::new();
    for &b in region {
        bmap.insert(b, func.new_block());
        let blk = func.block(b).clone();
        for v in blk
            .params
            .iter()
            .map(|p| p.0)
            .chain(blk.instructions.iter().map(|i| i.0))
        {
            let nv = func.new_value();
            if let Some(span) = func.span_map.get(&v).cloned() {
                func.span_map.insert(nv, span);
            }
            vmap.insert(v, nv);
        }
    }
    for &b in region {
        let blk = func.block(b).clone();
        let nb = bmap[&b];
        let params = blk.params.iter().map(|(v, t)| (vmap[v], *t)).collect();
        let instructions = blk
            .instructions
            .iter()
            .map(|(v, i)| {
                let mut i = i.clone();
                remap_inst(&mut i, &vmap);
                (vmap[v], i)
            })
            .collect();
        let mut term = blk.terminator.clone();
        remap_term(&mut term, &vmap);
        match &mut term {
            Terminator::Branch { target, .. } => {
                *target = bmap.get(target).copied().unwrap_or(*target);
            }
            Terminator::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                *true_target = bmap.get(true_target).copied().unwrap_or(*true_target);
                *false_target = bmap.get(false_target).copied().unwrap_or(*false_target);
            }
            _ => {}
        }
        let dst = func.block_mut(nb);
        dst.params = params;
        dst.instructions = instructions;
        dst.terminator = term;
    }
    (bmap, vmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_loop_name() {
        let interner = Interner::new();
        let pass = RangeLoop {
            interner: &interner,
            copy_descending: true,
        };
        assert_eq!(pass.name(), "range_loop");
    }
}
