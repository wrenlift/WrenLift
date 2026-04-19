/// Loop-Invariant Code Motion (LICM).
///
/// Hoists instructions out of loops when all their operands are defined
/// outside the loop (or are themselves loop-invariant) and the instruction
/// has no side effects. Moved instructions are placed in a preheader block
/// inserted before the loop header.
use std::collections::{HashMap, HashSet, VecDeque};

use super::MirPass;
use crate::mir::{BlockId, Instruction, MirFunction, Terminator, ValueId};

pub struct Licm;

impl MirPass for Licm {
    fn name(&self) -> &str {
        "licm"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        func.compute_predecessors();
        let rpo = compute_rpo(func);
        let idom = compute_dominators(func, &rpo);
        let loops = detect_loops(func, &idom);

        if loops.is_empty() {
            return false;
        }

        // Merge loops that share a header. Multiple back edges to the same
        // header (e.g. a `while` body with multiple `continue` paths) all
        // represent the same loop — processing them independently creates
        // orphaned preheader chains that hoist code into unreachable blocks.
        let merged = merge_loops_by_header(&loops);

        // Collect which block defines each value.
        let def_block = build_def_map(func);

        let mut changed = false;
        for lp in &merged {
            let body_set: HashSet<BlockId> = lp.body.iter().copied().collect();

            // Find loop-invariant instructions (fixpoint).
            let invariants = find_invariants(func, &body_set, &def_block);
            if invariants.is_empty() {
                continue;
            }

            // Insert a preheader block before the loop header.
            let preheader = insert_preheader(func, lp.header, &body_set);

            // Move invariant instructions to the preheader.
            if hoist_to_preheader(func, &invariants, &body_set, preheader) {
                changed = true;
            }
            // Refresh predecessor info so subsequent header processing sees
            // the redirected edges from this loop's preheader insertion.
            func.compute_predecessors();
        }

        if changed {
            func.compute_predecessors();
        }
        changed
    }
}

// ---------------------------------------------------------------------------
// Loop detection at MIR level
// ---------------------------------------------------------------------------

struct Loop {
    header: BlockId,
    #[allow(dead_code)]
    latch: BlockId,
    body: Vec<BlockId>,
}

/// Reverse post-order traversal of MIR blocks.
fn compute_rpo(func: &MirFunction) -> Vec<BlockId> {
    let n = func.blocks.len();
    let mut visited = vec![false; n];
    let mut post_order = Vec::with_capacity(n);

    fn dfs(
        func: &MirFunction,
        bid: BlockId,
        visited: &mut Vec<bool>,
        post_order: &mut Vec<BlockId>,
    ) {
        let idx = bid.0 as usize;
        if idx >= visited.len() || visited[idx] {
            return;
        }
        visited[idx] = true;
        for succ in func.block(bid).terminator.successors() {
            dfs(func, succ, visited, post_order);
        }
        post_order.push(bid);
    }

    dfs(func, func.entry_block(), &mut visited, &mut post_order);
    post_order.reverse();
    post_order
}

/// Cooper-Harvey-Kennedy iterative dominator algorithm for MIR blocks.
fn compute_dominators(func: &MirFunction, rpo: &[BlockId]) -> Vec<usize> {
    let n = func.blocks.len();
    let undef = usize::MAX;
    let mut idom = vec![undef; n];

    // Map BlockId -> RPO index for intersect.
    let mut rpo_index = vec![undef; n];
    for (i, &bid) in rpo.iter().enumerate() {
        rpo_index[bid.0 as usize] = i;
    }

    // Entry dominates itself.
    let entry = func.entry_block().0 as usize;
    idom[entry] = entry;

    let intersect = |mut a: usize, mut b: usize, idom: &[usize], rpo_idx: &[usize]| -> usize {
        while a != b {
            while rpo_idx[a] > rpo_idx[b] {
                a = idom[a];
            }
            while rpo_idx[b] > rpo_idx[a] {
                b = idom[b];
            }
        }
        a
    };

    let mut changed_any = true;
    while changed_any {
        changed_any = false;
        for &bid in rpo.iter().skip(1) {
            let b = bid.0 as usize;
            let preds = &func.block(bid).predecessors;

            // Find first processed predecessor.
            let mut new_idom = undef;
            for &pred in preds {
                let p = pred.0 as usize;
                if idom[p] != undef {
                    new_idom = p;
                    break;
                }
            }
            if new_idom == undef {
                continue;
            }

            // Intersect with remaining processed predecessors.
            for &pred in preds {
                let p = pred.0 as usize;
                if p != new_idom && idom[p] != undef {
                    new_idom = intersect(new_idom, p, &idom, &rpo_index);
                }
            }

            if idom[b] != new_idom {
                idom[b] = new_idom;
                changed_any = true;
            }
        }
    }

    idom
}

/// Check if block `a` dominates block `b`.
fn dominates(idom: &[usize], a: usize, b: usize) -> bool {
    let mut cur = b;
    loop {
        if cur == a {
            return true;
        }
        if idom[cur] == cur {
            return false; // reached entry without finding a
        }
        cur = idom[cur];
    }
}

/// Collapse loops that share a header into one. The merged body is the
/// union of each individual loop body; the latch is left as one of the
/// original latches (not used downstream — LICM only reads the header and
/// body).
fn merge_loops_by_header(loops: &[Loop]) -> Vec<Loop> {
    let mut by_header: HashMap<BlockId, (BlockId, HashSet<BlockId>)> = HashMap::new();
    for lp in loops {
        let entry = by_header
            .entry(lp.header)
            .or_insert_with(|| (lp.latch, HashSet::new()));
        for &b in &lp.body {
            entry.1.insert(b);
        }
    }
    let mut out: Vec<Loop> = by_header
        .into_iter()
        .map(|(header, (latch, body_set))| {
            let mut body: Vec<BlockId> = body_set.into_iter().collect();
            body.sort();
            Loop {
                header,
                latch,
                body,
            }
        })
        .collect();
    // Sort for deterministic processing order.
    out.sort_by_key(|lp| lp.header.0);
    out
}

/// Detect natural loops via back edges.
fn detect_loops(func: &MirFunction, idom: &[usize]) -> Vec<Loop> {
    let mut loops = Vec::new();

    for block in &func.blocks {
        let src = block.id.0 as usize;
        for succ in block.terminator.successors() {
            let target = succ.0 as usize;
            if dominates(idom, target, src) {
                // Back edge: src -> target. target is loop header.
                let body = compute_loop_body(func, succ, block.id);
                loops.push(Loop {
                    header: succ,
                    latch: block.id,
                    body,
                });
            }
        }
    }

    loops
}

/// Compute loop body: all blocks that can reach the latch without going
/// through the header.
fn compute_loop_body(func: &MirFunction, header: BlockId, latch: BlockId) -> Vec<BlockId> {
    let mut body: HashSet<BlockId> = HashSet::new();
    body.insert(header);
    body.insert(latch);

    let mut worklist = VecDeque::new();
    if latch != header {
        worklist.push_back(latch);
    }

    while let Some(bid) = worklist.pop_front() {
        for &pred in &func.block(bid).predecessors {
            if body.insert(pred) {
                worklist.push_back(pred);
            }
        }
    }

    let mut body_vec: Vec<BlockId> = body.into_iter().collect();
    body_vec.sort();
    body_vec
}

// ---------------------------------------------------------------------------
// Invariant analysis
// ---------------------------------------------------------------------------

/// Build a map from ValueId to the BlockId where it is defined.
fn build_def_map(func: &MirFunction) -> HashMap<ValueId, BlockId> {
    let mut map = HashMap::new();
    for block in &func.blocks {
        for &(vid, _) in &block.params {
            map.insert(vid, block.id);
        }
        for &(vid, _) in &block.instructions {
            map.insert(vid, block.id);
        }
    }
    map
}

/// Find all loop-invariant instructions via fixpoint iteration.
///
/// An instruction is loop-invariant if:
/// 1. It has no side effects
/// 2. It is not a BlockParam (block params receive values from predecessors)
/// 3. All its operands are either defined outside the loop or are themselves
///    loop-invariant
fn find_invariants(
    func: &MirFunction,
    body: &HashSet<BlockId>,
    def_block: &HashMap<ValueId, BlockId>,
) -> HashSet<ValueId> {
    let mut invariants: HashSet<ValueId> = HashSet::new();

    let is_outside = |vid: &ValueId| -> bool {
        match def_block.get(vid) {
            Some(bid) => !body.contains(bid),
            None => true, // unknown (e.g. function params) = outside
        }
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &bid in body {
            let block = func.block(bid);
            for &(vid, ref inst) in &block.instructions {
                if invariants.contains(&vid) {
                    continue;
                }

                // Skip side-effectful and block params.
                if inst.has_side_effects() {
                    continue;
                }
                if matches!(inst, Instruction::BlockParam(_)) {
                    continue;
                }
                // Skip guard instructions (they may deopt and depend on
                // being inside the loop for correctness).
                if matches!(
                    inst,
                    Instruction::GuardNum(_)
                        | Instruction::GuardBool(_)
                        | Instruction::GuardClass(_, _)
                ) {
                    continue;
                }
                // Skip module var reads (may change between iterations).
                if matches!(
                    inst,
                    Instruction::GetModuleVar(_) | Instruction::GetUpvalue(_)
                ) {
                    continue;
                }

                let ops = inst.operands();
                let all_invariant = ops
                    .iter()
                    .all(|op| is_outside(op) || invariants.contains(op));

                if all_invariant {
                    invariants.insert(vid);
                    changed = true;
                }
            }
        }
    }

    invariants
}

// ---------------------------------------------------------------------------
// Preheader insertion and hoisting
// ---------------------------------------------------------------------------

/// Insert a preheader block that jumps unconditionally to the loop header.
/// Redirects all non-loop predecessors of the header to the preheader.
fn insert_preheader(func: &mut MirFunction, header: BlockId, body: &HashSet<BlockId>) -> BlockId {
    let preheader = func.new_block();

    // The preheader jumps unconditionally to the header, forwarding any block
    // params the header expects.
    let header_params: Vec<(ValueId, _)> = func.block(header).params.clone();
    let mut preheader_params = Vec::new();
    let mut forward_args = Vec::new();
    for &(_, ty) in &header_params {
        let pv = func.new_value();
        preheader_params.push((pv, ty));
        forward_args.push(pv);
    }

    {
        let ph = func.block_mut(preheader);
        ph.params = preheader_params;
        ph.terminator = Terminator::Branch {
            target: header,
            args: forward_args,
        };
    }

    // Redirect non-loop predecessors of header to the preheader.
    let preds: Vec<BlockId> = func.block(header).predecessors.clone();
    for pred in preds {
        if body.contains(&pred) {
            continue; // keep back edges pointing at header
        }
        redirect_terminator(func, pred, header, preheader);
    }

    preheader
}

/// Rewrite all edges from `src` that target `old_target` to point at `new_target`.
fn redirect_terminator(
    func: &mut MirFunction,
    src: BlockId,
    old_target: BlockId,
    new_target: BlockId,
) {
    let term = &mut func.block_mut(src).terminator;
    match term {
        Terminator::Branch { target, .. } if *target == old_target => {
            *target = new_target;
        }
        Terminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            if *true_target == old_target {
                *true_target = new_target;
            }
            if *false_target == old_target {
                *false_target = new_target;
            }
        }
        _ => {}
    }
}

/// Move invariant instructions from loop body blocks into the preheader.
/// Instructions are inserted before the preheader's terminator.
/// Returns true if any instructions were moved.
fn hoist_to_preheader(
    func: &mut MirFunction,
    invariants: &HashSet<ValueId>,
    body: &HashSet<BlockId>,
    preheader: BlockId,
) -> bool {
    if invariants.is_empty() {
        return false;
    }

    // Collect instructions to hoist, preserving dependency order.
    // Walk blocks in sorted order and instructions in program order.
    let mut hoisted: Vec<(ValueId, Instruction)> = Vec::new();
    let mut body_sorted: Vec<BlockId> = body.iter().copied().collect();
    body_sorted.sort();

    for &bid in &body_sorted {
        let block = func.block(bid);
        for &(vid, ref inst) in &block.instructions {
            if invariants.contains(&vid) {
                hoisted.push((vid, inst.clone()));
            }
        }
    }

    if hoisted.is_empty() {
        return false;
    }

    // Remove hoisted instructions from their original blocks.
    for &bid in &body_sorted {
        let block = func.block_mut(bid);
        block
            .instructions
            .retain(|&(vid, _)| !invariants.contains(&vid));
    }

    // Insert into preheader (before terminator, which is already set).
    let ph = func.block_mut(preheader);
    for item in hoisted {
        ph.instructions.push(item);
    }

    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::interp::eval;
    use crate::mir::opt::MirPass;
    use crate::mir::{Instruction, MirFunction, MirType, Terminator};

    fn make_func(interner: &mut Interner) -> MirFunction {
        let name = interner.intern("test");
        MirFunction::new(name, 0)
    }

    /// Build a simple loop: sum = 0; i = 0; while (i < 5) { sum += invariant; i += 1 }
    /// where invariant = 10 + 20 (computable outside the loop).
    fn make_loop_with_invariant(interner: &mut Interner) -> (MirFunction, ValueId) {
        let mut f = make_func(interner);

        // bb0 (entry): compute invariant, jump to header
        let bb0 = f.new_block();
        let v_ten = f.new_value();
        let v_twenty = f.new_value();
        let v_invariant = f.new_value(); // 10 + 20, defined inside loop but invariant
        let v_zero = f.new_value();
        let v_five = f.new_value();
        let v_one = f.new_value();

        // bb1 (header): i, sum as block params; check i < 5
        let bb1 = f.new_block();
        let v_i = f.new_value(); // block param
        let v_sum = f.new_value(); // block param
        let v_cond = f.new_value();

        // bb2 (body): sum += invariant; i += 1; jump back to bb1
        let bb2 = f.new_block();
        let v_new_sum = f.new_value();
        let v_new_i = f.new_value();

        // bb3 (exit): return sum
        let bb3 = f.new_block();
        let v_exit_sum = f.new_value(); // block param

        // -- bb0: entry --
        {
            let b = f.block_mut(bb0);
            b.instructions.push((v_ten, Instruction::ConstNum(10.0)));
            b.instructions.push((v_twenty, Instruction::ConstNum(20.0)));
            b.instructions.push((v_zero, Instruction::ConstNum(0.0)));
            b.instructions.push((v_five, Instruction::ConstNum(5.0)));
            b.instructions.push((v_one, Instruction::ConstNum(1.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_zero, v_zero], // i=0, sum=0
            };
        }

        // -- bb1: loop header --
        {
            let b = f.block_mut(bb1);
            b.params = vec![(v_i, MirType::Value), (v_sum, MirType::Value)];
            b.instructions
                .push((v_cond, Instruction::CmpLt(v_i, v_five)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb2,
                true_args: vec![],
                false_target: bb3,
                false_args: vec![v_sum],
            };
        }

        // -- bb2: loop body (invariant computation is here) --
        {
            let b = f.block_mut(bb2);
            b.instructions
                .push((v_invariant, Instruction::Add(v_ten, v_twenty)));
            b.instructions
                .push((v_new_sum, Instruction::Add(v_sum, v_invariant)));
            b.instructions.push((v_new_i, Instruction::Add(v_i, v_one)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_new_i, v_new_sum],
            };
        }

        // -- bb3: exit --
        {
            let b = f.block_mut(bb3);
            b.params = vec![(v_exit_sum, MirType::Value)];
            b.terminator = Terminator::Return(v_exit_sum);
        }

        (f, v_invariant)
    }

    #[test]
    fn test_licm_hoists_invariant() {
        let mut interner = Interner::new();
        let (mut f, v_invariant) = make_loop_with_invariant(&mut interner);

        let before = eval(&f).unwrap();

        let licm = Licm;
        let changed = licm.run(&mut f);
        assert!(changed);

        let after = eval(&f).unwrap();
        assert_eq!(before, after, "LICM must preserve semantics");

        // The invariant instruction should no longer be in bb2 (the loop body).
        let bb2 = BlockId(2);
        let body_vals: Vec<ValueId> = f.block(bb2).instructions.iter().map(|&(v, _)| v).collect();
        assert!(
            !body_vals.contains(&v_invariant),
            "invariant should have been hoisted out of loop body"
        );
    }

    #[test]
    fn test_licm_preserves_side_effects() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);

        // Loop with a side-effecting call inside -- should NOT be hoisted.
        let bb0 = f.new_block();
        let v_zero = f.new_value();
        let v_five = f.new_value();
        let v_one = f.new_value();

        let bb1 = f.new_block();
        let v_i = f.new_value();
        let v_cond = f.new_value();

        let bb2 = f.new_block();
        let v_call = f.new_value();
        let v_new_i = f.new_value();

        let bb3 = f.new_block();

        // bb0
        {
            let b = f.block_mut(bb0);
            b.instructions.push((v_zero, Instruction::ConstNum(0.0)));
            b.instructions.push((v_five, Instruction::ConstNum(5.0)));
            b.instructions.push((v_one, Instruction::ConstNum(1.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_zero],
            };
        }

        // bb1: header
        {
            let b = f.block_mut(bb1);
            b.params = vec![(v_i, MirType::Value)];
            b.instructions
                .push((v_cond, Instruction::CmpLt(v_i, v_five)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb2,
                true_args: vec![],
                false_target: bb3,
                false_args: vec![],
            };
        }

        // bb2: body with call (side effect)
        {
            let method = interner.intern("print");
            let b = f.block_mut(bb2);
            b.instructions.push((
                v_call,
                Instruction::Call {
                    receiver: v_i,
                    method,
                    args: vec![],
                },
            ));
            b.instructions.push((v_new_i, Instruction::Add(v_i, v_one)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_new_i],
            };
        }

        // bb3: exit
        {
            f.block_mut(bb3).terminator = Terminator::ReturnNull;
        }

        let licm = Licm;
        let changed = licm.run(&mut f);
        // The call is side-effectful, should not be hoisted.
        // The Add(v_i, v_one) depends on v_i (loop-variant), should not be hoisted.
        assert!(!changed);
    }

    #[test]
    fn test_licm_no_loop_no_change() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);
        let bb = f.new_block();
        let v = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v, Instruction::ConstNum(42.0)));
        f.block_mut(bb).terminator = Terminator::Return(v);

        let licm = Licm;
        assert!(!licm.run(&mut f));
    }

    #[test]
    fn test_licm_chain_invariants() {
        // Two chained invariant computations: a = 2+3, b = a*4
        // Both should be hoisted.
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);

        let bb0 = f.new_block();
        let v_two = f.new_value();
        let v_three = f.new_value();
        let v_four = f.new_value();
        let v_zero = f.new_value();
        let v_limit = f.new_value();
        let v_one = f.new_value();

        let bb1 = f.new_block();
        let v_i = f.new_value();
        let v_acc = f.new_value();
        let v_cond = f.new_value();

        let bb2 = f.new_block();
        let v_a = f.new_value(); // 2 + 3 (invariant)
        let v_b = f.new_value(); // a * 4 (invariant, depends on v_a)
        let v_new_acc = f.new_value();
        let v_new_i = f.new_value();

        let bb3 = f.new_block();
        let v_result = f.new_value();

        // bb0
        {
            let b = f.block_mut(bb0);
            b.instructions.push((v_two, Instruction::ConstNum(2.0)));
            b.instructions.push((v_three, Instruction::ConstNum(3.0)));
            b.instructions.push((v_four, Instruction::ConstNum(4.0)));
            b.instructions.push((v_zero, Instruction::ConstNum(0.0)));
            b.instructions.push((v_limit, Instruction::ConstNum(3.0)));
            b.instructions.push((v_one, Instruction::ConstNum(1.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_zero, v_zero],
            };
        }

        // bb1
        {
            let b = f.block_mut(bb1);
            b.params = vec![(v_i, MirType::Value), (v_acc, MirType::Value)];
            b.instructions
                .push((v_cond, Instruction::CmpLt(v_i, v_limit)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb2,
                true_args: vec![],
                false_target: bb3,
                false_args: vec![v_acc],
            };
        }

        // bb2: both v_a and v_b are invariant
        {
            let b = f.block_mut(bb2);
            b.instructions.push((v_a, Instruction::Add(v_two, v_three)));
            b.instructions.push((v_b, Instruction::Mul(v_a, v_four)));
            b.instructions
                .push((v_new_acc, Instruction::Add(v_acc, v_b)));
            b.instructions.push((v_new_i, Instruction::Add(v_i, v_one)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_new_i, v_new_acc],
            };
        }

        // bb3
        {
            let b = f.block_mut(bb3);
            b.params = vec![(v_result, MirType::Value)];
            b.terminator = Terminator::Return(v_result);
        }

        let before = eval(&f).unwrap();
        let licm = Licm;
        let changed = licm.run(&mut f);
        assert!(changed);

        let after = eval(&f).unwrap();
        assert_eq!(before, after);

        // Both v_a and v_b should be out of bb2 now.
        let body_vals: Vec<ValueId> = f
            .block(BlockId(2))
            .instructions
            .iter()
            .map(|&(v, _)| v)
            .collect();
        assert!(!body_vals.contains(&v_a));
        assert!(!body_vals.contains(&v_b));
    }

    #[test]
    fn test_licm_does_not_hoist_loop_variant() {
        // i + 1 depends on i (block param), should NOT be hoisted.
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);

        let bb0 = f.new_block();
        let v_zero = f.new_value();
        let v_five = f.new_value();
        let v_one = f.new_value();

        let bb1 = f.new_block();
        let v_i = f.new_value();
        let v_cond = f.new_value();

        let bb2 = f.new_block();
        let v_new_i = f.new_value();

        let bb3 = f.new_block();

        // bb0
        {
            let b = f.block_mut(bb0);
            b.instructions.push((v_zero, Instruction::ConstNum(0.0)));
            b.instructions.push((v_five, Instruction::ConstNum(5.0)));
            b.instructions.push((v_one, Instruction::ConstNum(1.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_zero],
            };
        }

        // bb1
        {
            let b = f.block_mut(bb1);
            b.params = vec![(v_i, MirType::Value)];
            b.instructions
                .push((v_cond, Instruction::CmpLt(v_i, v_five)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb2,
                true_args: vec![],
                false_target: bb3,
                false_args: vec![],
            };
        }

        // bb2
        {
            let b = f.block_mut(bb2);
            b.instructions.push((v_new_i, Instruction::Add(v_i, v_one)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_new_i],
            };
        }

        // bb3
        {
            f.block_mut(bb3).terminator = Terminator::ReturnNull;
        }

        let licm = Licm;
        let changed = licm.run(&mut f);
        assert!(!changed, "loop-variant instructions should not be hoisted");
    }

    /// Regression: a loop body with multiple back edges (e.g. `while`
    /// with `continue` branches) produces several natural loops that all
    /// share the same header. LICM must treat them as one loop — processing
    /// each back edge independently used to leave invariant instructions in
    /// orphaned preheader blocks, breaking all uses of the hoisted value.
    #[test]
    fn test_licm_multiple_back_edges_same_header() {
        // CFG:
        //   bb0 -> bb1 (header, param i)
        //   bb1 -> bb2 (body) | bb3 (exit)
        //   bb2 -> bb4 (continue A) | bb5 (continue B)
        //   bb4: i = i + 1; jump bb1          <- back edge 1
        //   bb5 -> bb6: i = i + 1; jump bb1   <- back edge 2
        //   bb3: return i
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);

        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();
        let bb4 = f.new_block();
        let bb5 = f.new_block();
        let bb6 = f.new_block();

        let v_zero = f.new_value();
        let v_limit = f.new_value();
        let v_i = f.new_value();
        let v_cond = f.new_value();
        let v_branch_cond = f.new_value();
        let v_one_a = f.new_value();
        let v_new_i_a = f.new_value();
        let v_one_b = f.new_value();
        let v_new_i_b = f.new_value();
        let v_exit_i = f.new_value();

        {
            let b = f.block_mut(bb0);
            b.instructions.push((v_zero, Instruction::ConstNum(0.0)));
            b.instructions.push((v_limit, Instruction::ConstNum(3.0)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_zero],
            };
        }
        {
            let b = f.block_mut(bb1);
            b.params = vec![(v_i, MirType::Value)];
            b.instructions
                .push((v_cond, Instruction::CmpLt(v_i, v_limit)));
            b.terminator = Terminator::CondBranch {
                condition: v_cond,
                true_target: bb2,
                true_args: vec![],
                false_target: bb3,
                false_args: vec![v_i],
            };
        }
        {
            let b = f.block_mut(bb2);
            b.instructions
                .push((v_branch_cond, Instruction::ConstBool(true)));
            b.terminator = Terminator::CondBranch {
                condition: v_branch_cond,
                true_target: bb4,
                true_args: vec![],
                false_target: bb5,
                false_args: vec![],
            };
        }
        {
            let b = f.block_mut(bb3);
            b.params = vec![(v_exit_i, MirType::Value)];
            b.terminator = Terminator::Return(v_exit_i);
        }
        {
            let b = f.block_mut(bb4);
            b.instructions.push((v_one_a, Instruction::ConstNum(1.0)));
            b.instructions
                .push((v_new_i_a, Instruction::Add(v_i, v_one_a)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_new_i_a],
            };
        }
        {
            let b = f.block_mut(bb5);
            b.terminator = Terminator::Branch {
                target: bb6,
                args: vec![],
            };
        }
        {
            let b = f.block_mut(bb6);
            b.instructions.push((v_one_b, Instruction::ConstNum(1.0)));
            b.instructions
                .push((v_new_i_b, Instruction::Add(v_i, v_one_b)));
            b.terminator = Terminator::Branch {
                target: bb1,
                args: vec![v_new_i_b],
            };
        }

        let before = eval(&f).unwrap();
        let licm = Licm;
        licm.run(&mut f);
        let after = eval(&f).unwrap();
        assert_eq!(
            before, after,
            "LICM with multiple back edges must preserve semantics"
        );

        // Every use must have a dominating definition — no orphan uses.
        let defined: HashSet<ValueId> = f
            .blocks
            .iter()
            .flat_map(|blk| {
                blk.params
                    .iter()
                    .map(|&(v, _)| v)
                    .chain(blk.instructions.iter().map(|&(v, _)| v))
            })
            .collect();
        for blk in &f.blocks {
            for (_, inst) in &blk.instructions {
                for op in inst.operands() {
                    assert!(
                        defined.contains(&op),
                        "undefined value {:?} used in bb{}",
                        op,
                        blk.id.0
                    );
                }
            }
        }
    }

    #[test]
    fn test_dominators_simple() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);

        // bb0 -> bb1 -> bb2
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();

        let v = f.new_value();
        f.block_mut(bb0)
            .instructions
            .push((v, Instruction::ConstNull));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![],
        };
        f.block_mut(bb1).terminator = Terminator::Branch {
            target: bb2,
            args: vec![],
        };
        f.block_mut(bb2).terminator = Terminator::Return(v);

        f.compute_predecessors();
        let rpo = compute_rpo(&f);
        let idom = compute_dominators(&f, &rpo);

        assert!(dominates(&idom, 0, 1));
        assert!(dominates(&idom, 0, 2));
        assert!(dominates(&idom, 1, 2));
        assert!(!dominates(&idom, 2, 1));
    }

    #[test]
    fn test_loop_detection() {
        let mut interner = Interner::new();
        let mut f = make_func(&mut interner);

        // bb0 -> bb1 -> bb2 -> bb1 (loop), bb1 -> bb3 (exit)
        let bb0 = f.new_block();
        let bb1 = f.new_block();
        let bb2 = f.new_block();
        let bb3 = f.new_block();

        let v = f.new_value();
        let v_cond = f.new_value();
        f.block_mut(bb0)
            .instructions
            .push((v, Instruction::ConstBool(true)));
        f.block_mut(bb0).terminator = Terminator::Branch {
            target: bb1,
            args: vec![],
        };
        f.block_mut(bb1)
            .instructions
            .push((v_cond, Instruction::ConstBool(true)));
        f.block_mut(bb1).terminator = Terminator::CondBranch {
            condition: v_cond,
            true_target: bb2,
            true_args: vec![],
            false_target: bb3,
            false_args: vec![],
        };
        f.block_mut(bb2).terminator = Terminator::Branch {
            target: bb1,
            args: vec![],
        };
        f.block_mut(bb3).terminator = Terminator::ReturnNull;

        f.compute_predecessors();
        let rpo = compute_rpo(&f);
        let idom = compute_dominators(&f, &rpo);
        let loops = detect_loops(&f, &idom);

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, bb1);
        assert_eq!(loops[0].latch, bb2);
        assert!(loops[0].body.contains(&bb1));
        assert!(loops[0].body.contains(&bb2));
        assert!(!loops[0].body.contains(&bb0));
        assert!(!loops[0].body.contains(&bb3));
    }
}
