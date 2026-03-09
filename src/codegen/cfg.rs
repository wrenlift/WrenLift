/// Control Flow Graph analysis for MachFunc.
///
/// Extracts basic blocks from the flat instruction stream, builds
/// predecessor/successor edges, computes dominators, detects loops,
/// and provides RPO traversal ordering.
///
/// Used by:
/// - WASM emitter: structured control flow conversion (block/loop/br)
/// - Register allocator: liveness analysis improvements
/// - Optimization: loop-aware scheduling
use super::{Label, MachFunc, MachInst};
use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Basic block
// ---------------------------------------------------------------------------

/// A basic block in the CFG, identified by its position in the block list.
#[derive(Debug)]
pub struct CfgBlock {
    /// The label that starts this block (if any).
    pub label: Option<Label>,
    /// Instruction range `[start, end)` into `MachFunc.insts`.
    pub start: usize,
    pub end: usize,
    /// Successor block indices.
    pub succs: Vec<usize>,
    /// Predecessor block indices.
    pub preds: Vec<usize>,
}

impl CfgBlock {
    /// Number of instructions in this block.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

// ---------------------------------------------------------------------------
// CFG
// ---------------------------------------------------------------------------

/// Control flow graph extracted from a `MachFunc`.
pub struct Cfg {
    /// Basic blocks in original program order.
    pub blocks: Vec<CfgBlock>,
    /// Label → block index mapping.
    label_to_block: HashMap<Label, usize>,
}

impl Cfg {
    /// Build a CFG from a MachFunc's flat instruction stream.
    ///
    /// Identifies basic blocks by scanning for `DefLabel` markers and
    /// terminator instructions, then computes successor/predecessor edges.
    pub fn build(func: &MachFunc) -> Self {
        let insts = &func.insts;
        if insts.is_empty() {
            return Cfg {
                blocks: Vec::new(),
                label_to_block: HashMap::new(),
            };
        }

        // Phase 1: Identify block start positions.
        //
        // A new block starts at:
        //   - Index 0 (implicit entry)
        //   - Each DefLabel instruction
        //   - The instruction after any terminator (fall-through point)
        let mut starts: Vec<usize> = Vec::new();
        let mut label_positions: HashMap<usize, Label> = HashMap::new();

        // Always start at 0.
        starts.push(0);

        for (i, inst) in insts.iter().enumerate() {
            if let MachInst::DefLabel(l) = inst {
                if !starts.contains(&i) {
                    starts.push(i);
                }
                label_positions.insert(i, *l);
            } else if inst.is_terminator() || matches!(inst, MachInst::Ret) {
                let next = i + 1;
                if next < insts.len() && !starts.contains(&next) {
                    starts.push(next);
                }
            }
        }

        starts.sort_unstable();
        starts.dedup();

        // Phase 2: Create CfgBlock entries.
        let mut blocks = Vec::with_capacity(starts.len());
        let mut label_to_block: HashMap<Label, usize> = HashMap::new();

        for (idx, &start) in starts.iter().enumerate() {
            let end = starts.get(idx + 1).copied().unwrap_or(insts.len());
            let label = label_positions.get(&start).copied();

            if let Some(l) = label {
                label_to_block.insert(l, idx);
            }

            blocks.push(CfgBlock {
                label,
                start,
                end,
                succs: Vec::new(),
                preds: Vec::new(),
            });
        }

        // Phase 3: Compute edges from branch instructions.
        let num_blocks = blocks.len();
        for (i, block) in blocks.iter_mut().enumerate().take(num_blocks) {
            let (start, end) = (block.start, block.end);
            let mut ends_with_unconditional = false;

            for inst in &insts[start..end] {
                match inst {
                    MachInst::Jmp { target } => {
                        if let Some(&b) = label_to_block.get(target) {
                            if !block.succs.contains(&b) {
                                block.succs.push(b);
                            }
                        }
                        ends_with_unconditional = true;
                    }
                    MachInst::JmpIf { target, .. }
                    | MachInst::JmpZero { target, .. }
                    | MachInst::JmpNonZero { target, .. }
                    | MachInst::TestBitJmpZero { target, .. }
                    | MachInst::TestBitJmpNonZero { target, .. } => {
                        if let Some(&b) = label_to_block.get(target) {
                            if !block.succs.contains(&b) {
                                block.succs.push(b);
                            }
                        }
                        // Conditional branches also fall through.
                    }
                    MachInst::Ret | MachInst::Trap => {
                        ends_with_unconditional = true;
                    }
                    _ => {}
                }
            }

            // Fall-through to next block if no unconditional terminator.
            if !ends_with_unconditional && i + 1 < num_blocks
                && !block.succs.contains(&(i + 1)) {
                block.succs.push(i + 1);
            }
        }

        // Compute predecessors from successors.
        for i in 0..num_blocks {
            let succs = blocks[i].succs.clone();
            for s in succs {
                if !blocks[s].preds.contains(&i) {
                    blocks[s].preds.push(i);
                }
            }
        }

        Cfg {
            blocks,
            label_to_block,
        }
    }

    /// Number of basic blocks.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Look up which block a label belongs to.
    pub fn block_for_label(&self, label: Label) -> Option<usize> {
        self.label_to_block.get(&label).copied()
    }

    // -----------------------------------------------------------------------
    // RPO — Reverse Post-Order
    // -----------------------------------------------------------------------

    /// Compute reverse post-order traversal starting from the entry block (0).
    ///
    /// RPO is the standard iteration order for forward dataflow analyses
    /// and is required by the dominator algorithm.
    pub fn rpo(&self) -> Vec<usize> {
        let n = self.blocks.len();
        if n == 0 {
            return Vec::new();
        }

        let mut visited = vec![false; n];
        let mut post_order = Vec::with_capacity(n);

        // Iterative DFS (avoids stack overflow on deep CFGs).
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)]; // (block, next_succ_idx)
        visited[0] = true;

        while let Some((block, succ_idx)) = stack.last_mut() {
            let b = *block;
            if *succ_idx < self.blocks[b].succs.len() {
                let succ = self.blocks[b].succs[*succ_idx];
                *succ_idx += 1;
                if !visited[succ] {
                    visited[succ] = true;
                    stack.push((succ, 0));
                }
            } else {
                post_order.push(b);
                stack.pop();
            }
        }

        post_order.reverse();
        post_order
    }

    // -----------------------------------------------------------------------
    // Dominators
    // -----------------------------------------------------------------------

    /// Compute immediate dominators using the Cooper-Harvey-Kennedy algorithm.
    ///
    /// Returns `idom[i]` = immediate dominator of block `i`.
    /// The entry block (0) has `idom[0] = 0`.
    /// Unreachable blocks have `idom[i] = usize::MAX`.
    pub fn dominators(&self) -> Vec<usize> {
        let n = self.blocks.len();
        if n == 0 {
            return Vec::new();
        }

        let rpo = self.rpo();

        // Block index → RPO position (for comparison).
        let mut rpo_pos = vec![usize::MAX; n];
        for (pos, &block) in rpo.iter().enumerate() {
            rpo_pos[block] = pos;
        }

        let mut idom = vec![usize::MAX; n];
        idom[rpo[0]] = rpo[0]; // Entry dominates itself.

        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo.iter().skip(1) {
                // New idom = intersection of all processed predecessors.
                let mut new_idom = usize::MAX;
                for &p in &self.blocks[b].preds {
                    if idom[p] == usize::MAX {
                        continue; // Not yet processed.
                    }
                    if new_idom == usize::MAX {
                        new_idom = p;
                    } else {
                        new_idom = Self::intersect(&idom, &rpo_pos, new_idom, p);
                    }
                }
                if new_idom != usize::MAX && new_idom != idom[b] {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        idom
    }

    /// Intersect two dominator tree paths — walks up until they meet.
    fn intersect(idom: &[usize], rpo_pos: &[usize], mut a: usize, mut b: usize) -> usize {
        while a != b {
            while rpo_pos[a] > rpo_pos[b] {
                a = idom[a];
            }
            while rpo_pos[b] > rpo_pos[a] {
                b = idom[b];
            }
        }
        a
    }

    /// Check if block `a` dominates block `b`.
    pub fn dominates(&self, idom: &[usize], a: usize, b: usize) -> bool {
        if a == b {
            return true;
        }
        let mut cur = b;
        loop {
            if cur == a {
                return true;
            }
            let d = idom[cur];
            if d == cur {
                // Reached the root without finding `a`.
                return false;
            }
            cur = d;
        }
    }

    /// Build the dominator tree: `dom_children[i]` = blocks immediately
    /// dominated by block `i`.
    pub fn dominator_tree(&self, idom: &[usize]) -> Vec<Vec<usize>> {
        let n = idom.len();
        let mut children = vec![Vec::new(); n];
        for i in 0..n {
            if idom[i] != i && idom[i] != usize::MAX {
                children[idom[i]].push(i);
            }
        }
        children
    }

    // -----------------------------------------------------------------------
    // Loop detection
    // -----------------------------------------------------------------------

    /// Detect natural loops by finding back edges.
    ///
    /// A back edge `src → target` exists when `target` dominates `src`.
    /// The natural loop body is the set of blocks that can reach `src`
    /// without going through `target`.
    pub fn detect_loops(&self) -> Vec<Loop> {
        let idom = self.dominators();
        let n = self.blocks.len();
        let mut loops = Vec::new();

        for src in 0..n {
            for &target in &self.blocks[src].succs {
                if self.dominates(&idom, target, src) {
                    // Back edge: src → target. target is the loop header.
                    let body = self.compute_loop_body(target, src);
                    loops.push(Loop {
                        header: target,
                        latch: src,
                        body,
                    });
                }
            }
        }

        loops
    }

    /// Compute the natural loop body for a back edge `latch → header`.
    fn compute_loop_body(&self, header: usize, latch: usize) -> Vec<usize> {
        let mut body = HashSet::new();
        body.insert(header);

        if latch != header {
            body.insert(latch);
            let mut worklist = VecDeque::new();
            worklist.push_back(latch);

            while let Some(node) = worklist.pop_front() {
                for &pred in &self.blocks[node].preds {
                    if body.insert(pred) {
                        worklist.push_back(pred);
                    }
                }
            }
        }

        let mut body_vec: Vec<usize> = body.into_iter().collect();
        body_vec.sort_unstable();
        body_vec
    }

    /// Compute loop nesting depth for each block.
    /// Blocks not in any loop have depth 0.
    pub fn loop_depths(&self) -> Vec<u32> {
        let loops = self.detect_loops();
        let n = self.blocks.len();
        let mut depths = vec![0u32; n];

        for lp in &loops {
            for &block in &lp.body {
                depths[block] += 1;
            }
        }

        depths
    }

    /// Identify which blocks are loop headers.
    pub fn loop_header_set(&self) -> HashSet<usize> {
        self.detect_loops()
            .into_iter()
            .map(|lp| lp.header)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    /// Return the last terminator instruction in a block, if any.
    pub fn block_terminator<'a>(
        &self,
        block_idx: usize,
        func: &'a MachFunc,
    ) -> Option<&'a MachInst> {
        let b = &self.blocks[block_idx];
        if b.start >= b.end {
            return None;
        }
        let last = &func.insts[b.end - 1];
        if last.is_terminator() || matches!(last, MachInst::Ret) {
            Some(last)
        } else {
            None
        }
    }

    /// Get the instruction range excluding leading DefLabel.
    pub fn block_body_range(&self, block_idx: usize, func: &MachFunc) -> (usize, usize) {
        let b = &self.blocks[block_idx];
        let mut start = b.start;
        // Skip leading DefLabel instruction.
        if start < b.end
            && matches!(func.insts[start], MachInst::DefLabel(_)) {
            start += 1;
        }
        (start, b.end)
    }

    /// Pretty-print the CFG for debugging.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        for (i, block) in self.blocks.iter().enumerate() {
            out.push_str(&format!(
                "B{} (insts {}..{}) label={:?} preds={:?} succs={:?}\n",
                i, block.start, block.end, block.label, block.preds, block.succs
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------

/// A natural loop detected in the CFG.
#[derive(Debug, Clone)]
pub struct Loop {
    /// The loop header block (dominates all body blocks).
    pub header: usize,
    /// The latch block (source of the back edge to header).
    pub latch: usize,
    /// All blocks in the loop body (sorted, including header and latch).
    pub body: Vec<usize>,
}

impl Loop {
    /// Number of blocks in the loop body.
    pub fn num_blocks(&self) -> usize {
        self.body.len()
    }

    /// Check if a block is part of this loop.
    pub fn contains(&self, block: usize) -> bool {
        self.body.binary_search(&block).is_ok()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{Cond, MachFunc, MachInst, VReg};

    /// Helper: build a MachFunc from a list of instructions.
    fn make_func(insts: Vec<MachInst>) -> MachFunc {
        let mut mf = MachFunc::new("test".to_string());
        mf.insts = insts;
        mf
    }

    // --- Block identification ---

    #[test]
    fn test_single_block() {
        let r0 = VReg::gp(0);
        let func = make_func(vec![MachInst::LoadImm { dst: r0, bits: 42 }, MachInst::Ret]);
        let cfg = Cfg::build(&func);

        assert_eq!(cfg.num_blocks(), 1);
        assert_eq!(cfg.blocks[0].start, 0);
        assert_eq!(cfg.blocks[0].end, 2);
        assert!(cfg.blocks[0].succs.is_empty());
        assert!(cfg.blocks[0].preds.is_empty());
    }

    #[test]
    fn test_two_blocks_fallthrough() {
        let r0 = VReg::gp(0);
        let l1 = Label(1);
        let func = make_func(vec![
            MachInst::LoadImm { dst: r0, bits: 1 },
            MachInst::DefLabel(l1),
            MachInst::LoadImm { dst: r0, bits: 2 },
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);

        assert_eq!(cfg.num_blocks(), 2);
        // B0 falls through to B1.
        assert_eq!(cfg.blocks[0].succs, vec![1]);
        assert_eq!(cfg.blocks[1].preds, vec![0]);
    }

    #[test]
    fn test_unconditional_branch() {
        let r0 = VReg::gp(0);
        let l0 = Label(0);
        let l1 = Label(1);
        let func = make_func(vec![
            MachInst::DefLabel(l0),
            MachInst::LoadImm { dst: r0, bits: 1 },
            MachInst::Jmp { target: l1 },
            MachInst::DefLabel(l1),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);

        assert_eq!(cfg.num_blocks(), 2);
        // B0 jumps to B1, no fall-through (Jmp is unconditional).
        assert_eq!(cfg.blocks[0].succs, vec![1]);
        assert_eq!(cfg.blocks[1].preds, vec![0]);
    }

    #[test]
    fn test_conditional_branch_diamond() {
        //  B0: cmp + jmpIf L2
        //  B1: (fall-through) → ret
        //  B2: (label L2) → ret
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let l2 = Label(2);
        let func = make_func(vec![
            // B0
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Eq,
                target: l2,
            },
            // B1 (fall-through)
            MachInst::LoadImm { dst: r0, bits: 1 },
            MachInst::Ret,
            // B2
            MachInst::DefLabel(l2),
            MachInst::LoadImm { dst: r0, bits: 2 },
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);

        assert_eq!(cfg.num_blocks(), 3);
        // B0 → B1 (fall-through) + B2 (conditional jump)
        assert!(cfg.blocks[0].succs.contains(&1));
        assert!(cfg.blocks[0].succs.contains(&2));
        // B1 and B2 both terminate.
        assert!(cfg.blocks[1].succs.is_empty());
        assert!(cfg.blocks[2].succs.is_empty());
    }

    // --- RPO ---

    #[test]
    fn test_rpo_linear() {
        let l1 = Label(1);
        let l2 = Label(2);
        let func = make_func(vec![
            MachInst::Jmp { target: l1 },
            MachInst::DefLabel(l1),
            MachInst::Jmp { target: l2 },
            MachInst::DefLabel(l2),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let rpo = cfg.rpo();
        // Linear chain: RPO = [0, 1, 2].
        assert_eq!(rpo, vec![0, 1, 2]);
    }

    #[test]
    fn test_rpo_diamond() {
        // B0 → B1 (fall-through), B0 → B2 (jmpIf)
        // B1 → B3, B2 → B3
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let l1 = Label(1);
        let l2 = Label(2);
        let l3 = Label(3);
        let func = make_func(vec![
            // B0
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Eq,
                target: l2,
            },
            // B1
            MachInst::DefLabel(l1),
            MachInst::Jmp { target: l3 },
            // B2
            MachInst::DefLabel(l2),
            MachInst::Jmp { target: l3 },
            // B3
            MachInst::DefLabel(l3),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let rpo = cfg.rpo();

        // Entry comes first, exit comes last.
        assert_eq!(rpo[0], 0);
        assert_eq!(*rpo.last().unwrap(), 3);
        assert_eq!(rpo.len(), 4);
    }

    // --- Dominators ---

    #[test]
    fn test_dominators_linear() {
        let l1 = Label(1);
        let l2 = Label(2);
        let func = make_func(vec![
            MachInst::Jmp { target: l1 },
            MachInst::DefLabel(l1),
            MachInst::Jmp { target: l2 },
            MachInst::DefLabel(l2),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let idom = cfg.dominators();

        // Linear chain: idom[0] = 0, idom[1] = 0, idom[2] = 1.
        assert_eq!(idom[0], 0);
        assert_eq!(idom[1], 0);
        assert_eq!(idom[2], 1);
    }

    #[test]
    fn test_dominators_diamond() {
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let l1 = Label(1);
        let l2 = Label(2);
        let l3 = Label(3);
        let func = make_func(vec![
            // B0
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Eq,
                target: l2,
            },
            // B1
            MachInst::DefLabel(l1),
            MachInst::Jmp { target: l3 },
            // B2
            MachInst::DefLabel(l2),
            MachInst::Jmp { target: l3 },
            // B3
            MachInst::DefLabel(l3),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let idom = cfg.dominators();

        // B0 dominates everything.
        // B3's idom is B0 (not B1 or B2, since either path reaches B3).
        assert_eq!(idom[0], 0);
        assert_eq!(idom[1], 0);
        assert_eq!(idom[2], 0);
        assert_eq!(idom[3], 0);
    }

    #[test]
    fn test_dominates() {
        let l1 = Label(1);
        let l2 = Label(2);
        let func = make_func(vec![
            MachInst::Jmp { target: l1 },
            MachInst::DefLabel(l1),
            MachInst::Jmp { target: l2 },
            MachInst::DefLabel(l2),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let idom = cfg.dominators();

        assert!(cfg.dominates(&idom, 0, 0));
        assert!(cfg.dominates(&idom, 0, 1));
        assert!(cfg.dominates(&idom, 0, 2));
        assert!(cfg.dominates(&idom, 1, 2));
        assert!(!cfg.dominates(&idom, 2, 1));
        assert!(!cfg.dominates(&idom, 1, 0));
    }

    // --- Loop detection ---

    #[test]
    fn test_simple_loop() {
        // B0 → B1 (loop header)
        // B1 → B1 (self-loop via back edge) or B2 (exit)
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let l1 = Label(1);
        let l2 = Label(2);
        let func = make_func(vec![
            // B0: entry
            MachInst::Jmp { target: l1 },
            // B1: loop header
            MachInst::DefLabel(l1),
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Lt,
                target: l1,
            },
            // B2: fall-through exit
            MachInst::DefLabel(l2),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let loops = cfg.detect_loops();

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, 1);
        assert_eq!(loops[0].latch, 1); // Self-loop.
        assert!(loops[0].contains(1));
    }

    #[test]
    fn test_multi_block_loop() {
        // B0 → B1 (header)
        // B1 → B2 (body)
        // B2 → B1 (back edge) or B3 (exit)
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let l1 = Label(1);
        let l2 = Label(2);
        let l3 = Label(3);
        let func = make_func(vec![
            // B0
            MachInst::Jmp { target: l1 },
            // B1: loop header
            MachInst::DefLabel(l1),
            MachInst::Nop,
            MachInst::Jmp { target: l2 },
            // B2: loop body + latch
            MachInst::DefLabel(l2),
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Lt,
                target: l1,
            },
            // B3: exit (fall-through from B2)
            MachInst::DefLabel(l3),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let loops = cfg.detect_loops();

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, 1);
        assert_eq!(loops[0].latch, 2);
        assert!(loops[0].contains(1));
        assert!(loops[0].contains(2));
        assert!(!loops[0].contains(0)); // Entry not in loop.
        assert!(!loops[0].contains(3)); // Exit not in loop.
    }

    #[test]
    fn test_no_loops() {
        let r0 = VReg::gp(0);
        let func = make_func(vec![MachInst::LoadImm { dst: r0, bits: 42 }, MachInst::Ret]);
        let cfg = Cfg::build(&func);
        let loops = cfg.detect_loops();
        assert!(loops.is_empty());
    }

    #[test]
    fn test_loop_depths() {
        // B0 → B1 (outer header)
        // B1 → B2 (inner header)
        // B2 → B2 (inner back edge) or B1 (outer back edge via fall-through to B3)
        // Actually let's do a cleaner nested loop:
        // B0 → B1 (outer header) → B2 (inner header) → B2 (inner loop) or B3
        // B3 → B1 (outer back edge) or B4 (exit)
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let l1 = Label(1);
        let l2 = Label(2);
        let l3 = Label(3);
        let l4 = Label(4);
        let func = make_func(vec![
            // B0: entry
            MachInst::Jmp { target: l1 },
            // B1: outer loop header
            MachInst::DefLabel(l1),
            MachInst::Jmp { target: l2 },
            // B2: inner loop header (self-loop)
            MachInst::DefLabel(l2),
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Lt,
                target: l2,
            },
            // B3: outer loop body, back edge to B1
            MachInst::DefLabel(l3),
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Lt,
                target: l1,
            },
            // B4: exit
            MachInst::DefLabel(l4),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let depths = cfg.loop_depths();

        assert_eq!(depths[0], 0); // Entry: not in any loop.
        assert_eq!(depths[1], 1); // Outer loop header: depth 1.
        assert_eq!(depths[2], 2); // Inner loop header: depth 2.
        assert_eq!(depths[3], 1); // Outer loop body: depth 1.
        assert_eq!(depths[4], 0); // Exit: not in any loop.
    }

    // --- Dominator tree ---

    #[test]
    fn test_dominator_tree() {
        let r0 = VReg::gp(0);
        let r1 = VReg::gp(1);
        let l1 = Label(1);
        let l2 = Label(2);
        let l3 = Label(3);
        let func = make_func(vec![
            // B0
            MachInst::ICmp { lhs: r0, rhs: r1 },
            MachInst::JmpIf {
                cond: Cond::Eq,
                target: l2,
            },
            // B1
            MachInst::DefLabel(l1),
            MachInst::Jmp { target: l3 },
            // B2
            MachInst::DefLabel(l2),
            MachInst::Jmp { target: l3 },
            // B3
            MachInst::DefLabel(l3),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let idom = cfg.dominators();
        let tree = cfg.dominator_tree(&idom);

        // B0 dominates B1, B2, B3.
        let mut children_0 = tree[0].clone();
        children_0.sort();
        assert_eq!(children_0, vec![1, 2, 3]);
        assert!(tree[1].is_empty());
        assert!(tree[2].is_empty());
        assert!(tree[3].is_empty());
    }

    // --- Edge cases ---

    #[test]
    fn test_empty_func() {
        let func = make_func(vec![]);
        let cfg = Cfg::build(&func);
        assert_eq!(cfg.num_blocks(), 0);
        assert!(cfg.rpo().is_empty());
        assert!(cfg.dominators().is_empty());
    }

    #[test]
    fn test_block_for_label() {
        let l0 = Label(0);
        let l1 = Label(1);
        let func = make_func(vec![
            MachInst::DefLabel(l0),
            MachInst::Jmp { target: l1 },
            MachInst::DefLabel(l1),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        assert_eq!(cfg.block_for_label(l0), Some(0));
        assert_eq!(cfg.block_for_label(l1), Some(1));
        assert_eq!(cfg.block_for_label(Label(99)), None);
    }

    #[test]
    fn test_cfg_dump() {
        let l1 = Label(1);
        let func = make_func(vec![
            MachInst::Jmp { target: l1 },
            MachInst::DefLabel(l1),
            MachInst::Ret,
        ]);
        let cfg = Cfg::build(&func);
        let dump = cfg.dump();
        assert!(dump.contains("B0"));
        assert!(dump.contains("B1"));
        assert!(dump.contains("succs="));
    }
}
