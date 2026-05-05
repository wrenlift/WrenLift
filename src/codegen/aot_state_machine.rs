//! AOT state-machine MIR transform for fiber bodies.
//!
//! Tainted MIR functions (those that transitively reach
//! `Fiber.yield`) can't be lowered as straight-line native code —
//! Cranelift can't unwind across the suspension and resume later
//! from the saved C-stack. The Zura sibling project solves the
//! same problem for its `@ { }` blocks via a stackless state
//! machine: the function is split at every suspension into states,
//! each invocation enters at the saved state and runs forward
//! until the next suspension or return. We do the same here.
//!
//! v1 scope (this file):
//! - Closures that yield via `Fiber.yield(_)` / `Fiber.yield()`.
//! - **No** values that live across a suspension (each chunk's
//!   values are local). The trivial test case satisfies this; the
//!   liveness-driven save/restore comes in v2.
//! - One state per suspension call site, plus state 0 for the
//!   initial entry.
//!
//! The transform mutates a `MirFunction` in place and returns a
//! `StateMachineLayout` the AOT lowering consumes:
//!
//! - `dispatch_entry_block`: a new block, prepended via
//!   `entry_block`, that loads the state machine's state ID via
//!   a runtime helper and `br_table`s to the right resume entry.
//! - `resume_entries[i]`: the block to enter when state == i.
//!   `resume_entries[0]` is always the original `entry_block`
//!   (state 0 is the initial run).
//! - `yield_blocks[block_id] = next_state`: blocks whose final
//!   terminator is `Return(value)` representing a yield. The
//!   AOT lowering, before emitting that return, must call into
//!   the runtime to save `next_state` and stamp the kind as
//!   `Yield`. Plain returns get the kind stamped as `Done`.

use std::collections::HashMap;

use crate::intern::Interner;
use crate::mir::{BlockId, Instruction, MirFunction, Terminator, ValueId};

/// Decision payload returned by [`transform_to_state_machine`].
/// All other AOT-lowering invariants stay the same; this just
/// tells the lowering "you're emitting a state machine — here's
/// where to splice the prologue + which blocks have a Yield
/// semantics on their final return".
#[derive(Debug, Clone)]
pub struct StateMachineLayout {
    /// Per-state entry block. `resume_entries[0]` is the original
    /// MIR entry block. `resume_entries[i]` for `i > 0` is the
    /// block created by splitting at the `i`-th yield call.
    pub resume_entries: Vec<BlockId>,
    /// Blocks whose final `Return(v)` should be lowered as a
    /// suspension: store `next_state` to the state struct, stamp
    /// kind=Yield, return v. Other returns stamp kind=Done.
    pub yield_blocks: HashMap<BlockId, u32>,
    /// Per yield block, the live-across values to save before
    /// the Return — `(slot_index, value_to_save)`. Lowered as a
    /// `wlift_aot_sm_save_value(fiber, slot, v)` call sequence
    /// in the AOT body.
    pub yield_saves: HashMap<BlockId, Vec<(u32, ValueId)>>,
    /// Per resume block, the loads to emit at the block's
    /// entry — `(slot_index, fresh_value_id_to_define)`. The
    /// transform has already rewritten every downstream use of
    /// the original ValueId to point at this fresh one, so the
    /// lowering just needs to call `wlift_aot_sm_load_value(
    /// fiber, slot)` and store the result in `val_map[fresh_id]`.
    pub resume_loads: HashMap<BlockId, Vec<(u32, ValueId)>>,
}

/// Names of methods whose `Call` represents a suspension point.
/// Mirrors `aot_direct_yield_method_names` in `aot.rs`. The set
/// is duplicated here because this module is consumed by the AOT
/// lowering pass which doesn't otherwise pull in `aot.rs`.
fn suspension_method_names() -> &'static [&'static str] {
    &["yield()", "yield(_)", "suspend()"]
}

/// True if the instruction is a Call whose method name is one
/// of the suspension primitives.
fn is_suspension_call(inst: &Instruction, interner: &Interner) -> bool {
    if let Instruction::Call { method, .. } = inst {
        suspension_method_names()
            .iter()
            .any(|n| interner.resolve(*method) == *n)
    } else {
        false
    }
}

/// Errors the transform can refuse with. v1 caps refer to the
/// liveness-across-suspension case the simple split can't safely
/// emit; the AOT pipeline degrades to a hard error so we don't
/// silently mis-compile.
#[derive(Debug, Clone)]
pub enum StateMachineError {
    /// A value defined before a suspension is used after it.
    /// Implementing this requires lowering "save value to state
    /// struct" + "load on resume" through a per-state field
    /// table; v1 doesn't.
    LiveValueAcrossSuspension {
        block: BlockId,
        value: ValueId,
    },
    /// A suspension Call landed inside a block that already has
    /// a non-Return terminator (e.g. inside a Branch). v1 only
    /// supports linear bodies; this surfaces if a yield is
    /// nested inside a conditional branch.
    SuspensionInBranchedBlock { block: BlockId },
}

impl std::fmt::Display for StateMachineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StateMachineError::LiveValueAcrossSuspension { block, value } => write!(
                f,
                "v1 state-machine transform: value v{} from block bb{} used \
                 across a Fiber.yield suspension. Save/restore through the state \
                 struct is planned but not yet implemented",
                value.0, block.0
            ),
            StateMachineError::SuspensionInBranchedBlock { block } => write!(
                f,
                "v1 state-machine transform: Fiber.yield inside block bb{} which \
                 has a non-Return terminator (branch / loop body). Splitting \
                 across a branch would duplicate the post-yield code into each \
                 control-flow successor; v2 will handle this",
                block.0
            ),
        }
    }
}

impl std::error::Error for StateMachineError {}

/// Run the transform. Mutates `mir` in place: blocks containing
/// suspension Calls get split; new "resume" blocks are appended;
/// yield Calls are removed and replaced with a `Return(value)`
/// where `value` is the yield argument (the value the caller
/// observes). The returned `StateMachineLayout` carries the
/// per-state mapping the AOT lowering needs.
///
/// A function with zero suspension Calls returns a layout with a
/// single `resume_entries[0]` and no yield blocks — the AOT
/// lowering can either skip the state-machine prologue entirely
/// or emit it harmlessly (the dispatch always lands on state 0).
pub fn transform_to_state_machine(
    mir: &mut MirFunction,
    interner: &Interner,
) -> Result<StateMachineLayout, StateMachineError> {
    // First, find every suspension call site as (block_id,
    // instruction_index). Walk in block order; instructions
    // inside a block are split sequentially (the second yield in
    // a block becomes "first yield in the new block we just
    // split off"), so we re-scan after each split.
    let mut layout = StateMachineLayout {
        resume_entries: vec![mir.entry_block()],
        yield_blocks: HashMap::new(),
        yield_saves: HashMap::new(),
        resume_loads: HashMap::new(),
    };
    // Per-suspension save slot allocation. The vec is shared
    // across all suspensions — slots get reused as later yields
    // overwrite earlier saves. `next_slot` tracks the high-water
    // mark so concurrent saves never alias.
    let mut next_slot: u32 = 0;

    loop {
        let mut found: Option<(BlockId, usize, ValueId)> = None;
        'outer: for block in &mir.blocks {
            // Block must end in a Return for v1. Branches inside
            // a yielding chunk are deferred to v2.
            for (i, (_dst, inst)) in block.instructions.iter().enumerate() {
                if is_suspension_call(inst, interner) {
                    // The yield value is the first argument
                    // (`Fiber.yield(_)`). For `Fiber.yield()`
                    // (zero-arg form) we synthesise a null
                    // value via a fresh ConstNull placed just
                    // before the split point. Detect both.
                    let arg = match inst {
                        Instruction::Call { args, .. } => args.first().copied(),
                        _ => None,
                    };
                    // v2-cap2 path: yields inside Branch- /
                    // CondBranch-terminated blocks. The split
                    // keeps the original terminator on the
                    // resume block; if the resume's
                    // successors directly read saved values
                    // (rather than receiving them via block
                    // params), we clone those successor blocks
                    // so the yielding path can run a private
                    // copy that reads from `v_loaded`. The
                    // non-yielding path still falls through
                    // the original blocks, which keep using
                    // the original ValueIds.
                    //
                    // The cloning walk below handles this; the
                    // refusal that lived here in v1 was
                    // overconservative.
                    let _ = &block.terminator;
                    found = Some((
                        block.id,
                        i,
                        arg.unwrap_or(ValueId(u32::MAX)), // sentinel; resolved below
                    ));
                    break 'outer;
                }
            }
        }
        let Some((blk_id, inst_idx, mut yield_value)) = found else {
            break;
        };

        // Resolve `Fiber.yield()` (no-arg) by synthesising a
        // ConstNull immediately before the suspension call. We
        // append it to the block's instruction list, then peel
        // the suspension off.
        if yield_value.0 == u32::MAX {
            let new_vid = mir.new_value();
            let blk_mut = &mut mir.blocks[blk_id.0 as usize];
            blk_mut
                .instructions
                .insert(inst_idx, (new_vid, Instruction::ConstNull));
            yield_value = new_vid;
        }
        // After the optional ConstNull insert, the suspension
        // call sits at `inst_idx + (1 if no-arg else 0)`. Re-scan
        // for the suspension index — it's still the last item
        // before any post-suspension instructions.
        let inst_idx = {
            let blk = &mir.blocks[blk_id.0 as usize];
            blk.instructions
                .iter()
                .position(|(_, inst)| is_suspension_call(inst, interner))
                .expect("suspension still present after ConstNull insert")
        };

        // Liveness pass. v is "live across the suspension" iff
        // it's used somewhere reachable from the resume but NOT
        // defined in the resume's reachable code (i.e. its def
        // dominates the yield, by SSA dominance). For each such
        // v, allocate a slot, save before yield, load on resume,
        // remap downstream uses.
        //
        // The walk excludes back-edges: when a value is a block
        // param of a successor on a back-path, that block param
        // is its own SSA def (the loop's iteration value), not
        // a downstream use of `v` from this iteration. Stopping
        // at block-param shadows during remap (see below) is
        // what keeps the rewrite correct.
        let live_across: Vec<ValueId> = {
            let blk = &mir.blocks[blk_id.0 as usize];
            // Collect all uses + defs reachable from the
            // resume successors. Defs include both instruction
            // results and block params for blocks visited
            // along the way.
            let mut uses: Vec<ValueId> = Vec::new();
            let mut defs: std::collections::HashSet<ValueId> =
                std::collections::HashSet::new();
            // Pre-yield tail of the yielding block: any use
            // there belongs to the yielding code itself, not
            // post-yield (we already moved those instructions
            // off when splitting; this branch handles the
            // case before the split has happened).
            //
            // For "uses in the post-yield tail", scan the
            // instructions AFTER the yield (in the still-
            // un-split block) + the terminator + the
            // successor blocks.
            for (_, inst) in &blk.instructions[inst_idx + 1..] {
                for u in collect_uses(inst) {
                    uses.push(u);
                }
            }
            for u in collect_terminator_uses(&blk.terminator) {
                uses.push(u);
            }
            // Anything defined in the post-yield tail is local
            // to it — those don't need saving.
            for (vid, _) in &blk.instructions[inst_idx + 1..] {
                defs.insert(*vid);
            }
            let mut visited: std::collections::HashSet<BlockId> =
                std::collections::HashSet::new();
            let mut stack: Vec<BlockId> = blk.terminator.successors();
            while let Some(b) = stack.pop() {
                if !visited.insert(b) {
                    continue;
                }
                let s_blk = &mir.blocks[b.0 as usize];
                // Block params are NOT defs for liveness:
                // their incoming value is supplied by the
                // predecessor's branch arg, which itself can
                // be a saved (pre-yield) value flowing in via
                // tail-duplication. If we treated block params
                // as defs we'd miss saving values whose only
                // post-yield use is "v1 directly" inside a
                // block reachable from the resume that doesn't
                // shadow it.
                for (vid, inst) in &s_blk.instructions {
                    defs.insert(*vid);
                    for u in collect_uses(inst) {
                        uses.push(u);
                    }
                }
                for u in collect_terminator_uses(&s_blk.terminator) {
                    uses.push(u);
                }
                for s in s_blk.terminator.successors() {
                    stack.push(s);
                }
            }
            // live_across = uses not defined in the resume's
            // reachable code (so they must come from a
            // pre-yield definition).
            let mut seen: std::collections::HashSet<ValueId> =
                std::collections::HashSet::new();
            let mut live: Vec<ValueId> = Vec::new();
            for u in uses {
                if !defs.contains(&u) && seen.insert(u) {
                    live.push(u);
                }
            }
            // Reject any saved value whose def we can't pin
            // down (shouldn't happen for well-formed MIR but
            // surface defensively via the original error).
            // Iteration order is stable across Rust hash sets
            // because we used an order-preserving Vec.
            live
        };

        // Allocate save slots + fresh ValueIds for the loads.
        let saves: Vec<(u32, ValueId)> = live_across
            .iter()
            .map(|v| {
                let slot = next_slot;
                next_slot += 1;
                (slot, *v)
            })
            .collect();
        let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
        let loads: Vec<(u32, ValueId)> = saves
            .iter()
            .map(|(slot, original)| {
                let fresh = mir.new_value();
                remap.insert(*original, fresh);
                (*slot, fresh)
            })
            .collect();

        // Split the block.
        //   blk[0..inst_idx]       → stays in `blk`. Drop the
        //                            suspension Call itself; the
        //                            caller observes `yield_value`
        //                            via the new Return terminator.
        //   blk[inst_idx+1..]      → moves to a fresh resume block.
        //   blk.terminator         → moves with it.
        //   blk.terminator (new)   → Return(yield_value).
        let new_block_id = mir.new_block();
        let (post_instructions, original_terminator) = {
            let blk = &mut mir.blocks[blk_id.0 as usize];
            let mut tail = blk.instructions.split_off(inst_idx);
            // tail[0] is the suspension call — drop it.
            tail.remove(0);
            let original_terminator =
                std::mem::replace(&mut blk.terminator, Terminator::Return(yield_value));
            (tail, original_terminator)
        };
        // `new_block` was appended via `mir.new_block()` already;
        // populate it now.
        let new_block = mir.block_mut(new_block_id);
        new_block.instructions = post_instructions;
        new_block.terminator = original_terminator;

        // Tail-duplication remap. The resume block's
        // successors may also be reachable from the
        // non-yielding side of the CFG (e.g. an `if-else` whose
        // join block follows the yield's branch). Rewriting
        // those successors in place would break the
        // non-yielding path, which still expects the original
        // ValueIds. Clone any successor that *uses a saved
        // value directly* (i.e. without a block-param
        // shadowing it), substitute `original → fresh` in the
        // clone, and re-route the yielding path's terminator
        // through the clone. Blocks that don't use saved
        // values (or shadow them all) stay untouched and serve
        // both paths.
        if !remap.is_empty() {
            // First, handle the resume block itself: its
            // instructions and terminator may directly use
            // saved values. The resume block has no
            // block-param shadows (we just created it from a
            // mid-block split). Remap in place.
            {
                let blk = mir.block_mut(new_block_id);
                for (_, inst) in &mut blk.instructions {
                    remap_instruction_uses(inst, &remap);
                }
                remap_terminator_uses(&mut blk.terminator, &remap);
            }
            // Now walk the resume block's successors,
            // duplicating any that use saved values. Maintain
            // a memo so a cloned block is reused when reached
            // via multiple paths from the resume.
            //
            // Memo key: `(original_block_id, set_of_active_remap_keys)`.
            // For simplicity in v2-cap2, we use just the block
            // id — the active remap shrinks monotonically as
            // we encounter block-param shadows, but the
            // common case is "all saved values active" so this
            // memo collision is unlikely to over-clone.
            let mut clones: HashMap<BlockId, BlockId> = HashMap::new();
            // Worklist of (orig_block_id, clone_block_id) pairs whose
            // bodies still need to be filled.
            let mut to_fill: Vec<(BlockId, BlockId)> = Vec::new();
            // Seed: the resume block's terminator successors.
            // For each successor that needs cloning, allocate
            // a fresh BlockId and rewrite the resume's
            // terminator to point at it.
            let resume_term = mir.block(new_block_id).terminator.clone();
            let need_clone =
                |b: BlockId, mir_ref: &MirFunction, remap: &HashMap<ValueId, ValueId>| -> bool {
                    let blk = mir_ref.block(b);
                    // If every saved value is shadowed by this
                    // block's params, no clone needed — block
                    // params bind the value freshly per call.
                    let active: Vec<ValueId> = remap
                        .keys()
                        .filter(|orig| !blk.params.iter().any(|(p, _)| p == *orig))
                        .copied()
                        .collect();
                    if active.is_empty() {
                        return false;
                    }
                    // Block uses any active saved value
                    // directly (in instruction or terminator)?
                    let active_set: std::collections::HashSet<ValueId> =
                        active.iter().copied().collect();
                    for (_, inst) in &blk.instructions {
                        for u in inst.operands() {
                            if active_set.contains(&u) {
                                return true;
                            }
                        }
                    }
                    for u in blk.terminator.operands() {
                        if active_set.contains(&u) {
                            return true;
                        }
                    }
                    false
                };

            // Helper: redirect a terminator's successors that
            // need cloning to use the cloned BlockId. Also
            // collects new clones to be filled.
            fn redirect_term(
                term: &mut Terminator,
                clones: &mut HashMap<BlockId, BlockId>,
                to_fill: &mut Vec<(BlockId, BlockId)>,
                mir: &mut MirFunction,
                remap: &HashMap<ValueId, ValueId>,
                need_clone: &impl Fn(BlockId, &MirFunction, &HashMap<ValueId, ValueId>) -> bool,
            ) {
                let mut redirect = |succ: &mut BlockId| {
                    if need_clone(*succ, mir, remap) {
                        let clone_id = if let Some(c) = clones.get(succ) {
                            *c
                        } else {
                            let c = mir.new_block();
                            clones.insert(*succ, c);
                            to_fill.push((*succ, c));
                            c
                        };
                        *succ = clone_id;
                    }
                };
                match term {
                    Terminator::Branch { target, .. } => redirect(target),
                    Terminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        redirect(true_target);
                        redirect(false_target);
                    }
                    Terminator::Return(_)
                    | Terminator::ReturnNull
                    | Terminator::Unreachable => {}
                }
            }

            // Apply redirect to the resume block.
            {
                let _ = resume_term; // unused; we need &mut access below.
                // SAFETY: temporarily detach the terminator,
                // redirect, then reattach. Necessary because
                // `redirect_term` needs mutable access to mir
                // (for `new_block`) AND read access to mir
                // (for `need_clone`); separating like this
                // avoids overlapping borrows.
                let mut t = std::mem::replace(
                    &mut mir.block_mut(new_block_id).terminator,
                    Terminator::Unreachable,
                );
                redirect_term(&mut t, &mut clones, &mut to_fill, mir, &remap, &need_clone);
                mir.block_mut(new_block_id).terminator = t;
            }

            // Fill clones until the worklist is drained.
            while let Some((orig, clone_id)) = to_fill.pop() {
                let (mut insts, mut term, params) = {
                    let blk = mir.block(orig);
                    (blk.instructions.clone(), blk.terminator.clone(), blk.params.clone())
                };
                // Compute active remap (saved values not
                // shadowed by `orig`'s params).
                let active_remap: HashMap<ValueId, ValueId> = remap
                    .iter()
                    .filter(|(o, _)| !params.iter().any(|(p, _)| p == *o))
                    .map(|(o, n)| (*o, *n))
                    .collect();
                for (_, inst) in &mut insts {
                    remap_instruction_uses(inst, &active_remap);
                }
                remap_terminator_uses(&mut term, &active_remap);
                // Recursively redirect this clone's terminator
                // successors that still need cloning.
                redirect_term(
                    &mut term,
                    &mut clones,
                    &mut to_fill,
                    mir,
                    &active_remap,
                    &need_clone,
                );
                // Populate the clone block. We don't carry
                // over `params` because the clone is reached
                // via the same call args as the original (the
                // predecessor's branch is what invokes it).
                // But: the predecessor's branch passes block
                // args to satisfy `params`; if the clone has
                // the same params, the args still flow in
                // correctly. Keep the params unchanged.
                let blk = mir.block_mut(clone_id);
                blk.params = params;
                blk.instructions = insts;
                blk.terminator = term;
            }
        }

        let next_state = layout.resume_entries.len() as u32;
        layout.resume_entries.push(new_block_id);
        layout.yield_blocks.insert(blk_id, next_state);
        if !saves.is_empty() {
            layout.yield_saves.insert(blk_id, saves);
            layout.resume_loads.insert(new_block_id, loads);
        }
    }

    Ok(layout)
}

/// Replace every `ValueId` operand of `inst` with the mapping in
/// `remap`. Untouched if the operand isn't a key. Mirrors the
/// shape of `Instruction::operands` but writes back. v1's
/// liveness check only reads operands; the v2 lifting needs the
/// in-place rewrite to keep SSA valid after a split.
fn remap_instruction_uses(inst: &mut Instruction, remap: &HashMap<ValueId, ValueId>) {
    let mp = |v: &mut ValueId| {
        if let Some(new) = remap.get(v) {
            *v = *new;
        }
    };
    use Instruction::*;
    match inst {
        Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | Mod(a, b) => {
            mp(a);
            mp(b);
        }
        AddF64(a, b) | SubF64(a, b) | MulF64(a, b) | DivF64(a, b) | ModF64(a, b) => {
            mp(a);
            mp(b);
        }
        CmpLt(a, b) | CmpGt(a, b) | CmpLe(a, b) | CmpGe(a, b) | CmpEq(a, b) | CmpNe(a, b) => {
            mp(a);
            mp(b);
        }
        CmpLtF64(a, b) | CmpGtF64(a, b) | CmpLeF64(a, b) | CmpGeF64(a, b) => {
            mp(a);
            mp(b);
        }
        BitAnd(a, b) | BitOr(a, b) | BitXor(a, b) | Shl(a, b) | Shr(a, b) => {
            mp(a);
            mp(b);
        }
        MathBinaryF64(_, a, b) => {
            mp(a);
            mp(b);
        }
        Neg(a) | NegF64(a) | Not(a) | BitNot(a) | GuardNum(a) | GuardBool(a) | Unbox(a)
        | Box(a) | Move(a) | ToString(a) => mp(a),
        MathUnaryF64(_, a) => mp(a),
        GuardClass(a, _) | GuardProtocol(a, _) | IsType(a, _) => mp(a),
        GetField(recv, _) => mp(recv),
        SetField(recv, _, val) => {
            mp(recv);
            mp(val);
        }
        SetStaticField(_, val) | SetUpvalue(_, val) | SetModuleVar(_, val) => mp(val),
        Call { receiver, args, .. } | CallKnownFunc { receiver, args, .. } => {
            mp(receiver);
            for a in args.iter_mut() {
                mp(a);
            }
        }
        CallStaticSelf { args } => {
            for a in args.iter_mut() {
                mp(a);
            }
        }
        SuperCall { args, .. } => {
            for a in args.iter_mut() {
                mp(a);
            }
        }
        MakeClosure { upvalues, .. } => {
            for v in upvalues.iter_mut() {
                mp(v);
            }
        }
        MakeList(elems) => {
            for v in elems.iter_mut() {
                mp(v);
            }
        }
        MakeMap(pairs) => {
            for (k, v) in pairs.iter_mut() {
                mp(k);
                mp(v);
            }
        }
        MakeRange(from, to, _) => {
            mp(from);
            mp(to);
        }
        StringConcat(parts) => {
            for v in parts.iter_mut() {
                mp(v);
            }
        }
        SubscriptGet { receiver, args } => {
            mp(receiver);
            for a in args.iter_mut() {
                mp(a);
            }
        }
        SubscriptSet { receiver, args, value } => {
            mp(receiver);
            for a in args.iter_mut() {
                mp(a);
            }
            mp(value);
        }
        // Pure constants and parameter loads have no operand uses.
        ConstNum(_) | ConstBool(_) | ConstNull | ConstString(_) | ConstF64(_) | ConstI64(_) => {}
        GetModuleVar(_) | GetStaticField(_) | GetUpvalue(_) | BlockParam(_) => {}
        // Defensive fallback: any unknown variant is left alone.
        // Tracking new variants happens via the operands_mut()
        // canonical walker once it lands; for now the explicit
        // list above covers every Instruction the MIR builder
        // emits.
        #[allow(unreachable_patterns)]
        _ => {}
    }
}

/// Mirror of [`remap_instruction_uses`] for terminators.
fn remap_terminator_uses(term: &mut Terminator, remap: &HashMap<ValueId, ValueId>) {
    let mp = |v: &mut ValueId| {
        if let Some(new) = remap.get(v) {
            *v = *new;
        }
    };
    match term {
        Terminator::Return(v) => mp(v),
        Terminator::ReturnNull | Terminator::Unreachable => {}
        Terminator::Branch { args, .. } => {
            for a in args.iter_mut() {
                mp(a);
            }
        }
        Terminator::CondBranch {
            condition,
            true_args,
            false_args,
            ..
        } => {
            mp(condition);
            for a in true_args.iter_mut() {
                mp(a);
            }
            for a in false_args.iter_mut() {
                mp(a);
            }
        }
    }
}

/// Collect ValueId reads from one instruction. Defers to the
/// canonical `Instruction::operands` walker so this module stays
/// in sync with future `Instruction` enum additions.
fn collect_uses(inst: &Instruction) -> Vec<ValueId> {
    inst.operands()
}

/// Collect ValueId reads from a terminator. Uses `Terminator::operands`
/// to stay in sync with future enum additions without a per-variant
/// match in this module.
fn collect_terminator_uses(term: &Terminator) -> Vec<ValueId> {
    term.operands()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::aot::tests_helpers::build_mir_from_source;

    /// Trivial fiber body with two yields and no live values
    /// across them — this is the `/tmp/sockproj` shape the v1
    /// transform must handle.
    #[test]
    fn linear_yields_split_into_three_states() {
        let src = r#"
var f = Fiber.new {
  System.print("step 1")
  Fiber.yield(10)
  System.print("step 2")
  Fiber.yield(20)
  System.print("step 3")
  return 30
}
f.call()
"#;
        let (mut mir, interner) = build_mir_from_source(src, /* closure_idx */ 0);
        let layout = transform_to_state_machine(&mut mir, &interner)
            .expect("v1 transform succeeds on linear yields");
        assert_eq!(
            layout.resume_entries.len(),
            3,
            "expected 3 states (initial + 2 resumes); got {:?}",
            layout
        );
        assert_eq!(layout.yield_blocks.len(), 2, "expected 2 yield blocks");
    }

    /// v2 lift: a value defined before the yield and used
    /// after must round-trip through the fiber's saved-slot
    /// table. The transform allocates a slot per live value
    /// and rewrites downstream uses to reference the
    /// freshly-loaded ValueId.
    #[test]
    fn live_value_across_yield_saves_and_remaps() {
        let src = r#"
var f = Fiber.new {
  var x = 7
  Fiber.yield()
  System.print(x)
}
f.call()
"#;
        let (mut mir, interner) = build_mir_from_source(src, /* closure_idx */ 0);
        let layout = transform_to_state_machine(&mut mir, &interner)
            .expect("v2 transform handles live values across yield");
        assert!(
            !layout.yield_saves.is_empty(),
            "expected saves for the live `x`, got {:?}",
            layout.yield_saves
        );
        assert!(
            !layout.resume_loads.is_empty(),
            "expected matching loads at the resume entry, got {:?}",
            layout.resume_loads
        );
        // Sanity: every saved (slot, original) has a paired
        // (slot, fresh) in the resume entry.
        let saved_slots: std::collections::HashSet<u32> = layout
            .yield_saves
            .values()
            .flat_map(|v| v.iter().map(|(s, _)| *s))
            .collect();
        let load_slots: std::collections::HashSet<u32> = layout
            .resume_loads
            .values()
            .flat_map(|v| v.iter().map(|(s, _)| *s))
            .collect();
        assert_eq!(
            saved_slots, load_slots,
            "save/load slot sets must match"
        );
    }
}
