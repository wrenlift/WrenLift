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
use crate::intern::SymbolId;
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
    /// Per yielding block, the *kind* of suspension. Direct
    /// `Fiber.yield(_)` keeps the existing yield-stamp +
    /// `Return` lowering. Cross-fn tainted Calls get a more
    /// involved sequence (push frame + save args + invoke
    /// callee's poll + check kind + propagate Yield).
    pub block_kinds: HashMap<BlockId, BlockKind>,
    /// Per DirectYield resume block, the original ValueId that
    /// was the suspension Call's result (e.g. `var x =
    /// Fiber.yield(...)` in the source binds `x` to this
    /// ValueId). The transform drops the Call when splitting,
    /// so this ValueId has no MIR-level def afterwards. The
    /// backend rebinds it to the `resume_v` poll-fn parameter
    /// at the resume block's entry — that parameter carries
    /// whatever value `fiber.call(value)` passed back from the
    /// resumer.
    pub direct_yield_results: HashMap<BlockId, ValueId>,
    /// Per CrossFnCall done block, `(slot, call_dst)` for the
    /// suspension Call's result. The cranelift backend's
    /// done-branch lowering writes the runtime `ret` of
    /// `wlift_aot_invoke_sm_method` into `slot` via
    /// `wlift_aot_sm_save_value`; the done block's prologue
    /// loads it back, providing a dominating def for every
    /// downstream use. Without this, val_map.insert(result, ret)
    /// in the done branch only dominates the immediate done
    /// jump — successors reached via tail-duplicated paths or
    /// other CFG join points see an undefined SSA value, and
    /// Cranelift's verifier rejects with "uses value vN from
    /// non-dominating instM".
    pub cross_fn_results: HashMap<BlockId, (u32, ValueId)>,
}

/// What kind of suspension a block represents. AOT lowering
/// branches on this when emitting the pre-Return helpers (for
/// `DirectYield` / `CrossFnCallInit`) or replacing the body
/// entirely (for `CrossFnCallResume`).
#[derive(Debug, Clone)]
pub enum BlockKind {
    /// `Fiber.yield(_)` / `Fiber.suspend()`. Body sets kind=Yield
    /// and returns the yield value directly.
    DirectYield,
    /// First half of a cross-fn call site. Lowering emits the
    /// pre-call setup (advance own state, push child frame,
    /// save args) then jumps to the matching `CrossFnCallResume`
    /// block. Reached only via fall-through from the preceding
    /// state's code; the dispatcher's `br_table` does NOT land
    /// here.
    CrossFnCallInit {
        /// MIR BlockId of the matching `CrossFnCallResume`
        /// (synthetic) block. The init's lowering jumps here
        /// after pre-call setup.
        resume_check_block: BlockId,
        receiver: ValueId,
        args: Vec<ValueId>,
        result: ValueId,
        method_sym: SymbolId,
    },
    /// Second half of a cross-fn call site — synthetic block
    /// the dispatcher's `br_table` lands in on resume from a
    /// yielded child. Lowering emits `invoke_sm_method + peek +
    /// brif`; on Yield returns the propagated value, on Done
    /// pops the child frame and jumps to the post-call block
    /// (`done_block`). Has no MIR-level instructions; the
    /// terminator is set to `ReturnNull` as a placeholder that
    /// the lowering overrides.
    CrossFnCallResume {
        /// MIR BlockId of the post-call block (the original
        /// post-Call instructions + original terminator).
        done_block: BlockId,
        receiver: ValueId,
        args: Vec<ValueId>,
        result: ValueId,
        method_sym: SymbolId,
    },
}

/// Names of methods whose `Call` is a *direct* suspension —
/// `Fiber.yield` / `Fiber.suspend`. These get the simple
/// "stamp kind=Yield, return value" lowering. Cross-fn Calls
/// to other tainted methods route through a different (more
/// involved) sequence.
fn direct_yield_method_names() -> &'static [&'static str] {
    &["yield()", "yield(_)", "suspend()"]
}

/// Classification of a suspension Call. v2-cap3 distinguishes
/// `Fiber.yield(_)` from cross-fn calls into other tainted
/// methods; both are suspension points but get different
/// lowering.
#[derive(Debug, Clone, Copy)]
enum SuspensionKind {
    DirectYield,
    CrossFnCall,
}

/// Inspect a Call instruction and decide whether it's a
/// suspension point and which kind. `tainted_names` is the
/// whole-program transitive yield-method set.
fn classify_call(
    inst: &Instruction,
    interner: &Interner,
    tainted_names: &std::collections::HashSet<String>,
) -> Option<SuspensionKind> {
    let Instruction::Call { method, .. } = inst else {
        return None;
    };
    let name = interner.resolve(*method);
    if direct_yield_method_names().contains(&name) {
        return Some(SuspensionKind::DirectYield);
    }
    if tainted_names.contains(name) {
        return Some(SuspensionKind::CrossFnCall);
    }
    None
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
    LiveValueAcrossSuspension { block: BlockId, value: ValueId },
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

/// A single yield site, recorded during the planning pass before
/// any liveness, slot allocation, or cloning runs. Holds the post-
/// split block ids plus the call's metadata (yield value, cross-fn
/// receiver/args/result, etc.). Phase 2 fills in `saves` / `loads`
/// after liveness; phase 3 uses both for the per-clone remap.
#[derive(Debug, Clone)]
struct YieldPlan {
    /// Block that ended in the suspension Call. After Phase 1's
    /// split, terminator is `Return(yield_value)` (DirectYield) or
    /// `Branch(call_check_block)` (CrossFnCall).
    yielding_block: BlockId,
    /// Block the dispatcher's `br_table` lands on for state == K.
    /// DirectYield: the post-split tail. CrossFnCall: the synthetic
    /// `call_check_block`.
    resume_block: BlockId,
    /// 1-indexed state id (state 0 = bb0 / initial entry).
    state_id: u32,
    /// Per-kind metadata captured before the split.
    kind: YieldPlanKind,
    /// Live-across saves (filled in Phase 2). `(slot, orig_vid)`
    /// pairs; the slot is unique within this function.
    saves: Vec<(u32, ValueId)>,
    /// Paired loads for the resume entry (filled in Phase 2).
    /// `(slot, fresh_vid)`; downstream uses of `orig_vid` get
    /// remapped to `fresh_vid` along paths through this yield.
    loads: Vec<(u32, ValueId)>,
}

#[derive(Debug, Clone)]
enum YieldPlanKind {
    DirectYield {
        /// The value `Fiber.yield(_)` returns to the caller.
        /// Kept in metadata for completeness; the planning pass
        /// has already written it as the yielding block's
        /// `Return(yield_value)` terminator.
        #[allow(dead_code)]
        yield_value: ValueId,
        /// The Call's destination ValueId (i.e. `var x = Fiber.yield(...)`
        /// binds `x` here). Backend rebinds this to `resume_v` at the
        /// resume block's entry.
        result: ValueId,
    },
    CrossFnCall {
        /// The synthetic call-check block (also the br_table target).
        call_check_block: BlockId,
        /// The post-call tail block — call_check's MIR Branch target.
        done_block: BlockId,
        receiver: ValueId,
        args: Vec<ValueId>,
        result: ValueId,
        method_sym: SymbolId,
    },
}

/// Run the transform. Mutates `mir` in place: blocks containing
/// suspension Calls get split; new "resume" blocks are appended;
/// yield Calls are removed and replaced with a `Return(value)`
/// where `value` is the yield argument (the value the caller
/// observes). The returned `StateMachineLayout` carries the
/// per-state mapping the AOT lowering needs.
///
/// The transform is a four-phase pipeline:
///
/// 1. **Plan**: walk MIR, find every suspension Call. Split the
///    block at each call. For CrossFnCall, allocate the synthetic
///    call_check block and re-route. Record a `YieldPlan` per
///    site with the post-split block ids + per-kind metadata. No
///    liveness, no slot allocation, no remap, no cloning.
///
/// 2. **Liveness + slot allocation**: now that every block split
///    is done, compute `live_in` at each resume block. The live
///    set = saved values for that yield. Allocate slots + fresh
///    load vids per yield.
///
/// 3. **Path-sensitive cloning**: forward dataflow over the post-
///    split MIR computes the set of distinct reaching substitution
///    maps at each block. A block reached by N distinct subs maps
///    is cloned N-1 times; each clone gets its own remap applied.
///    Predecessor edges are redirected to the matching clone.
///
/// 4. **Layout metadata mirror + orphan prune**: populate
///    `StateMachineLayout` for the original yielding/resume blocks
///    and mirror onto every clone. Unreachable blocks (originals
///    whose predecessors all redirected away) get replaced with
///    `Unreachable` to remove stale ValueId references.
///
/// A function with zero suspension Calls returns a layout with a
/// single `resume_entries[0]` and no yield blocks — the AOT
/// lowering can either skip the state-machine prologue entirely
/// or emit it harmlessly (the dispatch always lands on state 0).
pub fn transform_to_state_machine(
    mir: &mut MirFunction,
    interner: &Interner,
    // Whole-program tainted method-name set. Calls to methods
    // in this set are treated as cross-fn suspensions in addition
    // to direct `Fiber.yield(_)`. Callers can pass an empty set
    // and only direct yields will be recognised as suspensions.
    tainted_names: &std::collections::HashSet<String>,
) -> Result<StateMachineLayout, StateMachineError> {
    // ===== Phase 1: PLAN =====
    //
    // Walk MIR; find every suspension Call; split at each call;
    // record a YieldPlan per site. No liveness, no slot allocation,
    // no remap, no cloning yet. After this phase, the MIR has the
    // final CFG structure (modulo the per-clone duplication done
    // in Phase 3) and every YieldPlan knows its yielding_block
    // and resume_block.
    let mut plans: Vec<YieldPlan> = Vec::new();
    plan_yields(mir, interner, tainted_names, &mut plans)?;

    // ===== Phase 2: LIVENESS + SLOT ALLOCATION =====
    //
    // For each plan's resume_block, compute live_in on the now-
    // complete CFG. Live values become saves; allocate slots and
    // fresh load vids. Liveness must run AFTER every split so a
    // value live across multiple yields is detected at each site.
    //
    // Slot numbering: per-function pool starting at `arity` so
    // SM-method arg slots (0..arity) don't collide with save slots.
    let mut next_slot: u32 = mir.arity as u32;
    compute_saves_loads(mir, &mut plans, &mut next_slot);

    // ===== Phase 3: PATH-SENSITIVE CLONING =====
    //
    // Forward dataflow over the post-split MIR computes the set
    // of distinct reaching substitution maps at each block. The
    // substitution map at block B records, for every saved value
    // v, which fresh load vid v should be remapped to on the path
    // that reaches B. Blocks with N > 1 distinct reaching maps
    // are cloned N times (one clone per map); predecessors are
    // redirected to the matching clone.
    //
    // Returns the `clone_index`: for each block B, a vec of
    // `(subs_key, clone_id)` pairs. `clone_index[B][0].1` is the
    // primary (= original block, kept reachable on the first
    // matching path). For blocks with only one reaching subs,
    // clone_index[B] has a single entry.
    let clone_index = clone_per_reaching_subs(mir, &plans);

    // ===== Phase 4: BUILD LAYOUT (mirror onto clones) =====
    //
    // Populate StateMachineLayout entries for every yielding /
    // resume block, then mirror onto each clone with the clone's
    // subs applied to saves / cross-fn results / direct-yield
    // results.
    let mut layout = StateMachineLayout {
        resume_entries: vec![mir.entry_block()],
        yield_blocks: HashMap::new(),
        yield_saves: HashMap::new(),
        resume_loads: HashMap::new(),
        block_kinds: HashMap::new(),
        direct_yield_results: HashMap::new(),
        cross_fn_results: HashMap::new(),
    };
    build_layout(mir, &plans, &clone_index, &mut layout);

    // ===== Phase 5: ORPHAN PRUNE (below) =====

    // Post-pass: prune blocks unreachable from any resume entry.
    //
    // The clone walker can leave orphan blocks behind in two
    // situations:
    //
    // 1. A block was cloned by yield N (with yield N's remap baked
    //    in), but no edge actually reaches the clone — yield N's
    //    redirect chain stopped at a different intermediate. The
    //    clone references yield N's fresh load vids, which are
    //    bound only at yield N's resume entry's prologue. Since
    //    the orphan is unreachable, its uses are never satisfied —
    //    but Cranelift's RPO walker still lowers it, and the
    //    verifier rejects with "undefined value vN in terminator".
    //
    // 2. The original yielding block has been split (its terminator
    //    is now `Return(yield_value)`), so its post-yield successors
    //    are unreachable through the original block. They survive
    //    only as the parent of any clones built off them. If the
    //    cloning process didn't pick up every path, the original
    //    post-yield block stays in `mir.blocks` with no live
    //    predecessors.
    //
    // Both cases produce the same symptom: orphan blocks with stale
    // ValueId references. Replacing their bodies with a single
    // `Unreachable` terminator removes all such references — the
    // Cranelift lowering emits a `trap` for `Terminator::Unreachable`,
    // which doesn't consult val_map, so the verifier is satisfied.
    //
    // We compute reachability forward from every resume entry (the
    // dispatcher's br_table targets are the only entry points at
    // runtime — `bb0` itself is `resume_entries[0]`).
    {
        use std::collections::HashSet;
        let mut reachable: HashSet<BlockId> = HashSet::new();
        let mut stack: Vec<BlockId> = layout.resume_entries.clone();
        let n = mir.blocks.len();
        while let Some(b) = stack.pop() {
            if !reachable.insert(b) {
                continue;
            }
            if (b.0 as usize) < n {
                for s in mir.blocks[b.0 as usize].terminator.successors() {
                    stack.push(s);
                }
            }
        }
        for (i, blk) in mir.blocks.iter_mut().enumerate() {
            let bid = BlockId(i as u32);
            if reachable.contains(&bid) {
                continue;
            }
            // Orphan. Replace body + terminator with a trap-equivalent
            // Unreachable, drop any layout metadata so the lowering
            // doesn't try to consult it.
            blk.instructions.clear();
            blk.params.clear();
            blk.terminator = Terminator::Unreachable;
            layout.yield_blocks.remove(&bid);
            layout.yield_saves.remove(&bid);
            layout.resume_loads.remove(&bid);
            layout.block_kinds.remove(&bid);
            layout.direct_yield_results.remove(&bid);
            layout.cross_fn_results.remove(&bid);
        }
    }

    Ok(layout)
}

// ---------------------------------------------------------------------------
// Phase 1 — PLAN
// ---------------------------------------------------------------------------

/// Walk the MIR; find every suspension Call; split at each call.
/// For CrossFnCall, also allocate the synthetic call_check block
/// and re-route the yielding block's terminator through it. No
/// liveness, no slot allocation, no remap, no cloning.
///
/// After this phase, every yielding block has a placeholder
/// terminator (Return(yield_value) for DirectYield, Branch(call_check)
/// for CrossFnCall) and the post-split tail lives in `resume_block`
/// (DirectYield) or `done_block` (CrossFnCall, reached via call_check).
fn plan_yields(
    mir: &mut MirFunction,
    interner: &Interner,
    tainted_names: &std::collections::HashSet<String>,
    plans: &mut Vec<YieldPlan>,
) -> Result<(), StateMachineError> {
    let mut planned_yields: std::collections::HashSet<BlockId> = std::collections::HashSet::new();

    loop {
        // Find the next unplanned suspension in any block reachable
        // from bb0 OR any prior resume_block. After splitting a
        // yielding block, its terminator is Return / Branch(call_check),
        // so the post-split tail isn't reachable via bb0 alone — it
        // becomes a resume_block (or downstream of one), which is
        // entered at runtime via the dispatcher's br_table. Seeding
        // reachability from every prior resume_block ensures we find
        // suspensions in those tails.
        let reachable: std::collections::HashSet<BlockId> = {
            use std::collections::HashSet;
            let mut visited: HashSet<BlockId> = HashSet::new();
            let mut stack: Vec<BlockId> = vec![mir.entry_block()];
            for plan in plans.iter() {
                stack.push(plan.resume_block);
            }
            let n = mir.blocks.len();
            while let Some(b) = stack.pop() {
                if !visited.insert(b) {
                    continue;
                }
                if (b.0 as usize) < n {
                    for s in mir.blocks[b.0 as usize].terminator.successors() {
                        stack.push(s);
                    }
                }
            }
            visited
        };
        let mut found: Option<(BlockId, usize, SuspensionKind)> = None;
        'outer: for block in &mir.blocks {
            if planned_yields.contains(&block.id) {
                continue;
            }
            if !reachable.contains(&block.id) {
                continue;
            }
            for (i, (_dst, inst)) in block.instructions.iter().enumerate() {
                if let Some(kind) = classify_call(inst, interner, tainted_names) {
                    found = Some((block.id, i, kind));
                    break 'outer;
                }
            }
        }
        let Some((blk_id, inst_idx, susp_kind)) = found else {
            break;
        };

        // Capture the call's metadata before mutating.
        let (yield_value, direct_result, cross_meta): (
            ValueId,
            Option<ValueId>,
            Option<(ValueId, Vec<ValueId>, ValueId, SymbolId)>,
        ) = match susp_kind {
            SuspensionKind::DirectYield => {
                let (call_dst, arg) = match &mir.blocks[blk_id.0 as usize].instructions[inst_idx] {
                    (dst, Instruction::Call { args, .. }) => (*dst, args.first().copied()),
                    _ => unreachable!("classify_call only returns Some for Call"),
                };
                let yield_value = if let Some(a) = arg {
                    a
                } else {
                    // `Fiber.yield()` with no args: synth a ConstNull
                    // just before the call so the Return has an operand.
                    let new_vid = mir.new_value();
                    mir.blocks[blk_id.0 as usize]
                        .instructions
                        .insert(inst_idx, (new_vid, Instruction::ConstNull));
                    new_vid
                };
                (yield_value, Some(call_dst), None)
            }
            SuspensionKind::CrossFnCall => {
                let (call_dst, call_inst) =
                    mir.blocks[blk_id.0 as usize].instructions[inst_idx].clone();
                let Instruction::Call {
                    receiver,
                    method,
                    args,
                    ..
                } = call_inst
                else {
                    unreachable!("classify_call only returns Some for Call");
                };
                (call_dst, None, Some((receiver, args, call_dst, method)))
            }
        };

        // Re-find inst_idx — the ConstNull insert above may have
        // shifted positions.
        let inst_idx = {
            let blk = &mir.blocks[blk_id.0 as usize];
            blk.instructions
                .iter()
                .position(|(_, inst)| classify_call(inst, interner, tainted_names).is_some())
                .expect("suspension still present after ConstNull insert")
        };

        // Split: allocate a new block for the post-call tail.
        let post_call_block = mir.new_block();
        let original_terminator = {
            let blk = &mut mir.blocks[blk_id.0 as usize];
            let mut tail = blk.instructions.split_off(inst_idx);
            tail.remove(0); // drop the suspension Call
            let orig_term =
                std::mem::replace(&mut blk.terminator, Terminator::Unreachable /* placeholder */);
            let new_block = &mut mir.blocks[post_call_block.0 as usize];
            new_block.instructions = tail;
            new_block.terminator = orig_term.clone();
            orig_term
        };
        let _ = original_terminator;

        // Per-kind wiring.
        match (susp_kind, cross_meta) {
            (SuspensionKind::DirectYield, _) => {
                // yielding_block: Return(yield_value).
                mir.blocks[blk_id.0 as usize].terminator = Terminator::Return(yield_value);
                let state_id = (plans.len() + 1) as u32;
                plans.push(YieldPlan {
                    yielding_block: blk_id,
                    resume_block: post_call_block,
                    state_id,
                    kind: YieldPlanKind::DirectYield {
                        yield_value,
                        result: direct_result
                            .expect("DirectYield always has a result vid"),
                    },
                    saves: Vec::new(),
                    loads: Vec::new(),
                });
            }
            (SuspensionKind::CrossFnCall, Some((receiver, args, result, method_sym))) => {
                // Allocate the synthetic call_check block (the
                // br_table target / dispatcher resume entry).
                let call_check_block = mir.new_block();
                // yielding_block: Branch(call_check_block).
                mir.blocks[blk_id.0 as usize].terminator = Terminator::Branch {
                    target: call_check_block,
                    args: Vec::new(),
                };
                // call_check_block: Branch(done_block).
                // The MIR Branch is just a structural successor so
                // rpo orders call_check before done_block. The
                // cranelift lowering of CrossFnCallResume overrides
                // both instructions and terminator.
                mir.blocks[call_check_block.0 as usize].terminator = Terminator::Branch {
                    target: post_call_block,
                    args: Vec::new(),
                };
                let state_id = (plans.len() + 1) as u32;
                plans.push(YieldPlan {
                    yielding_block: blk_id,
                    resume_block: call_check_block,
                    state_id,
                    kind: YieldPlanKind::CrossFnCall {
                        call_check_block,
                        done_block: post_call_block,
                        receiver,
                        args,
                        result,
                        method_sym,
                    },
                    saves: Vec::new(),
                    loads: Vec::new(),
                });
            }
            _ => unreachable!("classify_call invariants kept susp_kind and meta in sync"),
        }
        planned_yields.insert(blk_id);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2 — LIVENESS + SLOT ALLOCATION
// ---------------------------------------------------------------------------

/// For each yield's resume_block, compute live_in on the now-
/// complete CFG. The live set, filtered by "defined pre-yield"
/// (forward-reach from bb0 + every prior resume_block), is the
/// saved set for that yield. Allocate slots + fresh load vids.
fn compute_saves_loads(mir: &mut MirFunction, plans: &mut [YieldPlan], next_slot: &mut u32) {
    // Run liveness once over the whole function — we need live_in
    // for every resume_block. The per-block live_in is the
    // standard backward-dataflow fixed point.
    //
    // For yielding blocks, the suspension Call was dropped during
    // Phase 1's split — its operands no longer appear in the MIR.
    // But the runtime save_arg sequence (at CrossFnCallInit's
    // lowering) reads val_map[receiver] + val_map[args[i]], so
    // those vids ARE used at the yielding block. Without feeding
    // them back into the liveness pass, a value flowing from
    // pre-yield-K into yield K+1's call args goes uncounted —
    // saves[K] misses it, downstream use sees an undefined
    // val_map entry. Surfaces as App.serve_-shape closures'
    // "uses value vN from non-dominating instM" verifier reject.
    let mut extra_uses: HashMap<BlockId, Vec<ValueId>> = HashMap::new();
    for plan in plans.iter() {
        if let YieldPlanKind::CrossFnCall { receiver, args, .. } = &plan.kind {
            let entry = extra_uses.entry(plan.yielding_block).or_default();
            entry.push(*receiver);
            entry.extend(args.iter().copied());
        }
    }
    let all_live_in = compute_all_live_in_with_extra(mir, &extra_uses);

    // For "pre-yield defs" filter: a value v is "pre-yield"
    // (savable) only if its def is reachable from bb0 in the
    // post-split CFG (i.e., it exists before the yield runs at
    // runtime). The simplest sound choice is to take v defined
    // in ANY block — the dataflow in Phase 3 redirects clones
    // such that uses only see defs that actually dominate them.
    // But we need to be precise enough that we don't try to save
    // the result of a Call we just dropped, or a fresh resume-load
    // vid that won't exist when the save runs.
    //
    // Concretely, exclude:
    //  - The result vid of the CURRENT yield (no pre-yield def
    //    for that: the call producing it was dropped on split).
    //  - Already-planned cross-fn call results (lowering rebinds
    //    them via val_map at the matching Resume block, not via
    //    MIR-level defs — they're not savable as MIR ValueIds).
    //    BUT: they ARE valid SSA values at lowering time, so a
    //    later yield CAN save them; the saved runtime value is
    //    whatever the matching Resume bound. Don't exclude them.
    //  - Block params: ARE valid SSA values; CAN be saved. Don't
    //    exclude.
    //
    // The pre-yield-def set is computed by forward-reach from bb0
    // collecting every block's params + instruction defs. The
    // result includes block-params globally (every block's params
    // count as defs regardless of which subset of paths reach
    // them — Phase 3's per-path dataflow handles the per-path
    // visibility).
    // Per-plan pre_yield_defs: vids that are definitely defined
    // by the time plan K's yielding_block runs.
    //
    // Vids defined in a block reachable BACKWARD from K's
    // yielding_block (via CFG predecessors) ARE pre-yield: they're
    // computed on some path leading into yield K. Block params
    // also count (the predecessor's branch arg defines them).
    //
    // Cross-fn / direct-yield results from OTHER plans are pre-
    // yield ONLY IF that plan's resume_block is on a path from
    // bb0 to K's yielding_block (i.e., the other yield has
    // already fired by the time K fires). Without this scoping,
    // a later yield's result vid gets pulled into an earlier
    // yield's save set — the save reads val_map[result] which
    // isn't bound until the later yield's Resume, producing
    // garbage at runtime or a verifier reject.
    let mut predecessors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for blk in mir.blocks.iter() {
        for s in blk.terminator.successors() {
            predecessors.entry(s).or_default().push(blk.id);
        }
    }
    let plan_idx_by_resume: HashMap<BlockId, usize> = plans
        .iter()
        .enumerate()
        .map(|(i, p)| (p.resume_block, i))
        .collect();
    let plan_kinds: Vec<YieldPlanKind> = plans.iter().map(|p| p.kind.clone()).collect();
    let plan_yielding_blocks: Vec<BlockId> = plans.iter().map(|p| p.yielding_block).collect();

    for (k, plan) in plans.iter_mut().enumerate() {
        // Backward CFG walk from this plan's yielding_block.
        let backward: std::collections::HashSet<BlockId> = {
            use std::collections::HashSet;
            let mut visited: HashSet<BlockId> = HashSet::new();
            let mut stack = vec![plan.yielding_block];
            while let Some(b) = stack.pop() {
                if !visited.insert(b) {
                    continue;
                }
                if let Some(preds) = predecessors.get(&b) {
                    for p in preds {
                        stack.push(*p);
                    }
                }
            }
            visited
        };
        // Defs from the backward-reachable region.
        let mut defs: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
        for blk in mir.blocks.iter() {
            if !backward.contains(&blk.id) {
                continue;
            }
            for (vid, _) in &blk.params {
                defs.insert(*vid);
            }
            for (vid, _) in &blk.instructions {
                defs.insert(*vid);
            }
        }
        // Block params of every block — block params are
        // SSA-introduced at entry and are pre-yield for any
        // block that includes them via reachability.
        // Result vids of other plans whose resume_block sits on
        // a path from bb0 reaching this plan's yielding_block.
        // We approximate this by: result is pre-yield iff the
        // other plan's resume_block is in `backward`.
        for (j, kind) in plan_kinds.iter().enumerate() {
            if j == k {
                continue;
            }
            let resume = plans_resume(&plan_idx_by_resume, j).unwrap_or(plan_yielding_blocks[j]);
            if !backward.contains(&resume) {
                continue;
            }
            match kind {
                YieldPlanKind::DirectYield { result, .. } => {
                    defs.insert(*result);
                }
                YieldPlanKind::CrossFnCall { result, .. } => {
                    defs.insert(*result);
                }
            }
        }
        let pre_yield_defs = defs;

        // The result vid of the CURRENT yield isn't in the saved
        // set: there's no pre-yield MIR-level def (the call was
        // dropped during split). The Cranelift lowering rebinds
        // it at the resume's prologue (via resume_v param for
        // DirectYield, via runtime ret for CrossFnCall).
        let exclude_result = match &plan.kind {
            YieldPlanKind::DirectYield { result, .. } => Some(*result),
            YieldPlanKind::CrossFnCall { result, .. } => Some(*result),
        };

        let live_in_at_resume = all_live_in
            .get(&plan.resume_block)
            .cloned()
            .unwrap_or_default();

        let mut live: Vec<ValueId> = live_in_at_resume
            .into_iter()
            .filter(|v| Some(*v) != exclude_result)
            .filter(|v| pre_yield_defs.contains(v))
            .collect();
        live.sort_by_key(|v| v.0);

        // Allocate one slot + fresh-load vid per live value.
        let saves: Vec<(u32, ValueId)> = live
            .iter()
            .map(|v| {
                let slot = *next_slot;
                *next_slot += 1;
                (slot, *v)
            })
            .collect();
        let loads: Vec<(u32, ValueId)> = saves
            .iter()
            .map(|(slot, _)| (*slot, mir.new_value()))
            .collect();
        plan.saves = saves;
        plan.loads = loads;
    }
}

/// Helper: look up plan j's resume_block via the index map.
fn plans_resume(
    plan_idx_by_resume: &HashMap<BlockId, usize>,
    j: usize,
) -> Option<BlockId> {
    plan_idx_by_resume
        .iter()
        .find(|(_, idx)| **idx == j)
        .map(|(bid, _)| *bid)
}

/// Compute live_in for every block, accepting an `extra_uses`
/// table for vids referenced by metadata (not by MIR instructions).
/// CrossFnCallInit's receiver/args fall in this category — the Call
/// they came from was dropped during Phase 1's split, so liveness
/// over the post-split MIR alone misses them.
fn compute_all_live_in_with_extra(
    mir: &MirFunction,
    extra_uses: &HashMap<BlockId, Vec<ValueId>>,
) -> std::collections::HashMap<BlockId, std::collections::HashSet<ValueId>> {
    use std::collections::{HashMap as HMap, HashSet};
    let n = mir.blocks.len();
    let mut block_uses: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];
    let mut block_defs: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];
    let mut successors: Vec<Vec<BlockId>> = vec![Vec::new(); n];

    for (i, blk) in mir.blocks.iter().enumerate() {
        let mut local_defs = HashSet::new();
        for (vid, _) in &blk.params {
            local_defs.insert(*vid);
        }
        for (vid, inst) in &blk.instructions {
            for u in collect_uses(inst) {
                if !local_defs.contains(&u) {
                    block_uses[i].insert(u);
                }
            }
            local_defs.insert(*vid);
        }
        for u in collect_terminator_uses(&blk.terminator) {
            if !local_defs.contains(&u) {
                block_uses[i].insert(u);
            }
        }
        if let Some(extra) = extra_uses.get(&blk.id) {
            for u in extra {
                if !local_defs.contains(u) {
                    block_uses[i].insert(*u);
                }
            }
        }
        block_defs[i] = local_defs;
        successors[i] = blk.terminator.successors();
    }

    let mut live_in: Vec<HashSet<ValueId>> = vec![HashSet::new(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for i in (0..n).rev() {
            let mut live_out: HashSet<ValueId> = HashSet::new();
            for succ in &successors[i] {
                let s = succ.0 as usize;
                if s < n {
                    for v in &live_in[s] {
                        live_out.insert(*v);
                    }
                }
            }
            let mut new_live_in = block_uses[i].clone();
            for v in &live_out {
                if !block_defs[i].contains(v) {
                    new_live_in.insert(*v);
                }
            }
            if new_live_in != live_in[i] {
                live_in[i] = new_live_in;
                changed = true;
            }
        }
    }

    let mut out: HMap<BlockId, HashSet<ValueId>> = HMap::new();
    for (i, set) in live_in.into_iter().enumerate() {
        out.insert(BlockId(i as u32), set);
    }
    out
}

// ---------------------------------------------------------------------------
// Phase 3 — PATH-SENSITIVE CLONING
// ---------------------------------------------------------------------------

/// A canonical sorted-Vec form of `HashMap<ValueId, ValueId>` so
/// it can be used as a HashMap key. Each entry: (orig_vid,
/// fresh_load_vid) for some yield's saved value.
type SubsKey = Vec<(ValueId, ValueId)>;

fn subs_to_key(subs: &HashMap<ValueId, ValueId>) -> SubsKey {
    let mut v: SubsKey = subs.iter().map(|(k, v)| (*k, *v)).collect();
    v.sort_by_key(|(k, _)| k.0);
    v
}

fn key_to_subs(key: &SubsKey) -> HashMap<ValueId, ValueId> {
    key.iter().copied().collect()
}

/// Forward dataflow + per-(block, subs) cloning.
///
/// Returns a per-block index: for each original block B,
/// `clone_index[B]` is a Vec of (subs_key, clone_block_id). One
/// entry per distinct reaching subs. Index 0 is the "primary" —
/// the original block kept in place with its operands remapped
/// per that subs (or, if subs is empty, no remap applied).
fn clone_per_reaching_subs(
    mir: &mut MirFunction,
    plans: &[YieldPlan],
) -> HashMap<BlockId, Vec<(SubsKey, BlockId)>> {
    use std::collections::{HashMap as HMap, HashSet};

    // -- Build per-yield "subs delta" applied at the resume_block.
    // After the resume_block's loads, the subs becomes K's loads
    // (overriding any incoming subs for vids in K's saves).
    let mut yield_loads_subs: HMap<BlockId, HashMap<ValueId, ValueId>> = HMap::new();
    let mut yielding_block_of_plan: HMap<BlockId, usize> = HMap::new(); // yielding_block -> plan idx
    let mut resume_block_of_plan: HMap<BlockId, usize> = HMap::new(); // resume_block -> plan idx
    for (idx, plan) in plans.iter().enumerate() {
        let mut loads_subs: HashMap<ValueId, ValueId> = HashMap::new();
        for (i, (_slot, orig)) in plan.saves.iter().enumerate() {
            let (_slot2, fresh) = plan.loads[i];
            loads_subs.insert(*orig, fresh);
        }
        yield_loads_subs.insert(plan.resume_block, loads_subs);
        yielding_block_of_plan.insert(plan.yielding_block, idx);
        resume_block_of_plan.insert(plan.resume_block, idx);
    }

    // -- Forward dataflow: per-block set of reaching subs keys.
    //
    // Three kinds of blocks need special handling:
    //
    // 1. YIELDING blocks (terminator is Return for DirectYield or
    //    Branch(call_check) for CrossFnCall). The MIR successor —
    //    if any — is a STRUCTURAL marker for rpo ordering, NOT a
    //    runtime edge. At runtime, control leaves the yielding
    //    block via "stamp kind=Yield + return"; the resume_block
    //    is re-entered via the dispatcher's br_table on the next
    //    fiber.call. So subs from yielding blocks do NOT propagate
    //    to MIR successors.
    //
    // 2. RESUME blocks (= plan.resume_block: post-split tail for
    //    DirectYield, call_check for CrossFnCall). Entered ONLY via
    //    the dispatcher's br_table (initial entry = empty subs).
    //    The lowering applies K's loads at the resume block's
    //    prologue. So outgoing subs from a resume block = K's loads
    //    (subs reset). reaching_subs[R] is set to {empty} (the
    //    dispatcher's entry, before loads); we don't merge in any
    //    MIR-level reaching subs.
    //
    // 3. ORDINARY blocks: propagate incoming subs unchanged.
    //
    // Block params: NOT treated as subs-resetters; they're shadowing
    // saved values within the block's body but the per-clone remap
    // step is correct because remap_instruction_uses only renames
    // operands matching the subs key — block params have their own
    // SSA defs and aren't touched by the remap.
    let mut reaching_subs: HMap<BlockId, HashSet<SubsKey>> = HMap::new();
    // Seed bb0 (initial entry, empty subs).
    let mut worklist: Vec<(BlockId, SubsKey)> = vec![(mir.entry_block(), Vec::new())];
    // Seed every resume_block with {} (dispatcher entry, before
    // K's loads apply).
    for plan in plans {
        worklist.push((plan.resume_block, Vec::new()));
    }
    while let Some((blk, in_key)) = worklist.pop() {
        // Block params SHADOW incoming subs for matching keys.
        // The predecessor's branch arg re-binds the vid for this
        // block's scope, so within-block uses + downstream
        // propagation should not still treat v as the load's
        // fresh vid. Without this clearing, a loop header reached
        // both from the original back-edge (subs=empty) and from
        // the cloned region (subs={v1→v19}) ends up with two
        // distinct reaching subs and gets cloned — even though
        // bb1's param v1 SSA-shadows both and there is no
        // downstream observable difference. The doubled clone
        // collides with val_map[v1] at lowering time.
        let block_params = &mir.blocks[blk.0 as usize].params;
        let effective_in_key: SubsKey = if block_params.is_empty() {
            in_key
        } else {
            let mut m = key_to_subs(&in_key);
            for (p, _) in block_params {
                m.remove(p);
            }
            subs_to_key(&m)
        };
        let entry = reaching_subs.entry(blk).or_default();
        if !entry.insert(effective_in_key.clone()) {
            continue;
        }
        // YIELDING block: do not propagate subs to MIR successors.
        if yielding_block_of_plan.contains_key(&blk) {
            continue;
        }
        // Compute outgoing subs.
        let out_subs: SubsKey = if let Some(loads_subs) = yield_loads_subs.get(&blk) {
            // resume_block: outgoing = K's loads only (subs reset
            // at the resume's prologue).
            let mut combined: HashMap<ValueId, ValueId> = HashMap::new();
            for (k, v) in loads_subs.iter() {
                combined.insert(*k, *v);
            }
            subs_to_key(&combined)
        } else {
            effective_in_key.clone()
        };
        let succs = mir.blocks[blk.0 as usize].terminator.successors();
        for s in succs {
            worklist.push((s, out_subs.clone()));
        }
    }

    // -- Decide cloning. For each block with N distinct reaching
    // subs, allocate (N-1) clones; the original keeps the FIRST
    // subs in sorted order as its applied remap (or empty subs if
    // present in the set).
    let mut clone_index: HashMap<BlockId, Vec<(SubsKey, BlockId)>> = HashMap::new();
    let original_block_count = mir.blocks.len();
    for blk_idx in 0..original_block_count {
        let bid = BlockId(blk_idx as u32);
        let Some(keys_set) = reaching_subs.get(&bid).cloned() else {
            // Unreachable block — skip; orphan-prune handles it.
            continue;
        };
        let mut keys: Vec<SubsKey> = keys_set.into_iter().collect();
        // Sort: empty first, then by (key length, key content) for
        // determinism.
        keys.sort_by(|a, b| match a.len().cmp(&b.len()) {
            std::cmp::Ordering::Equal => a.cmp(b),
            other => other,
        });
        let mut entries: Vec<(SubsKey, BlockId)> = Vec::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            let clone_bid = if i == 0 {
                bid
            } else {
                mir.new_block()
            };
            entries.push((key.clone(), clone_bid));
        }
        clone_index.insert(bid, entries);
    }

    // -- Populate clone bodies. Each clone gets a copy of the
    // original's params + instructions + terminator with the
    // applicable subs map applied to operands.
    //
    // For a RESUME block, the applied subs is K's loads — the
    // lowering emits `wlift_aot_sm_load_value(slot)` calls at the
    // resume's prologue and binds val_map[fresh_load_vid] = ret,
    // so any pre-yield use of an original saved vid downstream of
    // the prologue must be rewritten to use the fresh load vid.
    // The dataflow's `reaching_subs[resume]` is the entry-side
    // (dispatcher = empty) subs, which we IGNORE here in favour of
    // K's loads.
    //
    // For ORDINARY blocks, applied subs = reaching subs (one of
    // the clone-index entries for that block).
    for (orig_bid, entries) in clone_index.iter() {
        let (params, insts, term) = {
            let b = mir.block(*orig_bid);
            (b.params.clone(), b.instructions.clone(), b.terminator.clone())
        };
        // Block params shadow saved values for the block's body.
        // A subs entry whose key matches a block param represents
        // a saved value that the predecessor's branch arg has
        // already re-bound — uses inside the block see the param,
        // not the load. Drop those keys before applying the subs.
        let param_vids: std::collections::HashSet<ValueId> =
            params.iter().map(|(v, _)| *v).collect();
        let filter_subs = |s: &HashMap<ValueId, ValueId>| -> HashMap<ValueId, ValueId> {
            s.iter()
                .filter(|(k, _)| !param_vids.contains(k))
                .map(|(k, v)| (*k, *v))
                .collect()
        };

        // RESUME blocks have only one clone-entry (the dispatcher
        // arrival), so we don't iterate over `entries` for them.
        let is_resume = yield_loads_subs.contains_key(orig_bid);
        if is_resume {
            // Apply K's loads subs to the (single) resume block.
            let raw_subs = yield_loads_subs.get(orig_bid).cloned().unwrap_or_default();
            let subs = filter_subs(&raw_subs);
            if !subs.is_empty() {
                let blk = mir.block_mut(*orig_bid);
                for (_, inst) in &mut blk.instructions {
                    remap_instruction_uses(inst, &subs);
                }
                remap_terminator_uses(&mut blk.terminator, &subs);
            }
            continue;
        }
        for (key, clone_bid) in entries.iter() {
            let raw_subs = key_to_subs(key);
            let subs = filter_subs(&raw_subs);
            if *clone_bid == *orig_bid {
                if !subs.is_empty() {
                    let blk = mir.block_mut(*clone_bid);
                    for (_, inst) in &mut blk.instructions {
                        remap_instruction_uses(inst, &subs);
                    }
                    remap_terminator_uses(&mut blk.terminator, &subs);
                }
                continue;
            }
            let mut clone_insts = insts.clone();
            let mut clone_term = term.clone();
            if !subs.is_empty() {
                for (_, inst) in &mut clone_insts {
                    remap_instruction_uses(inst, &subs);
                }
                remap_terminator_uses(&mut clone_term, &subs);
            }
            let blk = mir.block_mut(*clone_bid);
            blk.params = params.clone();
            blk.instructions = clone_insts;
            blk.terminator = clone_term;
        }
    }

    // -- Redirect terminator successors so each predecessor's
    // outgoing edge points to the matching clone.
    //
    // For each block B' (primary or clone) with terminator
    // successor T:
    //   outgoing_subs from B' = (subs applied to B') if B' is not
    //     a resume_block, else (yield K's loads).
    //   pick the clone of T whose key matches outgoing_subs.
    //
    // The keys in clone_index[T] cover every reachable outgoing
    // subs from any predecessor, so the match always finds one
    // (assuming the dataflow was sound and the input MIR was
    // structurally connected from bb0).
    let mut redirects: Vec<(BlockId, Terminator)> = Vec::new();
    for (orig_bid, entries) in clone_index.iter() {
        for (subs_key, clone_bid) in entries.iter() {
            // Compute this clone's outgoing subs — same rules as
            // the dataflow propagation step.
            let mut out_subs: HashMap<ValueId, ValueId> = if yield_loads_subs
                .contains_key(orig_bid)
            {
                yield_loads_subs.get(orig_bid).cloned().unwrap_or_default()
            } else {
                key_to_subs(subs_key)
            };
            // Block params shadow saved values; drop from outgoing.
            let block_params = &mir.blocks[orig_bid.0 as usize].params;
            for (p, _) in block_params {
                out_subs.remove(p);
            }
            let out_key = subs_to_key(&out_subs);
            // Redirect successors of *clone_bid.
            let mut term = mir.block(*clone_bid).terminator.clone();
            redirect_term_to_clone(&mut term, &out_key, &clone_index);
            redirects.push((*clone_bid, term));
        }
    }
    for (bid, term) in redirects {
        mir.block_mut(bid).terminator = term;
    }

    clone_index
}


fn redirect_term_to_clone(
    term: &mut Terminator,
    out_key: &SubsKey,
    clone_index: &HashMap<BlockId, Vec<(SubsKey, BlockId)>>,
) {
    let pick = |orig: BlockId| -> BlockId {
        let Some(entries) = clone_index.get(&orig) else {
            return orig; // unreachable / not in dataflow — keep as-is.
        };
        for (k, cid) in entries {
            if k == out_key {
                return *cid;
            }
        }
        // Fallback: no exact match (shouldn't happen if dataflow
        // was sound). Keep the primary (entry 0) — orphan-prune
        // will clear stale references if it's truly unreachable.
        entries[0].1
    };
    match term {
        Terminator::Branch { target, .. } => {
            *target = pick(*target);
        }
        Terminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            *true_target = pick(*true_target);
            *false_target = pick(*false_target);
        }
        Terminator::Return(_) | Terminator::ReturnNull | Terminator::Unreachable => {}
    }
}

// ---------------------------------------------------------------------------
// Phase 4 — BUILD LAYOUT (mirror onto clones)
// ---------------------------------------------------------------------------

fn build_layout(
    _mir: &MirFunction,
    plans: &[YieldPlan],
    clone_index: &HashMap<BlockId, Vec<(SubsKey, BlockId)>>,
    layout: &mut StateMachineLayout,
) {
    for plan in plans {
        // Per-clone resume entry — but resume_blocks have exactly
        // one entry per the dataflow ({} from dispatcher → K's loads).
        // So clone_index[resume_block] has one entry. We use the
        // primary clone as the canonical resume entry.
        let resume_entry = primary_clone(plan.resume_block, clone_index);

        layout.resume_entries.push(resume_entry);

        // Find every clone of the yielding_block; each may have a
        // different subs (so the saves operands differ per clone).
        // The state_id is shared (dispatcher hits this state_id
        // regardless of which clone yielded — runtime save table
        // is in the FRAME, identified by slot number, not vid).
        let yblk_clones = clones_of(plan.yielding_block, clone_index);
        for (subs_key, clone_bid) in yblk_clones.iter() {
            let subs = key_to_subs(subs_key);
            // yield_blocks: state_id mapping.
            layout.yield_blocks.insert(*clone_bid, plan.state_id);
            // yield_saves: saves with operand remapped.
            let saves: Vec<(u32, ValueId)> = plan
                .saves
                .iter()
                .map(|(slot, vid)| (*slot, subs.get(vid).copied().unwrap_or(*vid)))
                .collect();
            if !saves.is_empty() {
                layout.yield_saves.insert(*clone_bid, saves);
            }
            // block_kinds: per-kind metadata with operands remapped.
            match &plan.kind {
                YieldPlanKind::DirectYield { .. } => {
                    layout
                        .block_kinds
                        .insert(*clone_bid, BlockKind::DirectYield);
                }
                YieldPlanKind::CrossFnCall {
                    call_check_block,
                    receiver,
                    args,
                    result,
                    method_sym,
                    ..
                } => {
                    layout.block_kinds.insert(
                        *clone_bid,
                        BlockKind::CrossFnCallInit {
                            resume_check_block: primary_clone(*call_check_block, clone_index),
                            receiver: subs.get(receiver).copied().unwrap_or(*receiver),
                            args: args
                                .iter()
                                .map(|a| subs.get(a).copied().unwrap_or(*a))
                                .collect(),
                            result: *result, // result vid is shared across clones
                            method_sym: *method_sym,
                        },
                    );
                }
            }
        }

        // Resume-side metadata: one entry per resume_block (its
        // single clone). The dispatcher hits state_id and the
        // br_table jumps here.
        let resume_bid = resume_entry;
        // resume_loads: paired (slot, fresh_load_vid). Same for
        // every entry — not subs-dependent.
        if !plan.loads.is_empty() {
            layout.resume_loads.insert(resume_bid, plan.loads.clone());
        }
        // direct_yield_results / cross_fn block_kinds at resume side.
        match &plan.kind {
            YieldPlanKind::DirectYield { result, .. } => {
                layout.direct_yield_results.insert(resume_bid, *result);
            }
            YieldPlanKind::CrossFnCall {
                done_block,
                receiver,
                args,
                result,
                method_sym,
                ..
            } => {
                // The CrossFnCallResume's done_block primary is the
                // post-call tail's first clone. The cross_fn_results
                // is populated by the cranelift lowering once it
                // runs; here we just record kind + done_block.
                layout.block_kinds.insert(
                    resume_bid,
                    BlockKind::CrossFnCallResume {
                        done_block: primary_clone(*done_block, clone_index),
                        receiver: *receiver, // saved through slots; lowering reads from slots
                        args: args.clone(),
                        result: *result,
                        method_sym: *method_sym,
                    },
                );
            }
        }
    }
}

fn primary_clone(
    orig: BlockId,
    clone_index: &HashMap<BlockId, Vec<(SubsKey, BlockId)>>,
) -> BlockId {
    clone_index
        .get(&orig)
        .and_then(|entries| entries.first().map(|(_, cid)| *cid))
        .unwrap_or(orig)
}

fn clones_of<'a>(
    orig: BlockId,
    clone_index: &'a HashMap<BlockId, Vec<(SubsKey, BlockId)>>,
) -> Vec<(SubsKey, BlockId)> {
    clone_index
        .get(&orig)
        .cloned()
        .unwrap_or_else(|| vec![(Vec::new(), orig)])
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
        SubscriptSet {
            receiver,
            args,
            value,
        } => {
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
        let layout = transform_to_state_machine(&mut mir, &interner, &Default::default())
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
        let layout = transform_to_state_machine(&mut mir, &interner, &Default::default())
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
        assert_eq!(saved_slots, load_slots, "save/load slot sets must match");
    }
}
