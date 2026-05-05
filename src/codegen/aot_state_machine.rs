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
                    // Reject yields inside non-Return-terminated
                    // blocks for v1. The block's terminator
                    // currently is whatever bb_K had originally;
                    // if it's not a plain `Return(_)`/`ReturnNull`
                    // we'd need to thread the post-yield code
                    // through the original branch successors.
                    if !matches!(
                        block.terminator,
                        Terminator::Return(_) | Terminator::ReturnNull
                    ) {
                        return Err(StateMachineError::SuspensionInBranchedBlock {
                            block: block.id,
                        });
                    }
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

        // Liveness pass: collect values defined in this block
        // *before* the suspension that are used in either:
        //   - the rest of this block (after the yield), or
        //   - the trailing terminator, or
        //   - any block transitively reachable from the new
        //     resume block via terminator successors.
        // Each such value gets a save-before-yield + load-on-resume
        // pair in the layout; subsequent uses are remapped to the
        // freshly-loaded ValueId so SSA stays valid.
        let live_across: Vec<ValueId> = {
            let blk = &mir.blocks[blk_id.0 as usize];
            let mut defined_before: std::collections::HashSet<ValueId> =
                std::collections::HashSet::new();
            for (vid, _) in &blk.instructions[..inst_idx] {
                defined_before.insert(*vid);
            }
            // Block params don't survive a split here for v1
            // (entry-block params are mapped to function args at
            // lowering time and the post-resume code references
            // them indirectly through other instructions). v2's
            // more complete liveness analysis will need to fold
            // these in; for now they're unhandled and yield-in-a-
            // block-with-real-params errors out via the branch
            // refusal below.
            //
            // Walk reachable-from-yield-tail uses.
            let mut used_after: std::collections::HashSet<ValueId> =
                std::collections::HashSet::new();
            for (_, inst) in &blk.instructions[inst_idx + 1..] {
                for u in collect_uses(inst) {
                    used_after.insert(u);
                }
            }
            for u in collect_terminator_uses(&blk.terminator) {
                used_after.insert(u);
            }
            // BFS from the original terminator's successors.
            let mut visited: std::collections::HashSet<BlockId> =
                std::collections::HashSet::new();
            let mut stack: Vec<BlockId> = blk.terminator.successors();
            while let Some(b) = stack.pop() {
                if !visited.insert(b) {
                    continue;
                }
                let s_blk = &mir.blocks[b.0 as usize];
                for (_, inst) in &s_blk.instructions {
                    for u in collect_uses(inst) {
                        used_after.insert(u);
                    }
                }
                for u in collect_terminator_uses(&s_blk.terminator) {
                    used_after.insert(u);
                }
                for s in s_blk.terminator.successors() {
                    stack.push(s);
                }
            }
            // Stable ordering for deterministic slot assignment:
            // walk the pre-yield instruction list and pick those
            // that are also used after.
            let mut live: Vec<ValueId> = Vec::new();
            for (vid, _) in &blk.instructions[..inst_idx] {
                if used_after.contains(vid) {
                    live.push(*vid);
                }
            }
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

        // Remap every reference to a saved value in the new
        // resume block + every block transitively reachable from
        // it. Subsequent uses now read the freshly-loaded
        // ValueId from the resume block's prologue (emitted by
        // the AOT lowering from `layout.resume_loads`).
        if !remap.is_empty() {
            let mut visited: std::collections::HashSet<BlockId> =
                std::collections::HashSet::new();
            let mut stack: Vec<BlockId> = vec![new_block_id];
            while let Some(b) = stack.pop() {
                if !visited.insert(b) {
                    continue;
                }
                let blk = mir.block_mut(b);
                for (_, inst) in &mut blk.instructions {
                    remap_instruction_uses(inst, &remap);
                }
                remap_terminator_uses(&mut blk.terminator, &remap);
                for s in blk.terminator.successors() {
                    stack.push(s);
                }
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
