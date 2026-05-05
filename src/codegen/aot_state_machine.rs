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
    };

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

        // Liveness sanity: any value defined in this block
        // *before* the suspension and used in instructions
        // *after* it (inside this same block, or transitively
        // via the original terminator's successors) is a live
        // value across the suspension. v1 refuses; v2 will save
        // them. For now scan only inside this block — if a
        // post-yield instruction uses a pre-yield value, error.
        {
            let blk = &mir.blocks[blk_id.0 as usize];
            let mut defined_before: std::collections::HashSet<ValueId> =
                std::collections::HashSet::new();
            for (vid, _) in &blk.instructions[..inst_idx] {
                defined_before.insert(*vid);
            }
            for (_, inst) in &blk.instructions[inst_idx + 1..] {
                let uses = collect_uses(inst);
                for u in uses {
                    if defined_before.contains(&u) {
                        return Err(StateMachineError::LiveValueAcrossSuspension {
                            block: blk_id,
                            value: u,
                        });
                    }
                }
            }
            // The trailing terminator: also scan its uses for
            // any pre-yield-defined value.
            for u in collect_terminator_uses(&blk.terminator) {
                if defined_before.contains(&u) {
                    return Err(StateMachineError::LiveValueAcrossSuspension {
                        block: blk_id,
                        value: u,
                    });
                }
            }
        }

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

        let next_state = layout.resume_entries.len() as u32;
        layout.resume_entries.push(new_block_id);
        layout.yield_blocks.insert(blk_id, next_state);
    }

    Ok(layout)
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

    /// Fiber body with a yield that has a value live across it
    /// — must error with `LiveValueAcrossSuspension` so the AOT
    /// driver can degrade cleanly until v2 lands.
    #[test]
    fn live_value_across_yield_errors() {
        let src = r#"
var f = Fiber.new {
  var x = 7
  Fiber.yield()
  System.print(x)
}
f.call()
"#;
        let (mut mir, interner) = build_mir_from_source(src, /* closure_idx */ 0);
        let result = transform_to_state_machine(&mut mir, &interner);
        assert!(
            matches!(
                result,
                Err(StateMachineError::LiveValueAcrossSuspension { .. })
            ),
            "expected LiveValueAcrossSuspension; got {:?}",
            result
        );
    }
}
