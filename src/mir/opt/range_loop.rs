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
use super::{MirPass, replace_uses_in_func};
use crate::intern::Interner;
use crate::mir::{Instruction, MirFunction, Terminator, ValueId};
use std::collections::HashMap;

pub struct RangeLoop<'a> {
    pub interner: &'a Interner,
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
                if let Instruction::MakeRange(from, to, inclusive) = inst {
                    if !inclusive {
                        range_infos.push((vid, *from, *to));
                    }
                }
            }
        }

        let mut changed = false;
        for (range_vid, from_vid, to_vid) in range_infos {
            if self.try_optimize(func, range_vid, from_vid, to_vid, iterate_sym, iter_value_sym) {
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

        for (bi, block) in func.blocks.iter().enumerate() {
            for (ii, &(vid, ref inst)) in block.instructions.iter().enumerate() {
                if let Instruction::Call {
                    receiver, method, args,
                } = inst
                {
                    if *receiver != range_vid {
                        continue;
                    }
                    if method.index() == iterate_sym.index() && args.len() == 1 {
                        // Check if arg is ConstNull (initial call)
                        let arg = args[0];
                        let is_null_arg = func.blocks.iter().any(|b| {
                            b.instructions.iter().any(|&(v, ref i)| {
                                v == arg && matches!(i, Instruction::ConstNull)
                            })
                        });
                        if is_null_arg {
                            init_call = Some((bi, ii, vid));
                        } else {
                            body_iterate = Some((bi, ii, vid, arg));
                        }
                    } else if method.index() == iter_value_sym.index() && args.len() == 1 {
                        body_iter_value = Some((bi, ii, vid, args[0]));
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

        // body_iterate and body_iter_value should be in the same block
        if body_bi != body_val_bi {
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
        let not_pos = match cond_block.instructions.iter().position(|&(_, ref inst)| {
            matches!(inst, Instruction::Not(v) if *v == iter_param)
        }) {
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

        // Verify body block matches
        let body_block_id = func.blocks[body_bi].id;
        if body_bid != body_block_id {
            return false;
        }

        // Verify iter_val_arg == iter_param (iteratorValue receives the iterator)
        if iter_val_arg != iter_param {
            return false;
        }

        // === Apply transformation ===

        // 1. Replace initial iterate Call with Move(from_vid) — reuses init_vid
        func.blocks[init_bi].instructions[init_ii].1 = Instruction::Move(from_vid);

        // 2. In cond_bb: replace Not(iter_param) with CmpLt(iter_param, to_vid)
        //    Use boxed CmpLt (not CmpLtF64) because iter_param is NaN-boxed.
        //    TypeSpecialize will convert to CmpLtF64 on the Optimized tier.
        func.blocks[cond_bi].instructions[not_pos].1 =
            Instruction::CmpLt(iter_param, to_vid);

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
        func.blocks[body_bi].instructions[body_val_ii].1 = Instruction::Move(iter_param);

        // 4. Replace iterate Call with AddF64(iter_param, 1.0)
        //    Insert ConstNum(1.0) before the iterate call position
        let one_vid = func.new_value();
        let one_inst = (one_vid, Instruction::ConstNum(1.0));
        // Insert before body_iter_ii (which may have shifted if body_val_ii < body_iter_ii)
        let adjusted_iter_ii = if body_val_ii < body_iter_ii {
            body_iter_ii // no shift since we only replaced, not inserted above
        } else {
            body_iter_ii
        };
        func.blocks[body_bi]
            .instructions
            .insert(adjusted_iter_ii, one_inst);
        // The old iterate call is now at adjusted_iter_ii + 1
        // Use boxed Add (not AddF64) because iter_param is NaN-boxed.
        func.blocks[body_bi].instructions[adjusted_iter_ii + 1].1 =
            Instruction::Add(iter_param, one_vid);

        // 5. Replace all uses of iter_val_vid with iter_param
        //    (the iteratorValue result is now just iter_param)
        let mut replacements = HashMap::new();
        replacements.insert(iter_val_vid, iter_param);
        replace_uses_in_func(func, &replacements);

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_loop_name() {
        let interner = Interner::new();
        let pass = RangeLoop { interner: &interner };
        assert_eq!(pass.name(), "range_loop");
    }
}
