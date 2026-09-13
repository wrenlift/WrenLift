//! Guard Num math methods on receivers of unknown type.
//!
//! `x.sqrt` compiles to a method call whenever the specialiser cannot
//! prove `x` a Num, which is every value that came out of a call or a
//! field. The call is almost always on a Num, so the site becomes a
//! type test with the intrinsic on one side and the call on the other;
//! the specialiser and the backends then treat the fast side like any
//! other unboxed arithmetic. Runs once, before the specialiser, on JIT
//! compile clones only.

use std::collections::{HashMap, HashSet};

use super::inline_calls::split_after;
use super::MirPass;
use crate::intern::{Interner, SymbolId};
use crate::mir::{Instruction, MathBinaryOp, MathUnaryOp, MirFunction, MirType, Terminator};

pub struct MathGuard {
    unary: HashMap<SymbolId, MathUnaryOp>,
    binary: HashMap<SymbolId, MathBinaryOp>,
}

impl MathGuard {
    pub fn new(interner: &Interner) -> Self {
        let unary_methods: &[(&str, MathUnaryOp)] = &[
            ("abs", MathUnaryOp::Abs),
            ("acos", MathUnaryOp::Acos),
            ("asin", MathUnaryOp::Asin),
            ("atan", MathUnaryOp::Atan),
            ("cbrt", MathUnaryOp::Cbrt),
            ("ceil", MathUnaryOp::Ceil),
            ("cos", MathUnaryOp::Cos),
            ("floor", MathUnaryOp::Floor),
            ("round", MathUnaryOp::Round),
            ("sin", MathUnaryOp::Sin),
            ("sqrt", MathUnaryOp::Sqrt),
            ("tan", MathUnaryOp::Tan),
            ("log", MathUnaryOp::Log),
            ("log2", MathUnaryOp::Log2),
            ("exp", MathUnaryOp::Exp),
            ("truncate", MathUnaryOp::Trunc),
            ("fraction", MathUnaryOp::Fract),
            ("sign", MathUnaryOp::Sign),
        ];
        let binary_methods: &[(&str, MathBinaryOp)] = &[
            ("atan(_)", MathBinaryOp::Atan2),
            ("min(_)", MathBinaryOp::Min),
            ("max(_)", MathBinaryOp::Max),
            ("pow(_)", MathBinaryOp::Pow),
        ];
        let mut unary = HashMap::new();
        let mut binary = HashMap::new();
        for (name, op) in unary_methods {
            if let Some(id) = interner.lookup(name) {
                unary.insert(id, *op);
            }
        }
        for (name, op) in binary_methods {
            if let Some(id) = interner.lookup(name) {
                binary.insert(id, *op);
            }
        }
        Self { unary, binary }
    }
}

enum Site {
    Unary(MathUnaryOp),
    Binary(MathBinaryOp),
}

impl MirPass for MathGuard {
    fn name(&self) -> &str {
        "math-guard"
    }

    fn run(&self, func: &mut MirFunction) -> bool {
        if self.unary.is_empty() && self.binary.is_empty() {
            return false;
        }
        let mut changed = false;
        // The slow blocks hold the calls this pass moved; they are the
        // fallback, not another site. They are the false edge of a
        // branch on an `IsNum`, so an earlier run's blocks are skipped
        // too.
        let mut generated: HashSet<usize> = HashSet::new();
        for block in &func.blocks {
            if let Terminator::CondBranch {
                condition,
                false_target,
                ..
            } = &block.terminator
            {
                let on_is_num = block
                    .instructions
                    .iter()
                    .any(|(v, i)| v == condition && matches!(i, Instruction::IsNum(_)));
                if on_is_num {
                    generated.insert(false_target.0 as usize);
                }
            }
        }
        let mut bi = 0;
        while bi < func.blocks.len() {
            if generated.contains(&bi) {
                bi += 1;
                continue;
            }
            let mut k = 0;
            while k < func.blocks[bi].instructions.len() {
                let (dst, inst) = &func.blocks[bi].instructions[k];
                let Instruction::Call {
                    receiver,
                    method,
                    args,
                    ..
                } = inst
                else {
                    k += 1;
                    continue;
                };
                let site = match args.len() {
                    0 => self.unary.get(method).map(|op| Site::Unary(*op)),
                    1 => self.binary.get(method).map(|op| Site::Binary(*op)),
                    _ => None,
                };
                let Some(site) = site else {
                    k += 1;
                    continue;
                };
                let dst = *dst;
                let recv = *receiver;
                let arg = args.first().copied();
                let block = func.blocks[bi].id;

                // Everything after the call continues in `post`, which
                // takes the result as its parameter.
                let post = split_after(func, block, k);
                let (_, call) = func.block_mut(block).instructions.pop().expect("the call");
                func.block_mut(post).params.insert(0, (dst, MirType::Value));

                let fast = func.new_block();
                let slow = func.new_block();
                generated.insert(fast.0 as usize);
                generated.insert(slow.0 as usize);
                let slow_result = func.new_value();
                func.block_mut(slow).instructions.push((slow_result, call));
                func.block_mut(slow).terminator = Terminator::Branch {
                    target: post,
                    args: vec![slow_result],
                };

                // Type tests on the receiver and, for a binary op, the
                // argument; either failing takes the call.
                let recv_ok = func.new_value();
                func.block_mut(block)
                    .instructions
                    .push((recv_ok, Instruction::IsNum(recv)));
                let fast_entry = match (&site, arg) {
                    (Site::Binary(_), Some(a)) => {
                        let check = func.new_block();
                        generated.insert(check.0 as usize);
                        let arg_ok = func.new_value();
                        func.block_mut(check)
                            .instructions
                            .push((arg_ok, Instruction::IsNum(a)));
                        func.block_mut(check).terminator = Terminator::CondBranch {
                            condition: arg_ok,
                            true_target: fast,
                            true_args: vec![],
                            false_target: slow,
                            false_args: vec![],
                        };
                        check
                    }
                    _ => fast,
                };
                func.block_mut(block).terminator = Terminator::CondBranch {
                    condition: recv_ok,
                    true_target: fast_entry,
                    true_args: vec![],
                    false_target: slow,
                    false_args: vec![],
                };

                let ur = func.new_value();
                let r = func.new_value();
                let boxed = func.new_value();
                let fast_block = func.block_mut(fast);
                fast_block.instructions.push((ur, Instruction::Unbox(recv)));
                match (site, arg) {
                    (Site::Unary(op), _) => {
                        fast_block
                            .instructions
                            .push((r, Instruction::MathUnaryF64(op, ur)));
                    }
                    (Site::Binary(op), Some(a)) => {
                        let ua = func.new_value();
                        let fast_block = func.block_mut(fast);
                        fast_block.instructions.push((ua, Instruction::Unbox(a)));
                        fast_block
                            .instructions
                            .push((r, Instruction::MathBinaryF64(op, ur, ua)));
                    }
                    (Site::Binary(_), None) => unreachable!("binary site without an argument"),
                }
                let fast_block = func.block_mut(fast);
                fast_block.instructions.push((boxed, Instruction::Box(r)));
                fast_block.terminator = Terminator::Branch {
                    target: post,
                    args: vec![boxed],
                };
                changed = true;
                // The rest of the original block now lives in `post`,
                // which the outer loop reaches in its turn.
                break;
            }
            bi += 1;
        }
        if changed {
            func.compute_predecessors();
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::{BlockId, ValueId};

    #[test]
    fn guards_each_site_once() {
        let mut interner = Interner::new();
        let sqrt = interner.intern("sqrt");
        let pow = interner.intern("pow(_)");
        let name = interner.intern("f");
        let mut f = MirFunction::new(name, 1);
        let bb = f.new_block();
        let x = f.new_value();
        let a = f.new_value();
        let b = f.new_value();
        let c = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((x, Instruction::BlockParam(0)));
        f.block_mut(bb).instructions.push((
            a,
            Instruction::Call {
                receiver: x,
                method: sqrt,
                args: vec![],
                pure_call: false,
            },
        ));
        f.block_mut(bb).instructions.push((
            b,
            Instruction::Call {
                receiver: a,
                method: pow,
                args: vec![x],
                pure_call: false,
            },
        ));
        f.block_mut(bb).instructions.push((
            c,
            Instruction::Call {
                receiver: b,
                method: sqrt,
                args: vec![],
                pure_call: false,
            },
        ));
        f.block_mut(bb).terminator = Terminator::Return(c);
        let pass = MathGuard::new(&interner);
        assert!(pass.run(&mut f));
        // Three sites: a fast, slow and post block each, plus one
        // argument check for the binary site.
        assert_eq!(f.blocks.len(), 1 + 3 * 3 + 1);
        assert!(!pass.run(&mut f) || f.blocks.len() == 1 + 3 * 3 + 1);
        let calls = f
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter(|(_, i)| matches!(i, Instruction::Call { .. }))
            .count();
        assert_eq!(calls, 3);
        let _ = (BlockId(0), ValueId(0));
    }
}
