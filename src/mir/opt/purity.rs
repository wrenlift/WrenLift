/// Per-function effect summary — Phase 6 step 2.
///
/// A function is **pure** if its body has no observable side effects:
/// no field / module / static / upvalue / subscript writes, no
/// allocations (each alloc has identity, so the call would not be
/// substitutable), no `ToString` / `StringConcat` (also allocating),
/// no `MakeClosure`, and every internal call is also pure.
///
/// Internal-call purity is decided by a fixed-point closure: callees
/// start "unknown" and get marked pure as their bodies clear. Calls
/// to functions outside the analysed set are conservatively impure.
///
/// `Instruction::Call` is keyed by signature, not by callee
/// identity, so it cannot be propagated through here unless its
/// `pure_call` flag is already set (the MIR builder seeds that for
/// known builtin operators). User calls stay opaque until devirt
/// turns them into `CallKnownFunc { func_id }`.
///
/// Used by:
/// - `MirFunction::is_pure_self_recursive` (CSE in the optimizer
///   pipeline) for `CallStaticSelf`.
/// - The post-devirt CSE pass in codegen for `CallKnownFunc`.
use std::collections::HashMap;

use crate::mir::{Instruction, MirFunction};

/// Look up purity for a function identified by its `func_id`.
///
/// Returns `None` when the analyser has no information about the
/// callee (e.g. it lives in another module or isn't installed yet)
/// — callers must treat the absence as "may have side effects".
pub trait CalleePurity {
    fn is_pure(&self, func_id: u32) -> Option<bool>;
}

impl CalleePurity for HashMap<u32, bool> {
    fn is_pure(&self, func_id: u32) -> Option<bool> {
        self.get(&func_id).copied()
    }
}

impl<F: Fn(u32) -> Option<bool>> CalleePurity for F {
    fn is_pure(&self, func_id: u32) -> Option<bool> {
        (self)(func_id)
    }
}

/// True if every instruction in `mir` is non-side-effecting given
/// the callee-purity oracle. `CallKnownFunc { func_id }` looks up
/// `oracle.is_pure(func_id)`. `Call` is treated as pure only when
/// it carries `pure_call: true` from the MIR builder.
///
/// `CallStaticSelf` is treated as pure when `assume_self_pure` is
/// true — callers running a fixed-point pass set this to the
/// current best estimate of *this* function's purity.
pub fn function_body_is_pure(
    mir: &MirFunction,
    oracle: &dyn CalleePurity,
    assume_self_pure: bool,
) -> bool {
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            match inst {
                Instruction::Call { pure_call, .. } => {
                    if !*pure_call {
                        return false;
                    }
                }
                Instruction::CallKnownFunc { func_id, .. } => {
                    if oracle.is_pure(*func_id) != Some(true) {
                        return false;
                    }
                }
                Instruction::CallStaticSelf { .. } => {
                    if !assume_self_pure {
                        return false;
                    }
                }
                Instruction::SuperCall { .. } => return false,
                // Stores / mutations.
                Instruction::SetField(..)
                | Instruction::SetModuleVar(..)
                | Instruction::SetStaticField(..)
                | Instruction::SetUpvalue(..)
                | Instruction::SubscriptSet { .. } => return false,
                // Allocations (identity-bearing).
                Instruction::MakeList(..)
                | Instruction::MakeMap(..)
                | Instruction::MakeRange(..)
                | Instruction::MakeClosure { .. }
                | Instruction::StringConcat(..)
                | Instruction::ToString(..) => return false,
                _ => {}
            }
        }
    }
    true
}

/// Run a fixed-point analysis over a set of `(func_id, &MirFunction)`
/// pairs and return a purity map. A function in the input set with
/// `is_pure=true` in the result is observably side-effect free,
/// assuming every callee referenced via `CallKnownFunc { func_id }`
/// resolves to the same `func_id` in the input set.
///
/// External callees (not in the input set) are conservatively
/// impure — their `func_id` simply isn't found in the result map
/// and `function_body_is_pure` short-circuits.
pub fn compute_purity_map<'a, I>(funcs: I) -> HashMap<u32, bool>
where
    I: IntoIterator<Item = (u32, &'a MirFunction)>,
{
    let funcs: Vec<(u32, &MirFunction)> = funcs.into_iter().collect();
    let mut map: HashMap<u32, bool> = funcs.iter().map(|(id, _)| (*id, true)).collect();
    // Iterate to a fixed-point. Each pass can only flip `true` →
    // `false`, so the lattice has finite height; bound by N to
    // guard against pathological input.
    for _ in 0..funcs.len().saturating_add(1) {
        let mut changed = false;
        for (id, mir) in &funcs {
            // Tentatively assume this function is pure (so its
            // own `CallStaticSelf` doesn't disqualify it on the
            // first pass), then check the body.
            let prev = map.get(id).copied().unwrap_or(true);
            if !prev {
                continue;
            }
            let pure = function_body_is_pure(mir, &map, true);
            if !pure {
                map.insert(*id, false);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    map
}

/// True if every instruction in `mir` is alloc-free given the
/// callee oracle. Stricter than "no GC" only in that it also
/// considers `SuperCall` / `Call { method }` impure (their target
/// is unknown and might allocate).
///
/// Used by the IC fast path: a callee that's transitively
/// alloc-free can't fire a GC during its body, so register-passed
/// args stay valid even without JIT-frame stack maps. Impure
/// stores (`SetField`, etc.) are *fine* here — they don't move
/// anything in memory; they just write a slot.
pub fn function_body_is_alloc_free(
    mir: &MirFunction,
    oracle: &dyn CalleePurity,
    assume_self_alloc_free: bool,
) -> bool {
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            match inst {
                // Allocations / GC triggers.
                Instruction::MakeList(..)
                | Instruction::MakeMap(..)
                | Instruction::MakeRange(..)
                | Instruction::MakeClosure { .. }
                | Instruction::StringConcat(..)
                | Instruction::ToString(..) => return false,
                // Calls — only safe when the target is known and
                // also alloc-free (oracle map is shared with the
                // purity run, where alloc-free sits below pure).
                Instruction::Call { pure_call, .. } => {
                    if !*pure_call {
                        return false;
                    }
                }
                Instruction::CallKnownFunc { func_id, .. } => {
                    if oracle.is_pure(*func_id) != Some(true) {
                        return false;
                    }
                }
                Instruction::CallStaticSelf { .. } => {
                    if !assume_self_alloc_free {
                        return false;
                    }
                }
                Instruction::SuperCall { .. } => return false,
                // SetField / SetUpvalue / SubscriptSet / module
                // var stores: writes only, no allocation. Don't
                // bail.
                _ => {}
            }
        }
    }
    true
}

/// Same fixed-point shape as `compute_purity_map` but uses
/// `function_body_is_alloc_free`. Returned bits are: "this
/// function (transitively) doesn't trigger a GC."
pub fn compute_alloc_free_map<'a, I>(funcs: I) -> HashMap<u32, bool>
where
    I: IntoIterator<Item = (u32, &'a MirFunction)>,
{
    let funcs: Vec<(u32, &MirFunction)> = funcs.into_iter().collect();
    let mut map: HashMap<u32, bool> = funcs.iter().map(|(id, _)| (*id, true)).collect();
    for _ in 0..funcs.len().saturating_add(1) {
        let mut changed = false;
        for (id, mir) in &funcs {
            let prev = map.get(id).copied().unwrap_or(true);
            if !prev {
                continue;
            }
            let alloc_free = function_body_is_alloc_free(mir, &map, true);
            if !alloc_free {
                map.insert(*id, false);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::{Interner, SymbolId};
    use crate::mir::{BasicBlock, Instruction, Terminator};

    fn make_func(name: &str, interner: &mut Interner, arity: u8) -> MirFunction {
        let sym = interner.intern(name);
        MirFunction::new(sym, arity)
    }

    fn add_block(f: &mut MirFunction) -> crate::mir::BlockId {
        let id = f.new_block();
        // Replace with a fresh BasicBlock so we control its content.
        let _ = f.block(id);
        id
    }

    #[test]
    fn empty_function_is_pure() {
        let mut interner = Interner::new();
        let mut f = make_func("f", &mut interner, 0);
        let bb = add_block(&mut f);
        let v = f.new_value();
        f.block_mut(bb)
            .instructions
            .push((v, Instruction::ConstNum(1.0)));
        f.block_mut(bb).terminator = Terminator::Return(v);
        let oracle: HashMap<u32, bool> = HashMap::new();
        assert!(function_body_is_pure(&f, &oracle, true));
    }

    #[test]
    fn set_field_is_impure() {
        let mut interner = Interner::new();
        let mut f = make_func("f", &mut interner, 1);
        let bb = add_block(&mut f);
        let recv = f.new_value();
        let val = f.new_value();
        let _set = f.new_value();
        let block: &mut BasicBlock = f.block_mut(bb);
        block.instructions.push((recv, Instruction::BlockParam(0)));
        block.instructions.push((val, Instruction::ConstNum(0.0)));
        block
            .instructions
            .push((_set, Instruction::SetField(recv, 0, val)));
        block.terminator = Terminator::Return(val);
        let oracle: HashMap<u32, bool> = HashMap::new();
        assert!(!function_body_is_pure(&f, &oracle, true));
    }

    #[test]
    fn known_func_consults_oracle() {
        let mut interner = Interner::new();
        let mut f = make_func("f", &mut interner, 1);
        let bb = add_block(&mut f);
        let recv = f.new_value();
        let res = f.new_value();
        let block = f.block_mut(bb);
        block.instructions.push((recv, Instruction::BlockParam(0)));
        block.instructions.push((
            res,
            Instruction::CallKnownFunc {
                func_id: 7,
                method: SymbolId::from_raw(0),
                expected_class: 0,
                inline_getter_field: None,
                pure_leaf: false,
                receiver: recv,
                args: vec![],
            },
        ));
        block.terminator = Terminator::Return(res);

        // Oracle says callee impure → caller impure.
        let mut oracle: HashMap<u32, bool> = HashMap::new();
        oracle.insert(7, false);
        assert!(!function_body_is_pure(&f, &oracle, true));

        // Oracle says callee pure → caller pure.
        let mut oracle: HashMap<u32, bool> = HashMap::new();
        oracle.insert(7, true);
        assert!(function_body_is_pure(&f, &oracle, true));

        // Oracle has no entry → conservatively impure.
        let oracle: HashMap<u32, bool> = HashMap::new();
        assert!(!function_body_is_pure(&f, &oracle, true));
    }

    #[test]
    fn fixed_point_propagates_impurity() {
        let mut interner = Interner::new();

        // Function #0: SetField — directly impure.
        let mut f0 = make_func("impure", &mut interner, 1);
        let bb0 = add_block(&mut f0);
        let recv = f0.new_value();
        let val = f0.new_value();
        let _s = f0.new_value();
        let block = f0.block_mut(bb0);
        block.instructions.push((recv, Instruction::BlockParam(0)));
        block.instructions.push((val, Instruction::ConstNum(1.0)));
        block
            .instructions
            .push((_s, Instruction::SetField(recv, 0, val)));
        block.terminator = Terminator::Return(val);

        // Function #1: only calls #0 → should also be impure.
        let mut f1 = make_func("caller", &mut interner, 1);
        let bb1 = add_block(&mut f1);
        let recv1 = f1.new_value();
        let r = f1.new_value();
        let block = f1.block_mut(bb1);
        block.instructions.push((recv1, Instruction::BlockParam(0)));
        block.instructions.push((
            r,
            Instruction::CallKnownFunc {
                func_id: 0,
                method: SymbolId::from_raw(0),
                expected_class: 0,
                inline_getter_field: None,
                pure_leaf: false,
                receiver: recv1,
                args: vec![],
            },
        ));
        block.terminator = Terminator::Return(r);

        // Function #2: arithmetic + ConstStatic + return — pure.
        let mut f2 = make_func("pure", &mut interner, 0);
        let bb2 = add_block(&mut f2);
        let a = f2.new_value();
        let b = f2.new_value();
        let s = f2.new_value();
        let block = f2.block_mut(bb2);
        block.instructions.push((a, Instruction::ConstNum(1.0)));
        block.instructions.push((b, Instruction::ConstNum(2.0)));
        block.instructions.push((s, Instruction::Add(a, b)));
        block.terminator = Terminator::Return(s);

        let map = compute_purity_map([(0u32, &f0), (1u32, &f1), (2u32, &f2)]);
        assert_eq!(map.get(&0), Some(&false));
        assert_eq!(map.get(&1), Some(&false));
        assert_eq!(map.get(&2), Some(&true));
    }

    #[test]
    fn mutual_recursion_stays_pure_when_bodies_are_pure() {
        let mut interner = Interner::new();
        // Two functions that call each other via CallKnownFunc and
        // do nothing else — the fixed-point must converge to "both
        // pure" (no body-local impurity disqualifies them).
        let make_caller = |interner: &mut Interner, callee_id: u32| {
            let mut f = MirFunction::new(interner.intern("f"), 0);
            let bb = f.new_block();
            let r = f.new_value();
            let recv = f.new_value();
            let block = f.block_mut(bb);
            block.instructions.push((recv, Instruction::ConstNull));
            block.instructions.push((
                r,
                Instruction::CallKnownFunc {
                    func_id: callee_id,
                    method: SymbolId::from_raw(0),
                    expected_class: 0,
                    inline_getter_field: None,
                    pure_leaf: false,
                    receiver: recv,
                    args: vec![],
                },
            ));
            block.terminator = Terminator::Return(r);
            f
        };
        let f0 = make_caller(&mut interner, 1);
        let f1 = make_caller(&mut interner, 0);
        let map = compute_purity_map([(0u32, &f0), (1u32, &f1)]);
        // Both functions only call each other — neither has a
        // direct impurity, and the analyser treats unknown
        // self-recursion as tentatively pure. The result is the
        // greatest fixed point: both pure.
        assert_eq!(map.get(&0), Some(&true));
        assert_eq!(map.get(&1), Some(&true));
    }
}
