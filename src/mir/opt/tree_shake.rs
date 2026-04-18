//! Module-level reachability analysis ("tree shake").
//!
//! Walks the MIR function graph starting from every registered module's
//! top-level function *and* every class method FuncId found in a live
//! class object. Any registered function that isn't visited is — under
//! a conservative no-reflection assumption — unreachable at runtime.
//!
//! # Root set
//!
//! 1. Every module's `top_level` `FuncId`.
//! 2. Every FuncId bound into a live class's method table. Classes
//!    themselves live inside module vars; the analysis walks
//!    `module.vars` for `ObjClass` values and extracts the `fn_id`
//!    from each `Method::Closure` / `Method::Constructor` the class
//!    carries. Class methods never appear as `MakeClosure` operands
//!    in module MIR — the VM wires them directly in Rust at class-
//!    definition time — so without this step the analysis would
//!    incorrectly flag every class method as dead.
//!
//! # Cross-function edges
//!
//! Starting from the root set, the trace follows:
//!
//! * [`Instruction::MakeClosure`] — every user-defined function
//!   reached through a closure literal (nested closures, `Fn.new
//!   { ... }` blocks).
//! * [`Instruction::CallKnownFunc`] — devirtualised direct call from
//!   speculative IC data. Adds a direct edge the closure trace wouldn't
//!   cover on its own.
//! * `CallStaticSelf` carries an implicit "the current function" edge —
//!   walking the current function's body already covers that, so it's
//!   not listed explicitly below.
//!
//! # What this analysis does not handle
//!
//! * `Meta.eval` / any other runtime source-to-MIR path.
//! * Classes defined by native code (e.g. a host language embedding
//!   that calls `engine.register_function` for a method without putting
//!   the class object in any module's vars). The trace only walks
//!   `engine.modules[*].vars` when collecting class roots.
//! * Function values reached only through an IC entry the interpreter
//!   populated at runtime for a call site that we never lowered through
//!   `MakeClosure` or `CallKnownFunc` in this build — doesn't happen
//!   today, but worth naming as the assumption.
//! * Native methods registered directly from Rust (e.g. `List.add(_)`).
//!   Those aren't in `ExecutionEngine::functions`, so the analysis
//!   skips over them by construction.
//!
//! Callers must be confident none of the above happens before trusting
//! the "unreachable" verdict for memory-reclamation purposes.

use std::collections::HashSet;

use crate::mir::{Instruction, MirFunction};
use crate::runtime::engine::{ExecutionEngine, FuncId};
use crate::runtime::object::{Method, ObjClass, ObjClosure, ObjHeader, ObjType};

/// Collect the set of `FuncId`s reachable from any module's top-level
/// and every class method wired into a live class.
pub fn reachable_funcs(engine: &ExecutionEngine) -> HashSet<FuncId> {
    let mut reachable: HashSet<FuncId> = HashSet::new();
    let mut worklist: Vec<FuncId> = Vec::new();

    // Root 1: every module's top-level function.
    for module in engine.modules.values() {
        worklist.push(module.top_level);
    }

    // Root 2: every class method bound into a live class. Class objects
    // are reached via module vars (the language's only way to surface a
    // class reference at module scope). Methods are bound directly by
    // the VM in Rust rather than through MakeClosure, so the MIR trace
    // on its own cannot discover them.
    for module in engine.modules.values() {
        for var in &module.vars {
            collect_class_method_roots(*var, &mut worklist);
        }
    }

    while let Some(id) = worklist.pop() {
        if !reachable.insert(id) {
            continue;
        }
        if let Some(mir) = engine.get_mir(id) {
            visit_mir(&mir, &mut worklist);
        }
    }

    reachable
}

/// If `v` is an `ObjClass`, push every method's backing `FuncId` into
/// `worklist`. Safe to call on any `Value` — non-class values are
/// ignored. Static methods live in the same method table as regular
/// ones (prefixed with `static:`), so a single pass covers both.
fn collect_class_method_roots(v: crate::runtime::value::Value, worklist: &mut Vec<FuncId>) {
    let Some(ptr) = v.as_object() else {
        return;
    };
    // SAFETY: the module var holds a GC-rooted object. We only read
    // immutable header fields plus the methods table; we never follow
    // freed pointers because the VM keeps live module vars in the
    // nursery/old gen until shutdown.
    unsafe {
        let header = ptr as *const ObjHeader;
        if (*header).obj_type != ObjType::Class {
            return;
        }
        let class = ptr as *const ObjClass;
        for method in (*class).methods.iter().flatten() {
            // Match-ergonomics on a Copy field (`*mut ObjClosure`)
            // binds by value, so `&Method::*(p)` is the pattern that
            // gives us the raw pointer without a spurious deref.
            let closure_ptr: *mut ObjClosure = match method {
                &Method::Closure(p) | &Method::Constructor(p) => p,
                _ => continue,
            };
            if closure_ptr.is_null() {
                continue;
            }
            let fn_ptr = (*closure_ptr).function;
            if fn_ptr.is_null() {
                continue;
            }
            worklist.push(FuncId((*fn_ptr).fn_id));
        }
    }
}

fn visit_mir(mir: &MirFunction, worklist: &mut Vec<FuncId>) {
    for block in &mir.blocks {
        for (_, inst) in &block.instructions {
            match inst {
                Instruction::MakeClosure { fn_id, .. } => {
                    worklist.push(FuncId(*fn_id));
                }
                Instruction::CallKnownFunc { func_id, .. } => {
                    worklist.push(FuncId(*func_id));
                }
                _ => {}
            }
        }
    }
}

/// A dead-function report: every registered `FuncId` that the reachability
/// trace didn't visit.
#[derive(Debug, Default)]
pub struct DeadFuncReport {
    pub total: usize,
    pub reachable: usize,
    pub dead: Vec<FuncId>,
}

/// Produce a [`DeadFuncReport`] for the engine.
pub fn analyse(engine: &ExecutionEngine) -> DeadFuncReport {
    let reachable = reachable_funcs(engine);
    let total = engine.function_count();
    let mut dead = Vec::new();
    for i in 0..total {
        let id = FuncId(i as u32);
        if !reachable.contains(&id) {
            dead.push(id);
        }
    }
    DeadFuncReport {
        total,
        reachable: reachable.len(),
        dead,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intern::Interner;
    use crate::mir::{MirFunction, Terminator};
    use crate::runtime::engine::{ExecutionEngine, ExecutionMode, ModuleEntry};

    fn make_empty_fn(interner: &mut Interner, name: &str) -> MirFunction {
        let sym = interner.intern(name);
        let mut f = MirFunction::new(sym, 0);
        let bb = f.new_block();
        f.block_mut(bb).terminator = Terminator::ReturnNull;
        f
    }

    #[test]
    fn reports_unreferenced_function_as_dead() {
        let mut interner = Interner::new();
        let mut engine = ExecutionEngine::new(ExecutionMode::Interpreter);

        // Module top-level is a no-op.
        let top_level_mir = make_empty_fn(&mut interner, "<module>");
        let top_level = engine.register_function(top_level_mir);
        engine.modules.insert(
            "main".to_string(),
            ModuleEntry {
                top_level,
                vars: Vec::new(),
                var_names: Vec::new(),
            },
        );

        // A function that nothing refers to.
        let orphan_mir = make_empty_fn(&mut interner, "orphan");
        let orphan_id = engine.register_function(orphan_mir);

        let report = analyse(&engine);
        assert_eq!(report.total, 2);
        assert_eq!(report.reachable, 1);
        assert_eq!(report.dead, vec![orphan_id]);
    }

    #[test]
    fn closure_reference_keeps_function_alive() {
        let mut interner = Interner::new();
        let mut engine = ExecutionEngine::new(ExecutionMode::Interpreter);

        let target_mir = make_empty_fn(&mut interner, "target");
        let target_id = engine.register_function(target_mir);

        let top_sym = interner.intern("<module>");
        let mut top_mir = MirFunction::new(top_sym, 0);
        let bb = top_mir.new_block();
        let vid = top_mir.new_value();
        top_mir.block_mut(bb).instructions.push((
            vid,
            Instruction::MakeClosure {
                fn_id: target_id.0,
                upvalues: Vec::new(),
            },
        ));
        top_mir.block_mut(bb).terminator = Terminator::ReturnNull;
        let top_level = engine.register_function(top_mir);

        engine.modules.insert(
            "main".to_string(),
            ModuleEntry {
                top_level,
                vars: Vec::new(),
                var_names: Vec::new(),
            },
        );

        let report = analyse(&engine);
        assert_eq!(report.total, 2);
        assert_eq!(report.reachable, 2);
        assert!(report.dead.is_empty());
        assert!(reachable_funcs(&engine).contains(&target_id));
    }
}
