/// AST → MIR lowering.
///
/// Translates Wren AST into the SSA-based MIR. Each expression lowers to
/// one or more instructions producing a ValueId. Control flow (if, while,
/// for, short-circuit &&/||) creates multiple basic blocks with conditional
/// branches.
use std::collections::HashMap;

use crate::ast::*;
use crate::intern::{Interner, SymbolId};
use crate::mir::*;
use crate::sema::resolve::{ResolveResult, ResolvedName};

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct MirBuilder<'a> {
    func: MirFunction,
    interner: &'a mut Interner,
    resolutions: &'a HashMap<usize, ResolvedName>,
    module_vars: &'a [SymbolId],
    /// Sema upvalue info: scope_id → Vec<UpvalueInfo> for closures that capture variables.
    upvalue_map: &'a HashMap<usize, Vec<crate::sema::resolve::UpvalueInfo>>,
    /// Locals that are captured by nested closures, keyed by the
    /// scope_id they belong to. A boxed local lives inside a
    /// 1-element List so the inner closure's upvalue (which holds
    /// the list reference) sees writes from the outer scope and
    /// vice-versa. Populated by sema's `boxed_locals`.
    boxed_locals_map: &'a HashMap<usize, std::collections::HashSet<SymbolId>>,
    /// Names of locals to box in the *current* function being lowered.
    /// Derived from `boxed_locals_map` by looking up this function's
    /// scope_id when entering the builder.
    current_boxed: std::collections::HashSet<SymbolId>,
    /// Upvalues in the current function are always heap boxes (every
    /// captured local gets boxed), so upvalue reads / writes always
    /// go through a subscript. Stored as a flag rather than a per-
    /// upvalue set because the rule is uniform.
    upvalues_are_boxed: bool,
    current_block: BlockId,
    variables: HashMap<SymbolId, ValueId>,
    /// Break targets: (exit_block, tracked variable names for phi propagation)
    break_targets: Vec<(BlockId, Vec<SymbolId>)>,
    /// Continue targets: (header_block, tracked variable names for phi propagation)
    continue_targets: Vec<(BlockId, Vec<SymbolId>)>,
    /// Field name → index mapping for the current class (None if not in a method).
    field_map: Option<HashMap<SymbolId, u16>>,
    /// Compiled closure bodies collected during lowering.
    closures: Vec<MirFunction>,
    /// Base index for closure fn_ids (offset into module's closure list).
    closure_base: u32,
    /// Current enclosing class name when lowering a method body.
    current_class_name: Option<SymbolId>,
    /// Current method signature (for `super(args)` — uses same name as enclosing method).
    current_method_sig: Option<SymbolId>,
    /// Bare method name without arity suffix (e.g., "new" not "new(_,_)").
    current_method_base_name: Option<SymbolId>,
    /// Whether the current method is static.
    current_method_is_static: bool,
    /// Stack of shadowed variables per block scope.
    /// Each entry: (var name, previous value or None if newly introduced).
    block_shadows: Vec<Vec<(SymbolId, Option<ValueId>)>>,
    /// Type environment from sema inference. When present, enables emitting
    /// typed instructions (AddF64 instead of Add) for known-Num operands.
    type_env: Option<crate::sema::types::TypeEnv>,
}

impl<'a> MirBuilder<'a> {
    pub fn new(
        name: SymbolId,
        arity: u8,
        interner: &'a mut Interner,
        resolutions: &'a HashMap<usize, ResolvedName>,
        module_vars: &'a [SymbolId],
        upvalue_map: &'a HashMap<usize, Vec<crate::sema::resolve::UpvalueInfo>>,
    ) -> Self {
        // Keep the legacy constructor as a thin wrapper over the
        // full form so existing call sites that don't have a
        // `boxed_locals_map` keep working (closure capture just
        // won't box in that mode).
        static EMPTY_BOXED: std::sync::OnceLock<
            HashMap<usize, std::collections::HashSet<SymbolId>>,
        > = std::sync::OnceLock::new();
        let empty = EMPTY_BOXED.get_or_init(HashMap::new);
        // The `OnceLock` lives for the program's lifetime so we can
        // safely widen its lifetime to `'a`.
        let empty_ref: &'a HashMap<usize, std::collections::HashSet<SymbolId>> =
            unsafe { &*(empty as *const _) };
        Self::with_boxed(
            name,
            arity,
            interner,
            resolutions,
            module_vars,
            upvalue_map,
            empty_ref,
        )
    }

    pub fn with_boxed(
        name: SymbolId,
        arity: u8,
        interner: &'a mut Interner,
        resolutions: &'a HashMap<usize, ResolvedName>,
        module_vars: &'a [SymbolId],
        upvalue_map: &'a HashMap<usize, Vec<crate::sema::resolve::UpvalueInfo>>,
        boxed_locals_map: &'a HashMap<usize, std::collections::HashSet<SymbolId>>,
    ) -> Self {
        let mut func = MirFunction::new(name, arity);
        let entry = func.new_block();
        Self {
            func,
            interner,
            resolutions,
            module_vars,
            upvalue_map,
            boxed_locals_map,
            current_boxed: std::collections::HashSet::new(),
            upvalues_are_boxed: false,
            current_block: entry,
            variables: HashMap::new(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            field_map: None,
            closures: Vec::new(),
            closure_base: 0,
            current_class_name: None,
            current_method_sig: None,
            current_method_base_name: None,
            current_method_is_static: false,
            block_shadows: Vec::new(),
            type_env: None,
        }
    }

    /// If `name` is captured by a nested closure, wrap `val` in a
    /// 1-element List and return the list value. Otherwise return
    /// `val` unchanged. Callers use this when first binding a
    /// variable (param or `Stmt::Var`) so subsequent reads and
    /// writes go through the box uniformly.
    pub fn box_if_captured(&mut self, name: SymbolId, val: ValueId) -> ValueId {
        if self.current_boxed.contains(&name) {
            self.emit(Instruction::MakeList(vec![val]))
        } else {
            val
        }
    }

    /// Tell the builder which scope it's about to compile. Looks up
    /// the scope's captured-local set so we know what to box. Call
    /// this right after construction, before lowering the body.
    pub fn set_scope(&mut self, scope_id: usize, is_closure: bool) {
        self.current_boxed = self
            .boxed_locals_map
            .get(&scope_id)
            .cloned()
            .unwrap_or_default();
        // Inside a closure, every captured name is boxed (that's
        // how this whole scheme works), so upvalue reads/writes
        // always go through a subscript.
        self.upvalues_are_boxed = is_closure;
    }

    /// Set the type environment for type-directed instruction emission.
    pub fn with_type_env(mut self, env: crate::sema::types::TypeEnv) -> Self {
        self.type_env = Some(env);
        self
    }

    pub fn build_module(mut self, module: &Module) -> (MirFunction, Vec<MirFunction>) {
        for stmt in module {
            self.lower_stmt(stmt);
        }
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.func.block_mut(self.current_block).terminator = Terminator::ReturnNull;
        }
        self.func.compute_predecessors();
        (self.func, self.closures)
    }

    pub fn build_body(mut self, body: &Spanned<Stmt>, params: &[Spanned<SymbolId>]) -> MirFunction {
        // Emit BlockParams contiguously so register indices align
        // with the caller's argument positions. Only then wrap any
        // captured params in their List boxes.
        let raw_params: Vec<(SymbolId, ValueId)> = params
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let val = self.emit(Instruction::BlockParam(i as u16));
                (param.0, val)
            })
            .collect();
        for (name, val) in raw_params {
            let stored = self.box_if_captured(name, val);
            self.variables.insert(name, stored);
        }
        self.lower_stmt(body);
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.func.block_mut(self.current_block).terminator = Terminator::ReturnNull;
        }
        self.func.compute_predecessors();
        self.func
    }

    // -- Helpers ------------------------------------------------------------

    fn emit(&mut self, inst: Instruction) -> ValueId {
        let val = self.func.new_value();
        self.func
            .block_mut(self.current_block)
            .instructions
            .push((val, inst));
        val
    }

    fn set_terminator(&mut self, term: Terminator) {
        self.func.block_mut(self.current_block).terminator = term;
    }

    fn new_block(&mut self) -> BlockId {
        self.func.new_block()
    }

    fn switch_to(&mut self, block: BlockId) {
        self.current_block = block;
    }

    fn intern(&mut self, s: &str) -> SymbolId {
        self.interner.intern(s)
    }

    fn module_var_index(&self, name: SymbolId) -> Option<u16> {
        self.module_vars
            .iter()
            .position(|&n| n == name)
            .map(|i| i as u16)
    }

    /// Compute a Wren method signature from name + arity.
    /// 0 args → "name" (getter), 1+ args → "name(_)" / "name(_,_)" etc.
    fn method_sig(&mut self, name: SymbolId, arity: usize) -> SymbolId {
        self.method_sig_with_parens(name, arity, arity > 0)
    }

    /// Compute a Wren method signature, distinguishing 0-arg calls from getters.
    /// `has_parens` true + 0 args → "name()", false + 0 args → "name" (getter).
    fn method_sig_with_parens(
        &mut self,
        name: SymbolId,
        arity: usize,
        has_parens: bool,
    ) -> SymbolId {
        if arity == 0 && !has_parens {
            name
        } else {
            let name_str = self.interner.resolve(name).to_string();
            let params: Vec<&str> = std::iter::repeat_n("_", arity).collect();
            let sig = format!("{}({})", name_str, params.join(","));
            self.intern(&sig)
        }
    }

    // -- Statement lowering -------------------------------------------------

    fn lower_stmt(&mut self, stmt: &Spanned<Stmt>) {
        match &stmt.0 {
            Stmt::Expr(expr) => {
                self.lower_expr(expr);
            }

            Stmt::Var {
                name,
                initializer,
                attributes: _,
            } => {
                let val = if let Some(init) = initializer {
                    self.lower_expr(init)
                } else {
                    self.emit(Instruction::ConstNull)
                };
                // If this local is captured by a nested closure,
                // wrap it in a 1-element List so the closure's
                // upvalue (which will grab the list reference) and
                // the outer's writes share storage.
                let stored = if self.current_boxed.contains(&name.0) {
                    self.emit(Instruction::MakeList(vec![val]))
                } else {
                    val
                };
                // Inside a block scope, `var x` creates a local variable that
                // may shadow a module var or outer local. Only use the module
                // var path when at the top level (no block scope active).
                if self.block_shadows.is_empty() {
                    if let Some(idx) = self.module_var_index(name.0) {
                        self.emit(Instruction::SetModuleVar(idx, stored));
                    } else {
                        self.variables.insert(name.0, stored);
                    }
                } else {
                    // Record the shadow so we can restore on block exit.
                    if let Some(shadows) = self.block_shadows.last_mut() {
                        let old = self.variables.get(&name.0).copied();
                        shadows.push((name.0, old));
                    }
                    self.variables.insert(name.0, stored);
                }
            }

            Stmt::Block(stmts) => {
                self.block_shadows.push(Vec::new());
                for s in stmts {
                    self.lower_stmt(s);
                }
                // Restore shadowed variables and remove block-locals.
                if let Some(shadows) = self.block_shadows.pop() {
                    for (name, old_val) in shadows.into_iter().rev() {
                        match old_val {
                            Some(v) => {
                                self.variables.insert(name, v);
                            }
                            None => {
                                self.variables.remove(&name);
                            }
                        }
                    }
                }
            }

            Stmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.lower_if(condition, then_branch, else_branch.as_deref());
            }

            Stmt::While { condition, body } => {
                self.lower_while(condition, body);
            }

            Stmt::For {
                variable,
                iterator,
                body,
            } => {
                self.lower_for(variable, iterator, body);
            }

            Stmt::Break => {
                if let Some((target, tracked_vars)) = self.break_targets.last() {
                    let target = *target;
                    let args: Vec<ValueId> = tracked_vars
                        .iter()
                        .map(|name| self.variables[name])
                        .collect();
                    self.set_terminator(Terminator::Branch { target, args });
                    let dead = self.new_block();
                    self.switch_to(dead);
                }
            }

            Stmt::Continue => {
                if let Some((target, tracked_vars)) = self.continue_targets.last() {
                    let target = *target;
                    let args: Vec<ValueId> = tracked_vars
                        .iter()
                        .map(|name| self.variables[name])
                        .collect();
                    self.set_terminator(Terminator::Branch { target, args });
                    let dead = self.new_block();
                    self.switch_to(dead);
                }
            }

            Stmt::Return(expr) => {
                if let Some(e) = expr {
                    let val = self.lower_expr(e);
                    self.set_terminator(Terminator::Return(val));
                } else {
                    self.set_terminator(Terminator::ReturnNull);
                }
                let dead = self.new_block();
                self.switch_to(dead);
            }

            Stmt::Class(_) | Stmt::Import { .. } => {}
        }
    }

    fn lower_if(
        &mut self,
        condition: &Spanned<Expr>,
        then_branch: &Spanned<Stmt>,
        else_branch: Option<&Spanned<Stmt>>,
    ) {
        let cond_val = self.lower_expr(condition);

        let then_bb = self.new_block();
        let else_bb = self.new_block();
        let merge_bb = self.new_block();

        self.set_terminator(Terminator::CondBranch {
            condition: cond_val,
            true_target: then_bb,
            true_args: vec![],
            false_target: else_bb,
            false_args: vec![],
        });

        // Snapshot variables before branches
        let vars_before = self.variables.clone();

        // Lower then branch
        self.switch_to(then_bb);
        self.lower_stmt(then_branch);
        let then_vars = self.variables.clone();
        let then_exit_block = self.current_block;
        let then_returns = !matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        );

        // Restore variables for else branch
        self.variables = vars_before.clone();

        // Lower else branch
        self.switch_to(else_bb);
        if let Some(else_stmt) = else_branch {
            self.lower_stmt(else_stmt);
        }
        let else_vars = self.variables.clone();
        let else_exit_block = self.current_block;
        let else_returns = !matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        );

        // Find variables that differ between branches and create merge block params
        let mut merge_args_then = Vec::new();
        let mut merge_args_else = Vec::new();

        // Collect all variable names from both branches
        let all_names: Vec<SymbolId> = {
            let mut names: Vec<SymbolId> = vars_before.keys().copied().collect();
            for name in then_vars.keys() {
                if !names.contains(name) {
                    names.push(*name);
                }
            }
            for name in else_vars.keys() {
                if !names.contains(name) {
                    names.push(*name);
                }
            }
            names
        };

        for &name in &all_names {
            let then_val = then_vars.get(&name).copied();
            let else_val = else_vars.get(&name).copied();
            let before_val = vars_before.get(&name).copied();

            // If both branches agree, no phi needed
            if then_val == else_val {
                if let Some(v) = then_val {
                    self.variables.insert(name, v);
                }
                continue;
            }

            // Need a merge param
            let phi = self.func.new_value();
            self.func
                .block_mut(merge_bb)
                .params
                .push((phi, MirType::Value));

            // then branch value: use then_val if available, otherwise before_val, else null
            let tv = then_val.or(before_val).unwrap_or_else(|| {
                let saved = self.current_block;
                self.current_block = then_exit_block;
                let v = self.emit(Instruction::ConstNull);
                self.current_block = saved;
                v
            });
            merge_args_then.push(tv);

            // else branch value
            let ev = else_val.or(before_val).unwrap_or_else(|| {
                let saved = self.current_block;
                self.current_block = else_exit_block;
                let v = self.emit(Instruction::ConstNull);
                self.current_block = saved;
                v
            });
            merge_args_else.push(ev);

            self.variables.insert(name, phi);
        }

        // Wire branches to merge block
        if !then_returns {
            self.func.block_mut(then_exit_block).terminator = Terminator::Branch {
                target: merge_bb,
                args: merge_args_then,
            };
        }
        if !else_returns {
            self.func.block_mut(else_exit_block).terminator = Terminator::Branch {
                target: merge_bb,
                args: merge_args_else,
            };
        }

        self.switch_to(merge_bb);
    }

    fn lower_while(&mut self, condition: &Spanned<Expr>, body: &Spanned<Stmt>) {
        let cond_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();

        // Pre-create phi block params in cond_bb for ALL variables.
        // This ensures both condition and body use phi values, so mutations
        // in the body propagate correctly back to the condition via the back-edge.
        let mut vars_snapshot: Vec<(SymbolId, ValueId)> =
            self.variables.iter().map(|(&k, &v)| (k, v)).collect();
        // Sort by SymbolId to ensure deterministic block parameter ordering
        // across platforms. HashMap iteration order is non-deterministic,
        // which produces different (but semantically valid) parameter orderings
        // that can expose register allocation bugs on specific platforms.
        vars_snapshot.sort_by_key(|&(k, _)| k.index());

        let mut entry_args = Vec::new();
        let mut phi_map: Vec<(SymbolId, ValueId)> = Vec::new();
        let tracked_names: Vec<SymbolId> = vars_snapshot.iter().map(|&(k, _)| k).collect();

        for &(name, initial_val) in &vars_snapshot {
            let phi = self.func.new_value();
            self.func
                .block_mut(cond_bb)
                .params
                .push((phi, MirType::Value));
            entry_args.push(initial_val);
            phi_map.push((name, phi));
            self.variables.insert(name, phi);
        }

        // Also create phi params in exit_bb so break and normal exit can
        // propagate variable values to post-loop code.
        let mut exit_phi_map: Vec<(SymbolId, ValueId)> = Vec::new();
        for &(name, _) in &vars_snapshot {
            let exit_phi = self.func.new_value();
            self.func
                .block_mut(exit_bb)
                .params
                .push((exit_phi, MirType::Value));
            exit_phi_map.push((name, exit_phi));
        }

        // Branch from current block to cond_bb with initial values
        self.set_terminator(Terminator::Branch {
            target: cond_bb,
            args: entry_args,
        });

        // Lower condition (uses phi values)
        self.switch_to(cond_bb);
        let cond_val = self.lower_expr(condition);
        // Normal loop exit passes current (phi) values to exit_bb
        let exit_args: Vec<ValueId> = phi_map.iter().map(|&(_, phi)| phi).collect();
        self.set_terminator(Terminator::CondBranch {
            condition: cond_val,
            true_target: body_bb,
            true_args: vec![],
            false_target: exit_bb,
            false_args: exit_args,
        });

        // Lower body (cond_bb dominates body_bb, so phi values are accessible)
        self.switch_to(body_bb);
        self.break_targets.push((exit_bb, tracked_names.clone()));
        self.continue_targets.push((cond_bb, tracked_names));
        self.lower_stmt(body);
        self.break_targets.pop();
        self.continue_targets.pop();

        // Back-edge: pass current (possibly mutated) variable values
        let backedge_args: Vec<ValueId> = phi_map
            .iter()
            .map(|&(name, _)| self.variables[&name])
            .collect();

        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.set_terminator(Terminator::Branch {
                target: cond_bb,
                args: backedge_args,
            });
        }

        // Use exit_bb phi values for post-loop code (these receive values
        // from both the normal condition-false exit and any break statements).
        for &(name, exit_phi) in &exit_phi_map {
            self.variables.insert(name, exit_phi);
        }

        self.switch_to(exit_bb);
        // Emit a null so the while-statement "evaluates" to null (not a
        // variable phi).  Ensures closure implicit-return picks up null.
        self.emit(Instruction::ConstNull);
    }

    fn lower_for(
        &mut self,
        variable: &Spanned<SymbolId>,
        iterator: &Spanned<Expr>,
        body: &Spanned<Stmt>,
    ) {
        let iter_obj = self.lower_expr(iterator);

        let iterate_sym = self.intern("iterate");
        let iterate_sig = self.method_sig(iterate_sym, 1);
        let iter_value_sym = self.intern("iteratorValue");
        let iter_value_sig = self.method_sig(iter_value_sym, 1);

        let null_val = self.emit(Instruction::ConstNull);
        let iter_val = self.emit(Instruction::Call {
            receiver: iter_obj,
            method: iterate_sig,
            args: vec![null_val],
        });

        let cond_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();

        // Create phi block params in cond_bb for the iterator state AND all
        // live variables, so mutations in the body propagate through the
        // back-edge (same approach as lower_while).
        let mut vars_snapshot: Vec<(SymbolId, ValueId)> =
            self.variables.iter().map(|(&k, &v)| (k, v)).collect();
        // Deterministic ordering (see lower_while comment).
        vars_snapshot.sort_by_key(|&(k, _)| k.index());
        let tracked_names: Vec<SymbolId> = vars_snapshot.iter().map(|&(k, _)| k).collect();

        // First param: iterator state
        let iter_param = self.func.new_value();
        self.func
            .block_mut(cond_bb)
            .params
            .push((iter_param, MirType::Value));

        // Additional params: one per live variable
        let mut entry_args = vec![iter_val];
        let mut phi_map: Vec<(SymbolId, ValueId)> = Vec::new();

        for &(name, initial_val) in &vars_snapshot {
            let phi = self.func.new_value();
            self.func
                .block_mut(cond_bb)
                .params
                .push((phi, MirType::Value));
            entry_args.push(initial_val);
            phi_map.push((name, phi));
            self.variables.insert(name, phi);
        }

        // Create phi params in exit_bb for variable propagation from break + normal exit
        let mut exit_phi_map: Vec<(SymbolId, ValueId)> = Vec::new();
        for &(name, _) in &vars_snapshot {
            let exit_phi = self.func.new_value();
            self.func
                .block_mut(exit_bb)
                .params
                .push((exit_phi, MirType::Value));
            exit_phi_map.push((name, exit_phi));
        }

        self.set_terminator(Terminator::Branch {
            target: cond_bb,
            args: entry_args,
        });

        self.switch_to(cond_bb);
        let is_falsy = self.emit(Instruction::Not(iter_param));
        // Normal exit passes current phi values to exit_bb
        let exit_args: Vec<ValueId> = phi_map.iter().map(|&(_, phi)| phi).collect();
        self.set_terminator(Terminator::CondBranch {
            condition: is_falsy,
            true_target: exit_bb,
            true_args: exit_args,
            false_target: body_bb,
            false_args: vec![],
        });

        self.switch_to(body_bb);
        let elem_val = self.emit(Instruction::Call {
            receiver: iter_obj,
            method: iter_value_sig,
            args: vec![iter_param],
        });
        self.variables.insert(variable.0, elem_val);

        self.break_targets.push((exit_bb, tracked_names.clone()));
        self.continue_targets.push((cond_bb, tracked_names));
        self.lower_stmt(body);
        self.break_targets.pop();
        self.continue_targets.pop();

        let next_iter = self.emit(Instruction::Call {
            receiver: iter_obj,
            method: iterate_sig,
            args: vec![iter_param],
        });

        // Back-edge: pass iterator state + current (possibly mutated) variable values
        let mut backedge_args = vec![next_iter];
        for &(name, _) in &phi_map {
            backedge_args.push(self.variables[&name]);
        }

        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.set_terminator(Terminator::Branch {
                target: cond_bb,
                args: backedge_args,
            });
        }

        // Use exit_bb phi values for post-loop code
        for &(name, exit_phi) in &exit_phi_map {
            self.variables.insert(name, exit_phi);
        }

        self.switch_to(exit_bb);
        // Emit a null so the for-statement "evaluates" to null (not a
        // variable phi).  Ensures closure implicit-return picks up null.
        self.emit(Instruction::ConstNull);
    }

    // -- Expression lowering ------------------------------------------------

    fn lower_expr(&mut self, expr: &Spanned<Expr>) -> ValueId {
        let val = self.lower_expr_inner(expr);
        self.func.span_map.insert(val, expr.1.clone());
        val
    }

    fn lower_expr_inner(&mut self, expr: &Spanned<Expr>) -> ValueId {
        match &expr.0 {
            Expr::Num(n) => self.emit(Instruction::ConstNum(*n)),
            Expr::Str(s) => {
                let sym = self.interner.intern(s);
                self.emit(Instruction::ConstString(sym.index()))
            }
            Expr::Bool(b) => self.emit(Instruction::ConstBool(*b)),
            Expr::Null => self.emit(Instruction::ConstNull),
            Expr::This => {
                let this_sym = self.intern("this");
                self.variables
                    .get(&this_sym)
                    .copied()
                    .unwrap_or_else(|| self.emit(Instruction::ConstNull))
            }

            Expr::Ident(name) => {
                if let Some(&val) = self.variables.get(name) {
                    if self.current_boxed.contains(name) {
                        // Boxed local: the SSA value is a list ref.
                        // Deref via subscript so reads see whatever
                        // any nested closure most recently wrote.
                        let zero = self.emit(Instruction::ConstNum(0.0));
                        self.emit(Instruction::SubscriptGet {
                            receiver: val,
                            args: vec![zero],
                        })
                    } else {
                        self.emit(Instruction::Move(val))
                    }
                } else if let Some(resolved) = self.resolutions.get(&expr.1.start) {
                    match resolved {
                        ResolvedName::ModuleVar(idx) => self.emit(Instruction::GetModuleVar(*idx)),
                        ResolvedName::Upvalue(idx) => {
                            let raw = self.emit(Instruction::GetUpvalue(*idx));
                            if self.upvalues_are_boxed {
                                // The upvalue holds a list reference
                                // to a boxed outer-scope local. Deref
                                // element 0 to get the current value.
                                let zero = self.emit(Instruction::ConstNum(0.0));
                                self.emit(Instruction::SubscriptGet {
                                    receiver: raw,
                                    args: vec![zero],
                                })
                            } else {
                                raw
                            }
                        }
                        ResolvedName::ImplicitThis(method_name) => {
                            // Bare identifier in a method → implicit `this.name` getter.
                            let this_sym = self.intern("this");
                            let recv = self
                                .variables
                                .get(&this_sym)
                                .copied()
                                .unwrap_or_else(|| self.emit(Instruction::ConstNull));
                            self.emit(Instruction::Call {
                                receiver: recv,
                                method: *method_name,
                                args: vec![],
                            })
                        }
                        ResolvedName::Local(_) => {
                            // Local should have been in variables map; fallback
                            self.emit(Instruction::ConstNull)
                        }
                    }
                } else {
                    // Identifier not in variables map and not in resolutions.
                    // Try to find it as a module var by name (safety net).
                    let found = self.module_vars.iter().position(|s| s == name);
                    if let Some(idx) = found {
                        self.emit(Instruction::GetModuleVar(idx as u16))
                    } else {
                        self.emit(Instruction::ConstNull)
                    }
                }
            }

            Expr::Field(name) => {
                let this_sym = self.intern("this");
                let this_val = self
                    .variables
                    .get(&this_sym)
                    .copied()
                    .unwrap_or_else(|| self.emit(Instruction::ConstNull));
                let idx = self
                    .field_map
                    .as_ref()
                    .and_then(|m| m.get(name).copied())
                    .unwrap_or(0);
                self.emit(Instruction::GetField(this_val, idx))
            }
            Expr::StaticField(name) => self.emit(Instruction::GetStaticField(*name)),

            Expr::Interpolation(parts) => {
                let mut vals = Vec::new();
                for part in parts {
                    let v = self.lower_expr(part);
                    let s = self.emit(Instruction::ToString(v));
                    vals.push(s);
                }
                self.emit(Instruction::StringConcat(vals))
            }

            Expr::UnaryOp { op, operand } => {
                let val = self.lower_expr(operand);
                match op {
                    UnaryOp::Neg => self.emit(Instruction::Neg(val)),
                    UnaryOp::Not => self.emit(Instruction::Not(val)),
                    UnaryOp::BNot => self.emit(Instruction::BitNot(val)),
                }
            }

            Expr::BinaryOp { op, left, right } => {
                let lhs = self.lower_expr(left);
                let rhs = self.lower_expr(right);
                // Check if both operands are known-Num. Use AST structure for
                // literals (immune to span-key collisions in TypeEnv), TypeEnv
                // for variables/fields/calls.
                let is_num_expr = |e: &Spanned<Expr>, env: &crate::sema::types::TypeEnv| -> bool {
                    match &e.0 {
                        Expr::Num(_) => true,
                        Expr::UnaryOp {
                            op: UnaryOp::Neg,
                            operand,
                        } => matches!(operand.0, Expr::Num(_)),
                        _ => env.get_expr_type(e.1.start).is_num(),
                    }
                };
                let both_num = self
                    .type_env
                    .as_ref()
                    .is_some_and(|env| is_num_expr(left, env) && is_num_expr(right, env));
                if both_num {
                    // Emit unboxed f64 instructions directly — no CallRuntime.
                    match op {
                        BinaryOp::Add
                        | BinaryOp::Sub
                        | BinaryOp::Mul
                        | BinaryOp::Div
                        | BinaryOp::Mod => {
                            let ua = self.emit(Instruction::Unbox(lhs));
                            let ub = self.emit(Instruction::Unbox(rhs));
                            let r = self.emit(match op {
                                BinaryOp::Add => Instruction::AddF64(ua, ub),
                                BinaryOp::Sub => Instruction::SubF64(ua, ub),
                                BinaryOp::Mul => Instruction::MulF64(ua, ub),
                                BinaryOp::Div => Instruction::DivF64(ua, ub),
                                BinaryOp::Mod => Instruction::ModF64(ua, ub),
                                _ => unreachable!(),
                            });
                            self.emit(Instruction::Box(r))
                        }
                        BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => {
                            let ua = self.emit(Instruction::Unbox(lhs));
                            let ub = self.emit(Instruction::Unbox(rhs));
                            self.emit(match op {
                                BinaryOp::Lt => Instruction::CmpLtF64(ua, ub),
                                BinaryOp::Gt => Instruction::CmpGtF64(ua, ub),
                                BinaryOp::LtEq => Instruction::CmpLeF64(ua, ub),
                                BinaryOp::GtEq => Instruction::CmpGeF64(ua, ub),
                                _ => unreachable!(),
                            })
                        }
                        _ => {
                            // Eq/NotEq/BitOps: keep boxed (type-safe path)
                            let inst = match op {
                                BinaryOp::Eq => Instruction::CmpEq(lhs, rhs),
                                BinaryOp::NotEq => Instruction::CmpNe(lhs, rhs),
                                BinaryOp::BitAnd => Instruction::BitAnd(lhs, rhs),
                                BinaryOp::BitOr => Instruction::BitOr(lhs, rhs),
                                BinaryOp::BitXor => Instruction::BitXor(lhs, rhs),
                                BinaryOp::Shl => Instruction::Shl(lhs, rhs),
                                BinaryOp::Shr => Instruction::Shr(lhs, rhs),
                                _ => unreachable!(),
                            };
                            self.emit(inst)
                        }
                    }
                } else {
                    let inst = match op {
                        BinaryOp::Add => Instruction::Add(lhs, rhs),
                        BinaryOp::Sub => Instruction::Sub(lhs, rhs),
                        BinaryOp::Mul => Instruction::Mul(lhs, rhs),
                        BinaryOp::Div => Instruction::Div(lhs, rhs),
                        BinaryOp::Mod => Instruction::Mod(lhs, rhs),
                        BinaryOp::Lt => Instruction::CmpLt(lhs, rhs),
                        BinaryOp::Gt => Instruction::CmpGt(lhs, rhs),
                        BinaryOp::LtEq => Instruction::CmpLe(lhs, rhs),
                        BinaryOp::GtEq => Instruction::CmpGe(lhs, rhs),
                        BinaryOp::Eq => Instruction::CmpEq(lhs, rhs),
                        BinaryOp::NotEq => Instruction::CmpNe(lhs, rhs),
                        BinaryOp::BitAnd => Instruction::BitAnd(lhs, rhs),
                        BinaryOp::BitOr => Instruction::BitOr(lhs, rhs),
                        BinaryOp::BitXor => Instruction::BitXor(lhs, rhs),
                        BinaryOp::Shl => Instruction::Shl(lhs, rhs),
                        BinaryOp::Shr => Instruction::Shr(lhs, rhs),
                    };
                    self.emit(inst)
                }
            }

            Expr::LogicalOp { op, left, right } => self.lower_logical(*op, left, right),

            Expr::Is { value, type_name } => {
                // Lower `x is Klass` uniformly as `x.is(Klass)` so
                // it works for *any* class-valued RHS — a literal
                // class name, a parameter, a field access, or the
                // result of `.type`. The previous fast-path used
                // `IsType` with a compile-time symbol, which
                // silently returned false when the RHS wasn't a
                // literal class identifier (e.g. `is klass` where
                // `klass` was a local holding a class).
                //
                // `Object.is(_)` compares actual class pointers
                // while walking the superclass chain; a future
                // devirtualization pass can fold it back to
                // `IsType` when the RHS is statically known to
                // reference a core class.
                let val = self.lower_expr(value);
                let class_val = self.lower_expr(type_name);
                let sig = self.intern("is(_)");
                self.emit(Instruction::Call {
                    receiver: val,
                    method: sig,
                    args: vec![class_val],
                })
            }

            Expr::Assign { target, value } => {
                let val = self.lower_expr(value);
                self.lower_assign(target, val);
                val
            }

            Expr::CompoundAssign { op, target, value } => {
                let lhs = self.lower_expr(target);
                let rhs = self.lower_expr(value);
                let result = match op {
                    BinaryOp::Add => self.emit(Instruction::Add(lhs, rhs)),
                    BinaryOp::Sub => self.emit(Instruction::Sub(lhs, rhs)),
                    BinaryOp::Mul => self.emit(Instruction::Mul(lhs, rhs)),
                    BinaryOp::Div => self.emit(Instruction::Div(lhs, rhs)),
                    BinaryOp::Mod => self.emit(Instruction::Mod(lhs, rhs)),
                    _ => self.emit(Instruction::Add(lhs, rhs)),
                };
                self.lower_assign(target, result);
                result
            }

            Expr::Call {
                receiver,
                method,
                args,
                block_arg,
                has_parens,
            } => {
                let mut arg_vals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                if let Some(block) = block_arg {
                    let block_val = self.lower_expr(block);
                    arg_vals.push(block_val);
                }

                let sig = self.method_sig_with_parens(method.0, arg_vals.len(), *has_parens);
                let is_static_self_call = self.current_method_is_static
                    && self.current_method_sig == Some(sig)
                    && receiver.as_ref().is_some_and(|recv_expr| {
                        matches!(
                            &recv_expr.0,
                            Expr::Ident(sym) if Some(*sym) == self.current_class_name
                        )
                    });
                if is_static_self_call {
                    return self.emit(Instruction::CallStaticSelf { args: arg_vals });
                }

                let recv = if let Some(r) = receiver {
                    self.lower_expr(r)
                } else {
                    let this_sym = self.intern("this");
                    self.variables
                        .get(&this_sym)
                        .copied()
                        .unwrap_or_else(|| self.emit(Instruction::ConstNull))
                };
                self.emit(Instruction::Call {
                    receiver: recv,
                    method: sig,
                    args: arg_vals,
                })
            }

            Expr::SuperCall {
                method,
                args,
                has_parens,
            } => {
                // Include `this` as first arg so the VM can use it as receiver
                let this_sym = self.intern("this");
                let this_val = self
                    .variables
                    .get(&this_sym)
                    .copied()
                    .unwrap_or_else(|| self.emit(Instruction::ConstNull));
                let mut all_args = vec![this_val];
                all_args.extend(args.iter().map(|a| self.lower_expr(a)));

                let method_sym = if let Some(m) = method {
                    // super.method(args) or super.getter
                    self.method_sig_with_parens(m.0, args.len(), *has_parens)
                } else {
                    // super(args) — call same-named constructor/method on super.
                    // Use the base name with the CALLER's arg count, not the
                    // current method's full signature (which may have different arity).
                    if let Some(base) = self.current_method_base_name {
                        self.method_sig_with_parens(base, args.len(), *has_parens)
                    } else {
                        self.current_method_sig
                            .unwrap_or_else(|| self.intern("init"))
                    }
                };
                self.emit(Instruction::SuperCall {
                    method: method_sym,
                    args: all_args,
                })
            }

            Expr::Subscript { receiver, args } => {
                let recv = self.lower_expr(receiver);
                let arg_vals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                self.emit(Instruction::SubscriptGet {
                    receiver: recv,
                    args: arg_vals,
                })
            }

            Expr::SubscriptSet {
                receiver,
                index_args,
                value,
            } => {
                let recv = self.lower_expr(receiver);
                let arg_vals: Vec<ValueId> =
                    index_args.iter().map(|a| self.lower_expr(a)).collect();
                let val = self.lower_expr(value);
                self.emit(Instruction::SubscriptSet {
                    receiver: recv,
                    args: arg_vals,
                    value: val,
                })
            }

            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => self.lower_conditional(condition, then_expr, else_expr),

            Expr::ListLiteral(elements) => {
                let vals: Vec<ValueId> = elements.iter().map(|e| self.lower_expr(e)).collect();
                self.emit(Instruction::MakeList(vals))
            }

            Expr::MapLiteral(entries) => {
                let pairs: Vec<(ValueId, ValueId)> = entries
                    .iter()
                    .map(|(k, v)| (self.lower_expr(k), self.lower_expr(v)))
                    .collect();
                self.emit(Instruction::MakeMap(pairs))
            }

            Expr::Range {
                from,
                to,
                inclusive,
            } => {
                let from_val = self.lower_expr(from);
                let to_val = self.lower_expr(to);
                self.emit(Instruction::MakeRange(from_val, to_val, *inclusive))
            }

            Expr::Closure { params, body } => {
                let scope_id = expr.1.start;
                let fn_idx = self.compile_closure(params, body, scope_id);
                // Build upvalue capture list from sema info
                let upvalue_vals = if let Some(uv_info) = self.upvalue_map.get(&scope_id) {
                    uv_info
                        .iter()
                        .map(|info| {
                            if info.is_local {
                                // Capture a local variable from this scope by name
                                self.variables
                                    .get(&info.name)
                                    .copied()
                                    .unwrap_or_else(|| self.emit(Instruction::ConstNull))
                            } else {
                                // Capture an upvalue from this scope (we're inside a closure ourselves)
                                self.emit(Instruction::GetUpvalue(info.index))
                            }
                        })
                        .collect()
                } else {
                    vec![]
                };
                self.emit(Instruction::MakeClosure {
                    fn_id: fn_idx,
                    upvalues: upvalue_vals,
                })
            }
        }
    }

    fn lower_logical(
        &mut self,
        op: LogicalOp,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
    ) -> ValueId {
        let lhs = self.lower_expr(left);

        let rhs_bb = self.new_block();
        let merge_bb = self.new_block();

        let result = self.func.new_value();
        self.func
            .block_mut(merge_bb)
            .params
            .push((result, MirType::Value));

        match op {
            LogicalOp::And => {
                self.set_terminator(Terminator::CondBranch {
                    condition: lhs,
                    true_target: rhs_bb,
                    true_args: vec![],
                    false_target: merge_bb,
                    false_args: vec![lhs],
                });
            }
            LogicalOp::Or => {
                self.set_terminator(Terminator::CondBranch {
                    condition: lhs,
                    true_target: merge_bb,
                    true_args: vec![lhs],
                    false_target: rhs_bb,
                    false_args: vec![],
                });
            }
        }

        self.switch_to(rhs_bb);
        let rhs = self.lower_expr(right);
        self.set_terminator(Terminator::Branch {
            target: merge_bb,
            args: vec![rhs],
        });

        self.switch_to(merge_bb);
        result
    }

    fn lower_conditional(
        &mut self,
        condition: &Spanned<Expr>,
        then_expr: &Spanned<Expr>,
        else_expr: &Spanned<Expr>,
    ) -> ValueId {
        let cond = self.lower_expr(condition);

        let then_bb = self.new_block();
        let else_bb = self.new_block();
        let merge_bb = self.new_block();

        let result = self.func.new_value();
        self.func
            .block_mut(merge_bb)
            .params
            .push((result, MirType::Value));

        self.set_terminator(Terminator::CondBranch {
            condition: cond,
            true_target: then_bb,
            true_args: vec![],
            false_target: else_bb,
            false_args: vec![],
        });

        self.switch_to(then_bb);
        let then_val = self.lower_expr(then_expr);
        self.set_terminator(Terminator::Branch {
            target: merge_bb,
            args: vec![then_val],
        });

        self.switch_to(else_bb);
        let else_val = self.lower_expr(else_expr);
        self.set_terminator(Terminator::Branch {
            target: merge_bb,
            args: vec![else_val],
        });

        self.switch_to(merge_bb);
        result
    }

    fn lower_assign(&mut self, target: &Spanned<Expr>, value: ValueId) {
        match &target.0 {
            Expr::Ident(name) => {
                if self.variables.contains_key(name) {
                    if self.current_boxed.contains(name) {
                        // Boxed local: write through the list box
                        // instead of rebinding the SSA slot, so a
                        // previously-captured closure sees the new
                        // value.
                        let list = *self.variables.get(name).unwrap();
                        let zero = self.emit(Instruction::ConstNum(0.0));
                        self.emit(Instruction::SubscriptSet {
                            receiver: list,
                            args: vec![zero],
                            value,
                        });
                    } else {
                        self.variables.insert(*name, value);
                    }
                } else if let Some(resolved) = self.resolutions.get(&target.1.start) {
                    match resolved {
                        ResolvedName::ModuleVar(idx) => {
                            self.emit(Instruction::SetModuleVar(*idx, value));
                        }
                        ResolvedName::Upvalue(idx) => {
                            if self.upvalues_are_boxed {
                                // Upvalue holds a list box; write
                                // into element 0 so the outer
                                // scope's local reads see it.
                                let list = self.emit(Instruction::GetUpvalue(*idx));
                                let zero = self.emit(Instruction::ConstNum(0.0));
                                self.emit(Instruction::SubscriptSet {
                                    receiver: list,
                                    args: vec![zero],
                                    value,
                                });
                            } else {
                                self.emit(Instruction::SetUpvalue(*idx, value));
                            }
                        }
                        ResolvedName::ImplicitThis(method_name) => {
                            // Bare identifier assignment in a method → `this.name=(value)`
                            let this_sym = self.intern("this");
                            let recv = self
                                .variables
                                .get(&this_sym)
                                .copied()
                                .unwrap_or_else(|| self.emit(Instruction::ConstNull));
                            let setter_name =
                                format!("{}=(_)", self.interner.resolve(*method_name));
                            let setter_sym = self.intern(&setter_name);
                            self.emit(Instruction::Call {
                                receiver: recv,
                                method: setter_sym,
                                args: vec![value],
                            });
                        }
                        _ => {
                            self.variables.insert(*name, value);
                        }
                    }
                } else {
                    self.variables.insert(*name, value);
                }
            }
            Expr::Field(name) => {
                let this_sym = self.intern("this");
                let this_val = self
                    .variables
                    .get(&this_sym)
                    .copied()
                    .unwrap_or_else(|| self.emit(Instruction::ConstNull));
                let idx = self
                    .field_map
                    .as_ref()
                    .and_then(|m| m.get(name).copied())
                    .unwrap_or(0);
                self.emit(Instruction::SetField(this_val, idx, value));
            }
            Expr::StaticField(name) => {
                self.emit(Instruction::SetStaticField(*name, value));
            }
            Expr::Subscript { receiver, args } => {
                let recv = self.lower_expr(receiver);
                let arg_vals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                self.emit(Instruction::SubscriptSet {
                    receiver: recv,
                    args: arg_vals,
                    value,
                });
            }
            Expr::Call {
                receiver: Some(recv_expr),
                method,
                args,
                ..
            } if args.is_empty() => {
                // Property setter: obj.name = value → Call obj.name=(value)
                let recv = self.lower_expr(recv_expr);
                let setter_name = format!("{}=(_)", self.interner.resolve(method.0));
                let setter_sym = self.interner.intern(&setter_name);
                self.emit(Instruction::Call {
                    receiver: recv,
                    method: setter_sym,
                    args: vec![value],
                });
            }
            _ => {}
        }
    }

    /// Compile a closure body into a separate MirFunction, returning its index.
    fn compile_closure(
        &mut self,
        params: &[Spanned<SymbolId>],
        body: &Spanned<Stmt>,
        scope_id: usize,
    ) -> u32 {
        let fn_idx = self.closure_base + self.closures.len() as u32;

        let (closure_func, nested_closures) = compile_closure_body(
            params,
            body,
            self.interner,
            self.resolutions,
            self.module_vars,
            self.upvalue_map,
            self.boxed_locals_map,
            scope_id,
            fn_idx + 1, // nested closures start after this one
        );

        self.closures.push(closure_func);
        for nested in nested_closures {
            self.closures.push(nested);
        }

        fn_idx
    }
}

/// Standalone function to compile a closure body (avoids borrow conflicts).
#[allow(clippy::too_many_arguments)]
fn compile_closure_body(
    params: &[Spanned<SymbolId>],
    body: &Spanned<Stmt>,
    interner: &mut Interner,
    resolutions: &HashMap<usize, ResolvedName>,
    module_vars: &[SymbolId],
    upvalue_map: &HashMap<usize, Vec<crate::sema::resolve::UpvalueInfo>>,
    boxed_locals_map: &HashMap<usize, std::collections::HashSet<SymbolId>>,
    scope_id: usize,
    nested_base: u32,
) -> (MirFunction, Vec<MirFunction>) {
    let closure_name = interner.intern("<closure>");
    let arity = params.len() as u8;

    let mut builder = MirBuilder::with_boxed(
        closure_name,
        arity,
        interner,
        resolutions,
        module_vars,
        upvalue_map,
        boxed_locals_map,
    );
    builder.set_scope(scope_id, true);
    builder.closure_base = nested_base;

    // Emit BlockParams contiguously FIRST so register numbering
    // stays aligned with the caller's argument setup; then box
    // the ones that a deeper closure captures.
    let raw_params: Vec<(SymbolId, ValueId)> = params
        .iter()
        .enumerate()
        .map(|(i, param)| {
            let val = builder.emit(Instruction::BlockParam(i as u16));
            (param.0, val)
        })
        .collect();
    for (name, val) in raw_params {
        let stored = builder.box_if_captured(name, val);
        builder.variables.insert(name, stored);
    }

    // Lower body
    builder.lower_stmt(body);

    // If body is a single expression statement, make it the return value
    if matches!(
        builder.func.block(builder.current_block).terminator,
        Terminator::Unreachable
    ) {
        // Check if last instruction produced a value we can return
        // Also check block params (for &&/|| merge blocks)
        let last_val = builder
            .func
            .block(builder.current_block)
            .instructions
            .last()
            .map(|(id, _)| *id)
            .or_else(|| {
                builder
                    .func
                    .block(builder.current_block)
                    .params
                    .last()
                    .map(|(id, _)| *id)
            });
        if let Some(val) = last_val {
            builder.func.block_mut(builder.current_block).terminator = Terminator::Return(val);
        } else {
            builder.func.block_mut(builder.current_block).terminator = Terminator::ReturnNull;
        }
    }

    let closures = builder.closures;
    (builder.func, closures)
}

/// Compute a Wren method signature string from a MethodSig AST node.
fn wren_signature(sig: &MethodSig, interner: &Interner) -> String {
    match sig {
        MethodSig::Named { name, params } => {
            let n = interner.resolve(*name);
            if params.is_empty() {
                // Named with 0 params is a method call with parens but no args: "foo()"
                format!("{}()", n)
            } else {
                let us: Vec<&str> = std::iter::repeat_n("_", params.len()).collect();
                format!("{}({})", n, us.join(","))
            }
        }
        MethodSig::Getter(name) => interner.resolve(*name).to_string(),
        MethodSig::Setter { name, .. } => format!("{}=(_)", interner.resolve(*name)),
        MethodSig::Construct { name, params } => {
            let n = interner.resolve(*name);
            if params.is_empty() {
                format!("{}()", n)
            } else {
                let us: Vec<&str> = std::iter::repeat_n("_", params.len()).collect();
                format!("{}({})", n, us.join(","))
            }
        }
        MethodSig::Operator { op, params } => {
            let op_str = match op {
                Op::Plus => "+",
                Op::Minus | Op::Neg => "-",
                Op::Star => "*",
                Op::Slash => "/",
                Op::Percent => "%",
                Op::Lt => "<",
                Op::Gt => ">",
                Op::LtEq => "<=",
                Op::GtEq => ">=",
                Op::EqEq => "==",
                Op::BangEq => "!=",
                Op::BitAnd => "&",
                Op::BitOr => "|",
                Op::BitXor => "^",
                Op::Shl => "<<",
                Op::Shr => ">>",
                Op::Bang => "!",
                Op::Tilde => "~",
                Op::DotDot => "..",
                Op::DotDotDot => "...",
            };
            if params.is_empty() {
                op_str.to_string()
            } else {
                let us: Vec<&str> = std::iter::repeat_n("_", params.len()).collect();
                format!("{}({})", op_str, us.join(","))
            }
        }
        MethodSig::Subscript { params } => {
            let us: Vec<&str> = std::iter::repeat_n("_", params.len()).collect();
            format!("[{}]", us.join(","))
        }
        MethodSig::SubscriptSetter { params, .. } => {
            let us: Vec<&str> = std::iter::repeat_n("_", params.len()).collect();
            format!("[{}]=(_)", us.join(","))
        }
    }
}

/// Get the parameter list from a MethodSig for compiling the body.
/// Extract the bare method name (without arity) from a method signature.
fn method_base_name(sig: &MethodSig) -> Option<SymbolId> {
    match sig {
        MethodSig::Named { name, .. }
        | MethodSig::Construct { name, .. }
        | MethodSig::Getter(name)
        | MethodSig::Setter { name, .. } => Some(*name),
        _ => None,
    }
}

fn method_params(sig: &MethodSig) -> Vec<&Spanned<SymbolId>> {
    match sig {
        MethodSig::Named { params, .. }
        | MethodSig::Construct { params, .. }
        | MethodSig::Operator { params, .. }
        | MethodSig::Subscript { params } => params.iter().collect(),
        MethodSig::Setter { param, .. } => vec![param],
        MethodSig::SubscriptSetter { params, value } => {
            let mut v: Vec<&Spanned<SymbolId>> = params.iter().collect();
            v.push(value);
            v
        }
        MethodSig::Getter(_) => Vec::new(),
    }
}

/// Convenience: lower a parsed module to MIR (including class definitions).
///
/// Thin wrapper around [`lower_module_with_known_classes`] that passes an
/// empty known-classes table. Callers with a live VM should prefer the
/// `_with_known_classes` variant so cross-module inheritance works.
pub fn lower_module(
    module: &Module,
    interner: &mut Interner,
    resolve: &ResolveResult,
) -> ModuleMir {
    lower_module_with_known_classes(module, interner, resolve, &HashMap::new()).0
}

/// Lower a module to MIR, threading a cross-module field-layout table so
/// subclasses whose parent lives in a different module inherit the right
/// field slots.
///
/// `known_classes` maps class name → ordered list of field names (inherited +
/// own). Returned alongside the `ModuleMir` is a `new_layouts` map describing
/// every class defined in this module, ready to merge back into the caller's
/// registry before the next compile.
pub fn lower_module_with_known_classes(
    module: &Module,
    interner: &mut Interner,
    resolve: &ResolveResult,
    known_classes: &HashMap<String, Vec<String>>,
) -> (ModuleMir, HashMap<String, Vec<String>>) {
    // Run type inference to enable typed instruction emission.
    let type_env = crate::sema::types::infer_types(module);

    let name = interner.intern("<module>");
    let mut builder = MirBuilder::with_boxed(
        name,
        0,
        interner,
        &resolve.resolutions,
        &resolve.module_vars,
        &resolve.upvalues,
        &resolve.boxed_locals,
    )
    .with_type_env(type_env);
    // Module scope has scope_id=0 in sema; mirror that here so
    // top-level block locals that get captured get boxed too.
    builder.set_scope(0, false);
    let (top_level, closures) = builder.build_module(module);

    // Compile class definitions.
    // Track each class's full field map (inherited + own) so subclasses can
    // inherit correct field indices from their parent without re-assigning them.
    // `class_field_maps` is keyed by the class name's SymbolId *in this
    // module's interner* — newly-defined classes land directly. For classes
    // that only exist in another module, `known_classes` is consulted via
    // name lookup when resolving a superclass reference.
    let mut class_field_maps: HashMap<SymbolId, HashMap<SymbolId, u16>> = HashMap::new();
    // Full ordered field-name layout for every class we compile, so the
    // caller can extend its cross-module registry.
    let mut emitted_layouts: HashMap<String, Vec<String>> = HashMap::new();
    let mut classes = Vec::new();
    let mut all_closures = closures;
    for stmt in module {
        if let Stmt::Class(decl) = &stmt.0 {
            // Look up the parent. Prefer an in-module match (already in
            // `class_field_maps`); fall back to `known_classes` for an
            // imported parent.
            let parent_field_map: HashMap<SymbolId, u16> = if let Some(sc) = &decl.superclass {
                if let Some(map) = class_field_maps.get(&sc.0) {
                    map.clone()
                } else {
                    let sc_name = interner.resolve(sc.0).to_string();
                    match known_classes.get(&sc_name) {
                        Some(names) => names
                            .iter()
                            .enumerate()
                            .map(|(i, n)| (interner.intern(n), i as u16))
                            .collect(),
                        None => HashMap::new(),
                    }
                }
            } else {
                HashMap::new()
            };
            let inherited_fields = parent_field_map.len() as u16;
            let (class_mir, method_closures) = compile_class(
                decl,
                interner,
                resolve,
                inherited_fields,
                all_closures.len() as u32,
                &parent_field_map,
            );
            // Build the full field map for this class (inherited + own) so subclasses
            // can look up ALL fields (including inherited ones) with correct indices.
            let mut full_map = parent_field_map;
            let mut own_idx: u16 = 0;
            let mut seen: Vec<SymbolId> = Vec::new();
            for method_spanned in &decl.methods {
                if let Some(body) = &method_spanned.0.body {
                    scan_fields(body, interner, &mut seen);
                }
            }
            for sym in seen {
                if let std::collections::hash_map::Entry::Vacant(e) = full_map.entry(sym) {
                    e.insert(inherited_fields + own_idx);
                    own_idx += 1;
                }
            }
            // Reconstruct the ordered-by-slot list of field names for the
            // cross-module registry.
            let mut ordered: Vec<(u16, String)> = full_map
                .iter()
                .map(|(sym, idx)| (*idx, interner.resolve(*sym).to_string()))
                .collect();
            ordered.sort_by_key(|(idx, _)| *idx);
            let class_name = interner.resolve(decl.name.0).to_string();
            emitted_layouts.insert(
                class_name,
                ordered.into_iter().map(|(_, n)| n).collect(),
            );
            class_field_maps.insert(decl.name.0, full_map);
            classes.push(class_mir);
            all_closures.extend(method_closures);
        }
    }

    (
        ModuleMir {
            top_level,
            classes,
            closures: all_closures,
        },
        emitted_layouts,
    )
}

/// Compile a class declaration into ClassMir (plus any closures found in method bodies).
fn compile_class(
    decl: &ClassDecl,
    interner: &mut Interner,
    resolve: &ResolveResult,
    inherited_field_offset: u16,
    closure_base_offset: u32,
    parent_field_map: &HashMap<SymbolId, u16>,
) -> (ClassMir, Vec<MirFunction>) {
    let mut methods = Vec::new();
    let mut foreign_methods: Vec<ForeignMethodMir> = Vec::new();
    let mut field_names: Vec<SymbolId> = Vec::new();
    let mut all_closures: Vec<MirFunction> = Vec::new();

    // First pass: scan all methods for field accesses to determine field count.
    for method_spanned in &decl.methods {
        let method = &method_spanned.0;
        if let Some(body) = &method.body {
            scan_fields(body, interner, &mut field_names);
        }
    }

    // Second pass: compile method bodies.
    for method_spanned in &decl.methods {
        let method = &method_spanned.0;
        if method.is_foreign || method.body.is_none() {
            if method.is_foreign {
                foreign_methods.push(ForeignMethodMir {
                    signature: wren_signature(&method.signature, interner),
                    is_static: method.is_static,
                    symbol: extract_compile_time_string(&method.attributes, interner, "symbol"),
                });
            }
            continue;
        }

        let sig_str = wren_signature(&method.signature, interner);
        let is_constructor = matches!(&method.signature, MethodSig::Construct { .. });

        let method_name = interner.intern(&sig_str);
        let params = method_params(&method.signature);
        let arity = params.len() as u8 + 1; // +1 for receiver (this)

        // Method's scope_id matches what sema used: the method
        // AST node's span start.
        let method_scope_id = method_spanned.1.start;

        let mut builder = MirBuilder::with_boxed(
            method_name,
            arity,
            interner,
            &resolve.resolutions,
            &resolve.module_vars,
            &resolve.upvalues,
            &resolve.boxed_locals,
        );
        builder.set_scope(method_scope_id, false);
        builder.closure_base = closure_base_offset + all_closures.len() as u32;

        // Build the combined field map: inherited fields keep their parent indices;
        // only NEW fields (not present in parent) get fresh indices starting at
        // inherited_field_offset. This ensures subclass methods that reference
        // inherited fields use the correct slot indices.
        let mut fmap: HashMap<SymbolId, u16> = parent_field_map.clone();
        let mut own_idx: u16 = 0;
        for &sym in &field_names {
            if let std::collections::hash_map::Entry::Vacant(e) = fmap.entry(sym) {
                e.insert(inherited_field_offset + own_idx);
                own_idx += 1;
            }
        }
        builder.field_map = Some(fmap);
        builder.current_class_name = Some(decl.name.0);
        builder.current_method_sig = Some(method_name);
        builder.current_method_base_name = method_base_name(&method.signature);
        builder.current_method_is_static = method.is_static;

        // Bind 'this' as first block param (param index 0)
        let this_sym = builder.intern("this");
        let this_val = builder.emit(Instruction::BlockParam(0));
        builder.variables.insert(this_sym, this_val);

        // Emit BlockParams contiguously (registers 1..=N) before
        // allocating any List boxes, so argument register layout
        // matches the caller's Call instruction.
        let raw_params: Vec<(SymbolId, ValueId)> = params
            .iter()
            .enumerate()
            .map(|(i, param)| {
                let val = builder.emit(Instruction::BlockParam((i + 1) as u16));
                (param.0, val)
            })
            .collect();
        for (name, val) in raw_params {
            let stored = builder.box_if_captured(name, val);
            builder.variables.insert(name, stored);
        }

        // Lower the body
        builder.lower_stmt(method.body.as_ref().unwrap());

        // Ensure method returns something
        if matches!(
            builder.func.block(builder.current_block).terminator,
            Terminator::Unreachable
        ) {
            if is_constructor {
                // Constructors return 'this'
                builder.func.block_mut(builder.current_block).terminator =
                    Terminator::Return(this_val);
            } else {
                // Return last expression value if available (e.g. getters)
                // Check instructions first, then block params (for &&/|| merge blocks)
                let last_val = builder
                    .func
                    .block(builder.current_block)
                    .instructions
                    .last()
                    .map(|(id, _)| *id)
                    .or_else(|| {
                        builder
                            .func
                            .block(builder.current_block)
                            .params
                            .last()
                            .map(|(id, _)| *id)
                    });
                if let Some(val) = last_val {
                    builder.func.block_mut(builder.current_block).terminator =
                        Terminator::Return(val);
                } else {
                    builder.func.block_mut(builder.current_block).terminator =
                        Terminator::ReturnNull;
                }
            }
        }
        builder.func.compute_predecessors();

        // Collect closures from this method body
        all_closures.extend(builder.closures);

        methods.push(MethodMir {
            signature: sig_str,
            is_static: method.is_static,
            is_constructor,
            mir: builder.func,
            attributes: lower_runtime_attributes(&method.attributes, interner),
        });
    }

    // Compute protocol conformance from method signatures.
    let method_sigs: Vec<&str> = methods.iter().map(|m| m.signature.as_str()).collect();
    let superclass_protocols = crate::sema::protocol::ProtocolSet::EMPTY; // resolved at runtime
    let conformance =
        crate::sema::protocol::check_all_conformance(&method_sigs, superclass_protocols);

    (
        ClassMir {
            name: decl.name.0,
            superclass: decl.superclass.as_ref().map(|s| s.0),
            methods,
            num_fields: field_names
                .iter()
                .filter(|s| !parent_field_map.contains_key(s))
                .count() as u16,
            protocols: conformance.conforms,
            attributes: lower_runtime_attributes(&decl.attributes, interner),
            native_library: if decl.is_foreign {
                extract_compile_time_string(&decl.attributes, interner, "native")
            } else {
                None
            },
            foreign_methods,
        },
        all_closures,
    )
}

/// Extract a `#!key = "value"` string from an attribute list. Returns
/// `None` if no matching compile-time attribute exists or if the value
/// isn't a string literal. Used to pull build-time directives like
/// `#!native` and `#!symbol` that the runtime loader consumes directly
/// instead of exposing via reflection.
fn extract_compile_time_string(
    attrs: &[Attribute],
    interner: &Interner,
    key: &str,
) -> Option<String> {
    for attr in attrs {
        if attr.is_runtime {
            continue;
        }
        if interner.resolve(attr.name.0) != key {
            continue;
        }
        if let AttributeBody::Value((AttributeLiteral::Str(s), _)) = &attr.body {
            return Some(s.clone());
        }
    }
    None
}

/// Flatten AST attributes into MIR-side records, dropping compile-time
/// (`#!`) entries. Groups expand one entry per inner pair so the runtime
/// reflection builder sees a uniform `(group, key, value)` stream.
fn lower_runtime_attributes(attrs: &[Attribute], interner: &Interner) -> Vec<AttrEntry> {
    let mut out = Vec::new();
    for attr in attrs {
        if !attr.is_runtime {
            continue;
        }
        let outer = interner.resolve(attr.name.0).to_string();
        match &attr.body {
            AttributeBody::Flag => out.push(AttrEntry {
                group: None,
                key: outer,
                value: None,
            }),
            AttributeBody::Value((lit, _)) => out.push(AttrEntry {
                group: None,
                key: outer,
                value: Some(lower_attr_value(lit, interner)),
            }),
            AttributeBody::Group(pairs) => {
                for (key_sym, (lit, _)) in pairs {
                    out.push(AttrEntry {
                        group: Some(outer.clone()),
                        key: interner.resolve(key_sym.0).to_string(),
                        value: Some(lower_attr_value(lit, interner)),
                    });
                }
            }
        }
    }
    out
}

fn lower_attr_value(lit: &AttributeLiteral, interner: &Interner) -> AttrValue {
    match lit {
        AttributeLiteral::Num(n) => AttrValue::Num(*n),
        AttributeLiteral::Str(s) => AttrValue::Str(s.clone()),
        AttributeLiteral::Bool(b) => AttrValue::Bool(*b),
        AttributeLiteral::Null => AttrValue::Null,
        AttributeLiteral::Ident(sym) => AttrValue::Ident(interner.resolve(*sym).to_string()),
    }
}

/// Scan a statement body for field accesses (_name) and register them.
fn scan_fields(stmt: &Spanned<Stmt>, interner: &Interner, fields: &mut Vec<SymbolId>) {
    scan_stmt_fields(&stmt.0, interner, fields);
}

fn scan_stmt_fields(stmt: &Stmt, interner: &Interner, fields: &mut Vec<SymbolId>) {
    match stmt {
        Stmt::Expr(e) => scan_expr_fields(&e.0, interner, fields),
        Stmt::Var {
            initializer: Some(init),
            ..
        } => {
            scan_expr_fields(&init.0, interner, fields);
        }
        Stmt::Var {
            initializer: None, ..
        } => {}
        Stmt::Block(stmts) => {
            for s in stmts {
                scan_stmt_fields(&s.0, interner, fields);
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            scan_expr_fields(&condition.0, interner, fields);
            scan_stmt_fields(&then_branch.0, interner, fields);
            if let Some(eb) = else_branch {
                scan_stmt_fields(&eb.0, interner, fields);
            }
        }
        Stmt::While { condition, body } => {
            scan_expr_fields(&condition.0, interner, fields);
            scan_stmt_fields(&body.0, interner, fields);
        }
        Stmt::For { iterator, body, .. } => {
            scan_expr_fields(&iterator.0, interner, fields);
            scan_stmt_fields(&body.0, interner, fields);
        }
        Stmt::Return(Some(e)) => scan_expr_fields(&e.0, interner, fields),
        _ => {}
    }
}

fn scan_expr_fields(expr: &Expr, interner: &Interner, fields: &mut Vec<SymbolId>) {
    match expr {
        Expr::Field(name) if !fields.contains(name) => {
            fields.push(*name);
        }
        Expr::Assign { target, value } => {
            scan_expr_fields(&target.0, interner, fields);
            scan_expr_fields(&value.0, interner, fields);
        }
        Expr::BinaryOp { left, right, .. } | Expr::LogicalOp { left, right, .. } => {
            scan_expr_fields(&left.0, interner, fields);
            scan_expr_fields(&right.0, interner, fields);
        }
        Expr::UnaryOp { operand, .. } => scan_expr_fields(&operand.0, interner, fields),
        Expr::Call {
            receiver,
            args,
            block_arg,
            ..
        } => {
            if let Some(r) = receiver {
                scan_expr_fields(&r.0, interner, fields);
            }
            for a in args {
                scan_expr_fields(&a.0, interner, fields);
            }
            if let Some(b) = block_arg {
                scan_expr_fields(&b.0, interner, fields);
            }
        }
        Expr::Conditional {
            condition,
            then_expr,
            else_expr,
        } => {
            scan_expr_fields(&condition.0, interner, fields);
            scan_expr_fields(&then_expr.0, interner, fields);
            scan_expr_fields(&else_expr.0, interner, fields);
        }
        Expr::Is { value, type_name } => {
            scan_expr_fields(&value.0, interner, fields);
            scan_expr_fields(&type_name.0, interner, fields);
        }
        Expr::CompoundAssign { target, value, .. } => {
            scan_expr_fields(&target.0, interner, fields);
            scan_expr_fields(&value.0, interner, fields);
        }
        Expr::Subscript { receiver, args } => {
            scan_expr_fields(&receiver.0, interner, fields);
            for a in args {
                scan_expr_fields(&a.0, interner, fields);
            }
        }
        Expr::SubscriptSet {
            receiver,
            index_args,
            value,
        } => {
            scan_expr_fields(&receiver.0, interner, fields);
            for a in index_args {
                scan_expr_fields(&a.0, interner, fields);
            }
            scan_expr_fields(&value.0, interner, fields);
        }
        Expr::ListLiteral(elems) => {
            for e in elems {
                scan_expr_fields(&e.0, interner, fields);
            }
        }
        Expr::MapLiteral(entries) => {
            for (k, v) in entries {
                scan_expr_fields(&k.0, interner, fields);
                scan_expr_fields(&v.0, interner, fields);
            }
        }
        Expr::Range { from, to, .. } => {
            scan_expr_fields(&from.0, interner, fields);
            scan_expr_fields(&to.0, interner, fields);
        }
        Expr::Interpolation(parts) => {
            for p in parts {
                scan_expr_fields(&p.0, interner, fields);
            }
        }
        Expr::Closure { body, .. } => scan_stmt_fields(&body.0, interner, fields),
        Expr::SuperCall { args, .. } => {
            for a in args {
                scan_expr_fields(&a.0, interner, fields);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parser::parse;

    fn lower(source: &str) -> (MirFunction, Interner) {
        let mut result = parse(source);
        assert!(
            result.errors.is_empty(),
            "parse errors: {:?}",
            result.errors
        );
        let resolve_result = crate::sema::resolve::resolve(&result.module, &result.interner);
        let module_mir = lower_module(&result.module, &mut result.interner, &resolve_result);
        (module_mir.top_level, result.interner)
    }

    fn lower_full(source: &str) -> ModuleMir {
        let mut result = parse(source);
        assert!(
            result.errors.is_empty(),
            "parse errors: {:?}",
            result.errors
        );
        let resolve_result = crate::sema::resolve::resolve(&result.module, &result.interner);
        lower_module(&result.module, &mut result.interner, &resolve_result)
    }

    #[test]
    fn test_lower_class_attributes_runtime_only() {
        // `#runnable`  → runtime Flag
        // `#!internal` → compile-time, must be stripped
        // `#author = "Bob"` → runtime Value
        let module = lower_full("#runnable\n#!internal\n#author = \"Bob\"\nclass Foo {}");
        let class = &module.classes[0];
        assert_eq!(
            class.attributes.len(),
            2,
            "compile-time attr must be dropped"
        );
        assert_eq!(class.attributes[0].key, "runnable");
        assert!(class.attributes[0].value.is_none());
        assert_eq!(class.attributes[1].key, "author");
        matches!(class.attributes[1].value, Some(AttrValue::Str(_)));
    }

    #[test]
    fn test_lower_class_group_attribute_flattens() {
        let module =
            lower_full("#doc(brief = \"sum\", example = 42)\nclass Math {\n  static go() { 1 }\n}");
        let class = &module.classes[0];
        assert_eq!(class.attributes.len(), 2);
        assert_eq!(class.attributes[0].group.as_deref(), Some("doc"));
        assert_eq!(class.attributes[0].key, "brief");
        assert_eq!(class.attributes[1].group.as_deref(), Some("doc"));
        assert_eq!(class.attributes[1].key, "example");
    }

    #[test]
    fn test_lower_method_attributes_attached_to_method() {
        let source = "class C {\n  #pinned\n  foo() { 1 }\n}";
        let module = lower_full(source);
        let class = &module.classes[0];
        let method = class
            .methods
            .iter()
            .find(|m| m.signature == "foo()")
            .expect("foo() method");
        assert_eq!(method.attributes.len(), 1);
        assert_eq!(method.attributes[0].key, "pinned");
    }

    #[test]
    fn test_lower_native_library_on_foreign_class() {
        let source = "#!native = \"sqlite3\"\nforeign class Db {}";
        let module = lower_full(source);
        let class = &module.classes[0];
        assert_eq!(class.native_library.as_deref(), Some("sqlite3"));
    }

    #[test]
    fn test_lower_native_library_ignored_on_regular_class() {
        // `#!native` on a non-foreign class is meaningless — the loader
        // only consults it for foreign classes, so the MIR drops it.
        let source = "#!native = \"sqlite3\"\nclass Db {}";
        let module = lower_full(source);
        let class = &module.classes[0];
        assert!(class.native_library.is_none());
    }

    #[test]
    fn test_lower_foreign_methods_emitted() {
        let source = r#"
#!native = "sqlite3"
foreign class Db {
  #!symbol = "sqlite3_open_v2"
  foreign open(path)
  foreign close()
}
"#;
        let module = lower_full(source);
        let class = &module.classes[0];
        assert_eq!(class.methods.len(), 0, "no bodied methods");
        assert_eq!(class.foreign_methods.len(), 2);
        let open = class
            .foreign_methods
            .iter()
            .find(|m| m.signature == "open(_)")
            .expect("open(_)");
        assert_eq!(open.symbol.as_deref(), Some("sqlite3_open_v2"));
        let close = class
            .foreign_methods
            .iter()
            .find(|m| m.signature == "close()")
            .expect("close()");
        assert!(close.symbol.is_none(), "falls back to method name at load");
    }

    fn assert_has_instruction(func: &MirFunction, pred: impl Fn(&Instruction) -> bool) {
        for block in &func.blocks {
            for (_, inst) in &block.instructions {
                if pred(inst) {
                    return;
                }
            }
        }
        panic!("expected instruction not found in MIR");
    }

    #[test]
    fn test_lower_num() {
        let (func, _) = lower("42");
        assert_has_instruction(
            &func,
            |i| matches!(i, Instruction::ConstNum(n) if *n == 42.0),
        );
    }

    #[test]
    fn test_lower_string() {
        let (func, interner) = lower("\"hello\"");
        // ConstString now stores a SymbolId index (interned string)
        assert_has_instruction(&func, |i| {
            if let Instruction::ConstString(idx) = i {
                let sym = crate::intern::SymbolId::from_raw(*idx);
                interner.resolve(sym) == "hello"
            } else {
                false
            }
        });
    }

    #[test]
    fn test_lower_bool() {
        let (func, _) = lower("true");
        assert_has_instruction(&func, |i| matches!(i, Instruction::ConstBool(true)));
    }

    #[test]
    fn test_lower_null() {
        let (func, _) = lower("null");
        assert_has_instruction(&func, |i| matches!(i, Instruction::ConstNull));
    }

    #[test]
    fn test_lower_add() {
        let (func, _) = lower("1 + 2");
        // With type inference, Num + Num → Unbox + AddF64 + Box
        assert_has_instruction(&func, |i| matches!(i, Instruction::AddF64(..)));
    }

    #[test]
    fn test_lower_sub() {
        let (func, _) = lower("5 - 3");
        assert_has_instruction(&func, |i| matches!(i, Instruction::SubF64(..)));
    }

    #[test]
    fn test_lower_mul() {
        let (func, _) = lower("2 * 3");
        assert_has_instruction(&func, |i| matches!(i, Instruction::MulF64(..)));
    }

    #[test]
    fn test_lower_comparison() {
        let (func, _) = lower("1 < 2");
        assert_has_instruction(&func, |i| matches!(i, Instruction::CmpLtF64(..)));
    }

    #[test]
    fn test_lower_negation() {
        let (func, _) = lower("-42");
        assert_has_instruction(&func, |i| matches!(i, Instruction::Neg(..)));
    }

    #[test]
    fn test_lower_not() {
        let (func, _) = lower("!true");
        assert_has_instruction(&func, |i| matches!(i, Instruction::Not(..)));
    }

    #[test]
    fn test_lower_var_decl() {
        let (func, _) = lower("var x = 42");
        assert_has_instruction(
            &func,
            |i| matches!(i, Instruction::ConstNum(n) if *n == 42.0),
        );
    }

    #[test]
    fn test_lower_var_use() {
        // Module-level vars now use GetModuleVar/SetModuleVar instead of Move.
        let (func, _) = lower("var x = 1\nx");
        assert_has_instruction(&func, |i| matches!(i, Instruction::GetModuleVar(0)));
    }

    #[test]
    fn test_lower_if() {
        let (func, _) = lower("if (true) 1");
        assert!(func.blocks.len() >= 3);
        assert!(matches!(
            func.block(BlockId(0)).terminator,
            Terminator::CondBranch { .. }
        ));
    }

    #[test]
    fn test_lower_if_else() {
        let (func, _) = lower("if (true) 1 else 2");
        assert!(func.blocks.len() >= 4);
    }

    #[test]
    fn test_lower_while() {
        let (func, _) = lower("while (true) 1");
        assert!(func.blocks.len() >= 4);
    }

    #[test]
    fn test_lower_return() {
        let (func, _) = lower("return 42");
        assert!(matches!(
            func.block(BlockId(0)).terminator,
            Terminator::Return(..)
        ));
    }

    #[test]
    fn test_lower_return_null() {
        let (func, _) = lower("return");
        assert!(matches!(
            func.block(BlockId(0)).terminator,
            Terminator::ReturnNull
        ));
    }

    #[test]
    fn test_lower_logical_and() {
        let (func, _) = lower("true && false");
        assert!(func.blocks.len() >= 3);
    }

    #[test]
    fn test_lower_logical_or() {
        let (func, _) = lower("true || false");
        assert!(func.blocks.len() >= 3);
    }

    #[test]
    fn test_lower_conditional() {
        let (func, _) = lower("true ? 1 : 2");
        assert!(func.blocks.len() >= 4);
        let merge_found = func.blocks.iter().any(|b| !b.params.is_empty());
        assert!(merge_found);
    }

    #[test]
    fn test_lower_list() {
        let (func, _) = lower("[1, 2, 3]");
        assert_has_instruction(&func, |i| matches!(i, Instruction::MakeList(..)));
    }

    #[test]
    fn test_lower_map() {
        let (func, _) = lower("{\"a\": 1}");
        assert_has_instruction(&func, |i| matches!(i, Instruction::MakeMap(..)));
    }

    #[test]
    fn test_lower_range() {
        let (func, _) = lower("1..10");
        assert_has_instruction(&func, |i| matches!(i, Instruction::MakeRange(..)));
    }

    #[test]
    fn test_lower_method_call() {
        let (func, _) = lower("var x = null\nx.foo(1)");
        assert_has_instruction(&func, |i| matches!(i, Instruction::Call { .. }));
    }

    #[test]
    fn test_lower_subscript() {
        let (func, _) = lower("var x = null\nx[0]");
        assert_has_instruction(&func, |i| matches!(i, Instruction::SubscriptGet { .. }));
    }

    #[test]
    fn test_lower_is_type() {
        // `is` now always dispatches through `Object.is(_)` so it
        // works for both literal class idents and class-valued
        // variables. The static `IsType` opcode is kept for a
        // future devirt pass to re-introduce when the RHS is a
        // known core class.
        let (func, interner) = lower("var x = 1\nx is Num");
        let is_sym = interner.lookup("is(_)").expect("is(_) interned");
        assert_has_instruction(
            &func,
            |i| matches!(i, Instruction::Call { method, .. } if *method == is_sym),
        );
    }

    #[test]
    fn test_lower_pretty_print() {
        let (func, interner) = lower("var x = 1 + 2\nx");
        let output = func.pretty_print(&interner);
        assert!(output.contains("bb0"));
        assert!(output.contains("const.num"));
        assert!(output.contains("add"));
    }

    #[test]
    fn test_predecessors_computed() {
        let (func, _) = lower("if (true) 1 else 2");
        let has_preds = func.blocks.iter().any(|b| !b.predecessors.is_empty());
        assert!(has_preds);
    }

    #[test]
    fn test_module_var_indices() {
        let (func, _) = lower("var x = 1\nvar y = x");
        // x is module var 0, y is module var 1.
        // Should emit SetModuleVar(0, ...) for x, GetModuleVar(0) for the x reference,
        // and SetModuleVar(1, ...) for y.
        let mut set_indices = Vec::new();
        let mut get_indices = Vec::new();
        for block in &func.blocks {
            for (_, inst) in &block.instructions {
                match inst {
                    Instruction::SetModuleVar(idx, _) => set_indices.push(*idx),
                    Instruction::GetModuleVar(idx) => get_indices.push(*idx),
                    _ => {}
                }
            }
        }
        assert_eq!(
            set_indices,
            vec![0, 1],
            "SetModuleVar indices should be [0, 1]"
        );
        assert_eq!(
            get_indices,
            vec![0],
            "GetModuleVar index for x should be [0]"
        );
    }
}
