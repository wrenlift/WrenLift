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

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct MirBuilder<'a> {
    func: MirFunction,
    interner: &'a mut Interner,
    current_block: BlockId,
    variables: HashMap<SymbolId, ValueId>,
    break_targets: Vec<BlockId>,
    continue_targets: Vec<BlockId>,
}

impl<'a> MirBuilder<'a> {
    pub fn new(name: SymbolId, arity: u8, interner: &'a mut Interner) -> Self {
        let mut func = MirFunction::new(name, arity);
        let entry = func.new_block();
        Self {
            func,
            interner,
            current_block: entry,
            variables: HashMap::new(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
        }
    }

    pub fn build_module(mut self, module: &Module) -> MirFunction {
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
        self.func
    }

    pub fn build_body(mut self, body: &Spanned<Stmt>, params: &[Spanned<SymbolId>]) -> MirFunction {
        for param in params {
            let val = self.emit(Instruction::BlockParam(0));
            self.variables.insert(param.0, val);
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

    // -- Statement lowering -------------------------------------------------

    fn lower_stmt(&mut self, stmt: &Spanned<Stmt>) {
        match &stmt.0 {
            Stmt::Expr(expr) => {
                self.lower_expr(expr);
            }

            Stmt::Var { name, initializer } => {
                let val = if let Some(init) = initializer {
                    self.lower_expr(init)
                } else {
                    self.emit(Instruction::ConstNull)
                };
                self.variables.insert(name.0, val);
            }

            Stmt::Block(stmts) => {
                for s in stmts {
                    self.lower_stmt(s);
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
                if let Some(&target) = self.break_targets.last() {
                    self.set_terminator(Terminator::Branch {
                        target,
                        args: vec![],
                    });
                    let dead = self.new_block();
                    self.switch_to(dead);
                }
            }

            Stmt::Continue => {
                if let Some(&target) = self.continue_targets.last() {
                    self.set_terminator(Terminator::Branch {
                        target,
                        args: vec![],
                    });
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

        self.switch_to(then_bb);
        self.lower_stmt(then_branch);
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.set_terminator(Terminator::Branch {
                target: merge_bb,
                args: vec![],
            });
        }

        self.switch_to(else_bb);
        if let Some(else_stmt) = else_branch {
            self.lower_stmt(else_stmt);
        }
        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.set_terminator(Terminator::Branch {
                target: merge_bb,
                args: vec![],
            });
        }

        self.switch_to(merge_bb);
    }

    fn lower_while(&mut self, condition: &Spanned<Expr>, body: &Spanned<Stmt>) {
        let cond_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();

        self.set_terminator(Terminator::Branch {
            target: cond_bb,
            args: vec![],
        });

        self.switch_to(cond_bb);
        let cond_val = self.lower_expr(condition);
        self.set_terminator(Terminator::CondBranch {
            condition: cond_val,
            true_target: body_bb,
            true_args: vec![],
            false_target: exit_bb,
            false_args: vec![],
        });

        self.switch_to(body_bb);
        self.break_targets.push(exit_bb);
        self.continue_targets.push(cond_bb);
        self.lower_stmt(body);
        self.break_targets.pop();
        self.continue_targets.pop();

        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.set_terminator(Terminator::Branch {
                target: cond_bb,
                args: vec![],
            });
        }

        self.switch_to(exit_bb);
    }

    fn lower_for(
        &mut self,
        variable: &Spanned<SymbolId>,
        iterator: &Spanned<Expr>,
        body: &Spanned<Stmt>,
    ) {
        let iter_obj = self.lower_expr(iterator);

        let iterate_sym = self.intern("iterate");
        let iter_value_sym = self.intern("iteratorValue");

        let null_val = self.emit(Instruction::ConstNull);
        let iter_val = self.emit(Instruction::Call {
            receiver: iter_obj,
            method: iterate_sym,
            args: vec![null_val],
        });

        let cond_bb = self.new_block();
        let body_bb = self.new_block();
        let exit_bb = self.new_block();

        let iter_param = self.func.new_value();
        self.func
            .block_mut(cond_bb)
            .params
            .push((iter_param, MirType::Value));

        self.set_terminator(Terminator::Branch {
            target: cond_bb,
            args: vec![iter_val],
        });

        self.switch_to(cond_bb);
        let is_falsy = self.emit(Instruction::Not(iter_param));
        self.set_terminator(Terminator::CondBranch {
            condition: is_falsy,
            true_target: exit_bb,
            true_args: vec![],
            false_target: body_bb,
            false_args: vec![],
        });

        self.switch_to(body_bb);
        let elem_val = self.emit(Instruction::Call {
            receiver: iter_obj,
            method: iter_value_sym,
            args: vec![iter_param],
        });
        self.variables.insert(variable.0, elem_val);

        self.break_targets.push(exit_bb);
        self.continue_targets.push(cond_bb);
        self.lower_stmt(body);
        self.break_targets.pop();
        self.continue_targets.pop();

        let next_iter = self.emit(Instruction::Call {
            receiver: iter_obj,
            method: iterate_sym,
            args: vec![iter_param],
        });

        if matches!(
            self.func.block(self.current_block).terminator,
            Terminator::Unreachable
        ) {
            self.set_terminator(Terminator::Branch {
                target: cond_bb,
                args: vec![next_iter],
            });
        }

        self.switch_to(exit_bb);
    }

    // -- Expression lowering ------------------------------------------------

    fn lower_expr(&mut self, expr: &Spanned<Expr>) -> ValueId {
        match &expr.0 {
            Expr::Num(n) => self.emit(Instruction::ConstNum(*n)),
            Expr::Str(s) => {
                let idx = self.func.add_string(s.clone());
                self.emit(Instruction::ConstString(idx))
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
                    self.emit(Instruction::Move(val))
                } else {
                    self.emit(Instruction::GetModuleVar(0))
                }
            }

            Expr::Field(_) => self.emit(Instruction::GetField(ValueId(0), 0)),
            Expr::StaticField(_) => self.emit(Instruction::GetModuleVar(0)),

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

            Expr::LogicalOp { op, left, right } => self.lower_logical(*op, left, right),

            Expr::Is { value, type_name } => {
                let val = self.lower_expr(value);
                let class_sym = match &type_name.0 {
                    Expr::Ident(sym) => *sym,
                    _ => self.intern("<unknown>"),
                };
                self.emit(Instruction::IsType(val, class_sym))
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
            } => {
                let recv = if let Some(r) = receiver {
                    self.lower_expr(r)
                } else {
                    let this_sym = self.intern("this");
                    self.variables
                        .get(&this_sym)
                        .copied()
                        .unwrap_or_else(|| self.emit(Instruction::ConstNull))
                };

                let mut arg_vals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                if let Some(block) = block_arg {
                    let block_val = self.lower_expr(block);
                    arg_vals.push(block_val);
                }

                self.emit(Instruction::Call {
                    receiver: recv,
                    method: method.0,
                    args: arg_vals,
                })
            }

            Expr::SuperCall { method, args } => {
                let arg_vals: Vec<ValueId> = args.iter().map(|a| self.lower_expr(a)).collect();
                let method_sym = method
                    .as_ref()
                    .map(|m| m.0)
                    .unwrap_or_else(|| self.intern("init"));
                self.emit(Instruction::SuperCall {
                    method: method_sym,
                    args: arg_vals,
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

            Expr::Range { from, to, inclusive } => {
                let from_val = self.lower_expr(from);
                let to_val = self.lower_expr(to);
                self.emit(Instruction::MakeRange(from_val, to_val, *inclusive))
            }

            Expr::Closure { .. } => self.emit(Instruction::MakeClosure {
                fn_id: 0,
                upvalues: vec![],
            }),
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
                self.variables.insert(*name, value);
            }
            Expr::Field(_) => {
                self.emit(Instruction::SetField(ValueId(0), 0, value));
            }
            _ => {}
        }
    }
}

/// Convenience: lower a parsed module to MIR.
// TODO: accept a module name derived from the file path / namespace once
// we support multiple modules, instead of the hardcoded "<module>" sentinel.
pub fn lower_module(module: &Module, interner: &mut Interner) -> MirFunction {
    let name = interner.intern("<module>");
    let builder = MirBuilder::new(name, 0, interner);
    builder.build_module(module)
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
        assert!(result.errors.is_empty(), "parse errors: {:?}", result.errors);
        let func = lower_module(&result.module, &mut result.interner);
        (func, result.interner)
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
        assert_has_instruction(&func, |i| matches!(i, Instruction::ConstNum(n) if *n == 42.0));
    }

    #[test]
    fn test_lower_string() {
        let (func, _) = lower("\"hello\"");
        assert_has_instruction(&func, |i| matches!(i, Instruction::ConstString(0)));
        assert_eq!(func.strings[0], "hello");
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
        assert_has_instruction(&func, |i| matches!(i, Instruction::Add(..)));
    }

    #[test]
    fn test_lower_sub() {
        let (func, _) = lower("5 - 3");
        assert_has_instruction(&func, |i| matches!(i, Instruction::Sub(..)));
    }

    #[test]
    fn test_lower_mul() {
        let (func, _) = lower("2 * 3");
        assert_has_instruction(&func, |i| matches!(i, Instruction::Mul(..)));
    }

    #[test]
    fn test_lower_comparison() {
        let (func, _) = lower("1 < 2");
        assert_has_instruction(&func, |i| matches!(i, Instruction::CmpLt(..)));
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
        assert_has_instruction(&func, |i| matches!(i, Instruction::ConstNum(n) if *n == 42.0));
    }

    #[test]
    fn test_lower_var_use() {
        let (func, _) = lower("var x = 1\nx");
        assert_has_instruction(&func, |i| matches!(i, Instruction::Move(..)));
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
        let (func, _) = lower("var x = 1\nx is Num");
        assert_has_instruction(&func, |i| matches!(i, Instruction::IsType(..)));
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
}
